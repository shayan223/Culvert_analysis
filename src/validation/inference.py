"""
Used https://colab.research.google.com/drive/1Exoc3-A141_h8GKk-B6cJxoidJsgOZOZ?usp=sharing#scrollTo=anxv6VHFlQAK
for reference.
"""

import glob as glob
import os

import cv2
import numpy as np
import pandas as pd
import torch


from src.config import (
    CLASSES,
    DETECTION_THRESHOLD,
    INFER_FALSE_LABELS,
    NUM_CLASSES,
    NUM_QUERIES,
    NUM_EPOCHS,
    TEST_DIR,
    MODEL_OUT_DIR,
    CLASSIFIED_IMAGES_DIR,
    VALIDATION_RESULTS_DIR as VAL_RES_DIR,
)
from src.model import Model


def inference(args):
    """
    Perform inference on the test images.

    Args:
        - args: command line arguments (argparse).
    """

    # create directories
    if not os.path.exists(CLASSIFIED_IMAGES_DIR):
        os.makedirs(CLASSIFIED_IMAGES_DIR)
    if not os.path.exists(VAL_RES_DIR):
        os.makedirs(VAL_RES_DIR)

    # create model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Model(args.backbone, num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)

    # load model
    checkpoint = torch.load(
        os.path.join(MODEL_OUT_DIR, f"model{NUM_EPOCHS}.pth"),
        map_location=device,
    )
    model.model.load_state_dict(checkpoint, strict=False)
    model.model.eval()

    # get test images
    test_images = glob.glob(f"{TEST_DIR}/*")

    # create dataframe to store validation results
    col_names = ["file_name", "labels", "centroids", "x1", "x2", "y1", "y2"]
    validation_results = pd.DataFrame(columns=col_names)

    # iterate over test images
    for i, test_image in enumerate(test_images):
        # get image name
        image_name = test_image.split("/")[-1].split(".")[0]

        # add empty row to dataframe
        validation_results.loc[validation_results.shape[0]] = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        # get image name
        validation_results["file_name"][i] = image_name
        print(f"LOADING: {image_name}")

        # load image and normalize
        image = cv2.imread(test_image)
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)

        # propagate through the model
        with torch.no_grad():
            outputs, targets = model.model(image)

        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []

        centroids_list = []
        labels_list = []

        # iterate over predictions
        pred_logits = outputs["pred_logits"][0][:, :len(CLASSES)]
        pred_boxes = outputs["pred_boxes"][0]

        # keep only predictions with confidence above threshold
        max_output = pred_logits.softmax(-1).max(-1)
        topk = max_output.values.topk(15)

        # iterate over topk predictions
        pred_logits = pred_logits[topk.indices]
        pred_boxes = pred_boxes[topk.indices]

        # iterate over predictions
        for logits, box in zip(pred_logits, pred_boxes):
            # skip predictions with confidence below threshold
            cls = logits.argmax()
            if cls >= len(CLASSES):
                continue

            # get label
            label = CLASSES[cls]

            # skip false labels
            if label == "False" and not INFER_FALSE_LABELS:
                print("Skipping False Label!")
                continue

            # get bounding box coordinates
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # draw bounding box
            cv2.rectangle(
                orig_image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                16,
            )

            # draw label
            cv2.putText(
                orig_image,
                CLASSES[cls],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
            labels_list.append(CLASSES[cls])
            centroids_list.append(((x1 + x2) / 2, (y1 + y2) / 2))

        # show image
        cv2.imshow("Prediction", orig_image)
        cv2.waitKey(0)

        # new image path
        output_image_path = os.path.join(
            os.getcwd(),
            str(os.path.basename(CLASSIFIED_IMAGES_DIR)),
            os.path.split(image_name)[1] + ".jpg",
        )

        # output image path
        print(output_image_path)

        # save image
        print(f"Save complete: {cv2.imwrite(output_image_path, orig_image)}")

        validation_results["x1"][i] = x1_list
        validation_results["y1"][i] = y1_list
        validation_results["x2"][i] = x2_list
        validation_results["y2"][i] = y2_list
        validation_results["centroids"][i] = centroids_list
        validation_results["labels"][i] = labels_list

        print(f"Image {i + 1} done.")
        print("-" * 50)

        break

    validation_results.to_csv(f"{VAL_RES_DIR}/validation_results.csv")

    print("TEST PREDICTIONS COMPLETE")
    cv2.destroyAllWindows()