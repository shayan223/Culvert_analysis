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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Model(args.backbone, num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(device)
    model.model.load_state_dict(
        torch.load(
            os.path.join(MODEL_OUT_DIR, f"model{NUM_EPOCHS}.pth"),
            map_location=device,
        )
    )
    model.model.eval()

    test_images = glob.glob(f"{TEST_DIR}/*")
    # print(f"Test instances: {len(test_images)}")

    col_names = ["file_name", "labels", "centroids", "x1", "x2", "y1", "y2"]
    validation_results = pd.DataFrame(columns=col_names)

    for i in range(len(test_images)):
        image_name = test_images[i].split("/")[-1].split(".")[0]

        validation_results.loc[validation_results.shape[0]] = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        validation_results["file_name"][i] = image_name
        # print("LOADING:", test_images[i])

        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model.model(image)

        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []

        centroids_list = []
        labels_list = []

        if len(outputs["boxes"]) != 0:
            boxes = outputs["boxes"].data.numpy()
            scores = outputs["scores"].data.numpy()
            boxes = boxes[scores >= DETECTION_THRESHOLD].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [
                CLASSES[i] for i in outputs["labels"].cpu().numpy().astype(int)
            ]

            for j, box in enumerate(draw_boxes):
                if pred_classes[j] == "False" and INFER_FALSE_LABELS == False:
                    # print("Skipping False Label!")
                    continue

                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 0, 255),
                    16,
                )

                cv2.putText(
                    orig_image,
                    pred_classes[j],
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
                labels_list.append(pred_classes[j])
                centroids_list.append(((x1 + x2) / 2, (y1 + y2) / 2))

            cv2.imshow("Prediction", orig_image)
            cv2.waitKey(1)

            # print(
            #     os.path.join(
            #         os.getcwd(),
            #         str(os.path.basename(CLASSIFIED_IMAGES_DIR)),
            #         os.path.split(image_name)[1] + ".jpg",
            #     )
            # )

            # print(
            #     "Save complete: ",
            #     cv2.imwrite(
            #         os.path.join(
            #             os.getcwd(),
            #             CLASSIFIED_IMAGES_DIR.lstrip("./"),
            #             os.path.split(image_name)[1] + ".jpg",
            #         ),
            #         orig_image,
            #     ),
            # )

        validation_results["x1"][i] = x1_list
        validation_results["y1"][i] = y1_list
        validation_results["x2"][i] = x2_list
        validation_results["y2"][i] = y2_list
        validation_results["centroids"][i] = centroids_list
        validation_results["labels"][i] = labels_list

        # print(f"Image {i + 1} done...")
        # print("-" * 50)

    if not os.path.exists(CLASSIFIED_IMAGES_DIR):
        os.makedirs(CLASSIFIED_IMAGES_DIR)
    if not os.path.exists(VAL_RES_DIR):
        os.makedirs(VAL_RES_DIR)

    validation_results.to_csv(f"{VAL_RES_DIR}/validation_results.csv")

    # print("TEST PREDICTIONS COMPLETE")
    cv2.destroyAllWindows()