"""
NOTE: Relative paths are used in this file. These paths are relative to the root
directory. If you intend on running This code from this file specifically, you
must change them to go back a directory (ie. ../outputs/model rather than
./outputs/model)
"""

import glob as glob
import os

import cv2
import numpy as np
import pandas as pd
import torch

from config import (
    CLASSES,
    DETECTION_THRESHOLD,
    INFER_FALSE_LABELS,
    NUM_CLASSES,
    NUM_EPOCHS,
)
from model import create_model


def inference():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model().to(device)
    model.load_state_dict(
        torch.load("./outputs/model" + str(NUM_EPOCHS) + ".pth", map_location=device)
    )
    model.eval()

    DIR_TEST = "./big_geo_data/val"
    OUT_DIR = "./big_geo_data/classified_images"
    test_images = glob.glob(f"{DIR_TEST}/*")
    print(f"Test instances: {len(test_images)}")

    detection_threshold = DETECTION_THRESHOLD

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
        print("LOADING:", test_images[i])

        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(float)
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image)

        print(outputs)

        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []

        centroids_list = []
        labels_list = []

        if len(outputs["boxes"]) != 0:
            boxes = outputs["boxes"].data.numpy()
            scores = outputs["scores"].data.numpy()
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            pred_classes = [
                CLASSES[i] for i in outputs["labels"].cpu().numpy().astype(int)
            ]

            for j, box in enumerate(draw_boxes):
                print(pred_classes[j])
                if pred_classes[j] == "False" and INFER_FALSE_LABELS == False:
                    print("Skipping False Label!")
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

            print(
                os.path.join(
                    os.getcwd(),
                    str(os.path.basename(OUT_DIR)),
                    os.path.split(image_name)[1] + ".jpg",
                )
            )

            print(
                "Save complete: ",
                cv2.imwrite(
                    os.path.join(
                        os.getcwd(),
                        OUT_DIR.lstrip("./"),
                        os.path.split(image_name)[1] + ".jpg",
                    ),
                    orig_image,
                ),
            )

        validation_results["x1"][i] = x1_list
        validation_results["y1"][i] = y1_list
        validation_results["x2"][i] = x2_list
        validation_results["y2"][i] = y2_list
        validation_results["centroids"][i] = centroids_list
        validation_results["labels"][i] = labels_list

        print(f"Image {i + 1} done...")
        print("-" * 50)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists("./validation_results/"):
        os.makedirs("./validation_results/")
    validation_results.to_csv("./validation_results/validation_results.csv")

    print("TEST PREDICTIONS COMPLETE")
    cv2.destroyAllWindows()