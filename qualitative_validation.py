###############################################################################
# Author/Developer: Nicolas CHEN                                              #
# Filename: plotBox.py                                                        #
# Version: 1.0                                                                #
# Field of research: Deep Learning in medical imaging                         #
# Purpose: This Python script plots the boxes for each image from the dataset #
# Output: This Python script plots the boxes for each image and save it in    #
# a new directory                                                             #
###############################################################################

import os
from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd


def qualitative_validation():
    VAL_PATH = "./big_geo_data/classified_images/"

    def filterFiles(directoryPath, extension):
        relevant_path = directoryPath
        included_extensions = [extension]
        file_names = [
            file1
            for file1 in os.listdir(relevant_path)
            if any(file1.endswith(ext) for ext in included_extensions)
        ]
        numberOfFiles = len(file_names)
        listParams = [file_names, numberOfFiles]
        return listParams

    [image_names, _] = filterFiles(VAL_PATH, "jpg")

    trainRCNN = pd.read_csv("validation_results/val_labels.csv", sep=",")
    trainRCNN.columns = [
        "filename",
        "box_type",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
    ]
    trainRCNN["filename"] = trainRCNN["filename"].str.replace(".tif", ".jpg")
    print(trainRCNN)

    for imageFileName in image_names:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis("off")

        image = plt.imread(VAL_PATH + imageFileName)

        plt.imshow(image)

        iteration = trainRCNN[trainRCNN.filename == imageFileName].iterrows()
        for _, row in iteration:
            xmin = float(row.xmin)
            xmax = float(row.xmax)
            ymin = float(row.ymin)
            ymax = float(row.ymax)

            width = xmax - xmin
            height = ymax - ymin
            rect = None

            if not row.box_type:
                ax.annotate(
                    "False",
                    xy=(xmax - 40, ymin + 20),
                    fontsize=6,
                    color="green",
                )

                rect = patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    edgecolor="green",
                    facecolor="none",
                    linewidth=5,
                )
            elif row.box_type:
                ax.annotate(
                    "True",
                    xy=(xmax - 40, ymin + 20),
                    fontsize=6,
                    color="green",
                )
                rect = patches.Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    edgecolor="green",
                    facecolor="none",
                    linewidth=5,
                )
            else:
                print("nothing")

            try:
                ax.add_patch(rect)
            except Exception as e:
                raise e

            if not os.path.exists("validation_qualitative"):
                os.makedirs("validation_qualitative")

            img_name = os.path.splitext(imageFileName)[0]
            full_path = f"validation_qualitative/{img_name}.jpg"
            fig.savefig(
                full_path,
                dpi=90,
                bbox_inches="tight",
            )

        plt.close()
        print(f"ImageName: {imageFileName} is saved to validation_qualitative")

    print("PLOTBOX COMPLETED!")