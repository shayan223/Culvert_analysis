###############################################################################
# Author/Developer: Nicolas CHEN                                              #
# Filename: plotBox.py                                                        #
# Version: 1.0                                                                #
# Field of research: Deep Learning in medical imaging                         #
# Purpose: This Python script plots the boxes for each image from the dataset #
# Output: This Python script plots the boxes for each image and save it in a  #
# new directory                                                               #
###############################################################################

import os

from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd


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


[image_names, numberOfFiles] = filterFiles("ImageSets", "tif")

trainRCNN = pd.read_csv("test.csv", sep=",", header=None)
trainRCNN.columns = ["filename", "box_type", "xmin", "xmax", "ymin", "ymax"]


for imageFileName in image_names:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis("off")

    image = plt.imread("Images/" + imageFileName)
    plt.imshow(image)

    for _, row in trainRCNN[trainRCNN.filename == imageFileName].iterrows():
        xmin = float(row.xmin)
        xmax = float(row.xmax)
        ymin = float(row.ymin)
        ymax = float(row.ymax)

        width = xmax - xmin
        height = ymax - ymin
        ClassName = row.box_type

        rect = None

        if row.box_type == "False":
            ax.annotate("False", xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle(
                (xmin, ymin), width, height, edgecolor="r", facecolor="none"
            )
        elif row.box_type == "True":
            ax.annotate("True", xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle(
                (xmin, ymin), width, height, edgecolor="b", facecolor="none"
            )
        else:
            print("nothing")

        try:
            ax.add_patch(rect)
        except Exception as e:
            raise e

        if not os.path.exists("imagesBox"):
            os.makedirs("imagesBox")

        fig.savefig(
            "imagesBox/" + os.path.splitext(imageFileName)[0] + ".jpg",
            dpi=90,
            bbox_inches="tight",
        )
    plt.close()
    print("ImageName: " + imageFileName + " is saved in imagesBox folder")

print("PLOTBOX COMPLETED!")
