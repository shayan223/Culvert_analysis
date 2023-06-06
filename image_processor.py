"""
Source:
    https://stackoverflow.com/questions/55782857/how-to-detect-separate-figures-in-an-image
"""

import os

import cv2
from imutils import contours


def drawAllBoundingBox(image_path):
    img = cv2.imread(image_path)
    # original = img.copy()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    canny = cv2.Canny(
        image=img_gray, threshold1=100, threshold2=200
    )

    contour_list = []
    ROI_number = 0
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # roi = img[y : y + h, x : x + w]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # ROI = original[y : y + h, x : x + w]
        #
        contour_list.append(c)

        ROI_number += 1

    print("Contours Detected: {}".format(len(contour_list)))

    cv2.imwrite(
        os.path.join(
            os.getcwd(), "geo_data", "bounded_images_T", os.path.split(image_path)[1]
        ),
        img,
    )


def drawBiggestBoundingBox(image_path):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    canny = cv2.Canny(
        image=img_gray, threshold1=100, threshold2=200
    )

    contour_list = []
    ROI_number = 0
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    largest_vals1 = (0, 0, 0, 0)
    largest_vals2 = (0, 0, 0, 0)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > largest_vals1[2] * largest_vals1[3]:
            largest_vals2 = largest_vals1
            largest_vals1 = (x, y, w, h)

        print(w, h)

    topNBoxes = [largest_vals1, largest_vals2]
    for i in topNBoxes:
        x, y, w, h = i
        # roi = img[y : y + h, x : x + w]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # ROI = original[y : y + h, x : x + w]

        contour_list.append(c)

        ROI_number += 1

    print("Contours Detected: {}".format(len(contour_list)))
    cv2.imwrite(
        os.path.join(
            os.getcwd(),
            "geo_data",
            "bounded_images_T_Biggest",
            os.path.split(image_path)[1],
        ),
        img,
    )


directory = "./geo_data/preprocessing_data/T/"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        print(f)
        drawBiggestBoundingBox(f)