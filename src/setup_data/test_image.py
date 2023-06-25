import cv2

from src import config

img = cv2.imread(f"{TRAIN_DIR}/1001.tif", -1) / 255

cv2.imshow("test_image", img)
cv2.waitKey(0)