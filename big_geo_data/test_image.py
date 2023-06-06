import cv2


img = cv2.imread("./train/1001.tif", -1)
print(img)

print(img.shape)
img = img / 255
print(img)

cv2.imshow("test_image", img)
cv2.waitKey(0)