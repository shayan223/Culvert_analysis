import random
import cv2
from imutils import contours
import os



'''Source: https://stackoverflow.com/questions/55782857/how-to-detect-separate-figures-in-an-image'''
def drawAllBoundingBox(image_path):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    canny = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image

    # Find contours
    contour_list = []
    ROI_number = 0
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")
    for c in cnts:
        # Obtain bounding rectangle for each contour
        x,y,w,h = cv2.boundingRect(c)

        # Find ROI of the contour
        roi = img[y:y+h, x:x+w]

        # Draw bounding box rectangle, crop using Numpy slicing
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        ROI = original[y:y+h, x:x+w]
        #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        contour_list.append(c)
        ROI_number += 1

    print('Contours Detected: {}'.format(len(contour_list)))
    #cv2.imshow("image", img)
    #cv2.imshow("canny", canny)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(os.getcwd(), 'geo_data', 'bounded_images_T', os.path.split(image_path)[1]), img)


def drawBiggestBoundingBox(image_path):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    # Canny Edge Detection
    canny = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image

    # Find contours
    contour_list = []
    ROI_number = 0
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    #Hold on to largest 2 bounding boxes
    largest_vals1 = (0,0,0,0)
    largest_vals2 = (0,0,0,0)
    for c in cnts:
        # Obtain bounding rectangle for each contour
        x,y,w,h = cv2.boundingRect(c)
        if(w*h > largest_vals1[2]*largest_vals1[3]):
            #Demote previous largest and insert new largest
            largest_vals2 = largest_vals1
            largest_vals1 = (x,y,w,h)

        print(w,h)

    topNBoxes = [largest_vals1,largest_vals2]
    for i in topNBoxes:
        x,y,w,h = i
        # Find ROI of the contour
        roi = img[y:y+h, x:x+w]

        # Draw bounding box rectangle, crop using Numpy slicing
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        ROI = original[y:y+h, x:x+w]
        #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        contour_list.append(c)
        ROI_number += 1

    print('Contours Detected: {}'.format(len(contour_list)))
    #cv2.imshow("image", img)
    #cv2.imshow("canny", canny)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(os.getcwd(), 'geo_data', 'bounded_images_T_Biggest', os.path.split(image_path)[1]), img)



'''################################################################'''



# assign directory
directory = './geo_data/preprocessing_data/T/'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        #drawAllBoundingBox(f)
        drawBiggestBoundingBox(f)


