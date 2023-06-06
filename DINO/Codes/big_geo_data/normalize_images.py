import os

import numpy as np
import os
from PIL import Image
# import imageio as imageio
import imageio
from tqdm import tqdm
import cv2

def normalize_images():
    PATH = '/mnt/sdb1/udaykanth/DETR/CA'
    # OUT_PATH = './Sample800_norm'
    OUT_PATH = '/mnt/sdb1/udaykanth/DETR/Sample800_norm'
    FILE_TYPE = '.tif'

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)


    def apply_sobel(image,RGB=False):
        #Apply sobel filter to refine edges
        if(RGB == False):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

    def normalize_image(img_path,out_path):
        image = imageio.imread(img_path)  
        # image = cv2.imread(img_path)    
        array = np.array(image)
        normalized = (array.astype(np.uint16) - array.min()) * 255.0 / (array.max() - array.min())

        image = np.array(Image.fromarray(normalized.astype(np.uint8)))

        #image = apply_sobel(image)

        # convert back to PIL for saving
        image = Image.fromarray(image)
        image.save(out_path, "TIFF")
        return image


    file_list = os.listdir(PATH)

    for filename in tqdm(file_list):
        if filename.endswith(FILE_TYPE):
            # Normalise to range 0..255
            norm = normalize_image(PATH+'/'+filename,OUT_PATH+'/'+filename)

            #print(converted_image.save(OUT_PATH+filename))


