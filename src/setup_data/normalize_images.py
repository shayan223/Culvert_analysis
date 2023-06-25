import os

from PIL import Image
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from src import config

def normalize_images():
    PATH = config.DATA_PATH
    OUT_PATH = config.SAMPLES800_NORM_LOCATION
    FILE_TYPE = ".tif"

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    def normalize_image(img_path, out_path):
        image = imageio.imread(img_path)
        array = np.array(image)
        normalized = (
            (array.astype(np.uint16) - array.min())
            * 255.0
            / (array.max() - array.min())
        )

        image = np.array(Image.fromarray(normalized.astype(np.uint8)))

        image = Image.fromarray(image)
        image.save(out_path, "TIFF")

        return image

    file_list = os.listdir(PATH)

    for filename in tqdm(file_list):
        if filename.endswith(FILE_TYPE):
            normalize_image(
                os.path.join(PATH, filename),
                os.path.join(OUT_PATH, filename),
            )