import os
import shutil

import pandas as pd

from src import config


def seperate_files():
    IMAGE_DIR = config.SAMPLES800_NORM_LOCATION

    def seperate_images(file_path, source, destination, validation=False):
        file_list = pd.read_csv(file_path, dtype=str, names=["file"])
        file_list = file_list["file"].tolist()

        if not os.path.exists(destination):
            os.makedirs(destination)

        if validation:
            if not os.path.exists(f"{destination}_xml"):
                os.makedirs(f"{destination}_xml")

        for filename in os.listdir(source):
            current_file = os.path.splitext(filename)[0]

            if current_file in file_list:
                filename = os.path.join(source, filename)
                annotation_file = os.path.join(
                    config.ANNOTATIONS_LOCATION,
                    f"{current_file}.xml",
                )

                os.path.join(source, filename)
                shutil.copy(filename, destination)

                if not validation:
                    shutil.copy(annotation_file, destination)
                else:
                    shutil.copy(annotation_file, f"{destination}_xml")

    image_sets = config.IMAGE_SETS_LOCATION
    data_root = config.DATA_ROOT
    # seperate_images(f"{image_sets}/train.txt", IMAGE_DIR, f"{data_root}/train")
    # seperate_images(f"{image_sets}/test.txt", IMAGE_DIR, f"{data_root}/test")
    # seperate_images(f"{image_sets}/trainval.txt", IMAGE_DIR, f"{data_root}/trainval")
    seperate_images(f"{image_sets}/val.txt", IMAGE_DIR, f"{data_root}/val", validation=True)