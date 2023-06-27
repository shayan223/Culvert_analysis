import os

import pandas as pd

from src import config

csv_file_name = config.COORDINATES_BBOX_LOCATION


def xlsx_to_csv(datafile):
    if os.path.isfile(datafile):
        df = pd.read_excel(datafile, index_col=[0])
        df.to_csv(csv_file_name, sep=",")


def rename_files():
    folders_path = config.SAMPLES800_NORM_LOCATION
    for directname, _, files in os.walk(folders_path):
        for f in files:
            filename, ext = os.path.splitext(f)
            if "." in filename:
                new_name = filename.replace(".", "_")
                os.rename(
                    os.path.join(directname, f),
                    os.path.join(directname, new_name + ext),
                )

    xlsx_to_csv(config.COORDINATES_BBOX_XLS_LOCATION)
    df = pd.read_csv(csv_file_name)
    df.to_csv(csv_file_name)