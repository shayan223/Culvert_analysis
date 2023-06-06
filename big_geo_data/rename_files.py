import os

import pandas as pd

csv_file_name = "./coordinates_Bbox.csv"


def xlsx_to_csv(datafile):
    if os.path.isfile(datafile):
        df = pd.read_excel(datafile, index_col=[0])
        df.to_csv(csv_file_name, sep=",")


def rename_files():
    folders_path = "./Sample800"
    for directname, _, files in os.walk(folders_path):
        for f in files:
            filename, ext = os.path.splitext(f)
            if "." in filename:
                new_name = filename.replace(".", "_")
                os.rename(
                    os.path.join(directname, f),
                    os.path.join(directname, new_name + ext),
                )

    xlsx_to_csv("./data/coordinate in Bbox_Sept24.xlsx")
    df = pd.read_csv(csv_file_name)
    df.to_csv(csv_file_name)