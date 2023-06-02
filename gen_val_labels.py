from glob import glob
import xml.etree.ElementTree as ET

import pandas as pd


def gen_val_labels():
    annotations = glob("./big_geo_data/val_xml/*.xml")

    df = []
    cnt = 0
    for file in annotations:
        filename = file.split("\\")[-1]
        filename = filename.split(".")[0] + ".tif"

        row = []
        parsedXML = ET.parse(file)

        for node in parsedXML.getroot().iter("object"):
            box_type = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)

            row = [filename, box_type, xmin, xmax, ymin, ymax]
            df.append(row)
            cnt += 1

    data = pd.DataFrame(
        df, columns=["filename", "box_type", "xmin", "xmax", "ymin", "ymax"]
    )

    data[["filename", "box_type", "xmin", "xmax", "ymin", "ymax"]].to_csv(
        "./validation_results/val_labels.csv", index=False
    )