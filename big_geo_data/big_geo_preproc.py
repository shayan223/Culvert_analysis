import numpy as np
import pandas as pd
import os
from pascal_voc_writer import Writer


def big_geo_preproc():
    out_dir = "./annotations"
    df = pd.read_csv("./coordinates_Bbox.csv")

    BOUND_SIZE = 50
    image_height = 800
    image_width = 800
    df["Culvert Local Y"] = 800 - df["Culvert Local Y"]

    print(df)

    df = df.groupby(["Sample ID"]).aggregate(lambda x: list(x)).reset_index()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    def generate_xml(path, id, x, y, n):
        writer = Writer(path, image_width, image_height)
        for i in range(len(x)):
            xmin = int(np.clip(x[i] - n, 0, image_width))
            ymin = int(np.clip(y[i] - n, 0, image_height))
            xmax = int(np.clip(x[i] + n, 0, image_width))
            ymax = int(np.clip(y[i] + n, 0, image_height))

            writer.addObject("True", xmin, ymin, xmax, ymax)
            writer.save(out_dir + "/" + str(id) + ".xml")

    df.apply(
        lambda x: generate_xml(
            "./Sample800_norm/" + str(x["Sample ID"]),
            os.path.splitext(str(x["Sample ID"]))[0],
            x["Culvert Local X"],
            x["Culvert Local Y"],
            BOUND_SIZE,
        ),
        axis=1,
    )