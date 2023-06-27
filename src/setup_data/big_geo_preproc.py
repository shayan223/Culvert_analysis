import os

import numpy as np
import pandas as pd
from pascal_voc_writer import Writer

from src import config


def big_geo_preproc():
    out_dir = config.ANNOTATIONS_LOCATION
    df = pd.read_csv(config.COORDINATES_BBOX_LOCATION)

    BOUND_SIZE = 50

    image_height = 800
    image_width = 800

    df["Culvert Local Y"] = 800 - df["Culvert Local Y"]

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
            out_path = os.path.join(out_dir, f"{id}.xml")
            writer.save(out_path)

    df.apply(
        lambda x: generate_xml(
            f"{config.SAMPLES800_NORM_LOCATION}{x['Sample ID']}",
            os.path.splitext(str(x["Sample ID"]))[0],
            x["Culvert Local X"],
            x["Culvert Local Y"],
            BOUND_SIZE,
        ),
        axis=1,
    )
