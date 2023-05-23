"""
Based on the following tutorial:
    https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
"""

import torch

is_available = torch.cuda.is_available()
DEVICE = torch.device("cuda") if is_available else torch.device("cpu")

BATCH_SIZE = 4
RESIZE_TO = 800
NUM_EPOCHS = 1

TRAIN_DIR = "./big_geo_data/train"
VALID_DIR = "./big_geo_data/test"

IMAGE_TYPE = "tif"
CLASSES = ["Background", "True"]
NUM_CLASSES = len(CLASSES)
VISUALIZE_TRANSFORMED_IMAGES = False
OUT_DIR = "./outputs"
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2


IMAGE_SPLIT_DIM = (400)
INFER_FALSE_LABELS = False
DETECTION_THRESHOLD = 0.8
