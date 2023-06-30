import os

import torch

is_available = torch.cuda.is_available()
DEVICE = torch.device("cuda") if is_available else torch.device("cpu")

IMAGE_TYPE = "tif"
CLASSES = ["Background", "True"]
NUM_CLASSES = len(CLASSES)
NUM_QUERIES = 8
VISUALIZE_TRANSFORMED_IMAGES = True

SAVE_PLOTS_EPOCH = 1
SAVE_MODEL_EPOCH = 1

IMAGE_SPLIT_DIM = 400
INFER_FALSE_LABELS = True
DETECTION_THRESHOLD = 0.2

BATCH_SIZE = 10
RESIZE_TO = 800
NUM_EPOCHS = 10

DATA_ROOT = "data"

DATA_PATH = os.path.join(DATA_ROOT, "CA")

COORDINATES_BBOX_LOCATION = os.path.join(DATA_ROOT, "coordinates_Bbox.csv")
COORDINATES_BBOX_XLS_LOCATION = os.path.join(
    DATA_ROOT, "coordinate in Bbox_Sept24.xlsx"
)
SAMPLES800_NORM_LOCATION = os.path.join(DATA_ROOT, "Sample800_norm")
ANNOTATIONS_LOCATION = os.path.join(DATA_ROOT, "annotations")

IMAGE_SETS_LOCATION = os.path.join(DATA_ROOT, "ImageSets")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "test")

TEST_DIR = os.path.join(DATA_ROOT, "val")
TEST_XML_DIR = os.path.join(DATA_ROOT, "val_xml")

VALIDATION_RESULTS_DIR = os.path.join(DATA_ROOT, "validation_results")
VALIDATION_QUALITATIVE_DIR = os.path.join(DATA_ROOT, "validation_qualitative")
CLASSIFIED_IMAGES_DIR = os.path.join(DATA_ROOT, "classified_images")

MODEL_OUT_DIR = os.path.join(DATA_ROOT, "model")