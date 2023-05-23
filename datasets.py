import glob as glob
import os
from xml.etree import ElementTree as et

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, CLASSES, IMAGE_TYPE, RESIZE_TO, TRAIN_DIR, VALID_DIR
from utils import collate_fn, get_train_transform, get_valid_transform


class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        self.image_paths = glob.glob(f"{self.dir_path}/*." + IMAGE_TYPE)
        self.all_images = [image_path.split("/")[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, os.path.split(image_name)[1])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        annot_filename = image_name[:-4] + ".xml"
        annot_file_path = os.path.join(self.dir_path, os.path.split(annot_filename)[1])
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall("object"):
            labels.append(self.classes.index(member.find("name").text))

            xmin = int(member.find("bndbox").find("xmin").text)
            xmax = int(member.find("bndbox").find("xmax").text)
            ymin = int(member.find("bndbox").find("ymin").text)
            ymax = int(member.find("bndbox").find("ymax").text)

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            yamx_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(
                image=image_resized, bboxes=target["boxes"], labels=labels
            )
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


train_dataset = CustomDataset(
    TRAIN_DIR,
    RESIZE_TO,
    RESIZE_TO,
    CLASSES,
    get_train_transform(),
)
valid_dataset = CustomDataset(
    VALID_DIR,
    RESIZE_TO,
    RESIZE_TO,
    CLASSES,
    get_valid_transform(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

if __name__ == "__main__":
    dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
    print(f"Number of training images: {len(dataset)}")

    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = CLASSES[target["labels"][0]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            image,
            label,
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Image", image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)

