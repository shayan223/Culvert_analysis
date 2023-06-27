import math
import os
import random

from src import config


def split_files():
    images_path = config.SAMPLES800_NORM_LOCATION
    output_dir = config.IMAGE_SETS_LOCATION
    trainval_rate = 0.9
    train_rate = 0.8

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_names = os.listdir(images_path)
    images_list = []

    for img in images_names:
        images_list.append(img.split(".")[0])

    random.shuffle(images_list)

    annotation_num = len(images_list)
    trainval_num = int(math.ceil(trainval_rate * annotation_num))
    train_num = int(math.ceil(trainval_num * train_rate))
    trainval = images_list[0:trainval_num]
    test = images_list[trainval_num:]
    train = trainval[0:train_num]
    val = trainval[train_num:trainval_num]
    trainval = sorted(trainval)
    train = sorted(train)
    val = sorted(val)
    test = sorted(test)
    fout = open(os.path.join(output_dir, "trainval.txt"), "w")

    for line in trainval:
        fout.write(line + "\n")

    fout.close()
    fout = open(os.path.join(output_dir, "train.txt"), "w")

    for line in train:
        fout.write(line + "\n")

    fout.close()
    fout = open(os.path.join(output_dir, "val.txt"), "w")

    for line in val:
        fout.write(line + "\n")

    fout.close()
    fout = open(os.path.join(output_dir, "test.txt"), "w")

    for line in test:
        fout.write(line + "\n")

    fout.close()