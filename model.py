import torch
from torch import nn


def create_model(NUM_CLASSES):
    return torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)