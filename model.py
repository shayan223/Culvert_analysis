import torch


def create_model():
    return torch.hub.load("facebookresearch/detr", "detr_resnet101", pretrained=True)