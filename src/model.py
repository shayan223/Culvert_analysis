import torch


def create_model(backbone):
    return torch.hub.load("facebookresearch/detr", backbone, pretrained=True)