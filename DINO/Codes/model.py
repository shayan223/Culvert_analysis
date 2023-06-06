import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
# import Faster R-CNN with MobileNet
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # # get the number of input features
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # define a new head for the detector with required number of classes
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # for mobile net
    # # Load the pre-trained MobileNet backbone
    # print(torchvision.models.mobilenet_v2(pretrained=True))
    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    # # Set the number of output channels in the backbone
    # backbone.out_channels = 1280

    # # Define the anchor generator for the region proposal network (RPN)
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # # Create the Faster R-CNN model with MobileNet backbone and RPN anchor generator
    # model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator)

    # for resnet152
    # Load the pre-trained ResNet-150 backbone
    # backbone = torchvision.models.resnet152(pretrained=True).avgpool
    model=torchvision.models.resnet152(pretrained=True)
    # remove last layer of model and add it as backbone\
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    # print(backbone)
    # # Get the number of output channels in the backbone
    # in_channels = backbone.fc.in_features
    # # print(backbone)

    # # Replace the fully connected (fc) layer with a new one for the desired number of classes
    # backbone.fc = torch.nn.Linear(in_channels, num_classes)

    # print(backbone)
    backbone.out_channels = 2048

    # Define the anchor generator for the region proposal network (RPN)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # Create the Faster R-CNN model with ResNet-150 backbone and RPN anchor generator
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator)

    
    return model