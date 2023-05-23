import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
