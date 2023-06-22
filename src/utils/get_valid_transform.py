from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_valid_transform():
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )
