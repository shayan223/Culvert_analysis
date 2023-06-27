import cv2
import numpy as np

from src.config import DEVICE


def show_tranformed_image(train_loader, pred_boxes, num):
    if len(train_loader) < 0:
        return

    for _ in range(0, num):
        images, targets = next(iter(train_loader))

    images = list(image.to(DEVICE) for image in images)
    targets = [
        {k: v.to(DEVICE) for k, v in t.items()} for t in targets
    ]
    
    boxes = targets[num]["boxes"].cpu().numpy().astype(np.int32)
    pred_boxes = pred_boxes[num].detach().cpu().numpy().astype(np.int32)

    sample = images[num].permute(1, 2, 0).cpu().numpy()

    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        cv2.rectangle(
            sample,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

    for pred_box in pred_boxes:
        x1 = pred_box[0]
        y1 = pred_box[1]
        x2 = pred_box[2]
        y2 = pred_box[3]

        cv2.rectangle(
            sample,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    cv2.imshow("Transformed image", sample)
    cv2.waitKey(1)
    cv2.destroyAllWindows()