import cv2
import numpy as np

from src.config import DEVICE


def show_tranformed_image(train_loader, num):
    if len(train_loader) < 0:
        return

    print(num)
    for _ in range(0, num):
        images, targets = next(iter(train_loader))

    images = list(image.to(DEVICE) for image in images)
    targets = [
        {k: v.to(DEVICE) for k, v in t.items()} for t in targets
    ]
    boxes = targets[num]["boxes"].cpu().numpy().astype(np.int32)
    sample = images[num].permute(1, 2, 0).cpu().numpy()
    for box in boxes:
        cv2.rectangle(
            sample, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
        )

    cv2.imshow("Transformed image", sample)
    cv2.waitKey(0)
    cv2.destroyAllWindows()