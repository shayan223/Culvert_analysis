import os
import time

import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm.auto import tqdm

from src.config import DEVICE, MODEL_OUT_DIR, NUM_CLASSES
from src.config import NUM_EPOCHS, NUM_QUERIES
from src.config import VISUALIZE_TRANSFORMED_IMAGES
from src.config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from src.datasets import train_loader, valid_loader
from src.matcher import HungarianMatcher
from src.model import Model
from src.utils import Averager, show_tranformed_image

plt.style.use("ggplot")


class Engine:
    def __init__(self, backbone="detr_resnet101", num_classes=2, num_queries=512):
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = Model(
            backbone=self.backbone,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.train_loss_list = []
        self.val_loss_list = []

        self.train_loss_hist = Averager()
        self.val_loss_hist = Averager()

        self.train_itr = 0
        self.val_itr = 0

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            self.params,
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005,
        )

    def train(self, train_data_loader, matcher):
        print("Training")

        prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

        for data in prog_bar:
            self.optimizer.zero_grad()
            images, targets = data

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            outputs = self.model.model(images)

            indices = matcher.forward(outputs, targets)
            num_boxes = len(indices)
            losses = self.model.loss_boxes(
                outputs,
                targets,
                indices,
                num_boxes,
            )

            total_loss = sum(loss for loss in losses.values())
            loss_value = total_loss.item()

            self.train_loss_list.append(loss_value)
            self.train_loss_hist.send(loss_value)

            total_loss.backward()
            self.optimizer.step()

            if VISUALIZE_TRANSFORMED_IMAGES:
                show_tranformed_image(
                    train_loader, outputs["pred_boxes"], num=self.train_itr + 1
                )

            prog_bar.set_description(desc=f"Loss: {loss_value:.3f}")

        return self.train_loss_list

    def validate(self, valid_data_loader, matcher):
        print("Validating")

        prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

        for data in prog_bar:
            images, targets = data

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model.model(images)

            indices = matcher.forward(outputs, targets)
            num_boxes = len(indices)
            losses = self.model.loss_boxes(
                outputs,
                targets,
                indices,
                num_boxes,
            )

            total_loss = sum(loss for loss in losses.values())
            loss_value = total_loss.item()

            self.optimizer.step()
            self.val_itr += 1

            prog_bar.set_description(desc=f"Loss: {loss_value:.3f}")

        return self.val_loss_list


def main(args):
    engine = Engine(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
    )
    matcher = HungarianMatcher(cost_class=5, cost_bbox=5, cost_giou=5)

    if not os.path.exists(MODEL_OUT_DIR):
        os.makedirs(MODEL_OUT_DIR)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        engine.train_loss_hist.reset()
        engine.val_loss_hist.reset()

        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        start = time.time()

        train_loss = engine.train(train_loader, matcher)
        val_loss = engine.validate(valid_loader, matcher)

        train_loss_hist = engine.train_loss_hist.value
        val_loss_hist = engine.val_loss_hist.value

        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist:.4f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist:.4f}")

        end = time.time()

        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}")

        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(
                engine.model.state_dict(),
                f"{MODEL_OUT_DIR}/model{epoch + 1}.pth",
            )
            print("SAVING MODEL COMPLETE...\n")

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{MODEL_OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{MODEL_OUT_DIR}/valid_loss_{epoch + 1}.png")
            print("SAVING PLOTS COMPLETE.")

        if (epoch + 1) == NUM_EPOCHS:
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{MODEL_OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{MODEL_OUT_DIR}/valid_loss_{epoch + 1}.png")
            torch.save(
                engine.model.state_dict(),
                f"{MODEL_OUT_DIR}/model{epoch + 1}.pth",
            )

        plt.close("all")
