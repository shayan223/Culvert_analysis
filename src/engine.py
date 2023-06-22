import os
import time

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from src.model import create_model
from src.config import DEVICE, NUM_EPOCHS, OUT_DIR
from src.config import VISUALIZE_TRANSFORMED_IMAGES
from src.config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from src.datasets import train_loader, valid_loader
from src.model import create_model
from src.utils.averager import Averager

plt.style.use("ggplot")


class Engine:
    def __init__(self, backbone):
        self.backbone = backbone

        self.model = create_model(self.backbone).to(DEVICE)

        self.train_loss_list = []
        self.val_loss_list = []

        self.train_loss_hist = Averager()
        self.val_loss_hist = Averager()

        self.train_itr = 0
        self.val_itr = 0

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    def train(self, train_data_loader):
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        print("Training")

        prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()
            images, targets = data

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images)
            losses = [loss for loss in loss_dict.values()]

            for loss_tensor in losses:
                loss_value = loss_tensor.mean().item()
                self.train_loss_list.append(loss_value)
                self.train_loss_hist.send(loss_value)

            self.optimizer.step()
            self.train_itr += 1

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        def validate(self, valid_data_loader):
            print("Validating")

            prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

            for i, data in enumerate(prog_bar):
                images, targets = data

                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    loss_dict = self.model(images)

                losses = [loss for loss in loss_dict.values()]

                for loss_tensor in losses:
                    loss_value = loss_tensor.mean().item()
                    self.train_loss_list.append(loss_value)
                    self.train_loss_hist.send(loss_value)

                self.optimizer.step()
                self.val_itr += 1

                prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")


def main(backbone):
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image

        show_tranformed_image(train_loader)

    engine = Engine(backbone)
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        engine.train_loss_hist.reset()
        engine.val_loss_hist.reset()

        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        start = time.time()

        train_loss = engine.train(train_loader)
        val_loss = engine.validate(valid_loader)

        print(f"Epoch #{epoch} train loss: {engine.train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {engine.val_loss_hist.value:.3f}")

        end = time.time()

        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(engine.model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")
            print("SAVING MODEL COMPLETE...\n")

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            print("SAVING PLOTS COMPLETE...")

        if (epoch + 1) == NUM_EPOCHS:
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel("iterations")
            train_ax.set_ylabel("train loss")
            valid_ax.plot(val_loss, color="red")
            valid_ax.set_xlabel("iterations")
            valid_ax.set_ylabel("validation loss")
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            torch.save(engine.model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")

        plt.close("all")