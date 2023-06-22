#!/usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from sklearn.model_selection import KFold

from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, BATCH_SIZE
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from datasets import train_loader, valid_loader
from model import create_model
from utils import Averager

plt.style.use("ggplot")


def train(train_data_loader, model):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("Training")
    global train_itr
    global train_loss_list

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images)
        losses = [loss for loss in loss_dict.values()]

        for loss_tensor in losses:
            loss_value = loss_tensor.mean().item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)

        optimizer.step()
        train_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list


def validate(valid_data_loader, model):
    global val_itr
    global val_loss_list

    print("Validating")

    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images)

        losses = [loss for loss in loss_dict.values()]

        for loss_tensor in losses:
            loss_value = loss_tensor.mean().item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)

        optimizer.step()
        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list


if __name__ == "__main__":
    model = create_model().to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []

    MODEL_NAME = "model"

    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image

        show_tranformed_image(train_loader)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        train_loss_hist.reset()
        val_loss_hist.reset()

        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        start = time.time()

        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)

        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")

        end = time.time()

        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")
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
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")

        plt.close("all")