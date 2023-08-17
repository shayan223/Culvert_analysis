from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
# from model import create_model
from util.slconfig import DictAction, SLConfig
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt
import time
import os
plt.style.use('ggplot')
import argparse
from torch import nn


# function for running training iterations
def train(train_data_loader, model, optimizer, criterion):


    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print('Training')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    max_norm = 0.1
    model.train()
    criterion.train()
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        for j in range(len(targets)):
            boxes=targets[j]["boxes"].cpu().numpy()
            new_boxes=[]
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # chnage to center_x, center_y, width, height and normalize by image size
                new_box = [((x_min+x_max)/2)/800, ((y_min+y_max)/2)/800, (x_max-x_min)/800, (y_max-y_min)/800]
                new_boxes.append(new_box)

            boxes=torch.tensor(new_boxes, dtype=torch.float32)
            targets[j]["boxes"]=boxes

        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        output= model(images, targets)
        loss_dict=criterion(output, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # original backward function
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)

        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    # set model to evaluation mode
    model.eval()
    criterion.eval()
    for i, data in enumerate(prog_bar):
        images, targets = data
        for j in range(len(targets)):
            boxes=targets[j]["boxes"].cpu().numpy()
            new_boxes=[]
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # chnage to center_x, center_y, width, height and normalize by image size
                new_box = [((x_min+x_max)/2)/800, ((y_min+y_max)/2)/800, (x_max-x_min)/800, (y_max-y_min)/800]
                new_boxes.append(new_box)

            boxes=torch.tensor(new_boxes, dtype=torch.float32)
            targets[j]["boxes"]=boxes
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
           output= model(images, targets)
        loss_dict=criterion(output, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors





if __name__ == '__main__':
    # initialize the model and move to the computation device

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--device', '-m', type=str, required=True)
    args=parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))


    model, criterion, postprocessors = build_model_main(args)

    model = model.to(DEVICE)
    # changing the layers to load the pretrained weights from the checkpoint
    model.transformer.enc_out_class_embed = nn.Linear(in_features=256, out_features=91, bias=True)

    model.class_embed = nn.ModuleList([nn.Linear(in_features=256, out_features=91, bias=True) for i in range(6)])
    model.transformer.decoder.class_embed = nn.ModuleList([nn.Linear(in_features=256, out_features=91, bias=True) for i in range(6)])


    pretrained_weights=torch.load('/user/DETR/Culvert_analysis/checkpoint0029_4scale_swin.pth', map_location=torch.device('cuda'))

    pretrained_weights_model= pretrained_weights["model"]
    model.load_state_dict(pretrained_weights_model, strict=True)
    # free the gpu memory after loading the weights\
    del pretrained_weights
    del pretrained_weights_model
    torch.cuda.empty_cache()


    model.transformer.enc_out_class_embed = nn.Linear(in_features=256, out_features=2, bias=True)

    model.class_embed = nn.ModuleList([nn.Linear(in_features=256, out_features=2, bias=True) for i in range(6)])
    model.transformer.decoder.class_embed = nn.ModuleList([nn.Linear(in_features=256, out_features=2, bias=True) for i in range(6)])

    model=model.to(DEVICE)
    # model.load_state_dict(torch.load('/user/DETR/outputs/model_best_loss_finetuned.pth', map_location=torch.device('cuda')))
    BEST_VAL_LOSS = 10000000
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer

    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []
    train_list=[]
    val_list=[]
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image

        show_tranformed_image(train_loader)
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, optimizer,criterion)

        val_loss = validate(valid_loader, model)
        # update the learning rate
        lr_scheduler.step(val_loss_hist.value)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        val_list.append(val_loss_hist.value)
        train_list.append(train_loss_hist.value)

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the model for the current epoch ehen the validation loss is the lowest
        if val_loss_hist.value < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss_hist.value
            torch.save(model.state_dict(), f"{OUT_DIR}/model_best_loss_swin.pth")
            print('SAVING MODEL COMPLETE at BEST LOSS of {} at epoch {}...\n'.format(BEST_VAL_LOSS,epoch+1))

        # if (epoch + 1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
        #     torch.save(model.state_dict(), f"{OUT_DIR}/model_swin{epoch + 1}.pth")
        #     print('SAVING MODEL COMPLETE...\n')

        # if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:  # save loss plots after n epochs
        #     train_ax.plot(train_loss, color='blue')
        #     train_ax.set_xlabel('iterations')
        #     train_ax.set_ylabel('train loss')
        #     valid_ax.plot(val_loss, color='red')
        #     valid_ax.set_xlabel('iterations')
        #     valid_ax.set_ylabel('validation loss')
        #     figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
        #     figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
        #     print('SAVING PLOTS COMPLETE...')

        if (epoch + 1) == NUM_EPOCHS:  # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model_swin{epoch + 1}.pth")

        plt.close('all')
        # plot the training and validation loss graphs
        figure1, ax1 = plt.subplots()
        ax1.plot(train_list, color='blue')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('train loss')
        ax2 = ax1.twinx()
        ax2.plot(val_list, color='red')
        ax2.set_ylabel('validation loss')
        figure1.savefig(f"{OUT_DIR}/train_val_loss_swin.png")
        plt.close('all')
        print('SAVING TRAINING AND VALIDATION LOSS PLOT COMPLETE...')
        # save training and validation loss lists
        with open(f"{OUT_DIR}/train_loss_list_swin.txt", "w") as file:
            file.write(str(train_list))
        with open(f"{OUT_DIR}/val_loss_list_swin.txt", "w") as file:
            file.write(str(val_list))
        print('SAVING TRAINING AND VALIDATION LOSS LISTS COMPLETE...')


    print('TRAINING COMPLETE...')
