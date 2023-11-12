### Functions for showing predictions and utility
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision.transforms
from PIL import Image

from data_load import rooftops_dataset
from torchvision.transforms import ToTensor
from model_u_net import UNet
from torchvision.transforms.functional import to_pil_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def prepare_plot(true_image: np.ndarray,
                 true_mask: np.ndarray,
                 predicted_mask: np.ndarray):
    '''
    Generates three subplots in a row. First image, is the original image, second one the true segmentation mask,
    third the predicted mask. If true_mask is None, plot only image and predicted_mask.
    Args:
        true_image (numpy.ndarray): Original image with format H x W x C, expects RGB channels.
        true_mask (numpy.ndarray): True image segmentation mask in greyscale, i.e. H x W x 1.
        predicted_mask (numpy.ndarray): Predicted image segmentation mask in greyscale, i.e. H x W x 1.

    Returns: None
    '''

    if true_mask is not None:
        # Initalize figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,8))

        # Plot original image, the true mask and the predicted mask side by side
        ax1.imshow(true_image)
        ax2.imshow(true_mask)
        ax3.imshow(predicted_mask)

        # Set titles
        ax1.set_title("Original Image")
        ax2.set_title("True Mask")
        ax3.set_title("Predicted Mask")

        # Plot
        fig.tight_layout()
        plt.show()
    else:
        # Initalize figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

        # Plot original image, the true mask and the predicted mask side by side
        ax1.imshow(true_image)
        ax2.imshow(predicted_mask)

        # Set titles
        ax1.set_title("Original Image")
        ax2.set_title("Predicted Mask")

        # Plot
        fig.tight_layout()
        plt.show()

def generate_preds(model_path: str,
               model: torch.nn.Module,
               path: str,
               device: str,
               test_set: bool = False):
    '''
    Given the path to rooftop images, uses provided model to predict rooftops and obtains segmentation map. Plots image,
    true mask, and predicted mask side-by-side. Use test_set if no true label exists.

    Args:
        model_path (str): The path to the model to use for prediction
        model (torch.nn.Module): Model to populate with trained parameters
        path (str): Path to data directory
        device (str): The device to put input data on (either "cuda:0" or "cpu")
        test_set (bool, default is False): Whether the data set is a test set with no labels

    Returns: None
    '''
    # Get data in custom data set class without transformation
    pred_data = rooftops_dataset(data_dir=path, test_set=test_set)

    # Load model
    model.load_state_dict(torch.load(model_path))

    # Perform predicions, images need to be converted to tensor
    trans = ToTensor()

    preds_masks = []
    pred_mask_names = []
    with torch.no_grad():
        # Set model to eval mode
        model.eval()
        for i, sample in enumerate(pred_data):
            # Obtain image. For model, we need to add one dimension and transform to Tensor
            org_image = sample["image"] # H x W x C
            if not test_set:
                true_label = sample["label"] # H x W x 1
            else:
                true_label = None
            input = trans(org_image).to(device).unsqueeze(0) # 1 x C x H x W

            # Obtain logit mask
            prediction = model(input)

            # Output is logit and in format 1 x H x W. Use sigmoid to obtain probability mask, flatten,
            # and convert to numpy image.
            prediction_mask = torch.sigmoid(prediction).squeeze(0).cpu().numpy() # flatten + numpy transform
            prediction_mask = prediction_mask.swapaxes(0,-1) # H x W x 1
            prediction_mask = (prediction_mask >= 0.5) * 255 # convert binary values to black & white image values
            prediction_mask = prediction_mask.astype(np.uint8) # convert to int

            # For some reason output is W x H x 1, swap W and H
            prediction_mask = prediction_mask.swapaxes(0,1)

            # Perform plotting
            prepare_plot(org_image, true_label, prediction_mask)

            # Append prediction mask to list in case saving as to be performed
            preds_masks.append(prediction_mask)
            pred_mask_names.append(pred_data.filenames[i])

    # Save predicted segmentation masks for test set
    if test_set:
        for image, name in zip(preds_masks, pred_mask_names):
            im = Image.fromarray(image.squeeze(-1))
            im.save(os.path.join(path, "predictions", name + ".png"))
        print("Saving predictions complete.")



def plot_losses(values_path: str,
                colnames: dict = {"epoch_name": "epoch",
                                  "train_name" : "train_loss",
                                  "val_name" : "val_loss"},
                loss_type: str = "Binary Cross Entropy",
                skip: int = 2):
    '''
    Plot train and validation loss curves.
    Args:
        values_path (str): Path to the results csv for the model
        colnames (dict): Dictionary with column names for epoch, train and validation loss
        loss_type(str): Name of the loss function used for setting the title in plot.
        skip (int): How many epochs to skip from the beginning of training to avoid plotting outliers
    Returns: None
    '''
    # Get colnames
    epoch_name,train_name, val_name = colnames["epoch_name"], colnames["train_name"], colnames["val_name"]

    # Read results data, get model name and pivot data into long format
    plot_data = pd.read_csv(values_path)
    model_name = values_path.split(os.sep)[-1].split("_")[0]
    # Truncate plot_data to ignore train loss outliers at the beginning
    plot_data = plot_data.iloc[skip:, :]

    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(plot_data[epoch_name], plot_data[train_name])
    ax.plot(plot_data[epoch_name], plot_data[val_name])

    # Change title and axis labels
    ax.set_title("Training and Validation loss of {modelname} model with \n {loss_name} loss"\
                 .format(modelname=model_name, loss_name=loss_type))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_type + " loss value")
    ax.legend(["Train Loss", "Validation Loss"])

    # plot figure
    plt.show()

# soft dice loss
class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation; Credit to https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''

        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

# soft dice loss
class DiceCrossEntropy(nn.Module):
    '''
    soft-dice loss with cross entropy loss
    '''
    def __init__(self,
                 p=1,
                 smooth=1,
                 weight=0.5):
        super(DiceCrossEntropy, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = weight

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        # Get Probabilities from logits
        probs = torch.sigmoid(logits)

        # Compute BCE
        bce_loss_val = torch.nn.functional.binary_cross_entropy(probs, labels)

        # Compute Soft Dice
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        # dice_loss_val = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        dice_loss_val = (2 * numer + self.smooth) / (denor + self.smooth)

        # Add both values
        loss = 1 + bce_loss_val * self.weight - (1-self.weight) * dice_loss_val
        return loss


if __name__ == "__main__":
    # test
    model = UNet(256, 256).to(device)
    model_path = "..\\models\\base_u_net_lr_augment_dice_cross_loss_hsv_model"
    data_path = "..\\data\\test"
    results_path = "..\\models\\baseline_results.csv"

    # Prediction plots
    generate_preds(model_path, model,data_path, device, test_set=True)

    # Loss plot
    # plot_losses(results_path)



