### Script with functions used for training
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_load import rooftops_dataset
from model_u_net import UNet
import torchvision.transforms as T
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_fun(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              data_loaders: dict,
              criterion,
              device: str,
              model_path: str,
              scheduler: torch.optim.lr_scheduler = None,
              scheduler_type: str = "static",
              epochs: int = 10,
              patience: int = 4):
    '''
    Utility function that performs training of a model for image segmentation, given an optimizer,
    the train / validation data loaders, and a criterion (loss function) for the specified number of epochs.
    Utilizes early stopping.
    Args:
        model (torch.nn.Module): Neural network that should be trained. Should be on device.
        optimizer (torch.optim.Optimizer): Optimizer to use for training
        data_loaders (dict): Dictionary with two keys ["train", "validation"] with the respective data loaders as
        values
        criterion: Loss function to use in training, should behave like torch.nn.functional loss functions
        device (str): The device to put input data on (either "cuda:0" or "cpu")
        models_path (str): The path (with name) to save the model to
        scheduler (torch.optim.lr_scheduler, defaults to None): Learning rate scheduler to use
        scheduler_type (str): Either "static" or "dynamic", specifies if scheduler adjusts lr dynamically
        epochs (int, defaults to 10): Number of training rounds to perform
        patience (int, defaults to 4): Number of epochs to consider for early stopping
    Returns: A dictionary with keys ["values", "epoch_best_model"]. "values" contains the average train / validation
     loss per epoch (pandas.DataFrame), and "epoch_best_model" contains the epoch of the corresponding best model
     (according to validation loss). Model will be saved in folder "models".
    '''

    # Get train and validation data loaders
    trainloader = data_loaders["train"]
    valloader = data_loaders["validation"]

    train_loss = []
    val_loss = []
    current_min_loss = 1000
    epoch_with_min_loss = 0

    # Start counting at 1 not at 0
    for epoch in range(1, epochs+1):

        # losses per batch
        loss_per_batch_train = []

        # Set model to train mode
        model.train()
        # Perform training for all batches in trainloader
        for batch in trainloader:
            # Get image and label, put to device
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            # zero gradients
            optimizer.zero_grad()

            # Perform forward and backward pass, as well as optimizer step
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save loss per batch
            loss_per_batch_train.append(loss.item())

        # Perform validation loss calculation
        # Use no grad to save memory
        with torch.no_grad():
            # Set model to eval mode (just in case dropout or similar is used
            model.eval()

            # Val losses per batch
            loss_per_batch_val = []
            for batch in valloader:
                # Get image and label, put to device
                inputs, labels = batch["image"].to(device), batch["label"].to(device)

                # Perform forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Append loss
                loss_per_batch_val.append(loss.item())

        if scheduler is not None:
            if scheduler_type == "static":
                scheduler.step()
            elif scheduler_type == "dynamic":
                scheduler.step(sum(loss_per_batch_val) / len(loss_per_batch_val))
            else:
                raise ValueError(f"Scheduler type supplied should be either static or dynamic but got: {scheduler_type}")

        # Print statistics
        if (epoch % 5) == 0:
            if scheduler is not None:
                if scheduler_type =="static":
                    lr_val = scheduler.get_last_lr()[0]
                else:
                    lr_val = scheduler._last_lr[0]
            else:
                lr_val = optimizer.param_groups[-1]['lr']
            print("Epoch: {ep} | Avg. Train Loss: {train: .3f} | Avg. Val Loss: {val: .3f} | Learning Rate: {lr: .5f}"\
                  .format(ep=epoch, train=sum(loss_per_batch_train)/len(loss_per_batch_train),
                          val=sum(loss_per_batch_val)/len(loss_per_batch_val),
                          lr=lr_val))
        # Save values
        train_loss.append(sum(loss_per_batch_train)/len(loss_per_batch_train))
        val_loss.append(sum(loss_per_batch_val)/len(loss_per_batch_val))
        values = pd.DataFrame({
            "epoch" : np.arange(1, epoch+1),
            "train_loss" : train_loss,
            "val_loss" : val_loss
        })

        # Check if early stopping criterion is fulfilled
        if current_min_loss > val_loss[-1]:
            current_min_loss = val_loss[-1]
            epoch_with_min_loss = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        # if not fulfilled, check out if patience has been reached
        else:
            if epoch - epoch_with_min_loss > patience:
                # Early stopping initiated
                results = {"values": values,
                           "epoch_best_model": epoch_with_min_loss}
                # Save results to model_path as csv
                results["values"].to_csv(model_path + "_results.csv", index=False)

                # Save model parameter dict
                torch.save(best_model_state, model_path + "_model")
                return results

    # Early stopping was not initiated, but best model might not be the latest one
    results = {"values": values,
               "epoch_best_model": epoch_with_min_loss}
    # Save results to model_path as csv
    results["values"].to_csv(model_path + "_results.csv", index=False)

    # Save model parameter dict
    torch.save(best_model_state, model_path + "_model")
    return results



if __name__ == "__main__":
    ##
    # Params
    model = UNet(256, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=0.97)

    batch_size = 4
    # Augmentation params
    augment_prob = {
            "p_hflip": 0.02,
            "p_vflip": 0.02,
            "p_rotation": 0.15,
            "p_noise" : 0.2
        }
    augment_params = {
        "noise_mean": 0,
        "noise_std": 1,
        "noise_scale": 0.002,
        "rot_angles": [-90, 90, 180]
    }

    augment_dict = {"probs": augment_prob, "params": augment_params}

    train_data_loader = DataLoader(
        rooftops_dataset("..\\data\\train",
                         transform=T.ToTensor(),
                         augment=True,
                         augment_dict=augment_dict),
        shuffle=True,
        batch_size=batch_size
    )
    val_data_loader = DataLoader(
        rooftops_dataset("..\\data\\validation",
                         transform=T.ToTensor()),
        shuffle=False,
        batch_size=batch_size
    )
    dataloaders = {"train": train_data_loader,
                   "validation": val_data_loader}

    # Execute training
    results = train_fun(model=model,
                        optimizer=optimizer,
                        data_loaders=dataloaders,
                        criterion=criterion,
                        device=device,
                        model_path="..\\models\\baseline",
                        scheduler=scheduler,
                        epochs=100,
                        patience=8)

    # print(results["values"])
    # print(results["epoch_best_model"])