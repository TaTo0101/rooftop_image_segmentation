### Script that provides custom dataset and data loaders generator classes
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image

## Dataset
class rooftops_dataset(Dataset):
    """
    Rooftop satellite image data set. Input images have four channels (probably CMYK), transform to 3 channels. Labels
    do not have a channel, add dimension for consistency.
    """
    def __init__(self, data_dir: str, transform=None, test_set: bool=False,
                 augment: bool=False, augment_dict: dict = None,
                 add_filename: str = None):
        """

        Args:
            data_dir (str): Path to data directory
            transform (callable, optional): Optional transform to be applied, e.g. conversions.
            test_set (bool, default is False): Whether the data set is a test set with no labels
            augment (bool, default is False): Whether to perform data augmentation, for details see segmentation_transform
            augment_dict (dict, default is None): A dictionary with two keys "probs" and "params", with values being
            the dictionaries as described in segmentation_transform. If not none, requires that both keys have expected
            values, i.e. the full dictionaries.
            add_filename (str): Additional file names to retrieve from the "add_files" folder,
             only relevant for the wrongly labeled image
        """
        self.data_dir = data_dir
        self.filenames = os.listdir(os.path.join(self.data_dir, "images"))
        # Omit the folder name add_images
        self.filenames = [filename for filename in self.filenames if filename != "add"]
        self.transform = transform
        self.test = test_set
        self.augment = augment

        # Add additional filenames if specified
        if add_filename is not None:
            path = os.path.join("add", add_filename)
            self.filenames.append(path)

        # Get augment params, if not None
        self.augment_dict = augment_dict
        if self.augment_dict is not None:
            self.augment_probs = augment_dict["probs"]
            self.augment_params = augment_dict["params"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        # If iterator is tensor, convert to list
        if torch.is_tensor(item):
            item = item.tolist()

        image_dir = os.path.join(os.path.join(self.data_dir, "images"),
                                   self.filenames[item])
        image = Image.open(image_dir).convert("RGB")
        image.load()
        # image = np.asarray(image, dtype="int32")

        # If test set is loaded, there are no labels. Skip label reading then
        if not self.test:
            label_dir = os.path.join(os.path.join(self.data_dir, "labels"),
                                     self.filenames[item])
            label = Image.open(label_dir)
            label.load()
            sample = {"image": image, "label": label}
        else:
            sample = {"image": image}

        # Perform Augmentation if Specified
        if self.augment:
            # Augmentation is only required for non-test set images, hence raise error if test_set is set to true
            if self.test:
                raise ValueError("Parameter augment is set to true for test set with not labels.")
            # Check if augment params have been specified
            if self.augment_dict is not None:
                sample["image"], sample["label"] = segmentation_transform(input=sample["image"],
                                                                          label=sample["label"],
                                                                          probs=self.augment_probs,
                                                                          add_params=self.augment_params)
            else:
                sample["image"], sample["label"] = segmentation_transform(input=sample["image"],
                                                                          label=sample["label"])

        if self.transform:
            sample["image"] = self.transform(sample["image"])
            if not self.test:
                sample["label"] = self.transform(sample["label"])

        return sample


### Custom Augmentation Function
# Since augmentations need to be applied to both the input and the label, we need a custom augmentation function when
# we apply transformations in a random fashion
def segmentation_transform(input: Image,
                           label: Image,
                           probs: dict = {"p_hflip": 0.5,
                                          "p_vflip": 0.5,
                                          "p_rotation": 0.5,
                                          "p_noise": 0.5,
                                          "p_hsv": 0.5},
                           add_params: dict = {"noise_mean": 0,
                                               "noise_std": 1,
                                               "noise_scale": 0.01,
                                               "rot_angles": [-90, 90, 180],
                                               "hue_range": 0.5,
                                               "saturation_max": 0,
                                               "brightness_max": 0}):
    '''
    Performs augmentation of input images and labels in a random fashion. Allows for batched input.
    Augmentations include: Horizontal and vertical flipping, rotation in 90 degree steps, additive Gaussian noise,
    and random HSV jitter.

    Note: Gaussian noise and random HSV jitter is only applied on the input not on the mask for obvious reasons.

    Args:
        input (PIL.Image): Input image (either a single image or a batch of images)
        label (PIL.Image): Corresponding segmentation mask (either a single mask or a batch of masks)
        probs (dict): Dictionary with keys as in default, values set probability that the respective transformation is
        applied on image and label
        add_params (dict): Additional params for the transformations
    Returns: Transformed input and label (batch)
    '''

    # Horizontal Flip
    if random.random() <= probs["p_hflip"]:
        input = TF.hflip(input)
        label = TF.hflip(label)

    # Vertical Flip
    if random.random() <= probs["p_vflip"]:
        input = TF.vflip(input)
        label = TF.vflip(label)

    # Rotation
    if random.random() <= probs["p_rotation"]:
        angle = random.choice(add_params["rot_angles"])
        input = TF.rotate(input, angle)
        label = TF.rotate(label, angle)

    # Gaussian Noise
    if random.random() <= probs["p_noise"]:
        # Ugly but works: Convert PIL to tensor
        trans_tensor = T.ToTensor()
        trans_PIL = T.ToPILImage()
        input_conv = trans_tensor(input)
        # Draw noise and add to converted input after scaling
        noise = torch.randn(input_conv.size()) * add_params["noise_std"] + add_params["noise_mean"]
        input_conv = input_conv + add_params["noise_scale"] * noise

        # Convert back to PIL image
        input = trans_PIL(input_conv)
    if random.random() <= probs["p_hsv"]:
        trans_color = T.ColorJitter(brightness=add_params["brightness_max"],
                                    saturation=add_params["saturation_max"],
                                    hue=add_params["hue_range"])
        input = trans_color(input)

    return input, label

if __name__ == "__main__":
    # Test Class and also inspect the data (low volume allows for manual inspection)
    augment_prob = {
        "p_hflip": 0.1,
        "p_vflip": 0.1,
        "p_rotation": 0.1,
        "p_noise" : 0.1
    }
    augment_params = {
        "noise_mean": 0,
        "noise_std": 1,
        "noise_scale": 0.005,
        "rot_angles": [-90, 90, 180]
    }

    augment_dict = {"probs": augment_prob, "params": augment_params}

    rooftops_train = rooftops_dataset(data_dir="..\\data\\validation",
                                      augment=True,
                                      augment_dict=augment_dict)

    for i, sample in enumerate(rooftops_train):
        print(i, sample["image"].size, sample["label"].size)

        # Reswap axis for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        plt.interactive(False)
        ax1.imshow(sample["image"])
        ax1.axis("off")
        ax1.set_title(f"Sample {i}: Image")

        ax2.imshow(sample["label"])
        ax2.axis("off")
        ax2.set_title(f"Sample {i}: Label")

        plt.tight_layout()
        plt.show()
        if i ==5:
            break
# Notes: Inspection shows, that label for 278 is actually label for 270. Since we don't have a correct label for 278,
# we will use 278 in the test set as well and move one item from the validation set to the train set
# Furthermore, we can see that for some images black boxes were drawn, likely due to privacy reasons.
# Those boxes could be used by the model to cheat, see e.g. clever hans predictor. We'll first train the model
# and then check if this issue occurs, if so, we will add small noise to every pixel with value 0.
# Also note that for 379, 328, and 274 we do not see many rooftops (class imbalance). For those dice loss might be prefered
