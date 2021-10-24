import numpy as np
import torch
from skimage import filters

from src.data_preprocessing.helpers import cutdown, resize_padded


def load_and_normalize_image(image_fp) -> torch.Tensor:
    image = torch.from_numpy(np.load(image_fp)[np.newaxis, np.newaxis, :])
    image = (image - torch.mean(image)) / torch.std(image)
    return image


def simulate_augment_image(image, target_size, gaussian_sigma=0.8):
    image = filters.gaussian(image, gaussian_sigma, multichannel=False)
    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    image = resize_padded(image, target_size)

    return image
