import numpy as np
import os
import time
import torch
from skimage import color, filters, io

from constants import DEFAULT_CANVAS_SIZE
from src.data_preprocessing.helpers import cutdown, resize_padded


def preprocess_image_v0(image, simulate_augment=False):
    # convert to grayscale
    image_preprocessed = color.rgb2gray(image)

    # resize
    image_preprocessed = resize_padded(image_preprocessed, new_shape=DEFAULT_CANVAS_SIZE)

    # simulate augmentation -> this is to reduce the test-train mismatch when model was trained with data augmentation
    if simulate_augment:
        image_preprocessed = simulate_augment_image(image_preprocessed, DEFAULT_CANVAS_SIZE)

    # save the image - sanity check
    timestr = time.strftime("%Y%m%d_%H%M%S")
    io.imsave(os.path.join('./temp', 'sanity_check_' + str(timestr) + '.jpg'), image_preprocessed)

    # normalize
    image_preprocessed = (image_preprocessed - np.mean(image_preprocessed)) / np.std(image_preprocessed)

    # turn into NCHW as required by model
    while len(image_preprocessed.shape) < 4:
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

    return image_preprocessed


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
