import numpy as np
from skimage import filters

from skimage.color import rgb2gray as skimage_rgb2gray
from skimage.morphology import erosion as skimage_erosion
from skimage.exposure import adjust_gamma as skimage_adjust_gamma

from src.preprocessing.helpers import cutdown, resize_padded


def preprocess_image(image, target_size, version=1):
    if int(version) == 0:
        return preprocess_image_v0(image, target_size)

    if int(version) == 1:
        return preprocess_image_v1(image, target_size)

    raise ValueError('preprocessing version must be 0 or 1')


def preprocess_image_v0(image, target_size):
    # convert to grayscale
    image_preprocessed = skimage_rgb2gray(image)

    # resize
    image_preprocessed = resize_padded(image_preprocessed, new_shape=target_size)

    return image_preprocessed


def preprocess_image_v1(image, target_size):
    # convert to grayscale
    image_preprocessed = skimage_rgb2gray(image)

    # gamma correction
    image_preprocessed = skimage_erosion(image_preprocessed)
    image_preprocessed = skimage_adjust_gamma(image_preprocessed, gamma=3)

    # cutdown
    thresh_cut = np.percentile(image_preprocessed, 4)
    image_preprocessed = cutdown(img=image_preprocessed, threshold=thresh_cut)
    thresh_white = np.percentile(image_preprocessed, 8)
    image_preprocessed[image_preprocessed > thresh_white] = 1.0

    # resize
    image_preprocessed = resize_padded(image_preprocessed, new_shape=target_size)

    return image_preprocessed


def simulate_augment_image(image, target_size, gaussian_sigma=0.8):
    image = filters.gaussian(image, gaussian_sigma, multichannel=False)
    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    image = resize_padded(image, target_size)

    return image
