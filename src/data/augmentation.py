from dataclasses import dataclass
import numpy as np

from imutils import rotate_bound
from skimage import transform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import filters

from src.data.helpers import cutdown, resize_padded


@dataclass
class AugmentParameters:
    alpha_elastic_transform = 5
    sigma_elastic_transform = 10
    max_factor_skew = 1.5
    max_angle_rotate = 10
    gaussian_sigma = 0.8
    num_augment = 9


def elastic_transform(image, alpha=0, sigma=0, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    # pad image
    size_0 = image.shape[0]
    size_1 = image.shape[1]
    delta_0 = int(np.ceil(0.1 * size_0))
    delta_1 = int(np.ceil(0.1 * size_1))
    size_0_new = size_0 + 2 * delta_0

    image = np.concatenate((np.ones((delta_0, size_1)), image, np.ones((delta_0, size_1))), axis=0)
    image = np.concatenate((np.ones((size_0_new, delta_1)), image, np.ones((size_0_new, delta_1))), axis=1)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    img = map_coordinates(image, indices, order=1).reshape(shape)

    # remove padding prior to returning
    return img[delta_0:(size_0 + delta_0), delta_1:(size_1 + delta_1)]


# caution: changes images size -> apply cutdown after this step
def rotate(image, max_angle=0):
    angle = max_angle * (2 * np.random.rand() - 1)
    return 1 - rotate_bound(1 - image, angle)


def skew(image, max_factor):
    factor = (max_factor - 1) * np.random.rand() + 1
    skew_axis = np.random.binomial(1, 0.5, size=None)

    if skew_axis == 0:
        new_size = np.round((image.shape[0] * factor, image.shape[1]))
        image = transform.resize(image, new_size, anti_aliasing=True, mode='constant')[1:-1, :]
    else:
        new_size = np.round((image.shape[0], image.shape[1] * factor))
        image = transform.resize(image, new_size, anti_aliasing=True, mode='constant')[:, 1:-1]

    return image


def augment_image(image, alpha_elastic_transform, sigma_elastic_transform, max_factor_skew, max_angle_rotate,
                  target_size):
    image = elastic_transform(image, alpha=alpha_elastic_transform, sigma=sigma_elastic_transform)
    image = skew(image, max_factor=max_factor_skew)
    image = rotate(image, max_angle=max_angle_rotate)
    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    image = resize_padded(image, target_size)

    return image


def simulate_augment_image(image, target_size, gaussian_sigma=0.8):
    image = filters.gaussian(image, gaussian_sigma, multichannel=False)
    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    image = resize_padded(image, target_size)

    return image
