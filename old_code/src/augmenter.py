import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from cv2 import imwrite
import csv
from cv2 import imread
import os
from imutils import rotate_bound
from skimage import transform, filters
from helpers import *
import time
from config import DATA_DIR, DEBUG
from preprocess import preprocess_basic, cutdown, resize_padded, CANVAS_SIZE, augmented_CANVAS_SIZE, normalization






VISUALIZE_AUGMENTATION = False
if VISUALIZE_AUGMENTATION:
    create_directory("../data_preprocessing/visualized/augment", empty_dir=True)

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
    size_0 = CANVAS_SIZE[0]
    size_1 = CANVAS_SIZE[1]
    delta_0 = int(np.ceil(0.1 * size_0))
    delta_1 = int(np.ceil(0.1 * size_1))
    size_0_new = size_0 + 2*delta_0

    image = np.concatenate((np.ones((delta_0, size_1)), image, np.ones((delta_0, size_1))), axis=0)
    image = np.concatenate((np.ones((size_0_new, delta_1)), image, np.ones((size_0_new, delta_1))), axis=1)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    img = map_coordinates(image, indices, order=1).reshape(shape)

    # remove padding prior to returning
    return img[delta_0:(size_0+delta_0), delta_1:(size_1+delta_1)]


# caution: changes images size -> apply cutdown after this step
def rotate(image, max_angle=0):
    angle = max_angle * (2*np.random.rand() - 1)
    return 1 - rotate_bound(1 - image, angle)



def skew(image, max_factor):
    factor = (max_factor-1)*np.random.rand()+1
    skew_axis = np.random.binomial(1, 0.5, size=None)
    if skew_axis==0:
        new_size = np.round((CANVAS_SIZE[0] * factor, CANVAS_SIZE[1]))
        image = transform.resize(image, new_size, anti_aliasing=True, mode='constant')[1:-1,:]

    else:
        new_size = np.round((CANVAS_SIZE[0], CANVAS_SIZE[1] * factor))
        image = transform.resize(image, new_size, anti_aliasing=True, mode='constant')[:,1:-1]

    return image


def augment(image, alpha=None, sigma=5, max_factor=None, degrees=None):
    """Augment: function to augment input data_preprocessing using various procedures

    :param image: image (array-type)
    :param alpha: parameter to set intensity of elastic deformation (no deformation if None)
    :param sigma: smoothing parameter used in elastic deformation
    :param degrees: maximal rotation allowed (rotation randomized)
    :return: augmented image (array-type)
    """

    orig = np.copy(image)
    if alpha is not None:
        image = elastic_transform(image, alpha=alpha, sigma=sigma)
    step_elastic = np.copy(image)

    if max_factor is not None:
        image = skew(image, max_factor)
    step_skew = np.copy(image)

    if degrees is not None:
        image = rotate(image, max_angle=degrees)
    step_rotate = np.copy(image)

    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    step_cutdown = np.copy(image)

    image = resize_padded(image, augmented_CANVAS_SIZE)

    if VISUALIZE_AUGMENTATION:
        blurred = filters.gaussian(orig, 0.8, multichannel=False)
        imgs_to_file([orig,step_elastic,step_skew,step_rotate,step_cutdown,image,np.zeros((116,10)),blurred],"../data_preprocessing/visualized/augment/"+str(time.time())+".jpg")

    normalized = normalization(image)

    return normalized

def simulate_augment(image):
    """
    augmented data_preprocessing is different from original, e.g. a bit blurred
    this function tries to approximate that without actually doing any transformations
    """
    #
    image = filters.gaussian(image, 0.8, multichannel=False)
    thresh = np.percentile(image, 4)
    image = cutdown(image, thresh)
    image = resize_padded(image, augmented_CANVAS_SIZE)
    image = normalization(image)
    return image

def load_raw_data_aug(image_path, label_path, number):
    if(DEBUG):
        print("Loading dataset "+image_path)
    #read labels
    with open(label_path) as csv_file:
        labels_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in labels_reader]
        filenames = [row[0] for row in rows]
        labels = [row[1] for row in rows]


    start = np.int(np.ceil(np.random.uniform(0,np.shape(labels)[0]-number)))
    filenames = filenames[start:start+number]
    labels = labels[start:start+number]

    # read images
    valid_paths = [image_path + f for f in filenames]
    images = [imread(file) for file in valid_paths]

    return images, labels, filenames

def run_test(number=20):
    path = DATA_DIR + "augmented/"
    create_directory(path)
    images, _, filenames = load_raw_data_aug(DATA_DIR + "raw/Mexico/", DATA_DIR + "raw/mexico_trainval.csv", number)
    for i in range(number):
        img = images[i]
        img = preprocess_basic(img)
        img = augment(img, alpha=20, sigma=5, max_factor=3, degrees=20)
        filepath = path + "{}".format(filenames[i])
        imwrite(filepath, 255*augment(img, alpha=20, sigma=5, max_factor=2, degrees=20))
        imwrite(filepath, 255*img)



if __name__ == "__main__":
    run_test()
