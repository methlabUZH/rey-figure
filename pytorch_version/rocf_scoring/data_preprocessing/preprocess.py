import numpy as np
import pandas as pd
from cv2 import imwrite
from skimage.color import rgb2gray
from math import floor
import decimal
from skimage.morphology import erosion
from scipy.ndimage.filters import minimum_filter
from skimage.exposure import adjust_gamma
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
import warnings
from skimage import filters


from config import DATA_AUGMENTATION, DEBUG

if DATA_AUGMENTATION:
    CANVAS_SIZE = (464, 600)  # height, width
else:
    CANVAS_SIZE = (116, 150)
augmented_CANVAS_SIZE = (116, 150)

BIN_LOCATIONS = [(0, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30), (30, 32),
                 (32, 34), (34, 36), (36, 37)] # last bin for flawless images (36)


def resize_padded(img, new_shape, fill_cval=None, order=1, anti_alias = True):
    if fill_cval is None:
        fill_cval = np.max(img)
    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio / 2 for s in img.shape]).astype(np.int) * 2
    interm_img = resize(img, interm_shape, order=order, cval=fill_cval, anti_aliasing=anti_alias, mode='constant')

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[[slice(p, -p, None) if 0 != p else slice(None, None, None)
             for p in pad]] = interm_img

    return new_img


# function returning if image is already sufficiently clear
# img: grayscale image
def sufficient(img, windowsize=2, threshold=0.471):
    # compute starting points
    m = floor(img.shape[0]/2)  # rows
    n = floor(img.shape[1]/2)  # columns
    #  iterate over part of the image
    for i in range(m, img.shape[0]-windowsize, windowsize):
        for j in range(n, img.shape[1]-windowsize, windowsize):
            if np.mean(img[i:(i+windowsize), j:(j+windowsize)]) < threshold:
                return True
    return False


# normalize image to zero mean and unit variance
# rescale back into 0-1
def normalization(img):
    img = (img-np.mean(img))/np.std(img)
    return img


# make image interpretable again as grayscale image (=~ undo normalization)
def undo_normalization(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - min)/(max - min)
    return img


# gamma correction for images in [0,1]
def gamma_correct(img, gamma=3):
    img = erosion(img)
    img = adjust_gamma(img,gamma)
    return img


def procedure_min(img, size=3):
    if sufficient(img):
        return minimum_filter(img, size=2)
    else:
        return minimum_filter(img, size=size)


def procedure_erosion(img, threshold=90, windowsize=3, niter=1):
    if sufficient(img, threshold=threshold, windowsize=windowsize):
        return img
    else:
        for i in range(niter):
            enhanced_img = erosion(img)
            if sufficient(enhanced_img, threshold=threshold, windowsize=windowsize):
                return enhanced_img
        return enhanced_img


# To be used directly on original labels
def bin_numbers(labels):
    shape = np.shape(labels)
    # reshape to 1d
    shape_sum = np.prod(shape)
    labels_1d = np.reshape(labels,shape_sum)
    labels_1d = np.clip(labels_1d, 0, 36) # clip labels (otherwise you have -1 for <0 / >36)
    bins = pd.IntervalIndex.from_tuples(BIN_LOCATIONS, closed='left')
    categorical_object = pd.cut(labels_1d, bins)
    binned_data_1d = categorical_object.codes
    # reshape back to original shape and cast to float
    binned_data = np.reshape(binned_data_1d,shape).astype('float32')
    return binned_data

def bin_numbers_continuous(labels):
    labels = np.clip(labels, 0, 36) # clip labels (otherwise you have -1 for <0 / >36)

    def convert_value(val):
        bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
        bin = np.argmin(bins <= val) - 1
        bin_start = bins[bin]
        bin_end = bins[bin + 1]
        frac = (val - bin_start) / (bin_end - bin_start)
        return bin + frac - 0.5
    map_func = np.vectorize(convert_value)

    converted = map_func(labels)
    return converted


def postprocess_binned_labels(labels):
    """
    for continuous bins this will return the integer bin
    for integer bins this doesn't change anything (note that predictions are continuous nonetheless)
    note that python/numpy rounds 0.5 to 0 and 1.5 to 2, this is why the rounding here is done manually
    """
    def round_value(val):
        context = decimal.getcontext()
        context.rounding = decimal.ROUND_HALF_UP
        rounded_val = round(decimal.Decimal(float(val)), 0)
        clipped_val = np.float(np.clip(rounded_val,0,12))
        return clipped_val
    map_func = np.vectorize(round_value)

    converted = map_func(labels)
    return converted

# To be used directly on original labels
def one_hot_encoding(labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # suppress warnings in this section
        binned_data = bin_numbers(labels)
        onehot_encoder = OneHotEncoder(sparse=False)
        maxi = int(np.max(binned_data))
        onehot_encoder.fit(np.asarray(range(maxi + 1)).reshape(-1, 1))
        binned_data = binned_data.reshape(-1, 1)
        onehot_encoded = onehot_encoder.transform(binned_data)
    return onehot_encoded

# produces a weighted (not one-hot) encoding for a label for classification
# e.g. (0 0 0.05 0.1 0.7 0.1 0.05 0 0)
def weighted_classes_encoding(labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # suppress warnings in this section
        binned_data = bin_numbers(labels)
        encoded = np.zeros((binned_data.shape[0],len(BIN_LOCATIONS)))
        for i in range(binned_data.shape[0]):
            peak = int(binned_data[i])
            dist1 = [peak-1,peak+1]
            dist2 = [peak-2,peak+2]
            dist1 = [i for i in dist1 if i >=0 and i< len(BIN_LOCATIONS)]
            dist2 = [i for i in dist2 if i >=0 and i< len(BIN_LOCATIONS)]
            encoded[i, peak] = 3
            encoded[i, dist1] = 0.428571
            encoded[i, dist2] = 0.157895
        normalized = encoded/encoded.sum(axis=1,keepdims=1)
    return normalized

# produces a encoding for ordinal classification
# cf. http://www.cs.miami.edu/home/zwang/files/rank.pdf
# (1 1 1 1 0 0 0) instead of (0 0 0 1 0 0 0)
def ordinal_classification_encoding(labels):
    labels = np.reshape(labels, [np.shape(labels)[0], 1])  # reshape to (?, 1)
    labels = np.clip(labels, 0, 36)  # clip labels

    def convert_value(val):
        bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
        encoding = ((bins <= val)*1)[0:-1]
        return encoding

    converted = np.apply_along_axis(convert_value, 1, labels)
    return converted

# produces the bin number of the ordinal_classification_encoding (not just argmax anymore)
def ordinal_classification_bin_number(encoded):
    def convert_list_to_ind(encoded_line):
        # return last index with value above threshold
        threshold = 0.5
        ind = np.asarray(list(range(encoded_line.shape[0])))
        indices_above_threshold = ind[encoded_line[ind]>threshold]
        if len(indices_above_threshold) == 0:
            return 0
        return indices_above_threshold[-1]
    converted = np.apply_along_axis(convert_list_to_ind, 1, encoded)
    return converted

def cutdown(img, threshold=0.94):
    threshold = threshold+10*np.finfo(float).eps
    i = 0
    while np.min(img[i, :]) > threshold:
        i = i+1
        if i >= np.shape(img)[0]:
            i = 0
            break
    img = img[i:, :]

    i = 0
    while np.min(img[np.shape(img)[0]-1-i, :]) > threshold:
        i = i+1
        if i >= np.shape(img)[0]:
            i = 0
            break
    img = img[0:np.shape(img)[0]-i,:]

    i = 0
    while np.min(img[:, i]) > threshold:
        i = i+1
        if i >= np.shape(img)[1]:
            i = 0
            break
    img = img[:, i:]

    i = 0
    while np.min(img[:,np.shape(img)[1]-1-i]) > threshold:
        i = i+1
        if i >= np.shape(img)[1]:
            i = 0
            break
    img = img[:, 0:np.shape(img)[1]-i]

    return img

def blur_lines(image):
    new_img = 2*np.copy(image)
    for i in range(5):
        blurred = filters.gaussian(image, (i+1)*2, multichannel=False)
        new_img = np.add(new_img,blurred)
    return new_img


def preprocess(image):
    img_gray = rgb2gray(image)
    img_gray = gamma_correct(img_gray)
    thresh_cut = np.percentile(img_gray, 4)
    img_gray = cutdown(img_gray, thresh_cut)
    thresh_white = np.percentile(img_gray, 8)
    img_gray[img_gray>thresh_white] = 1
    resized_img = resize_padded(img_gray, CANVAS_SIZE)
    if not DATA_AUGMENTATION:
        # if data augmentation -> normalization is done later
        resized_img = normalization(resized_img)
    return resized_img


def preprocess_progress_plot(image):
    img_gray = rgb2gray(image)
    totim = img_gray
    pause = np.zeros((np.shape(totim)[0],20))
    img_gray = gamma_correct(img_gray)
    totim =np.concatenate((pause,totim,pause,img_gray,pause), axis=1)
    thresh_cut = np.percentile(img_gray, 4)
    img_gray = cutdown(img_gray, thresh_cut)
    add = np.zeros((np.shape(totim)[0],np.shape(img_gray)[1]))
    add[np.int((np.shape(totim)[0]-np.shape(img_gray)[0])/2):np.int((np.shape(totim)[0]-np.shape(img_gray)[0])/2)+np.shape(img_gray)[0],:]=img_gray
    totim=np.concatenate((totim,add,pause),axis=1)
    thresh_white = np.percentile(img_gray, 8)
    img_gray[img_gray>thresh_white] = 1
    add[np.int((np.shape(totim)[0]-np.shape(img_gray)[0])/2):np.int((np.shape(totim)[0]-np.shape(img_gray)[0])/2)+np.shape(img_gray)[0], :] = img_gray
    totim = np.concatenate((totim, add,pause), axis=1)
    resized_img = resize_padded(img_gray, CANVAS_SIZE)
    add = np.zeros((np.shape(totim)[0], np.shape(resized_img)[1]))
    add[np.int((np.shape(totim)[0]-np.shape(resized_img)[0])/2):np.int((np.shape(totim)[0]-np.shape(resized_img)[0])/2)+np.shape(resized_img)[0], :] = resized_img
    totim = np.concatenate((totim, add, pause), axis=1)
    #plt.figure()
    #plt.imshow(totim, cmap='Greys_r')
    #plt.axis('off')
    #plt.show()
    imwrite('../new_data/preprocess_progress/'+str(np.random.uniform(0,1))+'.jpg', totim*255)
    resized_img = normalization(resized_img)
    return resized_img

def preprocess_basic(image):
    """ only grayscale and resize, no more processing, used to compare to preprocessed image """
    img_gray = rgb2gray(image)
    fill_cval = 1  # color of the padded area
    resized_img = resize_padded(img_gray, (4*CANVAS_SIZE[0], 4*CANVAS_SIZE[1]), fill_cval=fill_cval)
    resized_img = np.concatenate((np.ones((100, resized_img.shape[1])), resized_img), axis=0)
    return resized_img



