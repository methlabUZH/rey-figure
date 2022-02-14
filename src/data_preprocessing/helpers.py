import numpy as np
from skimage.transform import resize
from skimage import filters


# import pandas as pd

# from math import floor
# # import decimal
# from skimage.morphology import erosion
# from scipy.ndimage.filters import minimum_filter
# from skimage.exposure import adjust_gamma
# from sklearn.preprocessing import OneHotEncoder


# import warnings

# from constants import BIN_LOCATIONS_DENSE, BIN_LOCATIONS, CANVAS_SIZE
# from src.dataloaders.helpers import check_if_in_range


def resize_padded(img, new_shape, fill_cval=None, order=1, anti_alias=True):
    if fill_cval is None:
        fill_cval = np.max(img)

    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio / 2 for s in img.shape]).astype(np.int) * 2
    interm_img = resize(img, interm_shape, order=order, cval=fill_cval, anti_aliasing=anti_alias, mode='constant')

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[tuple([slice(p, -p, None) if 0 != p else slice(None, None, None) for p in pad])] = interm_img

    return new_img


# normalize image to zero mean and unit variance
def normalize(img):
    img = (img - np.mean(img)) / np.std(img)
    return img


# make image interpretable again as grayscale image (=~ undo normalization)
def undo_normalization(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / (img_max - img_min)
    return img


def cutdown(img, threshold=0.94, pad=0):
    threshold = threshold + 10 * np.finfo(float).eps
    i = 0
    while np.min(img[i, :]) > threshold:
        i = i + 1
        if i >= np.shape(img)[0]:
            i = 0
            break
    img = img[max(0, i - pad):, :]

    i = 0
    while np.min(img[np.shape(img)[0] - 1 - i, :]) > threshold:
        i = i + 1
        if i >= np.shape(img)[0]:
            i = 0
            break
    img = img[0:np.shape(img)[0] - max(i - pad, 0), :]

    i = 0
    while np.min(img[:, i]) > threshold:
        i = i + 1
        if i >= np.shape(img)[1]:
            i = 0
            break
    img = img[:, max(i - pad, 0):]

    i = 0
    while np.min(img[:, np.shape(img)[1] - 1 - i]) > threshold:
        i = i + 1
        if i >= np.shape(img)[1]:
            i = 0
            break
    img = img[:, 0:np.shape(img)[1] - max(i - pad, 0)]

    return img


def blur_lines(image):
    new_img = 2 * np.copy(image)

    for i in range(5):
        blurred = filters.gaussian(image, (i + 1) * 2, multichannel=False)
        new_img = np.add(new_img, blurred)

    return new_img

# # gamma correction for images in [0,1]
# def gamma_correct(img, gamma=3):
#     img = erosion(img)
#     img = adjust_gamma(img, gamma)
#     return img


# # function returning if image is already sufficiently clear
# # img: grayscale image
# def sufficient(img, windowsize=2, threshold=0.471):
#     # compute starting points
#     m = floor(img.shape[0] / 2)  # rows
#     n = floor(img.shape[1] / 2)  # columns
#
#     #  iterate over part of the image
#     for i in range(m, img.shape[0] - windowsize, windowsize):
#         for j in range(n, img.shape[1] - windowsize, windowsize):
#
#             if np.mean(img[i:(i + windowsize), j:(j + windowsize)]) < threshold:
#                 return True
#
#     return False


# def procedure_min(img, size=3):
#     if sufficient(img):
#         return minimum_filter(img, size=2)
#
#     return minimum_filter(img, size=size)


# def procedure_erosion(img, threshold=90, windowsize=3, niter=1):
#     if sufficient(img, threshold=threshold, windowsize=windowsize):
#         return img
#
#     enhanced_img = None
#     for i in range(niter):
#
#         enhanced_img = erosion(img)
#         if sufficient(enhanced_img, threshold=threshold, windowsize=windowsize):
#             return enhanced_img
#
#     return enhanced_img


# # To be used directly on original labels
# def bin_numbers(labels, bin_locations):
#     shape = np.shape(labels)
#     # reshape to 1d
#     shape_sum = np.prod(shape)
#     labels_1d = np.reshape(labels, shape_sum)
#     labels_1d = np.clip(labels_1d, 0, 36)  # clip labels (otherwise you have -1 for <0 / >36)
#     bins = pd.IntervalIndex.from_tuples(bin_locations, closed='left')
#     categorical_object = pd.cut(labels_1d, bins)
#     binned_data_1d = categorical_object.codes
#     # reshape back to original shape and cast to float
#     binned_data = np.reshape(binned_data_1d, shape).astype('float32')
#     return binned_data
#
#
# def bin_numbers_continuous(labels):
#     labels = np.clip(labels, 0, 36)  # clip labels (otherwise you have -1 for <0 / >36)
#
#     def convert_value(val):
#         bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
#         binned = np.argmin(bins <= val) - 1
#         bin_start = bins[binned]
#         bin_end = bins[binned + 1]
#         frac = (val - bin_start) / (bin_end - bin_start)
#         return binned + frac - 0.5
#
#     map_func = np.vectorize(convert_value)
#
#     converted = map_func(labels)
#     return converted


# def postprocess_binned_labels(labels):
#     """
#     for continuous bins this will return the integer bin
#     for integer bins this doesn't change anything (note that predictions are continuous nonetheless)
#     note that python/numpy rounds 0.5 to 0 and 1.5 to 2, this is why the rounding here is done manually
#     """
#
#     def round_value(val):
#         context = decimal.getcontext()
#         context.rounding = decimal.ROUND_HALF_UP
#         rounded_val = round(decimal.Decimal(float(val)), 0)
#         clipped_val = float(np.clip(rounded_val, 0, 12))
#         return clipped_val
#
#     map_func = np.vectorize(round_value)
#
#     converted = map_func(labels)
#     return converted
#
#
# # To be used directly on original labels
# def one_hot_encoding(labels):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")  # suppress warnings in this section
#         binned_data = bin_numbers(labels)
#         onehot_encoder = OneHotEncoder(sparse=False)
#         maxi = int(np.max(binned_data))
#         onehot_encoder.fit(np.asarray(range(maxi + 1)).reshape(-1, 1))
#         binned_data = binned_data.reshape(-1, 1)
#         onehot_encoded = onehot_encoder.transform(binned_data)
#
#     return onehot_encoded


# # produces a weighted (not one-hot) encoding for a label for classification
# # e.g. (0 0 0.05 0.1 0.7 0.1 0.05 0 0)
# def weighted_classes_encoding(labels, bin_locations):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")  # suppress warnings in this section
#         binned_data = bin_numbers(labels)
#         encoded = np.zeros((binned_data.shape[0], len(bin_locations)))
#
#         for i in range(binned_data.shape[0]):
#             peak = int(binned_data[i])
#             dist1 = [peak - 1, peak + 1]
#             dist2 = [peak - 2, peak + 2]
#             dist1 = [i for i in dist1 if 0 <= i < len(bin_locations)]
#             dist2 = [i for i in dist2 if 0 <= i < len(bin_locations)]
#             encoded[i, peak] = 3
#             encoded[i, dist1] = 0.428571
#             encoded[i, dist2] = 0.157895
#
#         normalized = encoded / encoded.sum(axis=1, keepdims=1)
#
#     return normalized


# # produces a encoding for ordinal classification
# # cf. http://www.cs.miami.edu/home/zwang/files/rank.pdf
# # (1 1 1 1 0 0 0) instead of (0 0 0 1 0 0 0)
# def ordinal_classification_encoding(labels):
#     labels = np.reshape(labels, [np.shape(labels)[0], 1])  # reshape to (?, 1)
#     labels = np.clip(labels, 0, 36)  # clip labels
#
#     def convert_value(val):
#         bins = np.array([0, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 100])
#         encoding = ((bins <= val) * 1)[0:-1]
#         return encoding
#
#     converted = np.apply_along_axis(convert_value, 1, labels)
#     return converted
#
#
# # produces the bin number of the ordinal_classification_encoding (not just argmax anymore)
# def ordinal_classification_bin_number(encoded):
#     def convert_list_to_ind(encoded_line):
#         # return last index with value above threshold
#         threshold = 0.5
#         ind = np.asarray(list(range(encoded_line.shape[0])))
#         indices_above_threshold = ind[encoded_line[ind] > threshold]
#
#         if len(indices_above_threshold) == 0:
#             return 0
#
#         return indices_above_threshold[-1]
#
#     converted = np.apply_along_axis(convert_list_to_ind, 1, encoded)
#     return converted


# def preprocess(image, data_augmentation=False):
#     img_gray = rgb2gray(image)
#     img_gray = gamma_correct(img_gray)
#     thresh_cut = np.percentile(img_gray, 4)
#     img_gray = cutdown(img_gray, thresh_cut)
#     thresh_white = np.percentile(img_gray, 8)
#     img_gray[img_gray > thresh_white] = 1
#     resized_img = resize_padded(img_gray, CANVAS_SIZE)
#
#     if data_augmentation:
#         # normalization is done later
#         return resized_img
#
#     return normalization(resized_img)


# def preprocess_basic(image):
#     """ only grayscale and resize, no more processing, used to compare to preprocessed image """
#     img_gray = rgb2gray(image)
#     fill_cval = 1  # color of the padded area
#     resized_img = resize_padded(img_gray, (4 * CANVAS_SIZE[0], 4 * CANVAS_SIZE[1]), fill_cval=fill_cval)
#     # resized_img = resize_padded(img_gray, (4 * augmented_CANVAS_SIZE[0], 4 * augmented_CANVAS_SIZE[1]),
#     # fill_cval=fill_cval)
#     print("shape reshaped ")
#
#     resized_img = np.concatenate((np.ones((100, resized_img.shape[1])), resized_img), axis=0)
#     return resized_img


# def preprocess_dataset(figures, labels, files, set="train"):
#     """Loads raw data_preprocessing and preprocesses it.
#     :return: preprocessed images, files, labels
#     """
#
#     preprocessed_labels = preprocess_labels(labels)
#     preprocessed_labels = np.asarray(preprocessed_labels)
#     files = np.asarray(files)
#
#     # preprocessed_labels, files = shuffle(preprocessed_labels, files, random_state=RANDOM_STATE)
#
#     if len(figures) == 1:
#         figures = [figures]
#
#     images = [fig.getImage(set=set) for fig in figures]
#
#     preprocessed_images = preprocess_images(images)
#
#     #################################################
#     # TODO: create function to save serialized data_preprocessing  #
#     #################################################
#     if REDO_PREPROCESSING_LABELS or REDO_PREPROCESSING_IMAGES:
#         # save to disk for later use
#         if (DEBUG):
#             print("Writing preprocessed data_preprocessing to disk...")
#         create_directory(DATA_DIR + "serialized")
#
#     if REDO_PREPROCESSING_IMAGES:
#         # save images depending on data_preprocessing augmentation or not
#         if DATA_AUGMENTATION:
#             np.save(DATA_DIR + 'serialized/images_aug.npy', preprocessed_images)
#         else:
#             np.save(DATA_DIR + 'serialized/images.npy', preprocessed_images)
#
#     if REDO_PREPROCESSING_LABELS:
#         # save labels depending on LABEL_FORMAT
#         if LABEL_FORMAT == 'one-per-item':
#             np.save(DATA_DIR + 'serialized/labels_oneperitem.npy', preprocessed_labels)
#         elif LABEL_FORMAT == 'three-per-item':
#             np.save(DATA_DIR + 'serialized/labels_threeperitem.npy', preprocessed_labels)
#         else:
#             np.save(DATA_DIR + 'serialized/labels.npy', preprocessed_labels)
#
#         # save files
#         np.save(DATA_DIR + 'serialized/files.npy', files)
#
#     # change permission of written files
#     try:
#         if REDO_PREPROCESSING_IMAGES:
#             if DATA_AUGMENTATION:
#                 os.chmod(DATA_DIR + 'serialized/images_aug.npy', 0o777)
#             else:
#                 os.chmod(DATA_DIR + 'serialized/images.npy', 0o777)
#         if REDO_PREPROCESSING_LABELS:
#             if LABEL_FORMAT == 'one-per-item':
#                 os.chmod(DATA_DIR + 'serialized/labels_oneperitem.npy', 0o777)
#             elif LABEL_FORMAT == 'three-per-item':
#                 os.chmod(DATA_DIR + 'serialized/labels_threeperitem.npy', 0o777)
#             else:
#                 os.chmod(DATA_DIR + 'serialized/labels.npy', 0o777)
#             os.chmod(DATA_DIR + 'serialized/files.npy', 0o777)
#     except:
#         pass
#     #################################################
#     # TODO: create function to save serialized data_preprocessing #
#     #################################################
#
#     # TODO: Add TEST part of prepare_dataset() function
#
#     return preprocessed_images, preprocessed_labels, files
#
#
# def preprocess_images(images, set="train") -> object:
#     if (DEBUG):
#         print("Preprocessing " + str(len(images)) + " images...")
#     for i in range(len(images)):
#         images[i] = preprocess(images[i])
#         # if set == "test" and DATA_AUGMENTATION:
#         #    images[i] = simulate_augment(images[i])
#         if i != 0 and i % 100 == 0:
#             print("{}% ".format(i * 100 // len(images)), end='', flush=True)
#     print(" ")
#     return images
#
#
# def preprocess_labels(score):
#     final_score = score[-1]
#
#     classification_label = None
#     for i, _range in enumerate(BIN_LOCATIONS):
#         is_location = check_if_in_range(final_score, _range[1], _range[0])
#         if is_location:
#             classification_label = i
#
#     classification_label_dense = None
#     for i, _range in enumerate(BIN_LOCATIONS_DENSE):
#         is_location = check_if_in_range(final_score, _range[1], _range[0])
#         if is_location:
#             classification_label_dense = i
#
#     return classification_label, classification_label_dense, score
