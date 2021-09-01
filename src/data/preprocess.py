import numpy as np

from skimage.color import rgb2gray as skimage_rgb2gray
from skimage.morphology import erosion as skimage_erosion
from skimage.exposure import adjust_gamma as skimage_adjust_gamma

# from src.data.loading import load_raw_data
from src.data.helpers import cutdown, resize_padded


def preprocess_image(image, target_size):
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


# def check_if_in_range(value, upper_bound, lower_bound):
#     return lower_bound <= value < upper_bound

# def preprocess_labels(score):
#     overall_score = score[-1]
#
#     classification_label = None
#     for i, _range in enumerate(BIN_LOCATIONS):
#         is_location = check_if_in_range(overall_score, _range[1], _range[0])
#         if is_location:
#             classification_label = i
#
#     classification_label_dense = None
#     for i, _range in enumerate(BIN_LOCATIONS_DENSE):
#         is_location = check_if_in_range(overall_score, _range[1], _range[0])
#         if is_location:
#             classification_label_dense = i
#
#     return classification_label, classification_label_dense, score


# def preprocess_files(data_dir, mode='train', verbose=False):
#     """
#     preprocesses raw images from data_dir and saves them as .npy files in data_dir/<mode> together with a csv file
#         containing metadata on labels etc.
#
#         data_dir: str, directory which contains data
#         mode: str, must be either train or test
#         verbose: boolean
#     """
#     if mode not in ['train', 'test']:
#         raise ValueError(f'mode must be one of ["train", "test"]! got {mode}')
#
#     if not os.path.exists(data_dir):
#         raise NotADirectoryError(f'data_dir not found: {data_dir}')
#
#     df_dict = {
#         "base_img": [],
#         "filename": [],
#         "path_npy": [],
#         "classification_label": [],
#         "classification_label_dense": [],
#         "regression_label": [],
#     }
#
#     fn = f'{mode}_metadata.csv'
#
#     path_images_dir = os.path.join(data_dir, mode, 'serialized/')
#     csv_dir = os.path.join(data_dir, mode)
#
#     # create dir
#     if not os.path.exists(path_images_dir):
#         os.makedirs(path_images_dir)
#
#     # load data
#     figures, labels, files = load_raw_data(mode=mode)
#     num_files = len(files)
#
#     for figure, label, file in tqdm(zip(figures, labels, files), total=num_files):
#         preprocessed_img = preprocess_image(figure.get_image(mode=mode))
#         classification_label, classification_label_dense, regression_label = preprocess_labels(label)
#
#         npy_path = path_images_dir + figure.filename.split(".")[0] + ".npy"
#         np.save(npy_path, preprocessed_img)
#
#         df_dict["base_img"].append(figure.filename.split(".")[0])
#         df_dict["filename"].append(figure.filename)
#         df_dict["classification_label"].append(classification_label)
#         df_dict["classification_label_dense"].append(classification_label_dense)
#         df_dict["regression_label"].append(regression_label)
#         df_dict["path_npy"].append(npy_path)
#
#     df = pd.DataFrame.from_dict(df_dict)
#     df.to_csv(csv_dir + "/" + fn)
#
#     if verbose:
#         print(f'[INFO] Number of files found: {len(files)}')
#         print(f'[INFO] saved metadata as: {csv_dir + "/" + fn}')
#         print(f'[INFO] metadata:\n{df}')


### OLD
# def preprocess_image(image, data_augmentation):
# img_gray = rgb2gray(image)
# # img_gray = gamma_correct(img_gray)
# # img_gray = cutdown(img_gray, thresh_cut)
# # thresh_cut = np.percentile(img_gray, 4)
# # thresh_white = np.percentile(img_gray, 8)
# # img_gray[img_gray > thresh_white] = 1
# # resized_img = resize_padded(img_gray, CANVAS_SIZE)
#
# if data_augmentation:
#     # normalization is done later
#     return resized_img
#
# return normalize(resized_img)
