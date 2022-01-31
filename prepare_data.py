import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import random
from tabulate import tabulate
import time
from tqdm import tqdm

from src.utils import timestamp_human
from src.data_preprocessing.preprocess import preprocess_image, simulate_augment_image

from src.data_preprocessing.reyfigure import ReyFigure
from src.data_preprocessing.loading import join_ground_truth_files
from src.data_preprocessing.helpers import normalize
from src.data_preprocessing.augmentation import augment_image, AugmentParameters
from constants import DEFAULT_CANVAS_SIZE, AUGM_CANVAS_SIZE

"""
this script expects your data to be organized like this:

├──data_root
    ├── DBdumps
    ├── ProlificExport
    ├── ReyFigures
    │ ├── data2018
    │ │ ├── newupload
    │ │ ├── newupload_15_11_2018
    │ │ ├── newupload_9_11_2018
    │ │ ├── uploadFinal
    │ │ └── uploadFinalREF
    │ └── data2021
    │     ├── KISPI
    │     ├── Tino_cropped
    │     ├── Typeform
    │     ├── USZ_fotos
    │     └── USZ_scans
    ├── UserRatingData
    └── simulated
"""

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, required=True)
parser.add_argument('--dataset', type=str,
                    choices=['debug', 'scans-2018', 'scans-2018-2021', 'data-2018-2021', 'fotos-2021'])
parser.add_argument('--augment', action='store_true')
parser.add_argument('--preprocessing', default=1, choices=[0, 1], type=int,
                    help='type of preprocessing: 0 is minimal, 1 is erosion, contrast, cut whitespace, bg whitening')
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
args = parser.parse_args()

# only 2018 data
data_2018 = ['newupload_15_11_2018', 'newupload_9_11_2018', 'uploadFinal', 'newupload']
debug_data = ['newupload_15_11_2018']

# all 2021 data
data_2021_fotos = ['USZ_fotos', 'Typeform']
data_2021_scans = ['USZ_scans', 'Tino_cropped', 'KISPI']

# scanned images from 2018 and 2021
scans_2018_2021 = data_2018 + data_2021_scans

# all images from 2018 and 2021
data_2018_2021 = data_2018 + data_2021_scans + data_2021_fotos

# foto figures
data_2021_fotos = ['USZ_fotos', 'Typeform']

datasets = {'scans-2018': data_2018,
            'scans-2018-2021': scans_2018_2021,
            'data-2018-2021': data_2018_2021,
            'fotos-2021': data_2021_fotos,
            'debug': debug_data}

args.data_root = os.path.abspath(args.data_root)


def worker(figure, npy_filepath, label, augment_data, target_size, preprocessing_version, q):
    if augment_data:
        original_image_preprocessed = preprocess_image(figure.get_image(), target_size=AUGM_CANVAS_SIZE,
                                                       version=preprocessing_version)
    else:
        original_image_preprocessed = preprocess_image(figure.get_image(), target_size=target_size,
                                                       version=preprocessing_version)

    # augment image
    augm_npy_filepaths = []
    if augment_data:
        for i in range(AugmentParameters.num_augment):
            augm_image = augment_image(image=original_image_preprocessed,
                                       alpha_elastic_transform=AugmentParameters.alpha_elastic_transform,
                                       sigma_elastic_transform=AugmentParameters.sigma_elastic_transform,
                                       max_factor_skew=AugmentParameters.max_factor_skew,
                                       max_angle_rotate=AugmentParameters.max_angle_rotate,
                                       target_size=target_size)

            augm_image = normalize(augm_image)
            save_fp = os.path.splitext(npy_filepath)[0] + f'_augmented{i + 1}.npy'
            augm_npy_filepaths.append(save_fp)
            np.save(save_fp, augm_image)

    if augment_data:
        original_image_preprocessed = simulate_augment_image(image=original_image_preprocessed,
                                                             gaussian_sigma=AugmentParameters.gaussian_sigma,
                                                             target_size=target_size)

    np.save(npy_filepath, original_image_preprocessed)

    data = [{'figure_id': figure.figure_id,
             'filepath_image': figure.filepath,
             'filepath_npy': npy_filepath,
             'label': label,
             'augmented': False},
            *[{'figure_id': figure.figure_id + f'_augm{i + 1}',
               'filepath_image': figure.filepath,
               'filepath_npy': fp_npy,
               'label': label,
               'augmented': True} for i, fp_npy in enumerate(augm_npy_filepaths)]]

    q.put(data)


def listener(columns, save_as, q):
    df = pd.DataFrame(columns=columns)

    while 1:
        data = q.get()

        if data == "kill":
            df.set_index('figure_id', inplace=True)
            df.to_csv(save_as)
            break

        for d in data:
            figure_id = d.get('figure_id')
            filepath_image = d.get('filepath_image')
            fp_npy = d.get('filepath_npy')
            label = d.get('label')
            augmented = d.get('augmented')

            # add to df
            df.loc[-1] = [figure_id, filepath_image, fp_npy, augmented] + label
            df.index += 1
            df.sort_index()


def preprocess_data(data_root, dataset_name, image_size, preprocessing_version, augment_data=False):
    serialized_dir = os.path.join(data_root, 'serialized-data',
                                  f'{dataset_name}-{image_size[0]}x{image_size[1]}-pp{preprocessing_version}')

    if augment_data:
        serialized_dir += '-augmented'

    data_dirs = datasets[dataset_name]

    for data_dir in data_dirs:
        if data_dir in data_2018:
            os.makedirs(os.path.join(serialized_dir, 'data2018', data_dir))

        if data_dir in data_2021_scans or data_dir in data_2021_fotos:
            os.makedirs(os.path.join(serialized_dir, 'data2021', data_dir))

    # map filenames to paths (assumption: filenames are unique)
    figure_names_and_paths = []
    for (dirpath, dirname, filenames) in os.walk(os.path.join(data_root, 'ReyFigures')):
        rel_dirpath = './' + dirpath.split(os.path.normpath(data_root) + '/')[-1]
        dirpath_serialized = serialized_dir + '/' + rel_dirpath.split('./ReyFigures/')[-1]

        if os.path.split(dirpath)[-1] not in data_dirs:
            continue

        for fn in filenames:
            if fn.startswith('.'):  # exclude files like .DS_Store
                continue

            figure_id = os.path.splitext(fn)[0]
            figure_names_and_paths += [(figure_id,
                                        os.path.join(dirpath, fn),
                                        os.path.join(dirpath_serialized, str(figure_id) + '.npy'))]

    figure_ids = np.array(figure_names_and_paths)[:, 0].tolist()
    df_figure_paths = pd.DataFrame(data=figure_names_and_paths,
                                   columns=['figure_id', 'filepath_image', 'filepath_npy'])
    df_figure_paths = df_figure_paths.drop_duplicates(subset='figure_id')
    df_figure_paths = df_figure_paths.set_index('figure_id')
    figures_paths_fp = os.path.join(serialized_dir, 'figures_paths.csv')
    df_figure_paths.to_csv(figures_paths_fp)

    print(tabulate(df_figure_paths.head(10), headers='keys', tablefmt='psql'))
    print(f'number of unique figures: {len(df_figure_paths)}\n')

    # merge all user rating data files and save as csv
    df_user_ratings = join_ground_truth_files(labels_root=os.path.join(data_root, 'UserRatingData/'))
    df_user_ratings = df_user_ratings[df_user_ratings['figure_id'].isin(figure_ids)]
    df_user_ratings = df_user_ratings[df_user_ratings['FILE'].notna()]  # drop nan files
    user_rating_data_fp = os.path.join(serialized_dir, 'user_rating_data_merged.csv')
    df_user_ratings.to_csv(user_rating_data_fp)

    print(tabulate(df_user_ratings.head(10), headers='keys', tablefmt='psql'))
    print(f'\ntotal user rating rows without duplicates: {len(df_user_ratings)}')
    print(f'saved merged user rating data as {user_rating_data_fp}')

    # loop through all images and create figure objects
    print('\n* processing user rating data...')
    figures = {}
    for _, rating in tqdm(df_user_ratings.iterrows(), total=len(df_user_ratings)):
        figure_id = os.path.splitext(str(rating['FILE']))[0]
        relative_filepath = df_figure_paths.loc[figure_id]['filepath_image']

        try:
            abs_path_to_image = os.path.join(data_root, relative_filepath)
        except Exception as e:
            print(f'failed to load figure {figure_id}; err:\n\t{e}')
            continue

        if not os.path.exists(abs_path_to_image):
            print(f'file not found: {abs_path_to_image}')
            continue

        if figure_id not in figures:
            figures[figure_id] = ReyFigure(figure_id=figure_id, filepath=abs_path_to_image)

        assessment_id = rating['assessment_id']
        assessment = figures[figure_id].get_assessment(assessment_id)
        assessment.add_item(item_id=rating['part_id'],
                            score=rating['part_points'],
                            visible=rating['visible'],
                            right_place=rating['right_place'],
                            drawn_correct=rating['drawn_correct'])

    # convert to list
    figures = [figures[fig] for fig in figures if figures[fig].has_valid_assessment()]
    print(f'total number of figures with valid assessment:\t{len(figures)}')

    # extract labels with one-per-item scores; each label has shape (18 + 1,) one for each item + overall score
    labels = [fig.get_median_score_per_item() + [fig.get_median_score(), fig.get_sum_of_median_item_scores()]
              for fig in figures]

    # extract filepaths
    npy_filepaths = [df_figure_paths.loc[fig.figure_id]['filepath_npy'] for fig in figures]

    # prepocess figures and create label file
    label_cols = ['figure_id', 'image_filepath', 'serialized_filepath', 'augmented']
    label_cols += [f'score_item_{i + 1}' for i in range(18)]
    label_cols += ['median_score', 'summed_score']

    num_processes = mp.cpu_count() + 2

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    labels_csv = os.path.join(serialized_dir, 'labels.csv')
    _ = pool.apply_async(listener, (label_cols, labels_csv, q))

    start_time = time.time()
    print(f'\n* start preprocessing @ {timestamp_human()} ...')

    # fire off workers
    jobs = []
    for figure, label, npy_fp in zip(figures, labels, npy_filepaths):
        job = pool.apply_async(worker, (figure, npy_fp, label, augment_data, args.image_size, preprocessing_version, q))
        jobs.append(job)

    # collect results
    for i, job in enumerate(jobs):
        job.get()

    q.put('kill')
    pool.close()
    pool.join()

    end_time = time.time()
    print(f'\n* end preprocessing @ {timestamp_human()} ...')
    print(f'* total time for preprocessing:\t{end_time - start_time:.2f}s')

    df_labels = pd.read_csv(labels_csv)
    print(tabulate(df_labels.head(50), headers='keys', tablefmt='psql'))

    make_train_test_split(serialized_dir)
    compute_mean_and_std(serialized_dir, image_size)


def make_train_test_split(data_root):
    print('\n* generating train / test split...')

    labels_csv_fp = os.path.join(data_root, 'labels.csv')

    if not os.path.isfile(labels_csv_fp):
        raise FileNotFoundError(f'could not find labels csv in {labels_csv_fp}')

    labels_df = pd.read_csv(labels_csv_fp)
    labels_df_original = labels_df[labels_df.augmented == False]  # noqa
    labels_df_augmented = labels_df[labels_df.augmented == True]  # noqa
    num_original_datapoints = len(labels_df_original)

    train_indices = random.sample(list(range(num_original_datapoints)), k=int(num_original_datapoints * 0.8))
    test_indices = [i for i in range(num_original_datapoints) if i not in train_indices]
    assert set(train_indices).isdisjoint(test_indices)

    print(f'number of original datapoints:\t{num_original_datapoints}')
    print(f'number of original training datapoints:\t{len(train_indices)}')
    print(f'number of original testing datapoints:\t{len(test_indices)}')

    train_df_original = labels_df_original.iloc[train_indices]
    train_figure_ids = []
    for fid in train_df_original.figure_id.to_list():
        train_figure_ids.append(fid)
        for i in range(AugmentParameters.num_augment):
            train_figure_ids.append(fid + f'_augm{i + 1}')

    train_df = labels_df[labels_df.figure_id.isin(train_figure_ids)]
    test_df = labels_df_original.iloc[test_indices]

    assert len(test_df[test_df.figure_id.isin(train_figure_ids)]) == 0

    print(f'total number of original and augmented training datapoints:\t{len(train_df)}')

    train_df.to_csv(os.path.join(data_root, 'train_labels.csv'))
    test_df.to_csv(os.path.join(data_root, 'test_labels.csv'))


def compute_mean_and_std(data_root, image_size):
    train_labels = pd.read_csv(os.path.join(data_root, 'train_labels.csv'))

    filepaths = train_labels['serialized_filepath'].values.tolist()
    partial_sum = 0.0
    partial_sum_of_squares = 0.0
    num_images = 0

    print('\n* computing train set mean and std...')

    for fp in tqdm(filepaths):
        img = np.load(fp)
        image_shape = np.shape(img)

        if tuple(image_shape) != tuple(image_size):
            raise ValueError(f'image {fp} with inconsistent image size; expected {image_size}, got {image_shape}')

        partial_sum += np.sum(img)
        partial_sum_of_squares += np.sum(img ** 2)
        num_images += 1

    count = num_images * image_size[0] * image_size[1]
    total_mean = partial_sum / count
    total_std = np.sqrt((partial_sum_of_squares / count) - (total_mean ** 2))

    with open(os.path.join(data_root, 'trainset-stats.json'), 'w') as f:
        json.dump({'mean': total_mean, 'std': total_std}, f)
        print('saved mean and std in', os.path.join(data_root, 'trainset-stats.json'))

    print(f'training set mean:\t{total_mean}')
    print(f'training set std:\t{total_std}')


if __name__ == '__main__':
    preprocess_data(data_root=args.data_root,
                    dataset_name=args.dataset,
                    image_size=args.image_size,
                    preprocessing_version=args.preprocessing,
                    augment_data=args.augment)
