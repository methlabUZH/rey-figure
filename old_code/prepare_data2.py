import argparse
import json
import multiprocessing as mp

import cv2
import numpy as np
import os
import pandas as pd
import random
from tabulate import tabulate
import time
from tqdm import tqdm

from src.utils import timestamp_human

from src.data_preprocessing.reyfigure import ReyFigure
from src.data_preprocessing.loading import join_ground_truth_files
from src.data_preprocessing.helpers import resize_padded
from src.data_preprocessing.augmentation import AugmentParameters
from constants import DEFAULT_CANVAS_SIZE

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

_DEFAULT_DATA_ROOT = '/Users/maurice/phd/src/rey-figure/data/'

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, required=False, default=_DEFAULT_DATA_ROOT)
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
args = parser.parse_args()

_DATA_DIRECTORIES_18 = ['newupload_15_11_2018', 'newupload_9_11_2018', 'uploadFinal', 'newupload']
_DATA_DIRECTORIES_21 = ['USZ_fotos', 'Typeform', 'USZ_scans', 'Tino_cropped', 'KISPI']

_DATA_DIRECTORIES = _DATA_DIRECTORIES_18 + _DATA_DIRECTORIES_21
# _DATA_DIRECTORIES = ['newupload_15_11_2018']

args.data_root = os.path.abspath(args.data_root)


def worker(figure, jpg_fp, npy_fp, label, target_size, q):
    # load image as grayscale
    image = cv2.imread(figure.filepath, flags=cv2.IMREAD_GRAYSCALE).astype(float)

    # resize to target size
    image = resize_padded(image, new_shape=target_size)

    # save image as npy array
    cv2.imwrite(jpg_fp, image)
    np.save(file=npy_fp, arr=image[np.newaxis, :])

    data = [{'figure_id': figure.figure_id,
             'src_filepath': figure.filepath,
             'jpg_resized_image_fp': jpg_fp,
             'npy_resized_image_fp': npy_fp,
             'label': label,
             'augmented': False}]
    q.put(data)


def listener(save_as, q):
    columns = ['figure_id', 'src_filepath', 'jpg_resized_image_fp', 'npy_resized_image_fp', 'augmented']
    columns += [f'score_item_{i + 1}' for i in range(18)]
    columns += ['median_score', 'summed_score']
    df = pd.DataFrame(columns=columns)

    while 1:
        data = q.get()

        if data == "kill":
            df.set_index('figure_id', inplace=True)
            df.to_csv(save_as)
            break

        for d in data:
            figure_id = d.get('figure_id')
            src_fp = d.get('src_filepath')
            jpg_resized_fp = d.get('jpg_resized_image_fp')
            npy_resized_fp = d.get('npy_resized_image_fp')
            label = d.get('label')
            augmented = d.get('augmented')

            # add to df
            df.loc[-1] = [figure_id, src_fp, jpg_resized_fp, npy_resized_fp, augmented] + label
            df.index += 1
            df.sort_index()


def main(data_root, image_size):
    resized_data_dir = os.path.join(data_root, 'resized-data', f'{image_size[0]}x{image_size[1]}')

    # create dir structure
    for data_dir in _DATA_DIRECTORIES:
        if data_dir in _DATA_DIRECTORIES_18:
            os.makedirs(os.path.join(resized_data_dir, 'data2018', data_dir))
        else:
            os.makedirs(os.path.join(resized_data_dir, 'data2021', data_dir))

    # map filenames to paths (assumption: filenames are unique)
    figure_names_and_paths = []
    for (dirpath, dirname, filenames) in os.walk(os.path.join(data_root, 'ReyFigures')):
        rel_dirpath = './' + dirpath.split(os.path.normpath(data_root) + '/')[-1]
        dirpath_serialized = resized_data_dir + '/' + rel_dirpath.split('./ReyFigures/')[-1]

        if os.path.split(dirpath)[-1] not in _DATA_DIRECTORIES:
            continue

        for fn in filenames:
            if fn.startswith('.'):  # exclude files like .DS_Store
                continue

            figure_id = os.path.splitext(fn)[0]
            figure_names_and_paths += [(figure_id, os.path.join(dirpath, fn),
                                        os.path.join(dirpath_serialized, fn),
                                        os.path.join(dirpath_serialized, os.path.splitext(fn)[0] + '.npy'))]

    figure_ids = np.array(figure_names_and_paths)[:, 0].tolist()
    df_figure_paths = pd.DataFrame(
        data=figure_names_and_paths,
        columns=['figure_id', 'src_filepath', 'jpg_resized_image_fp', 'npy_resized_image_fp'])
    df_figure_paths = df_figure_paths.drop_duplicates(subset='figure_id')
    df_figure_paths = df_figure_paths.set_index('figure_id')
    figures_paths_fp = os.path.join(resized_data_dir, 'figures_paths.csv')
    df_figure_paths.to_csv(figures_paths_fp)

    print(tabulate(df_figure_paths.head(10), headers='keys', tablefmt='psql'))
    print(f'number of unique figures: {len(df_figure_paths)}\n')

    # merge all user rating data files and save as csv
    df_user_ratings = join_ground_truth_files(labels_root=os.path.join(data_root, 'UserRatingData/'))
    df_user_ratings = df_user_ratings[df_user_ratings['figure_id'].isin(figure_ids)]
    df_user_ratings = df_user_ratings[df_user_ratings['FILE'].notna()]  # drop nan files
    user_rating_data_fp = os.path.join(resized_data_dir, 'user_rating_data_merged.csv')
    df_user_ratings.to_csv(user_rating_data_fp)

    print(tabulate(df_user_ratings.head(10), headers='keys', tablefmt='psql'))
    print(f'\ntotal user rating rows without duplicates: {len(df_user_ratings)}')
    print(f'saved merged user rating data as {user_rating_data_fp}')

    # loop through all images and create figure objects
    print('\n* processing user rating data...')
    figures = {}
    for _, rating in tqdm(df_user_ratings.iterrows(), total=len(df_user_ratings)):
        figure_id = os.path.splitext(str(rating['FILE']))[0]
        relative_filepath = df_figure_paths.loc[figure_id]['src_filepath']

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

    num_processes = mp.cpu_count() + 2
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work first
    labels_csv = os.path.join(resized_data_dir, 'labels.csv')
    _ = pool.apply_async(listener, (labels_csv, q))

    start_time = time.time()
    print(f'\n* start preprocessing @ {timestamp_human()} ...')

    # extract filepaths
    jpg_resized_filepaths = [df_figure_paths.loc[fig.figure_id]['jpg_resized_image_fp'] for fig in figures]
    npy_resized_filepaths = [df_figure_paths.loc[fig.figure_id]['npy_resized_image_fp'] for fig in figures]

    # fire off workers
    jobs = []
    for figure, label, jpg_fp, npy_fp in zip(figures, labels, jpg_resized_filepaths, npy_resized_filepaths):
        job = pool.apply_async(worker, (figure, jpg_fp, npy_fp, label, args.image_size, q))
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

    make_train_test_split(resized_data_dir)


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

    print(f'total number of datapoints:\t{num_original_datapoints}')
    print(f'number of training datapoints:\t{len(train_indices)}')
    print(f'number of testing datapoints:\t{len(test_indices)}')

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


if __name__ == '__main__':
    main(data_root=args.data_root, image_size=args.image_size)
