import argparse

import cv2
from cv2 import imread, resize
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from constants import *
from src.data_preprocessing.helpers import cutdown

"""
this script is used to serialize simulated data_preprocessing. It is expected that the directory with simulated data_preprocessing is in the data_preprocessing 
root (see structure in prepare_data.py) and has the following contents: 

├──simulated
    ├── distorted_placed_correctly
    ├── max_score
    ├── warped_figures
    ├── warped_figures_distorted_placed_correctly
    ├── ScoresDistortedPlacedCorrectly_WARPED.csv
    ├── ScoresDistortedPlacedCorrectly.csv
    ├── ScoresMaxScore_WARPED.csv
    ├── ScoresMaxScore.csv
"""

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, required=False, default='../data')
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
args = parser.parse_args()

DATA_DIRS_AND_LABELS = [
    ('distorted_placed_correctly', 'ScoresDistortedPlacedCorrectly.csv'),
    ('max_score', 'ScoresMaxScore.csv'),
    ('warped_figures', 'ScoresMaxScore_WARPED.csv'),
    ('warped_figures_distorted_placed_correctly', 'ScoresDistortedPlacedCorrectly_WARPED.csv')
]

LABEL_DF_COLUMNS = ['image_filepath', 'serialized_filepath', 'summed_score']
LABEL_DF_COLUMNS += [f"score_item_{i + 1}" for i in range(N_ITEMS)]


def main(data_root, image_size):
    data_root = os.path.abspath(data_root)
    simulated_data_dir = os.path.join(data_root, 'simulated/')
    serialized_dir = os.path.join(data_root, f'serialized-data/simulated')

    # mirror dir structure in serialized dir
    for data_dir, _ in DATA_DIRS_AND_LABELS:
        serialized_subdir = os.path.join(serialized_dir, data_dir)
        if not os.path.exists(serialized_subdir):
            os.makedirs(serialized_subdir)

    # create single label df with all labels
    labels_df = pd.DataFrame(columns=LABEL_DF_COLUMNS)
    for data_dir, csv_file in DATA_DIRS_AND_LABELS:
        df = pd.read_csv(os.path.join(simulated_data_dir, csv_file))
        df['image_file'] = df['NAME'].apply(lambda fn: os.path.join(simulated_data_dir, data_dir, fn))
        df['image_file_serialized'] = df['NAME'].apply(lambda fn: os.path.join(
            serialized_dir, data_dir, str(fn).replace('.jpg', '.npy')))
        df = df.rename(columns={'tot_score': 'summed_score',
                                'image_file': 'image_filepath',
                                'image_file_serialized': 'serialized_filepath',
                                **{str(i + 1): f"score_item_{i + 1}" for i in range(N_ITEMS)}})
        labels_df = pd.concat([labels_df, df.drop(columns='NAME')], ignore_index=True)

    labels_df['augmented'] = False
    labels_df['median_score'] = labels_df.loc[:, ['summed_score']]
    labels_df['figure_id'] = labels_df.loc[:, ["image_filepath"]].applymap(
        lambda s: os.path.splitext(os.path.split(s)[-1])[0])
    labels_df_path = os.path.join(serialized_dir, 'simulated_labels.csv')
    labels_df.to_csv(labels_df_path)
    print(f'saved labels as {labels_df_path}')

    # loop through images and process each (currently this is just saving the images as npy files and optional resize)
    if image_size is not None:
        image_size = image_size[::-1]

    progress_bar = tqdm(labels_df.iterrows(), total=len(labels_df), leave=False)
    for idx, row in progress_bar:
        image = imread(row['image_filepath'], flags=cv2.IMREAD_GRAYSCALE)

        # cutdown
        image = cutdown(image, threshold=np.percentile(image, 0.1), pad=10)

        # resize
        if image_size is not None:
            image = resize(image, dsize=image_size, interpolation=cv2.INTER_AREA)

        np.save(row['serialized_filepath'], image)
        progress_bar.set_description(f'saved image as {row["serialized_filepath"]}', refresh=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    main(data_root=args.data_root, image_size=args.image_size)
