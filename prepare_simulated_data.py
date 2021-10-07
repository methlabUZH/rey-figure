import argparse

import cv2
from cv2 import imread, resize
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

"""
this script is used to serialize simulated data. It is expected that the directory with simulated data is in the data 
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
parser.add_argument('--image-size', nargs='+', default=(280, 200), help='height and width', type=int)
args = parser.parse_args()

DATA_DIRS_AND_LABELS = [
    ('distorted_placed_correctly', 'ScoresDistortedPlacedCorrectly.csv'),
    ('max_score', 'ScoresMaxScore.csv'),
    ('warped_figures', 'ScoresMaxScore_WARPED.csv'),
    ('warped_figures_distorted_placed_correctly', 'ScoresDistortedPlacedCorrectly_WARPED.csv')
]

LABEL_DF_COLUMNS = ['image_file', 'image_file_serialized', 'total_score'] + [str(i) for i in range(1, 19)]


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
        df = df.rename(columns={'tot_score': 'total_score'})
        df['image_file'] = df['NAME'].apply(lambda fn: os.path.join(simulated_data_dir, data_dir, fn))
        df['image_file_serialized'] = df['NAME'].apply(lambda fn: os.path.join(
            serialized_dir, data_dir, str(fn).replace('.jpg', '.npy')))
        labels_df = pd.concat([labels_df, df.drop(columns='NAME')], ignore_index=True)

    # loop through images and process each (currently this is just saving the images as npy files and optional resize)
    t = tqdm(labels_df.iterrows(), total=len(labels_df), leave=False)
    for idx, row in t:
        image = imread(row['image_file'], flags=cv2.IMREAD_GRAYSCALE)
        if image_size is not None:
            image = resize(image, dsize=image_size, interpolation=cv2.INTER_AREA)
        np.save(row['image_file_serialized'], image)
        t.set_description(f'saved image as {row["image_file_serialized"]}', refresh=True)


if __name__ == '__main__':
    main(data_root=args.data_root, image_size=args.image_size)
