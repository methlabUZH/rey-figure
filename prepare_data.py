import argparse
import cv2
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import shutil
from typing import *
from tqdm import tqdm

from constants import *
from src.preprocessing2 import resize_padded, create_label_files

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

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='./data')
parser.add_argument('--image-size', nargs='+', default=[232, 300], help='height and width', type=int)
parser.add_argument('--seed', type=int, default=1, help='random seed used for train and test split')
args = parser.parse_args()

# set numpy random seed (will be used by pandas)
np.random.seed(args.seed)


def main(data_root, image_size):
    serialized_dir = os.path.join(data_root, 'serialized-data', 'data_{}x{}-seed_{}'.format(*image_size, args.seed))
    serialized_data_dir_train = os.path.join(serialized_dir, "train")
    serialized_data_dir_test = os.path.join(serialized_dir, "test")

    if os.path.exists(serialized_data_dir_train):
        n_train = len(os.listdir(serialized_data_dir_train))

        if n_train > 0:
            print(f'dir {serialized_data_dir_train} already exists! found {n_train} files.')
            return

        shutil.rmtree(serialized_dir)
        print(f'no files in {serialized_dir}! removed {serialized_dir}')

    test_labels, train_labels = create_label_files(data_dir=data_root, test_fraction=TEST_FRACTION)

    os.makedirs(serialized_data_dir_test)
    os.makedirs(serialized_data_dir_train)

    # get list of filepaths
    figures_filepaths = {}
    for (dirpath, dirname, filenames) in os.walk(os.path.join(data_root, 'ReyFigures')):
        for fn in filenames:
            if fn.startswith('.'):  # exclude files like .DS_Store
                continue

            figures_filepaths[fn] = os.path.join(dirpath, fn)

    print(f'found {len(figures_filepaths)} files in {os.path.join(data_root, "ReyFigures")}')
    print(f'# test images: {len(test_labels)}')
    print(f'# train images: {len(train_labels)}')

    # process test data
    test_labels_fp = os.path.join(serialized_dir, 'test_labels.csv')
    _process_images(test_labels, figures_filepaths, test_labels_fp, serialized_data_dir_test)
    train_labels_fp = os.path.join(serialized_dir, 'train_labels.csv')
    _process_images(train_labels, figures_filepaths, train_labels_fp, serialized_data_dir_train)


def _process_images(labels: pd.DataFrame, figures_filepaths: Dict, labels_save_as: str, images_data_dir: str,
                    num_processes: int = 8):
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(num_processes)

    # put listener to work
    _ = pool.apply_async(_listener, (labels, labels_save_as, queue))

    # fire off workers
    worker_args = [(row, figures_filepaths, images_data_dir, queue) for _, row in labels.iterrows()]
    for _ in tqdm(pool.imap_unordered(_worker, worker_args), total=len(worker_args)):
        pass

    queue.put('kill')
    pool.close()
    pool.join()


def _worker(data):
    row = data[0]
    figures_filepaths = data[1]
    data_dir = data[2]
    queue = data[3]

    image_fp = figures_filepaths[row['FILE']]

    # load image and resize
    image_numpy = cv2.imread(image_fp, flags=cv2.IMREAD_GRAYSCALE).astype(float)

    if image_numpy is None:
        return

    image_numpy = resize_padded(image_numpy, new_shape=args.image_size)

    # convert to uint8
    image_numpy = image_numpy.astype(np.uint8)

    # save resized image
    serialized_image_fn = os.path.splitext(os.path.split(image_fp)[-1])[0] + '.npy'
    serialized_fp = os.path.join(data_dir, serialized_image_fn)
    np.save(serialized_fp, image_numpy)

    data = {'FILE': row['FILE'], 'image_filepath': image_fp, 'serialized_filepath': serialized_fp}
    queue.put(data)


def _listener(dataframe: pd.DataFrame, save_as, queue):
    dataframe['image_filepath'] = np.nan
    dataframe['serialized_filepath'] = np.nan
    dataframe = dataframe.set_index('FILE')

    while 1:
        data = queue.get()

        if data == 'kill':
            dataframe.reset_index(drop=False)
            dataframe.to_csv(save_as)
            break

        image_fn = data['FILE']
        dataframe.loc[image_fn, 'image_filepath'] = data['image_filepath']
        dataframe.loc[image_fn, 'serialized_filepath'] = data['serialized_filepath']


if __name__ == '__main__':
    main(data_root=args.data_root, image_size=args.image_size)
