import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from constants import *
from src.utils import map_to_score_grid, score_to_class
from src.dataloaders.transforms import NormalizeImage, ResizePadded

_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]


class TDRGDataset(Dataset):

    def __init__(self, data_root: str, labels: pd.DataFrame, image_size=DEFAULT_CANVAS_SIZE):
        self._labels_df = labels

        # round item scores to score grid {0, 0.5, 1.0, 2.0}
        self._item_scores = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: map_to_score_grid(x)))

        # round item scores to score grid {0, 0.5, 1.0, 2.0} and assign classes
        self._item_classes = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: score_to_class(map_to_score_grid(x))))

        # convert to global labels
        self._labels = np.multiply(np.ones_like(self._item_classes), np.arange(N_ITEMS)) * 4 + self._item_classes
        self._one_hot_labels = np.zeros(shape=(len(labels), N_ITEMS * 4))

        for i in range(len(labels)):
            self._one_hot_labels[i, self._labels[i]] = 1

        # total score = sum of item scores
        self._total_scores = np.sum(self._item_scores, axis=1)

        # normalizer (image-wise)
        self._normalize = NormalizeImage()
        self._resize = ResizePadded(size=image_size, fill=255)

        # get filepaths and ids
        self._image_files = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._image_ids = self._labels_df["figure_id"]

    def __getitem__(self, idx):
        image_tensor = read_image(path=self._image_files[idx], mode=ImageReadMode.RGB)

        image_tensor = self._resize(image_tensor)

        # normalize image
        image_tensor = image_tensor.type('torch.FloatTensor')
        image_tensor /= 255.0

        data = {'image': image_tensor, 'target': self._one_hot_labels[idx], 'name': self._image_files[idx]}

        return data

    def get_sample_weights(self):
        """ weights according to the distribution of total scores """
        scores, scores_counts = np.unique(self._total_scores, return_counts=True)
        n_samples = len(self._total_scores)

        weight_per_score = {s: n_samples / s_cnt for s, s_cnt in zip(scores, scores_counts)}
        sample_weights = [0] * n_samples

        for i, score in enumerate(self._total_scores):
            sample_weights[i] = weight_per_score[score]

        return sample_weights

    def __len__(self):
        return len(self._image_ids)

    @property
    def image_ids(self):
        return self._image_ids

    @property
    def image_files(self):
        return self._image_files

    @property
    def npy_filepaths(self):
        return self._images_npy


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from constants import DEBUG_DATADIR_SMALL, DEBUG_DATADIR_BIG

    labels_csv = os.path.join(DEBUG_DATADIR_BIG, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)
    ds = TDRGDataset(data_root=DEBUG_DATADIR_BIG, labels=labels_df, image_size=(150, 150))
    d = ds[0]
    image = d['image']
    print(d['target'])
    # image = np.squeeze(image.numpy())
    # image = np.moveaxis(image, 0, 2)
    # plt.imshow(image, cmap='gray')
    # plt.show()
