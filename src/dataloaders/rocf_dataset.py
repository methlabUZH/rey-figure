import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from constants import *
from src.utils import map_to_score_grid, score_to_class
from src.dataloaders.transforms import NormalizeImage

_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]


class ROCFDataset(Dataset):

    def __init__(self, data_root: str, labels: pd.DataFrame, label_type=CLASSIFICATION_LABELS, data_augmentation=False,
                 image_size=DEFAULT_CANVAS_SIZE):
        self._labels_df = labels

        # round item scores to score grid {0, 0.5, 1.0, 2.0}
        self._item_scores = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: map_to_score_grid(x)))

        # round item scores to score grid {0, 0.5, 1.0, 2.0} and assign classes
        self._item_classes = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: score_to_class(map_to_score_grid(x))))

        # total score = sum of item scores
        self._total_scores = np.sum(self._item_scores, axis=1)

        if label_type == CLASSIFICATION_LABELS:
            # for (multilabel) classification: labels are class of each item
            self._labels = self._item_classes
        elif label_type == REGRESSION_LABELS:
            # for regression: labels are score of each item + total score
            self._labels = np.concatenate([self._item_scores, np.expand_dims(self._total_scores, axis=1)], axis=1)
        else:
            raise ValueError(f'unknown label type {label_type}')

        # data augmentation
        self._do_augment = data_augmentation
        self._augment = transforms.RandomApply(transforms=[
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5, fill=255.0),
            transforms.Compose([transforms.RandomRotation(degrees=(-10, 10), expand=True, fill=255.0),
                                transforms.Resize(size=image_size)])],
            p=0.5)

        # normalizer (image-wise)
        self._normalize = NormalizeImage()

        # get filepaths and ids
        self._image_files = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._image_ids = self._labels_df["figure_id"]

    def __getitem__(self, idx):
        image_numpy = np.load(file=self._images_npy[idx])
        image_numpy = image_numpy[np.newaxis, :]
        image_tensor = torch.from_numpy(image_numpy)

        # augment image
        if self._do_augment:
            image_tensor = self._augment(image_tensor)

        # normalize image
        image_tensor = image_tensor.type('torch.FloatTensor')
        image_tensor /= 255.0
        image_tensor = self._normalize(image_tensor)

        return image_tensor, self._labels[idx]

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
    ds = ROCFDataset(data_root=DEBUG_DATADIR_BIG, labels=labels_df, label_type=CLASSIFICATION_LABELS,
                     data_augmentation=True, image_size=(232, 300))
    image, label = ds[1]
    image = np.squeeze(image.numpy())
    plt.imshow(image, cmap='gray')
    plt.show()
