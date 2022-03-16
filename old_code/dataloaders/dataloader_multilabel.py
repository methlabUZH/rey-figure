import pandas as pd
import torch
import numpy as np
import os
import cv2

from constants import N_ITEMS
from src.utils import map_to_score_grid, score_to_class

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_multilabel_dataloader(data_root: str, labels_df: pd.DataFrame, batch_size: int,
                              num_workers: int, shuffle: bool, prefectch_factor: int = 16, pin_memory: bool = True,
                              weighted_sampling=False, is_binary: bool = True):
    dataset = ROCFDatasetMultiLabelClassification(data_root, labels_df, None, is_binary=is_binary)

    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_sample_weights()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)


class Normalize:
    def __call__(self, img):
        return (img - np.mean(img)) / np.std(img)


class ROCFDatasetMultiLabelClassification(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transforms_list: transforms = None,
                 is_binary: bool = False, load_numpy=True):
        if transforms_list is None:
            self._transform = transforms.Compose([Normalize()])
        else:
            self._transform = transforms.Compose(transforms_list + [Normalize()])

        def scores_to_multiclass(x): return score_to_class(map_to_score_grid(x))
        def binarize_labels(x): return 1 if x > 0 else 0

        # get labels
        self._load_numpy = load_numpy
        self._labels_df = labels_df
        score_cols = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
        self._labels_df.loc[:, score_cols] = self._labels_df.loc[:, score_cols].applymap(scores_to_multiclass)
        self._total_scores = np.array(self._labels_df.loc[:, score_cols].sum(axis=1).values)

        if is_binary:
            self._labels_df.loc[:, score_cols] = self._labels_df.loc[:, score_cols].applymap(binarize_labels)

        for i, score_col in enumerate(score_cols):
            setattr(self, f"item-{i + 1}", np.array(self._labels_df.loc[:, score_col].values))

        # get filepaths and ids
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._images_jpeg = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._images_ids = self._labels_df["figure_id"]

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
        return len(self._total_scores)

    def __getitem__(self, idx):
        # load and normalize image
        if self._load_numpy:
            numpy_image = np.load(self._images_npy[idx]).astype(float)
        else:
            numpy_image = cv2.imread(self._images_jpeg[idx], flags=cv2.IMREAD_GRAYSCALE)
            numpy_image = numpy_image / 255.0

        image = self._transform(numpy_image)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).type('torch.FloatTensor')

        # load labels
        labels = torch.from_numpy(np.array([getattr(self, f"item-{i + 1}")[idx] for i in range(N_ITEMS)]))

        return image, labels

    # @staticmethod
    # def _normalize_single(image: torch.Tensor):
    #     return (image - torch.mean(image)) / torch.std(image)

    @property
    def image_ids(self):
        return self._images_ids

    @property
    def npy_filepaths(self):
        return self._images_npy

    @property
    def image_files(self):
        return self._images_jpeg
