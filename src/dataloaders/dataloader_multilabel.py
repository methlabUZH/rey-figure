import pandas as pd
import torch
import numpy as np
import os

from constants import N_ITEMS
from src.utils import map_to_score_grid, score_to_class

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_multilabel_dataloader(data_root: str, labels_df: pd.DataFrame, batch_size: int,
                              num_workers: int, shuffle: bool, mean: float = None, std: float = None,
                              prefectch_factor: int = 16, pin_memory: bool = True, weighted_sampling=False,
                              is_binary: bool = True):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetMultiLabelClassification(data_root, labels_df, transform, is_binary=is_binary)

    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_sample_weights()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)


class ROCFDatasetMultiLabelClassification(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms = None, is_binary: bool = False):
        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        def scores_to_multiclass(x): return score_to_class(map_to_score_grid(x))
        def binarize_labels(x): return 1 if x > 0 else 0

        # get labels
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
        numpy_image = np.load(self._images_npy[idx]).astype(float)[np.newaxis, :]
        torch_image = torch.from_numpy(numpy_image)
        image = self._transform(torch_image)

        # load labels
        labels = torch.from_numpy(np.array([getattr(self, f"item-{i + 1}")[idx] for i in range(N_ITEMS)]))

        return image, labels

    @staticmethod
    def _normalize_single(image: torch.Tensor):
        return (image - torch.mean(image)) / torch.std(image)

    @property
    def image_ids(self):
        return self._images_ids

    @property
    def npy_filepaths(self):
        return self._images_npy

    @property
    def jpeg_filepaths(self):
        return self._images_jpeg


if __name__ == '__main__':
    root = '/Users/maurice/phd/src/rey-figure/data/serialized-data/simulated'
    labels_csv = os.path.join(root, 'simulated_labels.csv')
    labels_df = pd.read_csv(labels_csv)
    ds = ROCFDatasetMultiLabelClassification(data_root=root, labels_df=labels_df)
    print(np.shape(ds[0][0]))

