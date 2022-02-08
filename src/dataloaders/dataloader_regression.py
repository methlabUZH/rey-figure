import pandas as pd
import torch
import numpy as np
import os

from src.utils import map_to_score_grid

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_regression_dataloader(data_root: str, labels_df: pd.DataFrame, batch_size: int, num_workers: int, shuffle: bool,
                              mean: float = None, std: float = None, prefectch_factor: int = 16,
                              weighted_sampling: bool = False, pin_memory: bool = True):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetRegression(data_root, labels_df=labels_df, transform=transform)

    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_weights_for_balanced_classes()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)


class ROCFDatasetRegression(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms = None):
        self._labels_df = labels_df
        self._labels_df.loc[:, 'items_sum'] = self._labels_df[[f'score_item_{i}' for i in range(1, 19)]].sum(axis=1)

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # read and process labels
        label_cols = [f'score_item_{i + 1}' for i in range(18)]  # + ['summed_score']
        labels = self._labels_df[label_cols].values.tolist()
        labels = np.array(list(list(map(map_to_score_grid, lab)) for lab in labels))
        self._labels = np.concatenate([labels, np.expand_dims(np.sum(labels, axis=1), axis=1)], axis=1)

        # label_cols = [f'score_item_{i + 1}' for i in range(18)] + ['summed_score']
        # self._labels = np.array(self._labels_df[label_cols].values, dtype=float)

        # get filepaths and ids
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._images_jpeg = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._images_ids = self._labels_df["figure_id"]

    def get_score_counts(self):
        _, scores_counts = np.unique(np.array(self._labels)[:, -1], return_counts=True)
        return scores_counts

    def get_weights_for_balanced_classes(self):
        scores, scores_counts = np.unique(np.array(self._labels)[:, -1], return_counts=True)
        n_samples = len(self._labels)

        weight_per_score = {s: n_samples / s_cnt for s, s_cnt in zip(scores, scores_counts)}
        sample_weights = [0] * n_samples

        for i, label in enumerate(np.array(self._labels)[:, -1]):
            sample_weights[i] = weight_per_score[label]

        return sample_weights

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        torch_image = torch.from_numpy(np.load(self._images_npy[idx])[np.newaxis, :])
        image = self._transform(torch_image)

        # load labels
        # label = torch.from_numpy(np.asarray(self._labels[idx])).type('torch.FloatTensor')
        label = torch.from_numpy(self._labels[idx]).type('torch.FloatTensor')

        return image, label

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

# if __name__ == '__main__':
# root = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'
# labels_csv = os.path.join(root, 'train_labels.csv')
# labels_df = pd.read_csv(labels_csv)
# ds = ROCFDatasetRegression(data_root=root, labels_df=labels_df)
# print(ds[0])
