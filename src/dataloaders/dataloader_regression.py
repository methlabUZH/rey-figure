import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_regression_dataloader_train(data_root: str, labels_df: pd.DataFrame, batch_size: int, num_workers: int, shuffle: bool,
                                    score_type: str, mean: float = None, std: float = None, prefectch_factor: int = 16,
                                    pin_memory: bool = True):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetRegression(data_root, labels_df=labels_df, transform=transform, score_type=score_type)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)


def get_regression_dataloader_eval(data_root: str, labels_df: pd.DataFrame, batch_size: int, num_workers: int, score_type: str,
                                   mean: float = None, std: float = None, prefectch_factor: int = 16, pin_memory: bool = True):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetRegressionEval(data_root, labels_df=labels_df, transform=transform, score_type=score_type)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)


class ROCFDatasetRegression(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms = None,
                 score_type: str = 'sum'):
        self._labels_df = labels_df
        self._labels_df.loc[:, 'items_sum'] = self._labels_df[[f'score_item_{i}' for i in range(1, 19)]].sum(axis=1)

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # read labels
        label_cols = [f'score_item_{i + 1}' for i in range(18)] + ['median_score', 'summed_score']
        all_labels = self._labels_df[label_cols].values.tolist()

        if score_type == 'sum':
            self._labels = np.array(all_labels)[:, [i for i in range(18)] + [19]].tolist()
        elif score_type == 'median':
            self._labels = np.array(all_labels)[:, :19].tolist()
        else:
            raise ValueError(f'invalid score type provided! got {score_type}; must be "sum" or "median"')

        # get filepaths
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        torch_image = torch.from_numpy(np.load(self._images_npy[idx])[np.newaxis, :])
        image = self._transform(torch_image)

        # load labels
        label = torch.from_numpy(np.asarray(self._labels[idx])).type('torch.FloatTensor')

        return image, label

    @staticmethod
    def _normalize_single(image: torch.Tensor):
        return (image - torch.mean(image)) / torch.std(image)


class ROCFDatasetRegressionEval(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms = None, score_type: str = 'sum'):
        if score_type != 'sum':
            raise NotImplementedError

        self._labels_df = labels_df

        # compute sum of items
        self._labels_df['items_sum'] = self._labels_df[[f'score_item_{i}' for i in range(1, 19)]].sum(axis=1)

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # read labels
        label_cols = [f'score_item_{i + 1}' for i in range(18)] + ['summed_score']
        self._labels = self._labels_df[label_cols].values.tolist()

        # get filepaths
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._images_jpeg = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]

        # get file_ids
        self._images_ids = self._labels_df["figure_id"]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        image_npy_fp = self._images_npy[idx]
        torch_image = torch.from_numpy(np.load(image_npy_fp)[np.newaxis, :])
        image = self._transform(torch_image)

        # load labels
        label = torch.from_numpy(np.asarray(self._labels[idx])).type('torch.FloatTensor')

        # get filepaths and id
        image_npy_fp = self._images_npy[idx]
        image_jpeg_fp = self._images_jpeg[idx]
        image_id = self._images_ids[idx]

        return image, label, image_npy_fp, image_jpeg_fp, image_id

    @staticmethod
    def _normalize_single(image: torch.Tensor):
        return (image - torch.mean(image)) / torch.std(image)
