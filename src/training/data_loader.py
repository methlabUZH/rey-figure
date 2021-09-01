import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from constants import LABEL_FORMATS


def get_dataloader(data_root: str, labels_csv: str, batch_size: int, num_workers: int, shuffle: bool, mean: float,
                   std: float):
    labels_df = pd.read_csv(labels_csv)

    if shuffle:
        labels_df = labels_df.sample(frac=1)

    transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetRegression(data_root=data_root, labels_df=labels_df, transform=transform)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)


class ROCFDatasetRegression(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms,
                 label_fmt: str = 'items-median-scores'):
        self._labels_df = labels_df
        self._labels_df['items_sum'] = self._labels_df[[f'score_item_{i}' for i in range(1, 19)]].sum(axis=1)
        self._transform = transform

        # read labels
        label_cols = [f'score_item_{i + 1}' for i in range(18)] + ['median_score', 'summed_score']
        all_labels = self._labels_df[label_cols].values.tolist()

        if label_fmt == 'items':
            # self._labels = all_labels[:, :18]
            raise NotImplementedError
        elif label_fmt == 'items-median-scores':
            self._labels = np.array(all_labels)[:, :19].tolist()
        elif label_fmt == 'items-sum-scores':
            self._labels = np.array(all_labels)[:, [i for i in range(18)] + [19]].tolist()
        else:
            raise ValueError(f'invalid label format provided! got {label_fmt}; must be one of {LABEL_FORMATS}')

        self._total_scores_median = np.array(all_labels)[:, 18].tolist()
        self._total_scores_sum = np.array(all_labels)[:, 19].tolist()

        # get filepaths
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        image = torch.from_numpy(np.load(self._images_npy[idx])[np.newaxis, :])
        image = self._transform(image)

        # load labels
        label = torch.from_numpy(np.asarray(self._labels[idx])).type('torch.FloatTensor')
        median_score = torch.from_numpy(np.asarray(self._total_scores_median[idx])).type('torch.FloatTensor')
        sum_score = torch.from_numpy(np.asarray(self._total_scores_sum[idx])).type('torch.FloatTensor')

        return image, label, median_score, sum_score
