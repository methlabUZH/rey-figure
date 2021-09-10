import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_dataloader(data_root: str, labels_df: pd.DataFrame, batch_size: int, num_workers: int, shuffle: bool,
                   score_type: str, mean: float = None, std: float = None):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetRegression(data_root, labels_df=labels_df, transform=transform, score_type=score_type)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class ROCFDatasetRegression(Dataset):

    def __init__(self, data_root: str, labels_df: pd.DataFrame, transform: transforms = None,
                 score_type: str = 'sum'):
        self._labels_df = labels_df
        self._labels_df['items_sum'] = self._labels_df[[f'score_item_{i}' for i in range(1, 19)]].sum(axis=1)

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
