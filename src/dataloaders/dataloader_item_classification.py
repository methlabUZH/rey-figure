import pandas as pd
import torch
import numpy as np
import os

from src.utils import map_to_score_grid, score_to_class

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_item_classification_dataloader(item_num: int, data_root: str, labels_df: pd.DataFrame, batch_size: int,
                                       num_workers: int, shuffle: bool, mean: float = None, std: float = None,
                                       prefectch_factor: int = 16, pin_memory: bool = True, weighted_sampling=False,
                                       is_binary: bool = True):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetItemClassification(item_num, data_root, labels_df=labels_df, transform=transform,
                                            binary=is_binary)

    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_weights_for_balanced_classes()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)


class ROCFDatasetItemClassification(Dataset):

    def __init__(self, item: int, data_root: str, labels_df: pd.DataFrame, transform: transforms = None
                 , binary: bool = False):
        assert 1 <= item <= 18, 'item must be an integer within the interval [1, 18] !'

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # get labels
        self._labels_df = labels_df
        item_scores = np.array(self._labels_df[f'score_item_{item}'].values)
        if binary:
            # only predict wheter or not the item is present
            self._labels = np.zeros(shape=(len(item_scores))).astype(int)
            self._labels[item_scores > 0] = 1
            self.num_classes = 2
        else:
            # predict the score of the item
            # class 0: not present, class 1: score 1/2, class 2: score 1, class 3: score 1.5, class 4: score 2.0
            scores_on_grid = list(map(map_to_score_grid, item_scores))
            self._labels = np.array(list(map(score_to_class, scores_on_grid)), dtype=int)
            self.num_classes = 4

        # get filepaths and ids
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._images_jpeg = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._images_ids = self._labels_df["figure_id"]

    def get_class_counts(self):
        class_counts = [0] * self.num_classes

        for label in self._labels:
            class_counts[label] += 1

        return class_counts

    def get_weights_for_balanced_classes(self):
        class_counts = [0] * self.num_classes
        n_samples = len(self._labels)

        for label in self._labels:
            class_counts[label] += 1

        weight_per_class = [n_samples / cnt for cnt in class_counts]
        sample_weights = [0] * n_samples

        for i, label in enumerate(self._labels):
            sample_weights[i] = weight_per_class[label]

        return sample_weights

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        torch_image = torch.from_numpy(np.load(self._images_npy[idx])[np.newaxis, :])
        image = self._transform(torch_image)

        # load labels
        label = torch.from_numpy(np.asarray(self._labels[idx]))

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
