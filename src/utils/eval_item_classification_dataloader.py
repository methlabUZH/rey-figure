import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_eval_item_classiciation_dataloader(item_num: int, data_root: str, labels_df: pd.DataFrame, batch_size: int,
                                           num_workers: int, mean: float = None, std: float = None,
                                           prefectch_factor: int = 16, pin_memory: bool = True, max_samples=-1):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetItemClassificationEval(item_num, data_root, labels_df=labels_df, transform=transform,
                                                max_samples=max_samples)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)


class ROCFDatasetItemClassificationEval(Dataset):

    def __init__(self, item: int, data_root: str, labels_df: pd.DataFrame, transform: transforms = None,
                 max_samples=-1):
        assert 1 <= item <= 18, 'item must be an integer within the interval [1, 18] !'

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # get one hot encoded labels
        self._labels_df = labels_df.loc[:max_samples, :] if max_samples > 0 else labels_df
        item_scores = np.array(self._labels_df[f'score_item_{item}'].values)
        self._labels = np.zeros(shape=(len(item_scores))).astype(int)
        self._labels[item_scores > 0] = 1

        # get filepaths
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]
        self._images_jpeg = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]

        # get file_ids
        self._images_ids = self._labels_df["figure_id"]

    def get_class_counts(self):
        class_counts = [0, 0]

        for label in self._labels:
            class_counts[label] += 1

        return class_counts

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        # load and normalize image
        torch_image = torch.from_numpy(np.load(self._images_npy[idx])[np.newaxis, :])
        image = self._transform(torch_image)

        # load labels
        label = torch.from_numpy(np.asarray(self._labels[idx]))

        # get filepaths and id
        image_npy_fp = self._images_npy[idx]
        image_jpeg_fp = self._images_jpeg[idx]
        image_id = self._images_ids[idx]

        return image, label, image_npy_fp, image_jpeg_fp, image_id

    @staticmethod
    def _normalize_single(image: torch.Tensor):
        return (image - torch.mean(image)) / torch.std(image)
