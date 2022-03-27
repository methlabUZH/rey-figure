import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataloaders.tdrg_dataset import TDRGDataset


def get_dataloader(data_root: str,
                   labels: pd.DataFrame,
                   batch_size: int,
                   image_size: Tuple[int, int],
                   num_workers: int,
                   shuffle: bool,
                   prefectch_factor: int = 16,
                   pin_memory: bool = True,
                   weighted_sampling=False,
                   augment=False):
    multilabel_dataset = TDRGDataset(data_root, labels, image_size=image_size)

    if weighted_sampling:
        sample_weights = multilabel_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights))
        DataLoader(dataset=multilabel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)

    return DataLoader(dataset=multilabel_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)
