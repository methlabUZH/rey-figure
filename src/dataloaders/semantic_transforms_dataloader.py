import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader

from src.dataloaders.semantic_transforms_dataset import STDataset


def get_dataloader(data_root: str,
                   labels: pd.DataFrame,
                   label_type: str,
                   batch_size: int,
                   image_size: Tuple[int, int],
                   num_workers: int,
                   shuffle: bool,
                   prefectch_factor: int = 16,
                   pin_memory: bool = True,
                   transform=None,
                   rotation_angles=None,
                   distortion_scale=None,
                   brightness_factor=None,
                   contrast_factor=None):
    multilabel_dataset = STDataset(data_root, labels, label_type=label_type, image_size=image_size,
                                   transform=transform, rotation_angles=rotation_angles,
                                   distortion_scale=distortion_scale, brightness_factor=brightness_factor,
                                   contrast_factor=contrast_factor)

    return DataLoader(dataset=multilabel_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)
