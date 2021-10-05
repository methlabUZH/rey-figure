import pandas as pd
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_item_classiciation_dataloader(item_num: int, data_root: str, labels_df: pd.DataFrame, batch_size: int,
                                      num_workers: int, shuffle: bool, mean: float = None, std: float = None,
                                      prefectch_factor: int = 16, pin_memory: bool = True, weighted_sampling=False):
    transform = None
    if mean is not None and std is not None:
        transform = transforms.Normalize(mean=[mean], std=[std])

    dataset = ROCFDatasetItemClassification(item_num, data_root, labels_df=labels_df, transform=transform)

    sampler = None
    if weighted_sampling:
        sample_weights = dataset.get_weights_for_balanced_classes()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor, sampler=sampler)


class ROCFDatasetItemClassification(Dataset):

    def __init__(self, item: int, data_root: str, labels_df: pd.DataFrame, transform: transforms = None):
        assert 1 <= item <= 18, 'item must be an integer within the interval [1, 18] !'

        if transform is None:
            self._transform = self._normalize_single
        else:
            self._transform = transform

        # get one hot encoded labels
        self._labels_df = labels_df
        item_scores = np.array(self._labels_df[f'score_item_{item}'].values)
        self._labels = np.zeros(shape=(len(item_scores))).astype(int)
        self._labels[item_scores > 0] = 1

        # get filepaths
        self._images_npy = [os.path.join(data_root, f) for f in self._labels_df["serialized_filepath"].tolist()]

    def get_class_counts(self):
        class_counts = [0, 0]

        for label in self._labels:
            class_counts[label] += 1

        return class_counts

    def get_weights_for_balanced_classes(self):
        class_counts = [0, 0]
        n_samples = len(self)

        for label in self._labels:
            class_counts[label] += 1

        weight_per_class = [n_samples / class_counts[0], n_samples / class_counts[1]]
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


# if __name__ == '__main__':
#     dataroot = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'
#     labels_csv = os.path.join(dataroot, 'train_labels.csv')
#     labels_df = pd.read_csv(labels_csv)
#
#     for i in range(1, 19):
#         ds = ROCFDatasetItemClassification(item=i, data_root=dataroot, labels_df=labels_df)
#         labels = ds._labels
#         pos_samples = sum(ds._labels)
#         total_samples = len(ds._labels)
#         print(f'item={i}, positive samples: {pos_samples / total_samples * 100:.2f}%, total samples: {total_samples}')
