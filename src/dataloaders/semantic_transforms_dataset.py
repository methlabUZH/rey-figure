import cv2
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from constants import *
from src.utils import map_to_score_grid, score_to_class
from src.dataloaders.transforms import *

_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
TF_ROTATION = 'rotation'
TF_PERSPECTIVE = 'perspective'
TF_BIRGHTNESS = 'brightness'
TF_CONTRAST = 'contrast'


class STDataset(Dataset):

    def __init__(self, data_root: str,
                 labels: pd.DataFrame,
                 label_type=CLASSIFICATION_LABELS,
                 image_size=DEFAULT_CANVAS_SIZE,
                 transform=TF_ROTATION,
                 rotation_angles=None,
                 distortion_scale=None,
                 brightness_factor=None,
                 contrast_factor=None):
        self._labels_df = labels

        # round item scores to score grid {0, 0.5, 1.0, 2.0}
        self._item_scores = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: map_to_score_grid(x)))

        # round item scores to score grid {0, 0.5, 1.0, 2.0} and assign classes
        self._item_classes = np.array(self._labels_df.loc[:, _SCORE_COLS].applymap(
            lambda x: score_to_class(map_to_score_grid(x))))

        # total score = sum of item scores
        self._total_scores = np.sum(self._item_scores, axis=1)

        if label_type == CLASSIFICATION_LABELS:
            # for (multilabel) classification: labels are class of each item
            self._labels = self._item_classes
        elif label_type == REGRESSION_LABELS:
            # for regression: labels are score of each item + total score
            self._labels = np.concatenate([self._item_scores, np.expand_dims(self._total_scores, axis=1)], axis=1)
        else:
            raise ValueError(f'unknown label type {label_type}')

        # semantic transformation
        if transform == TF_ROTATION:
            self._semantic_transform = RandomRotation(angles=rotation_angles, fill=255.0)
        elif transform == TF_PERSPECTIVE:
            self._semantic_transform = transforms.RandomPerspective(
                distortion_scale=distortion_scale, p=1.0, fill=255.0)
        elif transform == TF_BIRGHTNESS:
            self._semantic_transform = AdjustBrightness(brightness_factor)
        elif transform == TF_CONTRAST:
            self._semantic_transform = AdjustContrast(contrast_factor)
        else:
            raise ValueError

        # normalizer (image-wise)
        self._transform = transforms.Compose([
            self._semantic_transform,
            ResizePadded(size=image_size, fill=255)
        ])

        self._normalize = NormalizeImage()

        # get filepaths and ids
        self._image_files = [os.path.join(data_root, f) for f in self._labels_df["image_filepath"].tolist()]
        self._image_ids = self._labels_df["figure_id"]

    def __getitem__(self, idx):
        image = cv2.imread(self._image_files[idx], flags=cv2.IMREAD_GRAYSCALE)
        image = image[np.newaxis, :]

        image = torch.from_numpy(image)
        image = self._transform(image)

        # normalize image
        image = image.type('torch.FloatTensor')
        image /= 255.0
        image = self._normalize(image)

        return image, self._labels[idx]

    def __len__(self):
        return len(self._image_ids)

    @property
    def image_ids(self):
        return self._image_ids

    @property
    def image_files(self):
        return self._image_files


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from constants import DEBUG_DATADIR_SMALL, DEBUG_DATADIR_BIG

    labels_csv = os.path.join(DEBUG_DATADIR_BIG, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)
    ds = STDataset(data_root=DEBUG_DATADIR_BIG, labels=labels_df, label_type=CLASSIFICATION_LABELS,
                   transform=TF_CONTRAST, contrast_factor=1.5,
                   image_size=(348, 450))
    im, label = ds[-1]
    im = np.squeeze(im.numpy())
    print(np.shape(im))
    plt.imshow(im, cmap='gray')
    plt.show()
