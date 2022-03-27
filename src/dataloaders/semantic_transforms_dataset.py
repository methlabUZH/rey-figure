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
TF_BRIGHTNESS = 'brightness'
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
        elif transform == TF_BRIGHTNESS:
            self._semantic_transform = AdjustBrightness(brightness_factor)
        elif transform == TF_CONTRAST:
            self._semantic_transform = AdjustContrast(contrast_factor)
        else:
            self._semantic_transform = Identity()

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
    import seaborn as sns
    from constants import DEBUG_DATADIR_SMALL, DEBUG_DATADIR_BIG

    labels_csv = os.path.join(DEBUG_DATADIR_BIG, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)

    save_dir = '/Users/maurice/phd/src/rey-figure/code-main/analyze/figures/semantic-transformations/samples/'


    def get_ds(tf_spec):
        return STDataset(data_root=DEBUG_DATADIR_BIG, labels=labels_df, label_type=CLASSIFICATION_LABELS, **tf_spec,
                         image_size=(232, 300))


    transform_specs = [
        ([({'transform': TF_ROTATION, 'rotation_angles': (5.0, 10.0)}, r'Angle = $5^\circ-10^\circ$'),
          ({'transform': TF_ROTATION, 'rotation_angles': (25.0, 30.0)}, r'Angle = $25^\circ-30^\circ$'),
          ({'transform': TF_ROTATION, 'rotation_angles': (40.0, 45.0)}, r'Angle = $40^\circ-45^\circ$')],
         'rotation_samples.pdf'),

        ([({'transform': TF_PERSPECTIVE, 'distortion_scale': 0.1}, 'Distortion Scale=0.1'),
          ({'transform': TF_PERSPECTIVE, 'distortion_scale': 0.5}, 'Distortion Scale=0.5'),
          ({'transform': TF_PERSPECTIVE, 'distortion_scale': 1.0}, 'Distortion Scale=1.0')],
         'perspective_samples.pdf'),

        ([({'transform': TF_BRIGHTNESS, 'brightness_factor': 0.1}, 'Brightness=0.1'),
          ({'transform': TF_BRIGHTNESS, 'brightness_factor': 0.5}, 'Brightness=0.5'),
          ({'transform': TF_BRIGHTNESS, 'brightness_factor': 0.9}, 'Brightness=0.9')],
         'brightness_decrease_samples.pdf'),

        ([({'transform': TF_BRIGHTNESS, 'brightness_factor': 1.1}, 'Brightness=1.1'),
          ({'transform': TF_BRIGHTNESS, 'brightness_factor': 1.5}, 'Brightness=1.5'),
          ({'transform': TF_BRIGHTNESS, 'brightness_factor': 1.9}, 'Brightness=1.9')],
         'brightness_increase_samples.pdf'),

        ([({'transform': TF_CONTRAST, 'contrast_factor': 0.1}, 'Contrast=0.1'),
          ({'transform': TF_CONTRAST, 'contrast_factor': 0.6}, 'Contrast=0.6'),
          ({'transform': TF_CONTRAST, 'contrast_factor': 1.2}, 'Contrast=1.2')],
         'contrast_samples.pdf')
    ]

    image_idx = np.random.randint(0, 980)
    # image_idx = 0

    for specs, save_fn in transform_specs:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for ax, (spec, title) in zip(axes, specs):
            ds = get_ds(spec)
            im, label = ds[image_idx]
            im = np.squeeze(im.numpy())

            ax.imshow(im, cmap='gray')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(left=True, bottom=True, ax=ax)

        fig.tight_layout()
        save_fn = str(image_idx) + '-' + save_fn
        plt.savefig(save_dir + save_fn, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_dir + save_fn}')
        plt.close(fig)

        plt.show()
        plt.close(fig)
