import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
import pandas as pd
import shutil
from skimage.color import rgb2gray as skimage_rgb2gray
from skimage import io
from tqdm import tqdm
from typing import *

from tabulate import tabulate

import seaborn as sns

from constants import *
from src.utils import init_mpl

colors = init_mpl(sns_style='ticks', colorpalette='muted')

_CLUSTER_DATA_ROOT = '/cluster/work/zhang/webermau/rocf/psychology/'
_SCORE_COLUMNS = [f'score_item_{i + 1}' for i in range(N_ITEMS)] + ['total_score']
_CLASS_COLUMNS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]

_QUANTITY_SCORE_ERROR = 'total_score_error'
_QUANTITY_NUM_MISCLASSIFIED = 'num_misclassified'


def main(results_dir, save_as=None):
    predictions = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))
    ground_truths = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))

    # process dataframes
    file_columns = ['image_file', 'serialized_file']
    predictions = predictions.set_index('figure_id')
    predictions.loc[:, file_columns] = predictions.loc[:, file_columns].applymap(
        lambda s: s.replace(_CLUSTER_DATA_ROOT, ''))

    ground_truths = ground_truths.set_index('figure_id')
    ground_truths.loc[:, file_columns] = ground_truths.loc[:, file_columns].applymap(
        lambda s: s.replace(_CLUSTER_DATA_ROOT, ''))

    # get error rates
    predictions = compute_error_rates(predictions, ground_truths)

    # make figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.histplot(data=predictions, x=_QUANTITY_SCORE_ERROR, ax=axes[0], bins=np.arange(-10, 11), color=colors[0])
    sns.histplot(data=predictions, x=_QUANTITY_NUM_MISCLASSIFIED, ax=axes[1], bins=np.arange(0, 13), color=colors[1])

    # format axes
    axes[0].set_ylabel("# Samples")
    axes[1].set_ylabel("")
    axes[0].set_xlabel(r"Total Score Error ($\hat{y} - y$)")
    axes[1].set_xlabel("# Misclassified Items")
    axes[0].set_xlim(-10, 10)
    axes[0].xaxis.set_major_locator(plticker.MultipleLocator(base=2))
    axes[1].xaxis.set_major_locator(plticker.MultipleLocator(base=2))
    sns.despine(offset=10, trim=True, ax=axes[0])
    sns.despine(offset=10, trim=True, ax=axes[1])

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.clf()
        plt.close()
        return

    fig.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def compute_error_rates(preds, ground_truths) -> pd.DataFrame:
    preds[_QUANTITY_NUM_MISCLASSIFIED] = np.sum(preds.loc[:, _CLASS_COLUMNS] != ground_truths.loc[:, _CLASS_COLUMNS],
                                                axis=1)
    preds[_QUANTITY_SCORE_ERROR] = preds.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score']
    return preds


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir=results_root + 'final/rey-multilabel-classifier', save_as='./figures/error-distribution.pdf')
