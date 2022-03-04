import cv2
import numpy as np
import os
import pandas as pd
import shutil
from skimage.color import rgb2gray as skimage_rgb2gray
from skimage import io
from tqdm import tqdm
from typing import *

from tabulate import tabulate

from constants import *

_CLUSTER_DATA_ROOT = '/cluster/work/zhang/webermau/rocf/psychology/'
_SCORE_COLUMNS = [f'score_item_{i + 1}' for i in range(N_ITEMS)] + ['total_score']
_CLASS_COLUMNS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]

_QUANTITY_ABSOLUTE_ERROR = 'total_score_absolute_error'
_QUANTITY_NUM_MISCLASSIFIED = 'num_misclassified'


def main(results_dir):
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

    # get misclassification rates
    predictions = compute_error_rates(predictions, ground_truths)

    predictions.to_csv('./prediction_measures.csv',
                       columns=['image_file', _QUANTITY_NUM_MISCLASSIFIED, _QUANTITY_ABSOLUTE_ERROR])

    # # write to .txt
    # with open('./prediction_measures.csv', 'w') as f:
    #     for figure_id, row in predictions.iterrows():
    #         f.write(f"{figure_id},{row['image_file']},{}")


def compute_error_rates(preds, ground_truths) -> pd.DataFrame:
    preds[_QUANTITY_NUM_MISCLASSIFIED] = np.sum(preds.loc[:, _CLASS_COLUMNS] != ground_truths.loc[:, _CLASS_COLUMNS],
                                                axis=1)
    preds[_QUANTITY_ABSOLUTE_ERROR] = np.abs(preds.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    return preds


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir=results_root + 'final/rey-multilabel-classifier')
