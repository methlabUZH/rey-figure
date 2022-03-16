import numpy as np
import os
import pandas as pd

from tabulate import tabulate
from constants import *

_CLASS_COLS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]
_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
_Q_ACCURACY = 'accuracy'
_Q_ABS_ERR = 'absolute_error'


def main(res_dir):
    predictions = pd.read_csv(os.path.join(res_dir, 'test_predictions.csv'))
    ground_truths = pd.read_csv(os.path.join(res_dir, 'test_ground_truths.csv'))

    predictions = compute_error_rates(predictions, ground_truths, _Q_ABS_ERR)
    predictions = compute_error_rates(predictions, ground_truths, _Q_ACCURACY)

    item_accuracies = np.mean(predictions.loc[:, [f'{_Q_ACCURACY}-item-{i + 1}' for i in range(N_ITEMS)]], axis=0)
    item_mae = np.mean(predictions.loc[:, [f'{_Q_ABS_ERR}-item-{i + 1}' for i in range(N_ITEMS)]], axis=0)

    print(f'min item accuracy: {np.min(item_accuracies)}')
    print(f'max item accuracy: {np.max(item_accuracies)}')



def compute_error_rates(preds, ground_truths, quantity) -> pd.DataFrame:
    if quantity == _Q_ACCURACY:
        quantity_cols = [f'{_Q_ACCURACY}-item-{i + 1}' for i in range(N_ITEMS)]
        preds[quantity_cols] = preds.loc[:, _CLASS_COLS] == ground_truths.loc[:, _CLASS_COLS]
        return preds

    if quantity == _Q_ABS_ERR:
        quantity_cols = [f'{_Q_ABS_ERR}-item-{i + 1}' for i in range(N_ITEMS)]
        preds[quantity_cols] = np.abs(preds.loc[:, _CLASS_COLS] - ground_truths.loc[:, _CLASS_COLS])
        return preds

    raise ValueError(f'unknown quantity {quantity}')


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(res_dir=results_root + 'final/rey-multilabel-classifier')
