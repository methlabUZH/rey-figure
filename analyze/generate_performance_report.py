from sklearn import metrics
import numpy as np
import os
import pandas as pd
from tabulate import tabulate

from constants import *

_CLASS_COLS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]
_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
_Q_ACCURACY = 'accuracy'
_Q_ABS_ERR = 'absolute_error'
_TEST_PREDICTIONS_CSV = 'test_predictions.csv'
_TEST_GROUNDTRUTHS_CSV = 'test_ground_truths.csv'


class PerformanceMeasures:

    def __init__(self, ground_truths: pd.DataFrame, predictions: pd.DataFrame):
        # get figure ids
        self._figure_ids = np.array(predictions['figure_id'])

        # item class predictions
        self._item_class_preds = np.array(predictions.loc[:, _CLASS_COLS])
        self._item_class_gts = np.array(ground_truths.loc[:, _CLASS_COLS])

        # item score predictions
        self._item_score_preds = np.array(predictions.loc[:, _SCORE_COLS])
        self._item_score_gts = np.array(ground_truths.loc[:, _SCORE_COLS])

        # total score predictions
        self._total_score_preds = np.array(predictions.loc[:, _SCORE_COLS].sum(axis=1))
        self._total_score_gts = np.array(ground_truths.loc[:, _SCORE_COLS].sum(axis=1))

        # binarized multiclass labels
        self._binarized_classes_preds = _binarize_item_class_labels(self._item_class_preds)
        self._binarized_classes_gts = _binarize_item_class_labels(self._item_class_gts)

    def classification_report(self):
        print(metrics.classification_report(self._binarized_classes_gts, self._binarized_classes_preds))


def _binarize_item_class_labels(item_classes: np.ndarray) -> np.ndarray:
    """ function turns classes of individual items, each in {0, 1, 2, 3}, into an array of binary multilabel classes """
    labels = np.multiply(np.ones_like(item_classes), np.arange(N_ITEMS)) * 4 + item_classes
    binarized_classes = np.zeros(shape=(len(labels), N_ITEMS * 4))

    for i in range(len(labels)):
        binarized_classes[i, labels[i]] = 1

    return binarized_classes


if __name__ == '__main__':
    res_dir = '../results/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier'
    preds = pd.read_csv(os.path.join(res_dir, 'test_predictions.csv'))
    gts = pd.read_csv(os.path.join(res_dir, 'test_ground_truths.csv'))

    perf_meas = PerformanceMeasures(gts, preds)
    perf_meas.classification_report()
