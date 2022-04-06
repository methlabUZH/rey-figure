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

    def generate_report(self, latex_tables=False):
        self.high_level_metrics_report(latex_table=latex_tables)
        self.item_level_metrics_report(latex_table=latex_tables)

    def high_level_metrics_report(self, latex_table=False):
        # multilabel classification report
        cls_metrics = metrics.classification_report(
            self._binarized_classes_gts, self._binarized_classes_preds, output_dict=True, zero_division=0)

        # table to print
        cls_table = pd.DataFrame(data=[[
            cls_metrics[avg][score]
            for score in ["precision", "recall", "f1-score"]]
            for avg in ["micro avg", "macro avg", "weighted avg", "samples avg"]],
            columns=["precision", "recall", "f1-score"],
            index=["micro avg", "macro avg", "weighted avg", "samples avg"])

        print('\n' + '-' * 50)
        print("* High Level Multilabel Classification Metrics:")
        print(tabulate(cls_table, headers=cls_table.columns, tablefmt='presto'))

        # regression metrics
        reg_metrics = self._compute_regression_metrics_total_score()

        # table to print
        reg_table = pd.DataFrame(data=[[reg_metrics['mae'], reg_metrics['mse'], reg_metrics['r2']]],
                                 columns=['MAE', 'MSE', 'R2'])
        print('\n' + '-' * 50)
        print("* Total Score Regression Metrics:")
        print(tabulate(reg_table, headers=reg_table.columns, tablefmt='presto', showindex=False))

        if latex_table:
            # build latex tableß
            raise NotImplementedError

    def item_level_metrics_report(self, latex_table=False):
        # classification metrics
        item_accuraces = self._compute_classification_metrics_item_score()

        # regression metrics
        reg_metrics = self._compute_regression_metrics_item_score()

        # table to print
        reg_table = pd.DataFrame(data=[reg_metrics['mae'], reg_metrics['mse'], reg_metrics['r2'], item_accuraces],
                                 index=['MAE', 'MSE', 'R2', 'Accuracy'],
                                 columns=[f'Item {i + 1}' for i in range(N_ITEMS)])

        print('\n' + '-' * 50)
        print("* Item Specific Metrics:")
        print(tabulate(reg_table, headers=reg_table.columns, tablefmt='presto', showindex=True))

        if latex_table:
            # build latex tableß
            raise NotImplementedError

    def _compute_regression_metrics_total_score(self):
        # mean absolute error
        total_score_mae = np.mean(np.abs((self._total_score_preds - self._total_score_gts)))

        # mean sqaured error
        total_score_mse = np.mean((self._total_score_preds - self._total_score_gts) ** 2)

        # R2
        rss = np.sum((self._total_score_preds - self._total_score_gts) ** 2)
        tss = np.sum((self._total_score_gts - np.mean(self._total_score_gts, keepdims=True)) ** 2)
        r2 = 1 - rss / tss

        return {'mae': total_score_mae, 'mse': total_score_mse, 'r2': r2}

    def _compute_regression_metrics_item_score(self):
        # mean absolute error
        item_score_mae = np.mean(np.abs((self._item_score_preds - self._item_score_gts)), axis=0)

        # mean squared error
        item_score_mse = np.mean((self._item_score_preds - self._item_score_gts) ** 2, axis=0)

        # R2
        rss = np.sum((self._item_score_preds - self._item_score_gts) ** 2, axis=0)
        tss = np.sum((self._item_score_gts - np.mean(self._item_score_gts, keepdims=True, axis=0)) ** 2, axis=0)
        r2 = 1 - rss / tss

        return {'mae': item_score_mae, 'mse': item_score_mse, 'r2': r2}

    def _compute_classification_metrics_item_score(self):
        # accuracy
        item_acc = np.mean(self._item_class_preds == self._item_class_gts, axis=0)
        return item_acc


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
    perf_meas.generate_report()
