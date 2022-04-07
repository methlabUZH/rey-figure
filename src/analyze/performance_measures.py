from sklearn import metrics
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
from typing import Union

from constants import (CLASS_COLUMNS,
                       SCORE_COLUMNS,
                       N_ITEMS,
                       ERR_LEVEL_ITEM_SCORE,
                       ERR_LEVEL_TOTAL_SCORE,
                       ABSOLUTE_ERROR,
                       SQUARED_ERROR,
                       R_SQUARED)

__all__ = [
    'PerformanceMeasures'
]


class PerformanceMeasures:
    CONFIDENCE_LEVEL = 0.95

    def __init__(self, ground_truths: pd.DataFrame, predictions: pd.DataFrame):
        # get figure ids
        self._figure_ids = np.array(predictions['figure_id'])

        # item class predictions
        self._item_class_preds = np.array(predictions.loc[:, CLASS_COLUMNS])
        self._item_class_gts = np.array(ground_truths.loc[:, CLASS_COLUMNS])

        # item score predictions
        self._item_score_preds = np.array(predictions.loc[:, SCORE_COLUMNS])
        self._item_score_gts = np.array(ground_truths.loc[:, SCORE_COLUMNS])

        # total score predictions
        self._total_score_preds = np.array(predictions.loc[:, SCORE_COLUMNS].sum(axis=1))
        self._total_score_gts = np.array(ground_truths.loc[:, SCORE_COLUMNS].sum(axis=1))

        # binarized multiclass labels
        self._binarized_classes_preds = _binarize_item_class_labels(self._item_class_preds)
        self._binarized_classes_gts = _binarize_item_class_labels(self._item_class_gts)

    def compute_performance_measure(self, pmeasure, error_level, confidence_interval=False):
        if pmeasure == ABSOLUTE_ERROR:
            if confidence_interval:
                # compute a bias-corrected accelerated bootstrap confidence interval
                error_terms = self.absolute_error(error_level=error_level, reduce=None)
                ci = stats.bootstrap(data=(error_terms,),
                                     statistic=np.mean,
                                     confidence_level=self.CONFIDENCE_LEVEL,
                                     method='BCa', axis=0)
                ci = ci.confidence_interval
                mae = np.mean(error_terms, axis=0)
                return ci.low, mae, ci.high

            return self.absolute_error(error_level=error_level, reduce='mean')

        if pmeasure == SQUARED_ERROR:
            if confidence_interval:
                # compute a bias-corrected accelerated bootstrap confidence interval
                error_terms = self.squared_error(error_level=error_level, reduce=None)
                ci = stats.bootstrap(data=(error_terms,),
                                     statistic=np.mean,
                                     confidence_level=self.CONFIDENCE_LEVEL,
                                     method='BCa', axis=0)
                ci = ci.confidence_interval
                mse = np.mean(error_terms, axis=0)
                return ci.low, mse, ci.high

            return self.squared_error(error_level=error_level)

        if pmeasure == R_SQUARED:
            return self.r_squared(error_level=error_level)

        raise ValueError(f'param "pmeasure" must be one of {ABSOLUTE_ERROR}, {SQUARED_ERROR}, {R_SQUARED}')

    def absolute_error(self, error_level=ERR_LEVEL_TOTAL_SCORE, reduce: Union[str, None] = 'mean') -> np.ndarray:
        if error_level == ERR_LEVEL_TOTAL_SCORE:
            error_terms = np.abs(self._total_score_preds - self._total_score_gts)
        elif error_level == ERR_LEVEL_ITEM_SCORE:
            error_terms = np.abs(self._item_score_preds - self._item_score_gts)
        else:
            raise ValueError(f'param "error_level" must be one of {ERR_LEVEL_ITEM_SCORE}, {ERR_LEVEL_TOTAL_SCORE}')

        if reduce == 'mean':
            return np.mean(error_terms, axis=0)

        if reduce == 'sum':
            return np.sum(error_terms, axis=0)

        if reduce is None:
            return error_terms

        raise ValueError(f'param reduce must be one of "sum", "mean", or None; got {reduce}')

    def squared_error(self, error_level=ERR_LEVEL_TOTAL_SCORE, reduce: Union[str, None] = 'mean') -> np.ndarray:
        if error_level == ERR_LEVEL_TOTAL_SCORE:
            error_terms = (self._total_score_preds - self._total_score_gts) ** 2
        elif error_level == ERR_LEVEL_ITEM_SCORE:
            error_terms = (self._item_score_preds - self._item_score_gts) ** 2
        else:
            raise ValueError(f'param "error_level" must be one of {ERR_LEVEL_ITEM_SCORE}, {ERR_LEVEL_TOTAL_SCORE}')

        if reduce == 'mean':
            return np.mean(error_terms, axis=0)

        if reduce == 'sum':
            return np.sum(error_terms, axis=0)

        if reduce is None:
            return error_terms

        raise ValueError(f'param reduce must be one of "sum", "mean", or None; got {reduce}')

    def r_squared(self, error_level=ERR_LEVEL_TOTAL_SCORE) -> Union[float, np.ndarray]:
        if error_level == ERR_LEVEL_TOTAL_SCORE:
            rss = np.sum((self._total_score_preds - self._total_score_gts) ** 2)
            tss = np.sum((self._total_score_gts - np.mean(self._total_score_gts, keepdims=True)) ** 2)
            return 1. - rss / tss

        if error_level == ERR_LEVEL_ITEM_SCORE:
            rss = np.sum((self._item_score_preds - self._item_score_gts) ** 2, axis=0)
            tss = np.sum((self._item_score_gts - np.mean(self._item_score_gts, keepdims=True, axis=0)) ** 2, axis=0)
            return 1. - rss / tss

        raise ValueError(f'param "error_level" must be one of {ERR_LEVEL_ITEM_SCORE}, {ERR_LEVEL_TOTAL_SCORE}')

    def item_accuracies(self) -> np.ndarray:
        return np.mean(self._item_class_preds == self._item_class_gts, axis=0)

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
        print("* Multilabel Classification Metrics:")
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
        cls_metrics = self._compute_classification_metrics_item_score()

        # regression metrics
        reg_metrics = self._compute_regression_metrics_item_score()

        # table to print
        reg_table = pd.DataFrame(data=[reg_metrics['mae'], reg_metrics['mse'], reg_metrics['r2'], cls_metrics['acc']],
                                 index=['MAE', 'MSE', 'R2', 'Accuracy'],
                                 columns=[f'Item {i + 1}' for i in range(N_ITEMS)])

        print('\n' + '-' * 50)
        print("* Item Specific Metrics:")
        print(tabulate(reg_table, headers=reg_table.columns, tablefmt='presto', showindex=True))

        if latex_table:
            # build latex table
            raise NotImplementedError

    def _compute_regression_metrics_total_score(self):
        return {'mae': self.absolute_error(ERR_LEVEL_TOTAL_SCORE),
                'mse': self.squared_error(ERR_LEVEL_TOTAL_SCORE),
                'r2': self.r_squared(ERR_LEVEL_TOTAL_SCORE)}

    def _compute_regression_metrics_item_score(self):
        return {'mae': self.absolute_error(ERR_LEVEL_ITEM_SCORE),
                'mse': self.squared_error(ERR_LEVEL_ITEM_SCORE),
                'r2': self.r_squared(ERR_LEVEL_ITEM_SCORE)}

    def _compute_classification_metrics_item_score(self):
        return {'acc': self.item_accuracies()}


def _binarize_item_class_labels(item_classes: np.ndarray) -> np.ndarray:
    """ function turns classes of individual items, each in {0, 1, 2, 3}, into an array of binary multilabel classes """
    labels = np.multiply(np.ones_like(item_classes), np.arange(N_ITEMS)) * 4 + item_classes
    binarized_classes = np.zeros(shape=(len(labels), N_ITEMS * 4))

    for i in range(len(labels)):
        binarized_classes[i, labels[i]] = 1

    return binarized_classes

# def _bootstrap_confidence_interval(arr: np.ndarray, conf_level: float):
# mean = arr.mean()
# std = arr.std()
# t_val = np.abs(stats.t.ppf((1 - conf_level) / 2.0, n_samples - 1))
# err = std * t_val / np.sqrt(n_samples)
