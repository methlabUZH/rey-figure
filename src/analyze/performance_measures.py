import numpy as np
import os
import pandas as pd
from scipy import stats
from sklearn import metrics
from tabulate import tabulate
from typing import Union
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from src.analyze.utils import init_mpl
from constants import (CLASS_COLUMNS,
                       SCORE_COLUMNS,
                       N_ITEMS,
                       ITEM_SCORES,
                       ERR_LEVEL_ITEM_SCORE,
                       ERR_LEVEL_TOTAL_SCORE,
                       ABSOLUTE_ERROR,
                       SQUARED_ERROR,
                       R_SQUARED)

__all__ = [
    'PerformanceMeasures'
]

PLOT_COLORS = init_mpl(sns_style='ticks', colorpalette='muted')


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

    def generate_report(self, save_dir=None):
        stdout_default = sys.stdout
        f = None
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f = open(os.path.join(save_dir, 'performance_report.txt'), 'w')
            sys.stdout = f

        self.high_level_metrics_report()
        self.item_level_metrics_report()
        self.bin_level_metrics_report()

        sys.stdout = stdout_default

        if f is not None:
            print(f'saved report to {f.name}')
            f.close()

    def high_level_metrics_report(self):
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

        print('\n' + '=' * 200)
        print("* Multilabel Classification Metrics:")
        print(tabulate(cls_table, headers=cls_table.columns, tablefmt='presto', floatfmt=".3f"))

        # regression metrics
        reg_metrics = self._compute_regression_metrics_total_score()

        # table to print
        reg_table = pd.DataFrame(data=[[reg_metrics['mae'], reg_metrics['mse'], reg_metrics['r2']]],
                                 columns=['MAE', 'MSE', 'R2'])
        print('\n' + '=' * 200)
        print("* Total Score Regression Metrics:")
        print(tabulate(reg_table, headers=reg_table.columns, tablefmt='presto', showindex=False, floatfmt=".3f"))

    def item_level_metrics_report(self):
        # classification metrics
        cls_metrics = self._compute_classification_metrics_item_score()

        # regression metrics
        reg_metrics = self._compute_regression_metrics_item_score()

        # table to for total item score metrics
        reg_table = pd.DataFrame(data=[reg_metrics['mae'], reg_metrics['mse'], reg_metrics['r2'], cls_metrics['acc']],
                                 index=['MAE', 'MSE', 'R2', 'Accuracy'],
                                 columns=[f'Item {i + 1}' for i in range(N_ITEMS)])

        print('\n' + '=' * 200)
        print("* Item Specific Metrics:")
        print(tabulate(reg_table, headers=reg_table.columns, tablefmt='presto', showindex=True, floatfmt=".3f"))

        # class accuracies for each item
        cls_acc_table = pd.DataFrame(data=cls_metrics['class-acc'],
                                     columns=[f'Acc. Item Score {s}' for s in ITEM_SCORES],
                                     index=[f'Item {i + 1}' for i in range(N_ITEMS)])
        print('\n')
        print(tabulate(cls_acc_table, headers=cls_acc_table.columns, tablefmt='presto', showindex=True, floatfmt=".3f"))

    def bin_level_metrics_report(self, bin_granularity=3):
        # assign each score to a bin
        bins = np.arange(0, 37, step=bin_granularity)
        score_bins = np.digitize(self._total_score_gts, bins=bins, right=False)

        # build dataframe with errors per image
        table = pd.DataFrame(data=np.concatenate(
            [np.abs(self._total_score_preds - self._total_score_gts).reshape(-1, 1),
             ((self._total_score_preds - self._total_score_gts) ** 2).reshape(-1, 1),
             score_bins.reshape(-1, 1)], axis=1), columns=['MAE', 'MSE', 'bin'])

        # aggregate
        table = table.groupby('bin').agg({'MAE': 'mean', 'MSE': 'mean'})
        table['scores'] = ["[{}, {})".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)] + ['>=36']
        table = table[['scores', 'MAE', 'MSE']]

        print('\n' + '=' * 200)
        print("* Bin Specific Metrics:")
        print(tabulate(table, headers=table.columns, showindex=False))

    def create_figures(self, save_dir=None):
        self._plot_confusion_matrices(save_dir)
        self._plot_error_histogram(save_dir)

    def _plot_confusion_matrices(self, save_dir):
        # compute confusion matrix for each item
        confusion_matrices = [metrics.confusion_matrix(self._item_class_gts[:, i], self._item_class_preds[:, i])
                              for i in range(N_ITEMS)]

        # make figure
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 6))
        mat_idx = 0
        for i in range(3):
            for j in range(6):
                xticklabels = ITEM_SCORES if i == 0 else False
                yticklabels = ITEM_SCORES if j == 0 else False
                sns.heatmap(confusion_matrices[mat_idx], annot=True, fmt='g', ax=axes[i, j], xticklabels=xticklabels,
                            yticklabels=yticklabels, cbar=False)
                axes[i, j].tick_params(axis='both', which='major', labelbottom=False, bottom=False, top=False,
                                       left=False, labeltop=True)
                axes[i, j].set_xlabel(f'Item {mat_idx + 1}')
                mat_idx += 1

        plt.subplots_adjust(left=0.03, bottom=0.05, top=0.93, right=0.99, wspace=0.1, hspace=0.2)

        _save_or_show_figure(fig, save_dir, fig_name='confusion_matrices.pdf')

    def _plot_error_histogram(self, save_dir):
        # compute errors
        errors = self._total_score_preds - self._total_score_gts

        # setup figure
        fig = plt.figure()
        ax = plt.gca()

        # histogram
        sns.histplot(x=errors, bins=np.arange(-10, 11), ax=ax)

        # format axes
        ax.set_ylabel("# Samples")
        ax.set_xlabel(r"Total Score Error ($\hat{y} - y$)")
        ax.set_xlim(-8, 8)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2))
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2))
        sns.despine(offset=10, trim=True, ax=ax)
        sns.despine(offset=10, trim=True, ax=ax)
        plt.tight_layout()

        _save_or_show_figure(fig, save_dir, fig_name='error_histogram.pdf')

    def _compute_regression_metrics_total_score(self):
        return {'mae': self.absolute_error(ERR_LEVEL_TOTAL_SCORE),
                'mse': self.squared_error(ERR_LEVEL_TOTAL_SCORE),
                'r2': self.r_squared(ERR_LEVEL_TOTAL_SCORE)}

    def _compute_regression_metrics_item_score(self):
        return {'mae': self.absolute_error(ERR_LEVEL_ITEM_SCORE),
                'mse': self.squared_error(ERR_LEVEL_ITEM_SCORE),
                'r2': self.r_squared(ERR_LEVEL_ITEM_SCORE)}

    def _compute_classification_metrics_item_score(self):
        # compute accuracy for each item
        items_accs = self.item_accuracies()

        # compute accuracy for each item class {0, 0.5, 1, 2} per item
        confusion_matrices = [metrics.confusion_matrix(self._item_class_gts[:, i], self._item_class_preds[:, i])
                              for i in range(N_ITEMS)]
        items_class_accs = [m.diagonal() / m.sum(axis=1) for m in confusion_matrices]

        return {'acc': items_accs, 'class-acc': items_class_accs}


def _binarize_item_class_labels(item_classes: np.ndarray) -> np.ndarray:
    """ function turns classes of individual items, each in {0, 1, 2, 3}, into an array of binary multilabel classes """
    labels = np.multiply(np.ones_like(item_classes), np.arange(N_ITEMS)) * 4 + item_classes
    binarized_classes = np.zeros(shape=(len(labels), N_ITEMS * 4))

    for i in range(len(labels)):
        binarized_classes[i, labels[i]] = 1

    return binarized_classes


def _save_or_show_figure(fig, save_dir=None, fig_name=None):
    if save_dir is not None and fig_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_as = os.path.join(save_dir, fig_name)
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)
