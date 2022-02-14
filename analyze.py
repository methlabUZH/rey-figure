import pandas as pd
import numpy as np
import os

import matplotlib.ticker as plticker
from matplotlib.pyplot import Line2D
import matplotlib.pyplot as plt

from constants import *
from src.evaluate.utils import *
from src.utils import assign_bin, map_to_score_grid, init_mpl, score_to_class

_CLASS_ITEM_COLS = [f"class_item_{i + 1}" for i in range(N_ITEMS)]
_SCORE_ITEM_COLS = [f"score_item_{i + 1}" for i in range(N_ITEMS)]

_CI_ALPHA = 0.05
_BIN_LOCATIONS = BIN_LOCATIONS3_V2
_BINS = [i for i in range(1, len(_BIN_LOCATIONS))]

colors = init_mpl()


def item_plot(multilabel_scores, regression_scores, labels, ylabel, xlabel, ymax=None):
    x = np.arange(len(labels))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(17, 4))
    ax.bar(x - width / 2, multilabel_scores, width, label='Multilabel Classifier')
    ax.bar(x + width / 2, regression_scores, width, label='Regression')

    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ymax is not None:
        ax.set_ylim((0, ymax))
    ax.legend(ncol=2)

    fig.tight_layout()
    plt.show()
    plt.close()


def bin_plot_v1(multilabel_scores, multilabel_ci, regression_scores, regression_ci, multilabel_total_mse,
                multilabel_total_mse_ci, regression_total_mse, regression_total_mse_ci, labels, ylabel, xlabel,
                ymax=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19, 4), sharey=True, gridspec_kw={'width_ratios': [1, 19]})
    x = np.arange(len(labels))
    width = 0.35

    # -------------- bin scores plot --------------
    # confidence intervals
    yerr_ml = np.array(multilabel_ci).T - np.tile(np.array(multilabel_scores), (2, 1))
    yerr_ml[0, :] *= -1

    yerr_reg = np.array(regression_ci).T - np.tile(np.array(regression_scores), (2, 1))
    yerr_reg[0, :] *= -1

    ax = axes[1]
    ax.bar(x - width / 2, multilabel_scores, width, label='Multilabel Classifier', yerr=yerr_ml, capsize=4)
    ax.bar(x + width / 2, regression_scores, width, label='Regression', yerr=yerr_reg, capsize=4)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ymax is not None:
        ax.set_ylim((0, ymax))

    # legend
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + [Line2D([0], [0], color='black')]
    labels = labels + [f'{100 * (1 - _CI_ALPHA)}% Confidence Interval']
    ax.legend(handles, labels, ncol=3, fancybox=False, loc='upper center', bbox_to_anchor=(0.5, 1.25))

    # -------------- total scores plot --------------
    yerr_ml = np.array(multilabel_total_mse_ci).T - np.tile(np.array(multilabel_total_mse), (2, 1))
    yerr_ml[0, :] *= -1

    yerr_reg = np.array(regression_total_mse_ci).T - np.tile(np.array(regression_total_mse), (2, 1))
    yerr_reg[0, :] *= -1

    ax = axes[0]
    ax.bar([-width / 2], multilabel_total_mse, width, label='Multilabel Classifier', yerr=yerr_ml, capsize=4)
    ax.bar([width / 2], regression_total_mse, width, label='Regression', yerr=yerr_reg, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0])
    ax.set_xticklabels(['All Bins'])
    loc = plticker.MultipleLocator(base=2.0)
    ax.yaxis.set_major_locator(loc)

    fig.tight_layout()
    plt.show()
    plt.close()


def bin_plot_v2(multilabel_bin_scores, regression_bin_scores, multilabel_total_score,
                regression_total_score, labels, ylabel, xlabel, ymax=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19, 4), sharey=True, gridspec_kw={'width_ratios': [1, 19]})
    x = np.arange(len(labels))
    width = 0.35

    # -------------- bin scores plot --------------
    ax = axes[1]
    ax.bar(x - width, multilabel_bin_scores, width, label='Multilabel Classifier')
    ax.bar(x, regression_bin_scores, width, label='Regression')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ymax is not None:
        ax.set_ylim((0, ymax))

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, fancybox=False, loc='upper center', bbox_to_anchor=(0.5, 1.25))

    # -------------- total scores plot --------------
    ax = axes[0]
    ax.bar([-width], multilabel_total_score, width, label='Multilabel Classifier')
    ax.bar([0], regression_total_score, width, label='Regression')
    ax.set_ylabel(ylabel)
    ax.set_xticks([0])
    ax.set_xticklabels(['All Bins'])
    loc = plticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    fig.tight_layout()
    plt.show()
    plt.close()


def compute_scores(predictions, ground_truths):
    return {
        "item_mse_scores": compute_mse_scores(predictions, ground_truths, _SCORE_ITEM_COLS),
        "item_acc_scores": compute_accuracy_scores(predictions, ground_truths, _CLASS_ITEM_COLS),
        "bin_mse_scores": compute_bin_mse_scores(predictions, ground_truths, _BINS, True, alpha_ci=_CI_ALPHA),
        "total_score_mse": compute_mse_scores(predictions, ground_truths, ['total_score'], True, alpha_ci=_CI_ALPHA),
        "bin_f1_scores": compute_multilabel_f1_score(predictions, ground_truths, _BINS, 4),
        "total_f1_score": compute_multilabel_f1_score(predictions, ground_truths, None, 4)
    }


def main(multilabel_root, regression_root):
    # get predictions
    multilabel_preds = pd.read_csv(os.path.join(multilabel_root, 'test_predictions.csv'))
    regression_preds = pd.read_csv(os.path.join(regression_root, 'test_predictions.csv'))

    # get ground truths
    multilabel_gt = pd.read_csv(os.path.join(multilabel_root, 'test_ground_truths.csv'))
    regression_gt = pd.read_csv(os.path.join(regression_root, 'test_ground_truths.csv'))

    # assign bin to total scores in ground truth dfs
    for df in [multilabel_gt, regression_gt]:
        df[['bin']] = df[['total_score']].applymap(lambda x: assign_bin(x, _BIN_LOCATIONS))

    # compute scores
    multilabel_scores = compute_scores(multilabel_preds, multilabel_gt)
    regression_scores = compute_scores(regression_preds, regression_gt)

    # -------- Item MSE plot --------
    item_plot(multilabel_scores=multilabel_scores['item_mse_scores'],
              regression_scores=regression_scores['item_mse_scores'],
              labels=[f'Item {i + 1}' for i in range(N_ITEMS)],
              ylabel='MSE', xlabel=None)

    # -------- Item ACC plot --------
    item_plot(multilabel_scores=multilabel_scores['item_acc_scores'],
              regression_scores=regression_scores['item_acc_scores'],
              labels=[f'Item {i + 1}' for i in range(N_ITEMS)],
              ylabel='Classification Accuracy', xlabel=None, ymax=1.1)

    # -------- Bin MSE plot --------
    bin_plot_v1(multilabel_scores=multilabel_scores['bin_mse_scores'][0],
                multilabel_ci=multilabel_scores['bin_mse_scores'][1],
                regression_scores=regression_scores['bin_mse_scores'][0],
                regression_ci=regression_scores['bin_mse_scores'][1],
                multilabel_total_mse=multilabel_scores['total_score_mse'][0],
                multilabel_total_mse_ci=multilabel_scores['total_score_mse'][1],
                regression_total_mse=regression_scores['total_score_mse'][0],
                regression_total_mse_ci=regression_scores['total_score_mse'][1],
                labels=[f"{max(0, _BIN_LOCATIONS[i])}-{_BIN_LOCATIONS[i + 1]}" for i in range(len(_BIN_LOCATIONS) - 1)],
                ylabel='MSE', xlabel="Score Bins")

    # -------- Bin F1 Score plot --------
    bin_plot_v2(multilabel_bin_scores=multilabel_scores['bin_f1_scores'],
                regression_bin_scores=regression_scores['bin_f1_scores'],
                multilabel_total_score=multilabel_scores['total_f1_score'],
                regression_total_score=regression_scores['total_f1_score'],
                labels=[f"{max(0, _BIN_LOCATIONS[i])}-{_BIN_LOCATIONS[i + 1]}" for i in range(len(_BIN_LOCATIONS) - 1)],
                ylabel='F1', xlabel="Score Bins")


if __name__ == '__main__':
    results_root = './results/euler-results/data-2018-2021-116x150-pp0/final/'
    main(multilabel_root=results_root + 'rey-multilabel-classifier',
         regression_root=results_root + 'rey-regressor')
