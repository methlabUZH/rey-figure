import pandas as pd
import numpy as np
import os

import matplotlib.ticker as plticker
from matplotlib.pyplot import Line2D
import matplotlib.pyplot as plt

from constants import *
from src.evaluate.utils import *
from src.utils import assign_bin, init_mpl

_CLASS_ITEM_COLS = [f"class_item_{i + 1}" for i in range(N_ITEMS)]
_SCORE_ITEM_COLS = [f"score_item_{i + 1}" for i in range(N_ITEMS)]

_BAR_WIDTH = 0.25
_CI_ALPHA = 0.05
_BIN_LOCATIONS = BIN_LOCATIONS3_V2
_BINS = [i for i in range(1, len(_BIN_LOCATIONS))]

colors = init_mpl(colorpalette='muted')


def item_plot(multilabel_scores, multilabel_scores_sim, regression_scores, labels, ylabel, xlabel, ymax=None,
              save_as=None, title=None):
    x = np.arange(len(labels))
    width = _BAR_WIDTH  # the width of the bars

    fig, ax = plt.subplots(figsize=(17, 4))
    ax.bar(x - width, multilabel_scores_sim, width, label='Multilabel Classifier + simulated')
    ax.bar(x, multilabel_scores, width, label='Multilabel Classifier')
    ax.bar(x + width, regression_scores, width, label='Regression')

    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ymax is not None:
        ax.set_ylim((0, ymax))
    ax.set_title(title)
    ax.legend(ncol=3, fancybox=False, loc='lower center', bbox_to_anchor=(0.5, -0.45))

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)


def bin_plot_mse(multilabel_scores, multilabel_ci, multilabel_scores_sim, multilabel_ci_sim, regression_scores,
                 regression_ci, multilabel_total_mse, multilabel_total_mse_ci, multilabel_sim_total_mse,
                 multilabel_sim_total_mse_ci, regression_total_mse, regression_total_mse_ci, labels, ylabel, xlabel,
                 ymax=None, save_as=None, title=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19, 5), sharey=True, gridspec_kw={'width_ratios': [1, 19]})
    x = np.arange(len(labels))
    width = _BAR_WIDTH

    # -------------- bin scores plot --------------
    # confidence intervals
    yerr_ml_sim = np.array(multilabel_ci_sim).T - np.tile(np.array(multilabel_scores_sim), (2, 1))
    yerr_ml_sim[0, :] *= -1

    yerr_ml = np.array(multilabel_ci).T - np.tile(np.array(multilabel_scores), (2, 1))
    yerr_ml[0, :] *= -1

    yerr_reg = np.array(regression_ci).T - np.tile(np.array(regression_scores), (2, 1))
    yerr_reg[0, :] *= -1

    ax = axes[1]
    ax.bar(x - width, multilabel_scores_sim, width, label='Multilabel Classifier + simulated', yerr=yerr_ml_sim,
           capsize=4)
    ax.bar(x, multilabel_scores, width, label='Multilabel Classifier', yerr=yerr_ml, capsize=4)
    ax.bar(x + width, regression_scores, width, label='Regression', yerr=yerr_reg, capsize=4)

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
    ax.set_title(title)
    ax.legend(handles, labels, ncol=4, fancybox=False, loc='lower center', bbox_to_anchor=(0.5, -0.45))

    # -------------- total scores plot --------------
    yerr_ml_sim = np.array(multilabel_sim_total_mse_ci).T - np.tile(np.array(multilabel_sim_total_mse), (2, 1))
    yerr_ml_sim[0, :] *= -1

    yerr_ml = np.array(multilabel_total_mse_ci).T - np.tile(np.array(multilabel_total_mse), (2, 1))
    yerr_ml[0, :] *= -1

    yerr_reg = np.array(regression_total_mse_ci).T - np.tile(np.array(regression_total_mse), (2, 1))
    yerr_reg[0, :] *= -1

    ax = axes[0]
    ax.bar([-width], multilabel_sim_total_mse, width, label='Multilabel Classifier + simulated', yerr=yerr_ml_sim,
           capsize=4)
    ax.bar([0], multilabel_total_mse, width, label='Multilabel Classifier', yerr=yerr_ml, capsize=4)
    ax.bar([width], regression_total_mse, width, label='Regression', yerr=yerr_reg, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0])
    ax.set_xticklabels(['All Bins'])
    loc = plticker.MultipleLocator(base=2.0)
    ax.yaxis.set_major_locator(loc)

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close()


def bin_plot_f1(multilabel_bin_scores, multilabel_sim_bin_scores, regression_bin_scores, multilabel_total_score,
                multilabel_sim_total_score, regression_total_score, labels, ylabel, xlabel, ymax=None, save_as=None,
                title=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19, 5), sharey=True, gridspec_kw={'width_ratios': [1, 19]})
    x = np.arange(len(labels))
    width = _BAR_WIDTH

    # -------------- bin scores plot --------------
    ax = axes[1]
    ax.bar(x - width, multilabel_sim_bin_scores, width, label='Multilabel Classifier + simulated')
    ax.bar(x, multilabel_bin_scores, width, label='Multilabel Classifier')
    ax.bar(x + width, regression_bin_scores, width, label='Regression')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ymax is not None:
        ax.set_ylim((0, ymax))

    # legend
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=3, fancybox=False, loc='lower center', bbox_to_anchor=(0.5, -0.45))

    # -------------- total scores plot --------------
    ax = axes[0]
    ax.bar([-width], multilabel_sim_total_score, width, label='Multilabel Classifier + simulated')
    ax.bar([0], multilabel_total_score, width, label='Multilabel Classifier')
    ax.bar([width], regression_total_score, width, label='Regression')
    ax.set_ylabel(ylabel)
    ax.set_xticks([0])
    ax.set_xticklabels(['All Bins'])
    loc = plticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

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


def main(multilabel_root, multilabel_sim_root, regression_root):
    # get predictions
    multilabel_preds = pd.read_csv(os.path.join(multilabel_root, 'test_predictions.csv'))
    multilabel_sim_preds = pd.read_csv(os.path.join(multilabel_sim_root, 'test_predictions.csv'))
    regression_preds = pd.read_csv(os.path.join(regression_root, 'test_predictions.csv'))

    # get ground truths
    multilabel_gt = pd.read_csv(os.path.join(multilabel_root, 'test_ground_truths.csv'))
    multilabel_sim_gt = pd.read_csv(os.path.join(multilabel_sim_root, 'test_ground_truths.csv'))
    regression_gt = pd.read_csv(os.path.join(regression_root, 'test_ground_truths.csv'))

    # assign bin to total scores in ground truth dfs
    for df in [multilabel_gt, multilabel_sim_gt, regression_gt]:
        df[['bin']] = df[['total_score']].applymap(lambda x: assign_bin(x, _BIN_LOCATIONS))

    # compute scores
    multilabel_scores = compute_scores(multilabel_preds, multilabel_gt)
    multilabel_sim_scores = compute_scores(multilabel_sim_preds, multilabel_sim_gt)
    regression_scores = compute_scores(regression_preds, regression_gt)

    # -------- Item MSE plot --------
    item_plot(multilabel_scores=multilabel_scores['item_mse_scores'],
              multilabel_scores_sim=multilabel_sim_scores['item_mse_scores'],
              regression_scores=regression_scores['item_mse_scores'],
              labels=[f'Item {i + 1}' for i in range(N_ITEMS)],
              ylabel='MSE', xlabel=None, title='MSE per Item',
              save_as='./analyze/figures/item_mse.pdf'
              )

    # -------- Item ACC plot --------
    item_plot(multilabel_scores=multilabel_scores['item_acc_scores'],
              multilabel_scores_sim=multilabel_sim_scores['item_acc_scores'],
              regression_scores=regression_scores['item_acc_scores'],
              labels=[f'Item {i + 1}' for i in range(N_ITEMS)], title='Classification Accuracy per Item',
              ylabel='Classification Accuracy', xlabel=None, ymax=1.1,
              save_as='./analyze/figures/item_acc.pdf'
              )

    # -------- Bin MSE plot --------
    bin_plot_mse(multilabel_scores=multilabel_scores['bin_mse_scores'][0],
                 multilabel_ci=multilabel_scores['bin_mse_scores'][1],
                 multilabel_scores_sim=multilabel_sim_scores['bin_mse_scores'][0],
                 multilabel_ci_sim=multilabel_sim_scores['bin_mse_scores'][1],
                 regression_scores=regression_scores['bin_mse_scores'][0],
                 regression_ci=regression_scores['bin_mse_scores'][1],
                 multilabel_total_mse=multilabel_scores['total_score_mse'][0],
                 multilabel_total_mse_ci=multilabel_scores['total_score_mse'][1],
                 multilabel_sim_total_mse=multilabel_sim_scores['total_score_mse'][0],
                 multilabel_sim_total_mse_ci=multilabel_sim_scores['total_score_mse'][1],
                 regression_total_mse=regression_scores['total_score_mse'][0],
                 regression_total_mse_ci=regression_scores['total_score_mse'][1],
                 labels=[f"{max(0, _BIN_LOCATIONS[i])}-{_BIN_LOCATIONS[i + 1]}" for i in
                         range(len(_BIN_LOCATIONS) - 1)],
                 title='Total Score MSE per Score Bin',
                 ylabel='MSE', xlabel="Score Bins",
                 save_as='./analyze/figures/bin_mse.pdf'
                 )

    # -------- Bin F1 Score plot --------
    bin_plot_f1(multilabel_bin_scores=multilabel_scores['bin_f1_scores'],
                multilabel_sim_bin_scores=multilabel_sim_scores['bin_f1_scores'],
                regression_bin_scores=regression_scores['bin_f1_scores'],
                multilabel_total_score=multilabel_scores['total_f1_score'],
                multilabel_sim_total_score=multilabel_sim_scores['total_f1_score'],
                regression_total_score=regression_scores['total_f1_score'],
                labels=[f"{max(0, _BIN_LOCATIONS[i])}-{_BIN_LOCATIONS[i + 1]}" for i in range(len(_BIN_LOCATIONS) - 1)],
                title='F1 Score per Score Bin',
                ylabel='F1', xlabel="Score Bins",
                save_as='./analyze/figures/bin_f1.pdf'
                )


if __name__ == '__main__':
    results_root = './results/euler-results/data-2018-2021-116x150-pp0/'
    main(multilabel_root=results_root + 'final/rey-multilabel-classifier',
         multilabel_sim_root=results_root + 'final-simulated/rey-multilabel-classifier',
         regression_root=results_root + 'final/rey-regressor')
