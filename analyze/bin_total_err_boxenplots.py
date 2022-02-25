import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Tuple

from constants import *
from src.utils import assign_bin, init_mpl

colors = init_mpl(sns_style='ticks', colorpalette='muted')

_BIN_LOCATIONS = BIN_LOCATIONS3_V2


def model_comparison_predictions(results_dir_reg, results_dir_mlc, results_dir_reg_v2, quantity='num_misclassified'):
    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc, _ = compute_item_error_rates(predictions_mlc, ground_truths_mlc, quantity)

    # compute number of misclassified items for each sample for regressor
    predictions_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_predictions.csv'))
    ground_truths_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_ground_truths.csv'))
    predictions_reg, labels = compute_item_error_rates(predictions_reg, ground_truths_reg, quantity)

    # compute number of misclassified items for each sample for regressor-v2
    predictions_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_predictions.csv'))
    ground_truths_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_ground_truths.csv'))
    predictions_reg_v2, _ = compute_item_error_rates(predictions_reg_v2, ground_truths_reg_v2, quantity)

    # merde dataframes
    predictions_reg['hue_id'] = ['Regression-V1'] * len(predictions_reg)
    predictions_mlc['hue_id'] = ['Multilabel Classifier'] * len(predictions_mlc)
    predictions_reg_v2['hue_id'] = ['Regression-V2'] * len(predictions_reg_v2)

    return pd.concat([predictions_reg, predictions_mlc, predictions_reg_v2]), labels


def data_comparison_predictoins(results_dir, results_dir_sim, quantity='num_misclassified'):
    # compute number of misclassified items for each sample for multilabel classifier
    predictions0 = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))
    ground_truths0 = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))
    predictions0, _ = compute_item_error_rates(predictions0, ground_truths0, quantity)

    # compute number of misclassified items for each sample for regressor
    predictions_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_predictions.csv'))
    ground_truths_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_ground_truths.csv'))
    predictions_sim, labels = compute_item_error_rates(predictions_sim, ground_truths_sim, quantity)

    # merde dataframes
    predictions0['hue_id'] = ['Real Data'] * len(predictions0)
    predictions_sim['hue_id'] = ['Real Data + Simulated'] * len(predictions_sim)

    return pd.concat([predictions0, predictions_sim]), labels


def compute_item_error_rates(predictions, ground_truths, quantity) -> Tuple[pd.DataFrame, List[str]]:
    def _get_bin_label(idx, bin_locs):
        return f'{max(0, bin_locs[idx - 1])} - {bin_locs[idx]}'

    # assign bin to total scores in ground truth dfs
    predictions[['Total Score Bin']] = ground_truths[['total_score']].applymap(
        lambda x: _get_bin_label(assign_bin(x, _BIN_LOCATIONS), _BIN_LOCATIONS))

    if quantity == 'num_misclassified':
        # compute number of misclassifications for each sample
        class_columns = [f'class_item_{i + 1}' for i in range(18)]
        predictions[quantity] = np.sum(predictions.loc[:, class_columns] != ground_truths.loc[:, class_columns], axis=1)
    elif quantity == 'total_score_absolute_error':
        # compute total score mse for each sample
        predictions[quantity] = np.abs(predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    else:
        raise ValueError

    labels = [_get_bin_label(i, _BIN_LOCATIONS) for i in range(1, len(_BIN_LOCATIONS))]
    return predictions, labels


def make_plot(predictions, kind, labels, quantity, ylabel, hue_order, outliers=True, tickbase=1.0, save_as=None):
    fig = plt.figure(figsize=(20, 5))
    if kind == 'boxen':
        ax = sns.boxenplot(x='Total Score Bin', y=quantity, hue='hue_id', data=predictions, showfliers=outliers,
                           order=labels, hue_order=hue_order)

    elif kind == 'violin':
        ax = sns.violinplot(x='Total Score Bin', y=quantity, hue='hue_id', data=predictions, order=labels,
                            kind='violin', bw=0.4,
                            height=5, aspect=4, legend=False, cut=0, inner='box',
                            hue_order=hue_order)

    else:
        raise ValueError

    # remove legend
    ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .98), fancybox=False)

    ax.set_ylabel(ylabel)
    loc = plticker.MultipleLocator(base=tickbase)
    ax.yaxis.set_major_locator(loc)
    sns.despine(offset=10, trim=True)
    ax.grid(axis='y')

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close()
        return

    fig.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'

    # num misclassified plots ------------------------------------------------------------------------------------------
    save_as = './figures/num_miscl_per_bin_model.pdf'
    # save_as = None
    preds, labels = model_comparison_predictions(results_dir_reg=results_root + 'final/rey-regressor',
                                                 results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
                                                 results_dir_reg_v2=results_root + 'final/rey-regressor-v2',
                                                 quantity='num_misclassified')
    make_plot(preds, kind='violin', labels=labels, quantity='num_misclassified', ylabel='# Misclassified Items',
              outliers=True, tickbase=2.0, save_as=save_as,
              hue_order=['Multilabel Classifier', 'Regression-V1', 'Regression-V2'])

    save_as = './figures/absolute_error_per_bin_model.pdf'
    # save_as = None
    preds, labels = model_comparison_predictions(results_dir_reg=results_root + 'final/rey-regressor',
                                                 results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
                                                 results_dir_reg_v2=results_root + 'final/rey-regressor-v2',
                                                 quantity='total_score_absolute_error')
    make_plot(preds, kind='violin', labels=labels, quantity='total_score_absolute_error',
              ylabel='Total Score Absolute Error', outliers=True, tickbase=2.0, save_as=save_as,
              hue_order=['Multilabel Classifier', 'Regression-V1', 'Regression-V2'])

    # data comparison --------------------------------------------------------------------------------------------------
    save_as = './figures/num_miscl_per_bin_data.pdf'
    # save_as = None
    preds, labels = data_comparison_predictoins(results_dir=results_root + 'final/rey-regressor-v2',
                                                results_dir_sim=results_root + 'final-simulated/rey-regressor-v2',
                                                quantity='num_misclassified')
    make_plot(preds, kind='violin', labels=labels, quantity='num_misclassified', ylabel='# Misclassified Items',
              outliers=True, tickbase=2.0, save_as=save_as, hue_order=['Real Data', 'Real Data + Simulated'])

    save_as = './figures/absolute_error_per_bin_data.pdf'
    # save_as = None
    preds, labels = data_comparison_predictoins(results_dir=results_root + 'final/rey-regressor-v2',
                                                results_dir_sim=results_root + 'final-simulated/rey-regressor-v2',
                                                quantity='total_score_absolute_error')
    make_plot(preds, kind='violin', labels=labels, quantity='total_score_absolute_error',
              ylabel='Total Score Absolute Error', outliers=True, tickbase=2.0, save_as=save_as,
              hue_order=['Real Data', 'Real Data + Simulated'])
