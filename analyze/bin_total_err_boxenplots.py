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


def main(results_dir_reg, results_dir_mlc, results_dir_mlc_sim, quantity='num_misclassified',
         ylabel='# Mislcassified Items', save_as=None, outliers=True, tickbase=1.0):
    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_predictions.csv'))
    ground_truths_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_ground_truths.csv'))
    predictions_mlc_sim, _ = compute_item_error_rates(predictions_mlc_sim, ground_truths_mlc_sim, quantity)

    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc, _ = compute_item_error_rates(predictions_mlc, ground_truths_mlc, quantity)

    # compute number of misclassified items for each sample for regressor
    predictions_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_predictions.csv'))
    ground_truths_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_ground_truths.csv'))
    predictions_reg, labels = compute_item_error_rates(predictions_reg, ground_truths_reg, quantity)

    # merde dataframes
    predictions_reg['model'] = ['Regression'] * len(predictions_reg)
    predictions_mlc['model'] = ['Multilabel Classifier'] * len(predictions_mlc)
    predictions_mlc_sim['model'] = ['Multilabel Classifier w/ Simulated Data'] * len(predictions_mlc_sim)
    predictions = pd.concat([predictions_reg, predictions_mlc, predictions_mlc_sim])
    fig = plt.figure(figsize=(20, 5))
    ax = sns.boxenplot(x='Total Score Bin', y=quantity, hue='model', data=predictions, showfliers=outliers,
                       order=labels,
                       hue_order=['Multilabel Classifier w/ Simulated Data', 'Multilabel Classifier', 'Regression'])

    # modify legend
    ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, 0.98), fancybox=False)

    ax.set_ylabel(ylabel)
    loc = plticker.MultipleLocator(base=tickbase)
    ax.yaxis.set_major_locator(loc)
    sns.despine(offset=10, trim=True)
    ax.grid(axis='y')

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close(fig)


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


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir_reg=results_root + 'final/rey-regressor',
         results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
         results_dir_mlc_sim=results_root + 'final-simulated/rey-multilabel-classifier',
         quantity='num_misclassified', ylabel='# Misclassified Items', outliers=True, tickbase=2.0,
         save_as='./figures/num_miscl_boxen_binned.pdf'
         )

    main(results_dir_reg=results_root + 'final/rey-regressor',
         results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
         results_dir_mlc_sim=results_root + 'final-simulated/rey-multilabel-classifier',
         quantity='total_score_absolute_error', ylabel='Total Score Absolute Error', outliers=False, tickbase=1.0,
         save_as='./figures/abs_err_boxen_binned.pdf'
         )
