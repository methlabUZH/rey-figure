import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Tuple

from src.utils import init_mpl

colors = init_mpl(sns_style='ticks', colorpalette='muted')


def main(results_dir_reg, results_dir_mlc, results_dir_mlc_sim, quantity='num_misclassified',
         ylabel='# Mislcassified Items', save_as=None):
    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_predictions.csv'))
    ground_truths_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_ground_truths.csv'))
    predictions_mlc_sim = compute_item_error_rates(predictions_mlc_sim, ground_truths_mlc_sim, quantity)

    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc = compute_item_error_rates(predictions_mlc, ground_truths_mlc, quantity)

    # compute number of misclassified items for each sample for regressor
    predictions_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_predictions.csv'))
    ground_truths_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_ground_truths.csv'))
    predictions_reg = compute_item_error_rates(predictions_reg, ground_truths_reg, quantity)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    sns.boxenplot(y=quantity, data=predictions_mlc_sim, ax=axes[0], color=colors[0], showfliers=False)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('Multilabel Classifier w/ Simulated Data')
    axes[0].set_xticks([])

    sns.boxenplot(y=quantity, data=predictions_mlc, ax=axes[1], color=colors[1], showfliers=False)
    axes[1].set_ylabel('')
    axes[1].set_title('Multilabel Classifier')
    axes[1].set_xticks([])
    axes[1].tick_params(colors='white', which='both')

    sns.boxenplot(y=quantity, data=predictions_reg, ax=axes[2], color=colors[2], showfliers=False)
    axes[2].set_ylabel('')
    axes[2].set_title('Regression')
    axes[2].set_xticks([])
    axes[2].tick_params(colors='white', which='both')

    sns.despine(offset=10, trim=True, bottom=True, ax=axes[0])
    sns.despine(offset=10, trim=True, bottom=True, left=True, ax=axes[1])
    sns.despine(offset=10, trim=True, bottom=True, left=True, ax=axes[2])
    axes[0].grid(axis='y')
    axes[1].grid(axis='y')
    axes[2].grid(axis='y')

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close()


def compute_item_error_rates(predictions, ground_truths, quantity) -> Tuple[pd.DataFrame, List[str]]:
    if quantity == 'num_misclassified':
        # compute number of misclassifications for each sample
        class_columns = [f'class_item_{i + 1}' for i in range(18)]
        predictions[quantity] = np.sum(
            predictions.loc[:, class_columns] != ground_truths.loc[:, class_columns], axis=1)
    elif quantity == 'total_score_absolute_error':
        # compute total score mse for each sample
        predictions[quantity] = np.abs(predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    else:
        raise ValueError

    return predictions


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir_reg=results_root + 'final/rey-regressor',
         results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
         results_dir_mlc_sim=results_root + 'final-simulated/rey-multilabel-classifier',
         quantity='num_misclassified',
         ylabel='# Misclassified Items',
         save_as='./figures/num_miscl_boxen.pdf'
         )

    main(results_dir_reg=results_root + 'final/rey-regressor',
         results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
         results_dir_mlc_sim=results_root + 'final-simulated/rey-multilabel-classifier',
         quantity='total_score_absolute_error',
         ylabel='Total Score Absolute Error',
         save_as='./figures/abs_err_boxen.pdf'
         )
