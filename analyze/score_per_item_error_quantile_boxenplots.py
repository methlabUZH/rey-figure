import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Tuple

from src.utils import init_mpl

init_mpl(sns_style='ticks', colorpalette='muted')


def main(results_dir_reg, results_dir_mlc, results_dir_mlc_sim, save_as=None):
    # compute number of misclassified items for each sample for multilabel classifier with simulations
    predictions_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_predictions.csv'))
    ground_truths_mlc_sim = pd.read_csv(os.path.join(results_dir_mlc_sim, 'test_ground_truths.csv'))
    predictions_mlc_sim, labels_mlc_sim = compute_quantiles(predictions_mlc_sim, ground_truths_mlc_sim)

    # compute number of misclassified items for each sample for multilabel classifier
    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc, labels_mlc = compute_quantiles(predictions_mlc, ground_truths_mlc)

    # compute number of misclassified items for each sample for regressor
    predictions_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_predictions.csv'))
    ground_truths_reg = pd.read_csv(os.path.join(results_dir_reg, 'test_ground_truths.csv'))
    predictions_reg, labels_reg = compute_quantiles(predictions_reg, ground_truths_reg)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    sns.boxenplot(x='n_miscl_quantile', y='true_total_score', data=predictions_mlc_sim, order=labels_mlc_sim,
                  ax=axes[0])
    axes[0].set_ylabel('Total Score')
    axes[0].set_xlabel('# Misclassified Items')
    axes[0].set_title('Multilabel Classifier w/ Simulated Data')

    sns.boxenplot(x='n_miscl_quantile', y='true_total_score', data=predictions_mlc, order=labels_mlc, ax=axes[1])
    axes[1].set_ylabel('')
    axes[1].set_xlabel('# Misclassified Items')
    axes[1].set_title('Multilabel Classifier')
    axes[1].tick_params(colors='white', which='both', axis='y')

    sns.boxenplot(x='n_miscl_quantile', y='true_total_score', data=predictions_reg, order=labels_reg, ax=axes[2])
    axes[2].set_ylabel('')
    axes[2].set_xlabel('# Misclassified Items')
    axes[2].set_title('Regression')
    axes[2].tick_params(colors='white', which='both', axis='y')

    sns.despine(offset=10, trim=False, ax=axes[0])
    sns.despine(offset=10, trim=False, left=True, ax=axes[1])
    sns.despine(offset=10, trim=False, left=True, ax=axes[2])

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


def compute_quantiles(predictions, ground_truths) -> Tuple[pd.DataFrame, List[str]]:
    # compute number of misclassifications for each sample
    class_columns = [f'class_item_{i + 1}' for i in range(18)]
    predictions['num_miscl'] = np.sum(
        predictions.loc[:, class_columns] != ground_truths.loc[:, class_columns], axis=1)

    # write true total score to predictions df
    predictions['true_total_score'] = ground_truths.loc[:, 'total_score']
    scores_q0, scores_q1, scores_q2 = predictions['num_miscl'].quantile([0.25, 0.5, 0.75])

    # assign each sample to bin
    labels = [f'≤ {scores_q0}', f'{scores_q0} - {scores_q1}', f'{scores_q1} - {scores_q2}', f'> {scores_q2}']
    predictions['n_miscl_quantile'] = predictions.loc[:, ['num_miscl']].applymap(
        lambda x: labels[np.digitize(x, bins=[-np.inf, scores_q0, scores_q1, scores_q2, np.inf], right=True) - 1])

    return predictions, labels


if __name__ == '__main__':
    results_root = './results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir_reg=results_root + 'final/rey-regressor',
         results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
         results_dir_mlc_sim=results_root + 'final-simulated/rey-multilabel-classifier',
         save_as='./analyze/figures/score_quantiles_plot.pdf'
         )
