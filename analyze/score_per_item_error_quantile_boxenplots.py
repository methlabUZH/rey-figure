import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Tuple

from src.utils import init_mpl

init_mpl(sns_style='ticks', colorpalette='muted')


def get_predictions_model_comparison(results_dir_reg_v1, results_dir_mlc, results_dir_reg_v2):
    predictions_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_predictions.csv'))
    ground_truths_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_ground_truths.csv'))
    predictions_reg_v2, labels_reg_v2 = compute_quantiles_figures(predictions_reg_v2, ground_truths_reg_v2)

    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc, labels_mlc = compute_quantiles_figures(predictions_mlc, ground_truths_mlc)

    predictions_reg_v1 = pd.read_csv(os.path.join(results_dir_reg_v1, 'test_predictions.csv'))
    ground_truths_reg_v1 = pd.read_csv(os.path.join(results_dir_reg_v1, 'test_ground_truths.csv'))
    predictions_reg_v1, labels_reg_v1 = compute_quantiles_figures(predictions_reg_v1, ground_truths_reg_v1)

    return [(predictions_mlc, labels_mlc, 'Multilabel Classsifier'),
            (predictions_reg_v1, labels_reg_v1, 'Regression-V1'),
            (predictions_reg_v2, labels_reg_v2, 'Regression-V2')]


def get_predictions_data_comparison(results_dir0, results_dir_sim):
    predictions_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_predictions.csv'))
    ground_truths_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_ground_truths.csv'))
    predictions_sim, labels_sim = compute_quantiles_figures(predictions_sim, ground_truths_sim)

    predictions0 = pd.read_csv(os.path.join(results_dir0, 'test_predictions.csv'))
    ground_truths0 = pd.read_csv(os.path.join(results_dir0, 'test_ground_truths.csv'))
    predictions0, labels_reg0 = compute_quantiles_figures(predictions0, ground_truths0)

    return [(predictions_sim, labels_sim, 'Real Data'), (predictions0, labels_reg0, 'Real Data + Simulated')]


def make_plot(predictions_list, save_as=None, kind='boxen'):
    fig, axes = plt.subplots(nrows=1, ncols=len(predictions_list), sharey=True, figsize=(15, 5))

    for i, ((predictions, labels, title), ax) in enumerate(zip(predictions_list, axes)):
        if kind == 'boxen':
            sns.boxenplot(x='n_miscl_quantile', y='true_total_score', data=predictions, order=labels, ax=ax)
        elif kind == 'violin':
            sns.violinplot(x='n_miscl_quantile', y='true_total_score', data=predictions, order=labels, ax=ax, cut=0,
                           width=1.0)
        else:
            raise ValueError

        if i == 0:
            ax.set_ylabel('Total Score')
            sns.despine(offset=10, trim=False, ax=ax)
        else:
            ax.tick_params(colors='white', which='both', axis='y')
            ax.set_ylabel('')
            sns.despine(offset=10, trim=False, left=True, ax=ax)

        ax.set_xlabel('Quartiles (# Misclassified Items)')
        ax.set_title(title)
        ax.grid(axis='y')

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close()


def compute_quantiles_figures(predictions, ground_truths) -> Tuple[pd.DataFrame, List[str]]:
    # compute number of misclassifications for each sample
    class_columns = [f'class_item_{i + 1}' for i in range(18)]
    predictions['num_miscl'] = np.sum(
        predictions.loc[:, class_columns] != ground_truths.loc[:, class_columns], axis=1)

    # write true total score to predictions df
    predictions['true_total_score'] = ground_truths.loc[:, 'total_score']
    predictions['true_total_score'] = ground_truths.loc[:, 'total_score']
    scores_q0, scores_q1, scores_q2 = predictions['num_miscl'].quantile([0.25, 0.5, 0.75])

    # assign each sample to bin
    labels = [f'≤ {scores_q0}', f'{scores_q0} - {scores_q1}', f'{scores_q1} - {scores_q2}', f'> {scores_q2}']
    predictions['n_miscl_quantile'] = predictions.loc[:, ['num_miscl']].applymap(
        lambda x: labels[np.digitize(x, bins=[-np.inf, scores_q0, scores_q1, scores_q2, np.inf], right=True) - 1])

    return predictions, labels


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'

    # model comparison plots -------------------------------------------------------------------------------------------
    preds_list = get_predictions_model_comparison(
        results_dir_reg_v1=results_root + 'final/rey-regressor',
        results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
        results_dir_reg_v2=results_root + 'final/rey-regressor-v2')
    make_plot(preds_list, save_as='./figures/score_n_miscl_quantiles_models.pdf', kind='violin')

    # data comparison plots -------------------------------------------------------------------------------------------
    preds_list = get_predictions_data_comparison(
        results_dir0=results_root + 'final/rey-regressor-v2',
        results_dir_sim=results_root + 'final/rey-multilabel-classifier')
    make_plot(preds_list, save_as='./figures/score_n_miscl_quantiles_data.pdf', kind='violin')
