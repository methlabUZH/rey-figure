import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.utils import init_mpl

colors = init_mpl(sns_style='ticks', colorpalette='muted')


def get_model_comparison_predictions(results_dir_reg_v1, results_dir_mlc, results_dir_reg_v2,
                                     quantity='num_misclassified'):
    predictions_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_predictions.csv'))
    ground_truths_reg_v2 = pd.read_csv(os.path.join(results_dir_reg_v2, 'test_ground_truths.csv'))
    predictions_reg_v2 = compute_error_rates(predictions_reg_v2, ground_truths_reg_v2, quantity)

    predictions_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_predictions.csv'))
    ground_truths_mlc = pd.read_csv(os.path.join(results_dir_mlc, 'test_ground_truths.csv'))
    predictions_mlc = compute_error_rates(predictions_mlc, ground_truths_mlc, quantity)

    predictions_reg_v1 = pd.read_csv(os.path.join(results_dir_reg_v1, 'test_predictions.csv'))
    ground_truths_reg_v1 = pd.read_csv(os.path.join(results_dir_reg_v1, 'test_ground_truths.csv'))
    predictions_reg_v1 = compute_error_rates(predictions_reg_v1, ground_truths_reg_v1, quantity)

    return [(predictions_mlc, 'Multilabel Classifier'),
            (predictions_reg_v1, 'Regression-V1'),
            (predictions_reg_v2, 'Regression-V2')]


def get_data_comparison_predictions(results_dir0, results_dir_sim, quantity):
    predictions_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_predictions.csv'))
    ground_truths_sim = pd.read_csv(os.path.join(results_dir_sim, 'test_ground_truths.csv'))
    predictions_sim = compute_error_rates(predictions_sim, ground_truths_sim, quantity)

    predictions0 = pd.read_csv(os.path.join(results_dir0, 'test_predictions.csv'))
    ground_truths0 = pd.read_csv(os.path.join(results_dir0, 'test_ground_truths.csv'))
    predictions0 = compute_error_rates(predictions0, ground_truths0, quantity)

    return [(predictions_sim, 'Real Data'), (predictions0, 'Real Data + Simulated')]


def make_plot(list_of_predictions_df, quantity, ylabel, save_as=None, kind='boxen', bw=0.4):
    fig, axes = plt.subplots(nrows=1, ncols=len(list_of_predictions_df), sharey=True, figsize=(15, 5))

    for i, ((predictions, title), ax) in enumerate(zip(list_of_predictions_df, axes)):
        if kind == 'boxen':
            sns.boxenplot(y=quantity, data=predictions, ax=ax, color=colors[i], showfliers=False)
        elif kind == 'violin':
            sns.violinplot(y=quantity, data=predictions, ax=ax, color=colors[i], cut=0, bw=bw)
        else:
            raise ValueError

        # annotate mean and median
        props = dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='gray')

        mean = np.mean(predictions.loc[:, quantity])
        median = np.median(predictions.loc[:, quantity])
        text = '\n'.join((f'mean: {mean:.2f}', f'median: {median:.2f}'))
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(axis='y')
        if i == 0:
            ax.set_ylabel(ylabel)
            sns.despine(offset=10, trim=False, bottom=True, ax=ax)
        else:
            ax.set_ylabel('')
            ax.tick_params(colors='white', which='both')
            sns.despine(offset=10, trim=False, bottom=True, left=True, ax=ax)

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close(fig)
        return

    plt.show()
    plt.close()


def compute_error_rates(predictions, ground_truths, quantity) -> pd.DataFrame:
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

    # model comparison plots -------------------------------------------------------------------------------------------
    # save_as='./figures/num_miscl_boxen.pdf'
    save_as = None
    preds_list = get_model_comparison_predictions(results_dir_reg_v1=results_root + 'final/rey-regressor',
                                                  results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
                                                  results_dir_reg_v2=results_root + 'final/rey-regressor-v2',
                                                  quantity='num_misclassified')

    make_plot(preds_list, quantity='num_misclassified', ylabel='# Misclassified Items', kind='violin', bw=0.35)

    save_as = None
    preds_list = get_model_comparison_predictions(results_dir_reg_v1=results_root + 'final/rey-regressor',
                                                  results_dir_mlc=results_root + 'final/rey-multilabel-classifier',
                                                  results_dir_reg_v2=results_root + 'final/rey-regressor-v2',
                                                  quantity='total_score_absolute_error')

    make_plot(preds_list, quantity='total_score_absolute_error', ylabel='Total Score Absolute Error', kind='violin',
              bw=0.35)

    # data comparison plots --------------------------------------------------------------------------------------------
    # save_as='./figures/num_miscl_boxen.pdf'
    save_as = None
    preds_list = get_data_comparison_predictions(results_dir0=results_root + 'final/rey-regressor-v2',
                                                 results_dir_sim=results_root + 'final-simulated/rey-regressor-v2',
                                                 quantity='num_misclassified')

    make_plot(preds_list, quantity='num_misclassified', ylabel='# Misclassified Items', kind='violin', bw=0.35)

    save_as = None
    preds_list = get_data_comparison_predictions(results_dir0=results_root + 'final/rey-regressor-v2',
                                                 results_dir_sim=results_root + 'final-simulated/rey-regressor-v2',
                                                 quantity='total_score_absolute_error',
                                                 )

    make_plot(preds_list, quantity='total_score_absolute_error', ylabel='Total Score Absolute Error', kind='violin',
              bw=0.35)
