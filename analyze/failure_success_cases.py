from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from constants import RESULTS_DIR, N_ITEMS
from src.inference.preprocess import load_and_normalize_image
from src.inference.utils import assign_score
from src.utils import print_dataframe

RESULTS_DIR = os.path.join(RESULTS_DIR, 'scans-2018-116x150-augmented/id-1/')
ID_TO_PATH = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150/figures_paths.csv'
ID_TO_PATH = pd.read_csv(ID_TO_PATH).set_index('figure_id')['filepath_npy'].to_dict()


def get_num_equal_per_row(row):
    predictions = row[[f'pred_item_{k}_present' for k in range(1, N_ITEMS + 1)]]
    ground_truths = row[[f'item_{k}_present' for k in range(1, N_ITEMS + 1)]]
    n_correct = np.sum(predictions.values == ground_truths.values)
    return n_correct


def get_mse_per_row(row):
    predictions = row[[f'true_score_item_{k}' for k in range(1, N_ITEMS + 1)]]
    ground_truths = row[[f'pred_score_item_{k}' for k in range(1, N_ITEMS + 1)]]
    mse = np.sum((ground_truths.values - np.clip(predictions.values, 0, 2)) ** 2)
    return mse


def get_fail_success_instances(regressor_predictions_csv: str, classifiers_predictions_csv: str, metric: str,
                               n_failure: int = 20, n_success: int = 20):
    assert metric in ['mse', 'correct-presence']

    regression_predictions = pd.read_csv(regressor_predictions_csv, index_col=0)
    classifiers_predictions = pd.read_csv(classifiers_predictions_csv, index_col=0)
    predictions = pd.merge(classifiers_predictions, regression_predictions,
                           on=['figure_id', 'image_file', 'serialized_file'])
    predictions['mse'] = predictions.apply(get_mse_per_row, axis=1)
    predictions['correct-presence'] = predictions.apply(get_num_equal_per_row, axis=1)
    predictions = predictions.sort_values(by=metric, ascending=True)
    failure_cases = predictions.head(n_failure)
    success_cases = predictions.tail(n_success)

    failure_cases_dict, success_cases_dict = {}, {}
    for (_, fail_row), (_, succ_row) in zip(failure_cases.iterrows(), success_cases.iterrows()):
        failure_cases_dict[fail_row['figure_id']] = {
            'items': [k for k in range(1, N_ITEMS + 1)],
            'classifier': fail_row[[f'pred_item_{k}_present' for k in range(1, N_ITEMS + 1)]].values,
            'regressor': fail_row[[f'pred_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values,
            'true_scores': fail_row[[f'true_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values
        }
        success_cases_dict[succ_row['figure_id']] = {
            'items': [k for k in range(1, N_ITEMS + 1)],
            'classifier': succ_row[[f'pred_item_{k}_present' for k in range(1, N_ITEMS + 1)]].values,
            'regressor': succ_row[[f'pred_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values,
            'true_scores': succ_row[[f'true_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values
        }

    return failure_cases_dict, success_cases_dict


def generate_images_and_tables(predictions_dict, save_dir) -> str:
    latex = ''
    for figure_id, data in predictions_dict.items():
        # process and save image
        image_fp = ID_TO_PATH[figure_id]
        image = load_and_normalize_image(image_fp)
        image = np.squeeze(image.numpy())
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, figure_id + '.pdf'), dpi=250, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # write as dataframe and compute total scores
        data = deepcopy(predictions_dict[figure_id])
        preds_df = pd.DataFrame.from_dict(data)
        preds_df = preds_df.set_index('items')
        preds_df['regressor'] = preds_df['regressor'].apply(assign_score)
        ground_truth_binary = np.zeros_like(preds_df['true_scores'].values)
        ground_truth_binary[preds_df['true_scores'].values > 0] = 1.0
        classifier_correct = np.sum(preds_df['classifier'].values == ground_truth_binary)
        hybrid_score = (preds_df['regressor'][:-1] * preds_df['classifier'][:-1]).sum()
        regressor_score = preds_df['regressor'].sum()
        true_score = preds_df['true_scores'].sum()
        preds_df.loc['total score', :] = [classifier_correct, regressor_score, true_score]
        preds_df.loc['total hybrid', :] = [hybrid_score, hybrid_score, true_score]
        preds_df = preds_df.reset_index()

        # generate table
        latex += '\\begin{table}[H]\n'
        latex += '\\begin{minipage}[c]{0.45\\textwidth}\n\\centering\n\\includegraphics[width=\\linewidth]'
        latex += '{figures/examples/' + figure_id + '.pdf}\n'
        latex += '\\end{minipage}\\hfill\n'
        latex += '\\begin{minipage}[c]{0.45\\textwidth}\n\\centering\n'
        latex += preds_df.to_latex(index=False)
        latex += '\\end{minipage}\n'
        latex += '\\end{table}\n'

    latex = latex.replace('llrl', 'lccc')
    return latex


def main():
    failure_cases, success_cases = get_fail_success_instances(os.path.join(RESULTS_DIR, 'regressor_predictions.csv'),
                                                              os.path.join(RESULTS_DIR, 'classifiers_predictions.csv'),
                                                              metric='correct-presence')

    # generate latex code for failure cases
    failure_figures_dir = os.path.join(RESULTS_DIR, 'figures/failure')
    if not os.path.exists(failure_figures_dir):
        os.makedirs(failure_figures_dir)
    latex_code = generate_images_and_tables(failure_cases, failure_figures_dir)
    with open(os.path.join(failure_figures_dir, 'tables-failure.txt'), 'w') as f:
        f.write(latex_code)
    print(f'==> saved failure cases in {failure_figures_dir}')

    # generate latex code for success cases
    success_figures_dir = os.path.join(RESULTS_DIR, 'figures/success')
    if not os.path.exists(success_figures_dir):
        os.makedirs(success_figures_dir)
    latex_code = generate_images_and_tables(success_cases, success_figures_dir)
    with open(os.path.join(success_figures_dir, 'tables-success.txt'), 'w') as f:
        f.write(latex_code)
    print(f'==> saved success cases in {success_figures_dir}')


if __name__ == '__main__':
    main()
