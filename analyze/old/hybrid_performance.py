import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.inference.model_initialization import get_classifiers_checkpoints, init_regressor, init_classifier
from src.inference.predict import do_score_image
from constants import RESULTS_DIR, N_ITEMS
from analyze0.failure_success_cases import get_fail_success_instances
from src.inference.preprocess import load_and_normalize_image

RESULTS_DIR = os.path.join(RESULTS_DIR, 'scans-2018-116x150-augmented/id-1/')
CLASSIFIERS_DIR = os.path.join(RESULTS_DIR, 'item-classifier')
REGRESSOR_DIR = os.path.join(RESULTS_DIR, 'rey-regressor')

# preprocessing
ID_TO_PATH = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150/figures_paths.csv'
DEFAULT_IMG = '/Users/maurice/phd/src/rey-figure/preprocessing/serialized-preprocessing/scans-2018-116x150/data2018/'
DEFAULT_IMG += 'uploadFinal/Colombia092_f2_NaN.npy'
IMAGE_SIZE = 116, 150
NORM_LAYER = 'batch_norm'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='path to image file', default=None)
args = parser.parse_args()


def display_predictions(predictions, ground_truth, image, show_figure=True, print_table=True):
    # print results
    if ground_truth is None:
        data = [[item, present, score] for item, (present, score) in predictions.items()]
    else:
        data = [[item, present, score, ground_truth[item - 1]] for item, (present, score) in predictions.items()]

    # write scores to df
    df = pd.DataFrame(data, columns=['item', 'classifier', 'regressor', 'ground_truth'])
    df = df.set_index('item')
    ground_truth_binary = np.zeros_like(ground_truth)
    ground_truth_binary[ground_truth > 0] = 1.0
    classifier_correct = np.sum(df['classifier'].values == ground_truth_binary)
    hybrid_score = (df['regressor'][:-1] * df['classifier'][:-1]).sum()
    regressor_score = df['regressor'].sum()
    true_score = df['ground_truth'].sum()
    df.loc['total score', :] = [classifier_correct, regressor_score, true_score]
    df.loc['total hybrid', :] = [hybrid_score, hybrid_score, true_score]

    if print_table:
        print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    # plot figure and scores
    if show_figure:
        image = np.squeeze(image)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0].axis('off')

        text = 'item {0:2}: classifier / regressor / ground truth\n'.format('k')
        for item, row in df.iterrows():
            text += 'item {0:2}: {1} / {2} / {3}\n'.format(item, row['classifier'], row['regressor'],
                                                           row['ground_truth'])

        axes[1].text(0.1, 0.9, text, horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)
        axes[1].axis('off')

        plt.show()
        plt.close()

    return df


def predict_single(regressor, classifiers, image_fp, ground_truth=None, show_figure=True, predict_table=True):
    input_tensor = load_and_normalize_image(image_fp)
    final_scores = do_score_image(input_tensor, regressor, classifiers)
    predictions_df = display_predictions(final_scores, ground_truth, input_tensor.numpy(), show_figure, predict_table)
    return predictions_df, np.squeeze(input_tensor.numpy())


def predict_from_csv(regressor, classifiers):
    id_to_path = pd.read_csv(ID_TO_PATH)
    id_to_path = id_to_path.set_index('figure_id')['filepath_npy'].to_dict()

    failure_cases, success_cases = get_fail_success_instances(os.path.join(RESULTS_DIR, 'regressor_predictions.csv'),
                                                              os.path.join(RESULTS_DIR, 'classifiers_predictions.csv'),
                                                              metric='correct-presence')
    print('\n//-- classifiers failure instances --//\n')
    print(tabulate(failure_cases, headers='keys', tablefmt='presto', floatfmt=".3f"))
    print('\n//-- classifiers success instances --//\n')
    print(tabulate(success_cases, headers='keys', tablefmt='presto', floatfmt=".3f"))

    failure_figures_and_predictions = []
    for _, row in tqdm(failure_cases.iterrows(), total=len(failure_cases)):
        figure_id = row['figure_id']
        figure_fp = id_to_path[figure_id]
        ground_truth = row[[f'true_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values
        predictions_df, image_npy = predict_single(regressor, classifiers, figure_fp, ground_truth, False, False)
        failure_figures_and_predictions.append([image_npy, figure_id, predictions_df])

    success_figures_and_predictions = []
    for _, row in tqdm(success_cases.iterrows(), total=len(success_cases)):
        figure_id = row['figure_id']
        figure_fp = id_to_path[figure_id]
        ground_truth = row[[f'true_score_item_{k}' for k in range(1, N_ITEMS + 1)]].values
        predictions_df, image_npy = predict_single(regressor, classifiers, figure_fp, ground_truth, False, False)
        success_figures_and_predictions.append([image_npy, figure_id, predictions_df])

    # generate latex code for failure cases
    failure_figures_dir = os.path.join(RESULTS_DIR, 'figures/failure')
    if not os.path.exists(failure_figures_dir):
        os.makedirs(failure_figures_dir)
    latex_code = generate_images_and_tables(failure_figures_and_predictions, failure_figures_dir)
    with open(os.path.join(failure_figures_dir, 'tables-failure.txt'), 'w') as f:
        f.write(latex_code)

    # generate latex code for success cases
    success_figures_dir = os.path.join(RESULTS_DIR, 'figures/success')
    if not os.path.exists(success_figures_dir):
        os.makedirs(success_figures_dir)
    latex_code = generate_images_and_tables(success_figures_and_predictions, success_figures_dir)
    with open(os.path.join(success_figures_dir, 'tables-success.txt'), 'w') as f:
        f.write(latex_code)


def generate_images_and_tables(figures_and_predictions, save_dir) -> str:
    latex = ''
    for image_npy, figure_id, preds_df in figures_and_predictions:
        # process and save image
        image = (image_npy - np.min(image_npy)) / (np.max(image_npy) - np.min(image_npy))
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, figure_id + '.pdf'), dpi=250, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # generate table
        latex += '\\begin{table}[H]\n'
        latex += '\\begin{minipage}[c]{0.45\\textwidth}\n\\centering\n\\includegraphics[width=\\linewidth]'
        latex += '{figures/examples/' + figure_id + '.pdf}\n'
        latex += '\\end{minipage}\\hfill\n'
        latex += '\\begin{minipage}[c]{0.45\\textwidth}\n\\centering\n'
        latex += preds_df.reset_index().to_latex(index=False)
        latex += '\\end{minipage}\n'
        latex += '\\end{table}\n'

    latex = latex.replace('lrrr', 'lccc')
    return latex


def main():
    reg_ckpt_fp = os.path.join(REGRESSOR_DIR, 'checkpoints/model_best.pth.tar')
    items_and_cls_ckpt_files = get_classifiers_checkpoints(CLASSIFIERS_DIR)

    # init regressor
    regressor = init_regressor(reg_ckpt_fp)

    # init item classifiers
    classifiers = {i: init_classifier(ckpt_fp) for i, ckpt_fp in items_and_cls_ckpt_files}

    predict_from_csv(regressor, classifiers)


if __name__ == '__main__':
    main()
