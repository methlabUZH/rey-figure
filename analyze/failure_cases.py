import cv2
import numpy as np
import os
import pandas as pd
import shutil
from skimage.color import rgb2gray as skimage_rgb2gray
from skimage import io
from tqdm import tqdm
from typing import *

from constants import *

_DATA_ROOT = '/Users/maurice/phd/src/rey-figure/data/'
_TEX_REPORTS_ROOT = os.path.join(ROOT_DIR, 'tex_reports/failure_cases/')
_CLUSTER_DATA_ROOT = '/cluster/work/zhang/webermau/rocf/psychology/'
_SCORE_COLUMNS = [f'score_item_{i + 1}' for i in range(N_ITEMS)] + ['total_score']
_NUM_CASES = 200
_REFERENCE_FILE = os.path.join(RESOURCES_DIR, 'reference.pdf')

_QUANTITY_ABSOLUTE_ERROR = 'total_score_absolute_error'
_QUANTITY_NUM_MISCLASSIFIED = 'num_misclassified'


def main(results_dir, quantity=_QUANTITY_NUM_MISCLASSIFIED):
    predictions = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))
    ground_truths = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))

    # process dataframes
    file_columns = ['image_file', 'serialized_file']
    predictions = predictions.set_index('figure_id')
    predictions.loc[:, file_columns] = predictions.loc[:, file_columns].applymap(
        lambda s: s.replace(_CLUSTER_DATA_ROOT, _DATA_ROOT))

    ground_truths = ground_truths.set_index('figure_id')
    ground_truths.loc[:, file_columns] = ground_truths.loc[:, file_columns].applymap(
        lambda s: s.replace(_CLUSTER_DATA_ROOT, _DATA_ROOT))

    # get misclassification rates
    predictions = compute_error_rates(predictions, ground_truths, quantity=quantity)

    # sort wrt number of misclassified items
    predictions = predictions.sort_values(quantity, ascending=False)

    # setup dir structure
    tex_reports_dir = os.path.join(_TEX_REPORTS_ROOT, quantity)
    if not os.path.exists(tex_reports_dir):
        os.makedirs(tex_reports_dir)
        os.makedirs(os.path.join(tex_reports_dir, 'preprocessed_images'))

    # copy reference.jpg to dir
    shutil.copy(_REFERENCE_FILE, os.path.join(tex_reports_dir, 'reference.pdf'))

    tex_doc = _init_doc()
    image_files = []

    for figure_id, row in tqdm(predictions.head(n=_NUM_CASES).iterrows(), total=_NUM_CASES):
        image_fp = row['image_file']
        image_fp_preprocessed = os.path.join(tex_reports_dir, f'preprocessed_images/{figure_id}.jpg')

        image_files.append(image_fp.replace(_DATA_ROOT, ''))

        # preprocess and save image
        image = cv2.imread(image_fp)
        image_preprocessed = skimage_rgb2gray(image)
        io.imsave(image_fp_preprocessed, (image_preprocessed * 255).astype(np.uint8))

        # get statistics
        predicted_scores = list(row[_SCORE_COLUMNS])
        true_scores = list(ground_truths.loc[figure_id, _SCORE_COLUMNS])
        correct_predictions = np.array(np.array(predicted_scores[:-1]) == np.array(true_scores[:-1])).astype(int)
        corr_preds_strings = [
            r"\textcolor{red}{\xmark}" if x == 0 else r"\textcolor{green}{\cmark}" for x in correct_predictions
        ]
        corr_preds_strings += [sum(correct_predictions)]

        absolute_errors = list(np.array(predicted_scores) - np.array(true_scores))
        tex_doc += create_page(figure_id, image_fp_preprocessed, predicted_scores, true_scores, corr_preds_strings,
                               absolute_errors)

    tex_doc += "\n" + r"\end{document}"

    # write image files to text
    txt_file = os.path.join(tex_reports_dir, 'image_files.txt')
    with open(txt_file, 'w') as f:
        f.write("\n".join(image_files))

    # write to .tex and compile
    tex_file = os.path.join(tex_reports_dir, 'main.tex')
    with open(tex_file, 'w') as f:
        f.write(tex_doc)

    # compile
    os.chdir(tex_reports_dir)
    os.system(f"pdflatex {tex_file}")


def create_page(figure_id, figure_path, predicted_scores: List[float], true_scores: List[float],
                correct_predictions: List, absolute_errors: List[float]):
    return "\n".join([
        rf"\subsection*{{ID: {figure_id}}}".replace("_", r"\_").replace(r"^", r"\^"),
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[height=7.5cm]{{{figure_path}}}",
        r"\end{figure}",
        r"\begin{figure}[h]",
        r"\centering",
        r"\includegraphics[height=5cm]{reference.pdf}",
        r"\end{figure}",
        r"\begin{table}[!th]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c c c c c c c c c c c c c c c c c c c}",
        r"\toprule",
        rf"Item & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 & 17 & 18 & Total\\",
        r"\toprule",
        r"Prediction ($\hat{{y}}$)&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}\\".format(
            *predicted_scores),
        r"\midrule",
        r"True Score ($y$)&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}\\".format(*true_scores),
        r"\midrule",
        r"Correct&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}/18\\".format(*correct_predictions),
        r"\midrule",
        r"Error ($\hat{{y}} - y$)&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}\\".format(*absolute_errors),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "\n",
        r"\newpage"
    ])


def _init_doc():
    return r"""
\documentclass[a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{a4wide}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{pifont}
\usepackage{xcolor}
\geometry{a4paper, total={170mm,257mm}, top=20mm}

\newcommand{\cmark}{\ding{51}}%
\newcommand{\xmark}{\ding{55}}%

\title{Failure Cases for Multilabel Classifier}
\date{\today}

\begin{document}
\maketitle

"""


def compute_error_rates(preds, ground_truths, quantity) -> pd.DataFrame:
    if quantity == _QUANTITY_NUM_MISCLASSIFIED:
        # compute number of misclassifications for each sample
        class_columns = [f'class_item_{i + 1}' for i in range(18)]
        preds[quantity] = np.sum(preds.loc[:, class_columns] != ground_truths.loc[:, class_columns], axis=1)
    elif quantity == _QUANTITY_ABSOLUTE_ERROR:
        # compute total score absolute error for each sample
        preds[quantity] = np.abs(preds.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    else:
        raise ValueError

    return preds


if __name__ == '__main__':
    results_root = '../results/euler-results/data-2018-2021-116x150-pp0/'
    main(results_dir=results_root + 'final/rey-multilabel-classifier', quantity=_QUANTITY_NUM_MISCLASSIFIED)
