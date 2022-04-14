import pandas as pd
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from typing import List, Tuple

from constants import ABSOLUTE_ERROR, ERR_LEVEL_TOTAL_SCORE, ERROR_TO_LABEL, R_SQUARED
from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures

_FIG_SIZE = (5, 4)

colors = init_mpl()


def make_plot(resolutions_and_res_dirs: List[Tuple[str, str]], pmeasure=ABSOLUTE_ERROR, save_as=None):
    x_labels = []
    y_values_no_augm, y_values_augm = [], []

    ci = False if pmeasure == R_SQUARED else True

    # compute errors
    for resolution, res_dir in resolutions_and_res_dirs:
        # without data augmentation
        preds = pd.read_csv(os.path.join(res_dir, 'test_predictions.csv'))
        gts = pd.read_csv(os.path.join(res_dir, 'test_ground_truths.csv'))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_no_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                               confidence_interval=ci))

        # with data augmentation
        preds = pd.read_csv(os.path.join(res_dir.replace('/final/', '/final-aug/'), 'test_predictions.csv'))
        gts = pd.read_csv(os.path.join(res_dir.replace('/final/', '/final-aug/'), 'test_ground_truths.csv'))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                            confidence_interval=ci))

        x_labels.append(resolution)

    plt.figure(figsize=_FIG_SIZE)

    # separate confidence intervals
    y_values_augm = np.array(y_values_augm)
    y_values_no_augm = np.array(y_values_no_augm)
    if ci:
        y_errs_augm = np.abs(
            y_values_augm[:, np.array([0, 2])] - np.repeat(y_values_augm[:, 1].reshape(-1, 1), repeats=2, axis=1))

        y_errs_no_augm = np.abs(
            y_values_no_augm[:, np.array([0, 2])] - np.repeat(y_values_no_augm[:, 1].reshape(-1, 1), repeats=2, axis=1))

        plt.errorbar(range(len(x_labels)), y_values_no_augm[:, 1], yerr=y_errs_no_augm.T, elinewidth=1.0, capsize=5,
                     capthick=2, ls='--', marker='o', color=colors[0], label='without Data Augmentation')
        plt.errorbar(range(len(x_labels)), y_values_augm[:, 1], yerr=y_errs_augm.T, elinewidth=1.0, capsize=5,
                     capthick=2, ls='-.', marker='d', color=colors[1], label='with Data Augmentation')

    else:
        plt.plot(range(len(x_labels)), y_values_no_augm, label='without Data Augmentation', ls='--', marker='o')
        plt.plot(range(len(x_labels)), y_values_augm, label='with Data Augmentation', ls='--', marker='d')

    plt.legend(fancybox=False, fontsize=12)
    plt.ylabel(ERROR_TO_LABEL[pmeasure])
    plt.xlabel('Image Resolution')
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(True)
    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')
