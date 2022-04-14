import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Iterable

from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures
from constants import (ABSOLUTE_ERROR,
                       ERR_LEVEL_TOTAL_SCORE,
                       ERROR_TO_LABEL,
                       TF_ROTATION, TF_CONTRAST, TF_PERSPECTIVE, TF_BRIGHTNESS)

__all__ = ['make_plot']

_FIG_SIZE = (7, 4)
_TF_TO_LABEL = {
    TF_ROTATION: 'Rotation Angle',
    TF_PERSPECTIVE: 'Perspective Distortion',
    TF_CONTRAST: 'Contrast Change'
}

colors = init_mpl()


def make_plot(results_dir, results_dir_aug, transform, transform_params, xlabel=None, pmeasure=ABSOLUTE_ERROR,
              save_as=None):
    x_labels = []
    y_values_no_augm, y_values_augm = [], []

    # compute errors
    for params in transform_params:
        gts_csv, preds_csv, label = _get_predictions_filenames_and_label(transform, params)

        # without data augmentation
        gts = pd.read_csv(os.path.join(results_dir, gts_csv))
        preds = pd.read_csv(os.path.join(results_dir, preds_csv))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_no_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                               confidence_interval=True))

        # with data augmentation
        gts = pd.read_csv(os.path.join(results_dir_aug, gts_csv))
        preds = pd.read_csv(os.path.join(results_dir_aug, preds_csv))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                            confidence_interval=True))

        x_labels.append(label)

    plt.figure(figsize=_FIG_SIZE)

    # separate confidence intervals
    y_values_augm = np.array(y_values_augm)
    y_values_no_augm = np.array(y_values_no_augm)

    y_errs_augm = np.abs(
        y_values_augm[:, np.array([0, 2])] - np.repeat(y_values_augm[:, 1].reshape(-1, 1), repeats=2, axis=1))

    y_errs_no_augm = np.abs(
        y_values_no_augm[:, np.array([0, 2])] - np.repeat(y_values_no_augm[:, 1].reshape(-1, 1), repeats=2, axis=1))

    plt.errorbar(range(len(x_labels)), y_values_no_augm[:, 1], yerr=y_errs_no_augm.T,
                 label='without Data Augmentation', elinewidth=1.0, capsize=5, capthick=2, ls='--', marker='o')
    plt.errorbar(range(len(x_labels)), y_values_augm[:, 1], yerr=y_errs_augm.T, label='with Data Augmentation',
                 elinewidth=1.0, capsize=5, capthick=2, ls='-.', marker='d')

    plt.ylabel(ERROR_TO_LABEL[pmeasure])
    plt.xlabel(_TF_TO_LABEL[transform] if xlabel is None else xlabel)
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(True)
    plt.legend(fancybox=False, fontsize=12)
    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')


def _get_predictions_filenames_and_label(transform, params) -> Tuple[str, str, str]:
    params = [params] if not isinstance(params, Iterable) else params
    if transform == TF_ROTATION:
        csv_pattern = "rotation_[{}, {}]-{}.csv"
        label = "{}-{}".format(*[int(p) for p in params])
    elif transform == TF_CONTRAST:
        csv_pattern = "contrast_{}-{}.csv"
        label = '{}'.format(*params)
    elif transform == TF_PERSPECTIVE:
        csv_pattern = "perspective_{}-{}.csv"
        label = '{}'.format(*params)
    elif transform == TF_BRIGHTNESS:
        csv_pattern = "brightness_{}-{}.csv"
        label = '{}'.format(*params)
    else:
        raise ValueError('unknown transformation')

    # return csv_pattern.format(*params, 'test_ground_truths'), csv_pattern.format(*params, 'test_predictions')
    return "test_ground_truths.csv", csv_pattern.format(*params, 'test_predictions'), label
