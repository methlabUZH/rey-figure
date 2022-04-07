import pandas as pd
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from constants import ABSOLUTE_ERROR, ERR_LEVEL_TOTAL_SCORE
from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures

_FIG_SIZE = (5, 4)

colors = init_mpl()


def make_plot(resolutions_and_res_dirs: List[Tuple[str, str]], pmeasure=ABSOLUTE_ERROR, save_as=None):
    x_labels = []
    y_values_no_augm, y_values_augm = [], []

    # compute errors
    for resolution, res_dir in resolutions_and_res_dirs:
        # without data augmentation
        preds = pd.read_csv(os.path.join(res_dir, 'test_predictions.csv'))
        gts = pd.read_csv(os.path.join(res_dir, 'test_ground_truths.csv'))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_no_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                               confidence_interval=True))

        # with data augmentation
        preds = pd.read_csv(os.path.join(res_dir.replace('/final/', '/final-aug/'), 'test_predictions.csv'))
        gts = pd.read_csv(os.path.join(res_dir.replace('/final/', '/final-aug/'), 'test_ground_truths.csv'))
        pm = PerformanceMeasures(gts, preds)

        # low, mean, high confidence interval
        y_values_augm.append(pm.compute_performance_measure(pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE,
                                                            confidence_interval=True))

        x_labels.append(resolution)

    #

    plt.figure(figsize=_FIG_SIZE)
    plt.plot(range(len(x_labels)), y_values_no_augm, marker='o', label='without Data Augmentation')
    plt.plot(range(len(x_labels)), y_values_augm, marker='d', label='with Data Augmentation')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Image Resolution')
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
