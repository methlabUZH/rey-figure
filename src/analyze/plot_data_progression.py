import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Iterable

from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures
from constants import ABSOLUTE_ERROR, ERR_LEVEL_TOTAL_SCORE, ERROR_TO_LABEL

_FIG_SIZE = (7, 4)

colors = init_mpl()


def make_plot(dir_configs, pmeasure=ABSOLUTE_ERROR, save_as=None):
    plt.figure(figsize=_FIG_SIZE)

    x_labels = []

    for cfg in dir_configs:
        label = cfg['label']
        linestyle = cfg['ls']
        marker = cfg['marker']
        color = colors[cfg['color_idx']]
        x_labels, y_values = [], []

        for res_dir, num_data in cfg['res-dirs']:
            gts = pd.read_csv(os.path.join(res_dir, 'test_ground_truths.csv'))
            preds = pd.read_csv(os.path.join(res_dir, 'test_predictions.csv'))
            pm = PerformanceMeasures(gts, preds)
            y_values.append(pm.compute_performance_measure(
                pmeasure=pmeasure, error_level=ERR_LEVEL_TOTAL_SCORE, confidence_interval=True))
            x_labels.append(num_data)

        # separate confidence intervals
        y_values = np.array(y_values)
        y_errs = np.abs(y_values[:, np.array([0, 2])] - np.repeat(y_values[:, 1].reshape(-1, 1), repeats=2, axis=1))

        plt.errorbar(range(len(x_labels)), y_values[:, 1], yerr=y_errs.T, elinewidth=1.0, capsize=5, capthick=2,
                     ls=linestyle, marker=marker, label=label, color=color)

    plt.ylabel(ERROR_TO_LABEL[pmeasure])
    plt.xlabel('# Training Samples')
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
