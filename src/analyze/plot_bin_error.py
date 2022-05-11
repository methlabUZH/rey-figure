import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures
from constants import ABSOLUTE_ERROR, ERROR_TO_LABEL

_FIG_SIZE = (8, 4)

colors = init_mpl()


def make_plot(results_dir, pmeasure=ABSOLUTE_ERROR, save_as=None, label=None):
    try:
        with open(os.path.join(results_dir, 'args.json'), 'r') as f:
            args = json.load(f)
    except FileNotFoundError:
        args = {}

    num_classes = args.get('n_classes', 4)
    gts = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))
    preds = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))

    # compute model errors for each bin
    pm = PerformanceMeasures(gts, preds, num_classes=num_classes)
    bin_metrics = pm.bin_level_metrics_report(bin_granularity=4, compute_ci=True)
    y_values = bin_metrics['MAE' if pmeasure == ABSOLUTE_ERROR else 'MSE']
    y_cis = bin_metrics['MAE_CI' if pmeasure == ABSOLUTE_ERROR else 'MSE_CI']

    plt.figure(figsize=_FIG_SIZE)

    x_labels = bin_metrics['scores']
    y_values = np.array(y_values).reshape(-1, 1)
    y_errs = np.array([np.array(ci) for ci in y_cis])
    y_errs = y_errs[:, [0, 2]]
    y_errs = np.abs(y_errs - np.repeat(y_values, repeats=2, axis=1))

    plt.errorbar(range(len(x_labels)), y_values, yerr=y_errs.T, elinewidth=1.0, capsize=5, capthick=2,
                 ls='-.', marker='d', label=label or 'CNN with Data Augmentation', color=colors[1])

    plt.ylabel(ERROR_TO_LABEL[pmeasure])
    plt.xlabel('Total Scores')
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
