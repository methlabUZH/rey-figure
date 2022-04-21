import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.analyze.utils import init_mpl
from src.analyze.performance_measures import PerformanceMeasures
from constants import N_ITEMS

_FIG_SIZE = (18, 6)

colors = init_mpl()


def make_plot(results_dir, save_as=None):
    with open(os.path.join(results_dir, 'args.json'), 'r') as f:
        args = json.load(f)

    num_classes = args.get('n_classes', 4)
    gts = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))
    preds = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))

    # compute model errors for each bin
    pm = PerformanceMeasures(gts, preds, num_classes=num_classes)
    item_metrics, _, confidence_intervals = pm.item_level_metrics_report()

    mae_per_item = np.array(item_metrics.loc['MAE', :]).reshape(-1, 1)
    mae_errors = np.array(confidence_intervals.loc[['MAE_LOW', 'MAE_HIGH'], :]).T
    mae_errors = np.abs(mae_errors - np.repeat(mae_per_item, repeats=2, axis=1))

    acc_per_item = np.array(item_metrics.loc['Accuracy', :]).reshape(-1, 1)
    acc_errors = np.array(confidence_intervals.loc[['Accuracy_LOW', 'Accuracy_HIGH'], :]).T
    acc_errors = np.abs(acc_errors - np.repeat(acc_per_item, repeats=2, axis=1))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=_FIG_SIZE)

    barwidth = 0.25
    bars_pos = np.arange(len(mae_per_item))

    # bars for mae
    axes[0].bar(bars_pos, mae_per_item.reshape(-1), width=barwidth, color=colors[0], edgecolor='black', yerr=mae_errors.T,
                capsize=7)
    axes[0].set_xticks(bars_pos)
    axes[0].set_xticklabels([f'Item {i + 1}' for i in range(N_ITEMS)])
    axes[0].set_title('Mean Absolute Error per Item')
    axes[0].grid(True)

    # bars for accuracy
    axes[1].bar(bars_pos, acc_per_item.reshape(-1), width=barwidth, color=colors[1], edgecolor='black', yerr=acc_errors.T,
                capsize=7)
    axes[1].set_xticks(bars_pos)
    axes[1].set_xticklabels([f'Item {i + 1}' for i in range(N_ITEMS)])
    axes[1].set_title('Classification Accuracy per Item')
    axes[1].set_ylim(0.75, 1.0)
    axes[1].grid(True)

    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')
