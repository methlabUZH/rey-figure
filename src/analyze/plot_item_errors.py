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


def make_plot(results_dir1, results_dir2=None, label1=None, label2=None, save_as=None, num_classes=4, color_idx=0):
    gts = pd.read_csv(os.path.join(results_dir1, 'test_ground_truths.csv'))
    preds = pd.read_csv(os.path.join(results_dir1, 'test_predictions.csv'))

    # compute model errors for each bin
    pm = PerformanceMeasures(gts, preds, num_classes=num_classes)
    item_metrics, _, confidence_intervals = pm.item_level_metrics_report(compute_ci=True)

    mae_per_item = np.array(item_metrics.loc['MAE', :]).reshape(-1, 1)
    mae_errors = np.array(confidence_intervals.loc[['MAE_LOW', 'MAE_HIGH'], :]).T
    mae_errors = np.abs(mae_errors - np.repeat(mae_per_item, repeats=2, axis=1))

    acc_per_item = np.array(item_metrics.loc['Accuracy', :]).reshape(-1, 1)
    acc_errors = np.array(confidence_intervals.loc[['Accuracy_LOW', 'Accuracy_HIGH'], :]).T
    acc_errors = np.abs(acc_errors - np.repeat(acc_per_item, repeats=2, axis=1))

    if results_dir2 is not None:
        gts2 = pd.read_csv(os.path.join(results_dir2, 'test_ground_truths.csv'))
        preds2 = pd.read_csv(os.path.join(results_dir2, 'test_predictions.csv'))

        pm = PerformanceMeasures(gts2, preds2, num_classes=num_classes)
        item_metrics2, _, confidence_intervals2 = pm.item_level_metrics_report(compute_ci=True)

        mae_per_item2 = np.array(item_metrics2.loc['MAE', :]).reshape(-1, 1)
        mae_errors2 = np.array(confidence_intervals2.loc[['MAE_LOW', 'MAE_HIGH'], :]).T
        mae_errors2 = np.abs(mae_errors2 - np.repeat(mae_per_item2, repeats=2, axis=1))

        acc_per_item2 = np.array(item_metrics2.loc['Accuracy', :]).reshape(-1, 1)
        acc_errors2 = np.array(confidence_intervals2.loc[['Accuracy_LOW', 'Accuracy_HIGH'], :]).T
        acc_errors2 = np.abs(acc_errors2 - np.repeat(acc_per_item2, repeats=2, axis=1))
    else:
        acc_errors2 = mae_errors2 = mae_per_item2 = acc_per_item2 = None

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=_FIG_SIZE)

    barwidth = 0.5 if results_dir2 is None else 0.25
    shift = 0 if results_dir2 is None else barwidth / 2.0
    bars_pos = np.arange(len(mae_per_item))

    # bars for mae
    axes[0].bar(bars_pos - shift, mae_per_item.reshape(-1), width=barwidth, color=colors[color_idx], edgecolor='black',
                yerr=mae_errors.T, capsize=7, label=label1)

    if mae_per_item2 is not None:
        axes[0].bar(bars_pos + shift, mae_per_item2.reshape(-1), width=barwidth, color=colors[color_idx+1], edgecolor='black',
                    yerr=mae_errors2.T, capsize=7, label=label2)

    axes[0].set_xticks(bars_pos)
    axes[0].set_xticklabels([f'Item {i + 1}' for i in range(N_ITEMS)])
    axes[0].set_title('Mean Absolute Error per Item')
    axes[0].grid(True)

    # bars for accuracy
    axes[1].bar(bars_pos - shift, acc_per_item.reshape(-1), width=barwidth, color=colors[color_idx], edgecolor='black',
                yerr=acc_errors.T, capsize=7, label=label1)

    if acc_per_item2 is not None:
        axes[1].bar(bars_pos + shift, acc_per_item2.reshape(-1), width=barwidth, color=colors[color_idx+1], edgecolor='black',
                    yerr=acc_errors2.T, capsize=7, label=label2)

    axes[1].set_xticks(bars_pos)
    axes[1].set_xticklabels([f'Item {i + 1}' for i in range(N_ITEMS)])
    axes[1].set_title('Classification Accuracy per Item')
    axes[1].set_ylim(0.75, 1.0)
    axes[1].grid(True)

    if label1 is not None or label2 is not None:
        axes[0].legend(fancybox=False, fontsize=12)

    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')
