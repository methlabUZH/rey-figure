import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import List, Tuple

from analyze.plot_utils import init_mpl

init_mpl(sns_style='ticks', colorpalette='muted')


def get_errors(results_dir, rotation_angles=None):
    errors = []
    for rot in rotation_angles:
        prefix = f"rot={rot}"
        predictions = pd.read_csv(os.path.join(results_dir, prefix + '-test_predictions.csv'))
        ground_truths = pd.read_csv(os.path.join(results_dir, prefix + '-test_ground_truths.csv'))

        # compute MAE
        mae = np.mean(np.abs(predictions['total_score'] - ground_truths['total_score']))
        errors.append(mae)

    return rotation_angles, errors


def make_plot(x_values, y_values, save_as=None):
    plt.plot(x_values, y_values, marker='o')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Rotation Angle')
    plt.grid(True)
    sns.despine()
    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close()
        return

    plt.show()
    plt.close()


if __name__ == '__main__':
    res_dir = '../results/euler-results/data-2018-2021-116x150-pp0/final/rey-multilabel-classifier'

    rot_angles = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    rot_angles, errors = get_errors(results_dir=res_dir, rotation_angles=rot_angles)
    make_plot(rot_angles, errors, './figures/rotation-robustness.pdf')
