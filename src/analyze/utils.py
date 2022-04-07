import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

from constants import CLASS_COLUMNS, ABSOLUTE_ERROR, SQUARED_ERROR, NUM_MISCLASS

__all__ = [
    'compute_errors',
    'init_mpl'
]


def compute_errors(predictions, ground_truths):
    results_df = pd.DataFrame(columns=['figure_id', ABSOLUTE_ERROR, SQUARED_ERROR, NUM_MISCLASS])
    results_df['figure_id'] = predictions['figure_id']

    results_df[ABSOLUTE_ERROR] = np.abs(predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    results_df[SQUARED_ERROR] = (predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score']) ** 2
    results_df[NUM_MISCLASS] = np.sum(predictions.loc[:, CLASS_COLUMNS] != ground_truths.loc[:, CLASS_COLUMNS], axis=1)

    results_df = results_df.set_index('figure_id')

    return results_df


def init_mpl(sns_style="ticks", colorpalette='muted', fontsize=16, grid_lw=1.0):
    sns.set_style(sns_style)
    sns.set_palette(colorpalette)
    colors = sns.color_palette(colorpalette)
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["grid.linewidth"] = grid_lw / 2.0
    mpl.rcParams["axes.linewidth"] = grid_lw
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.
    return colors
