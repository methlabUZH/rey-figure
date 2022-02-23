import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as plticker

from constants import *
from src.utils import init_mpl, map_to_score_grid

colors = init_mpl(sns_style='ticks', colorpalette='muted')


def main(train_labels_csv, simulated_labels_csv):
    train_labels = pd.read_csv(train_labels_csv)
    simulated_labels = pd.read_csv(simulated_labels_csv)

    score_cols = [f'score_item_{i}' for i in range(1, N_ITEMS + 1)]
    train_labels.loc[:, score_cols] = train_labels.loc[:, score_cols].applymap(map_to_score_grid)

    # indicate whether or not figure is simulated
    train_labels['simulated'] = [0] * len(train_labels)
    simulated_labels['simulated'] = [1] * len(simulated_labels)

    plot_total_score(train_labels, simulated_labels)
    # plot_item_score(train_labels)
    # plot_item_score(simulated_labels)


def plot_total_score(train_labels, simulated_labels):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 7))

    sns.histplot(train_labels, x='summed_score', ax=axes[0], bins=np.arange(0, 37, 1), color=colors[0])
    sns.histplot(simulated_labels, x='summed_score', ax=axes[1], bins=np.arange(0, 37, 1), color=colors[1])

    loc = plticker.MultipleLocator(base=1000)
    axes[0].yaxis.set_major_locator(loc)
    axes[0].set_xlabel('')
    axes[0].set_xlim((0, 36))
    axes[0].set_title('Real Training Data')

    axes[1].yaxis.set_major_locator(loc)
    axes[1].set_xlabel('Total Score')
    axes[1].set_xlim((0, 36))
    axes[1].set_title('Simulated Figures')
    sns.despine(offset=5, trim=False)

    fig.tight_layout()
    plt.show()


def plot_item_score(labels):
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(15, 7), sharey=True, sharex=True)
    axes = list(itertools.chain(*axes))
    for i, ax in enumerate(axes):
        scores, counts = np.unique(labels[f'score_item_{i + 1}'], return_counts=True)
        sns.barplot(x=scores, y=counts, ax=ax)
    plt.show()


if __name__ == '__main__':
    main(train_labels_csv='../results/euler-results/data-2018-2021-116x150-pp0/train_labels.csv',
         simulated_labels_csv='../results/euler-results/data-2018-2021-116x150-pp0/simulated_labels.csv')
