import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from analyze.plot_utils import init_mpl
from src.utils import map_to_score_grid
from constants import N_ITEMS, DATA_DIR
from src.preprocessing2.process_ratings import merge_rating_files

COLORS = init_mpl(sns_style="ticks", fontsize=14)

RATINGS_COLUMNS = ['assessment_id', 'prolific_pid', 'drawing_id', 'FILE', 'part_id', 'part_points']
SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]
ID_LENGTH = 24

MODE_STD = 'std'
MODE_DISAGREE_FRACTION = 'disagree_fraction'


def main(model_predictions_csv, figure_quality_mode='disagree_fraction', save_as=None):
    clinicians_ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings_clinicians.csv'))
    model_predictions = pd.read_csv(model_predictions_csv)
    mt_ratings = merge_rating_files(DATA_DIR)

    # get rid of weird prolific_pids
    clinicians_ratings = clinicians_ratings[clinicians_ratings.prolific_pid.str.len() == ID_LENGTH]
    mt_ratings = mt_ratings[mt_ratings.prolific_pid.str.len() == ID_LENGTH]

    # compute quality of aggregated ratings per figure
    rating_qualities = compute_ratings_quality(mt_ratings, mode=figure_quality_mode)
    ground_truth = compute_ground_truth(mt_ratings)

    # remove professional raters from all ratings
    mt_ratings = mt_ratings[~mt_ratings.prolific_pid.isin(clinicians_ratings.prolific_pid)]

    # compute performance of model
    model_errors = compute_model_errors(model_predictions, ground_truth)
    model_errors = pd.merge(model_errors, rating_qualities, on='FILE')

    # compute average performance of raters
    raters_errors = compute_rater_errors(mt_ratings, ground_truth)
    raters_errors = pd.merge(raters_errors, rating_qualities, on='FILE')
    raters_errors = raters_errors.sort_values(by=['FILE'])

    # compute average performance of clinicians
    clinicians_errors = compute_rater_errors(clinicians_ratings, ground_truth)
    clinicians_errors = pd.merge(clinicians_errors, rating_qualities, on='FILE')
    clinicians_errors = clinicians_errors.sort_values(by=['FILE'])

    # subsample figure from model test set
    raters_errors = raters_errors[raters_errors['FILE'].isin(model_errors['FILE'])]
    clinicians_errors = clinicians_errors[clinicians_errors['FILE'].isin(model_errors['FILE'])]

    # assign bins and compute mean
    model_errors['binned_quality'], bins = pd.cut(model_errors['quality'], bins=10, retbins=True)
    model_lines = model_errors.groupby('binned_quality').agg(mean_absolute_error=('absolute_error', 'mean'))
    model_lines = model_lines.reset_index()

    raters_errors['binned_quality'] = pd.cut(raters_errors['quality'], bins=bins)
    raters_lines = raters_errors.groupby('binned_quality').agg(mean_absolute_error=('absolute_error', 'mean'))
    raters_lines = raters_lines.reset_index()

    clinicians_errors['binned_quality'] = pd.cut(clinicians_errors['quality'], bins=bins)
    clinicians_lines = clinicians_errors.groupby('binned_quality').agg(mean_absolute_error=('absolute_error', 'mean'))
    clinicians_lines = clinicians_lines.reset_index()

    # xtick labels = mean of bin boundaries
    xticks = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]

    # make plot
    fig = plt.figure()
    ax = plt.gca()

    # scatterplot with errors
    sns.lineplot(x=xticks, y=model_lines['mean_absolute_error'], marker='p', label='CNN', ax=ax,
                 markeredgecolor=COLORS[0])
    sns.lineplot(x=xticks, y=clinicians_lines['mean_absolute_error'], marker='o', label='Clinicians', ax=ax,
                 markeredgecolor=COLORS[1])
    sns.lineplot(x=xticks, y=raters_lines['mean_absolute_error'], marker='d', label='Crowd Source', ax=ax,
                 markeredgecolor=COLORS[2])
    ax.set_ylabel('Total Score MAE')
    ax.set_xlabel('Ratings Standard Deviation' if figure_quality_mode == MODE_STD else 'Distinct Ratings Fraction')

    # histogram of human rater errors
    ax2 = plt.twinx(ax)
    sns.histplot(x=model_errors['quality'], color='gray', alpha=0.4, bins=bins, ax=ax2)
    ax2.set_ylabel('# Figures')

    ax.set_xticks([max(0, b) for b in bins])
    ax.set_xticklabels([f"{max(0, b):.1f}" for b in bins])

    ax.set_zorder(1)
    ax.patch.set_visible(False)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fancybox=False, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.02))

    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close()
        return

    plt.show()
    plt.close()


def compute_model_errors(predictions, ground_truths):
    predictions['FILE'] = predictions.loc[:, ['image_file']].applymap(lambda s: os.path.split(s)[-1])
    predictions = predictions.rename(columns={'total_score': 'model_total_score'})
    predictions = pd.merge(predictions, ground_truths, on='FILE')
    predictions = predictions[['FILE', 'model_total_score', 'total_score']]
    predictions['absolute_error'] = (predictions['model_total_score'] - predictions['total_score']).abs()
    return predictions


def compute_rater_errors(ratings, ground_truths):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(rater_total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = pd.merge(ratings_aggregated, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['absolute_error'] = (
            ratings_aggregated['rater_total_score'] - ratings_aggregated['total_score']
    ).abs()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(absolute_error=('absolute_error', 'mean'))
    ratings_aggregated = ratings_aggregated.reset_index()
    return ratings_aggregated


def compute_ratings_quality(ratings: pd.DataFrame, mode='agree_fraction'):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    if mode == 'disagree_fraction':
        ratings_aggregated = ratings_aggregated.groupby('FILE').agg(
            quality=('total_score', lambda rows: len(rows.unique()) / len(rows)))
    elif mode == 'std':
        ratings_aggregated = ratings_aggregated.groupby('FILE').agg(
            quality=('total_score', lambda rows: rows.std()))
    else:
        raise ValueError

    return ratings_aggregated.reset_index()


def compute_ground_truth(ratings: pd.DataFrame) -> pd.DataFrame:
    # compute median score per item
    ratings_aggregated = ratings.groupby(['FILE', 'part_id']).agg(item_score=('part_points', 'median'))
    ratings_aggregated = ratings_aggregated.reset_index()

    # map item score to {0, 0.5, 1, 2} score grid
    ratings_aggregated.item_score = ratings_aggregated.loc[:, ['item_score']].applymap(lambda s: map_to_score_grid(s))

    # rearrange table so that each row corresponds to a single figure
    ratings_aggregated = ratings_aggregated.sort_values(by=['FILE', 'part_id'])
    ratings_aggregated = ratings_aggregated.pivot_table(values='item_score', index=['FILE'], columns='part_id')
    ratings_aggregated = ratings_aggregated.reset_index()

    # rename columns for readability
    ratings_aggregated = ratings_aggregated.rename(
        columns={c: f'score_item_{c}' for c in ratings_aggregated.columns if isinstance(c, int)})

    # compute total score
    ratings_aggregated['total_score'] = ratings_aggregated.loc[:, SCORE_COLS].sum(axis=1)

    return ratings_aggregated


def print_table(table: pd.DataFrame, n=10):
    print(tabulate(table.head(n=n), headers=table.columns))


if __name__ == '__main__':
    results_root = '/Users/maurice/phd/src/rey-figure/code-main/results/spaceml-results/'

    main(model_predictions_csv=os.path.join(
        results_root, 'data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier/test_predictions.csv'),
        figure_quality_mode=MODE_STD, save_as='./figures/human_comparison_std.pdf')

    main(model_predictions_csv=os.path.join(
        results_root, 'data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier/test_predictions.csv'),
        figure_quality_mode=MODE_DISAGREE_FRACTION, save_as='./figures/human_comparison_dis_frac.pdf')
