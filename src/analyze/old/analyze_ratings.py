import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.utils import map_to_score_grid

from constants import N_ITEMS, DATA_DIR

from src.preprocessing2.process_ratings import merge_rating_files

COLORS = sns.color_palette()

RATINGS_COLUMNS = ['assessment_id', 'prolific_pid', 'drawing_id', 'FILE', 'part_id', 'part_points']
SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]

unique_columns = []


def main(model_predictions_csv, figure_quality_mode='disagree_fraction'):
    clinicians_ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings_clinicians.csv'))
    mt_ratings = merge_rating_files(DATA_DIR)
    ground_truth = compute_ground_truth(mt_ratings)

    clinicians_ratings = aggregate_predictions(clinicians_ratings, ground_truth)
    mt_ratings = aggregate_predictions(mt_ratings, ground_truth)

    # compute errors
    clinicians_ratings['abs_error'] = (clinicians_ratings['pred_total_score'] - clinicians_ratings['total_score']).abs()
    mt_ratings['abs_error'] = (mt_ratings['pred_total_score'] - mt_ratings['total_score']).abs()

    # compute MAE for each professional rater
    clinicians_errors = clinicians_ratings.groupby(['prolific_pid']).agg(mae=('abs_error', 'mean'))
    mean_clinician_error = clinicians_errors.values.mean()
    print(mean_clinician_error)

    mt_errors = mt_ratings.groupby(['prolific_pid']).agg(mae=('abs_error', 'mean'))
    mean_mt_error = mt_errors.values.mean()
    print(mean_mt_error)
    quit()

    # remove professional raters from all ratings
    mt_ratings = mt_ratings[~mt_ratings.prolific_pid.isin(clinicians_ratings.prolific_pid)]

    # compute average performance of raters
    raters_errors = compute_rater_errors(mt_ratings, ground_truth)
    raters_errors = raters_errors.sort_values(by=['FILE'])

    # compute average performance of clinicians
    clinicians_errors = compute_rater_errors(clinicians_ratings, ground_truth)
    clinicians_errors = clinicians_errors.sort_values(by=['FILE'])


def aggregate_predictions(ratings: pd.DataFrame, ground_truth: pd.DataFrame):
    ratings = ratings.groupby(['FILE', 'prolific_pid']).agg(pred_total_score=('part_points', 'sum'))
    ratings = ratings.reset_index()
    ratings = pd.merge(ratings, ground_truth, on='FILE')
    return ratings


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
        figure_quality_mode='std')
