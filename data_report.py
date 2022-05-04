import numpy as np
import os
import pandas as pd
from tabulate import tabulate

from constants import DATA_DIR, SCORE_COLUMNS
from src.preprocessing2.process_ratings import merge_rating_files
from src.utils import map_to_score_grid

ID_LENGTH = 24


def main():
    ratings_clinicians = pd.read_csv(os.path.join(DATA_DIR, 'raters_clinicians_merged.csv'))
    ratings_clinicians = ratings_clinicians.loc[:, ['ID', 'Name', 'Score', 'FigureID', 'drawing_id']]
    ratings_clinicians = ratings_clinicians.dropna(axis=0, subset=['Score'])
    ratings_clinicians = ratings_clinicians[ratings_clinicians.Score <= 36.0]
    ratings_clinicians = ratings_clinicians.rename(
        columns={'ID': 'clinician_id', 'Name': 'FILE', 'Score': 'clinician_total_score'}
    )

    ratings_crowd_source = merge_rating_files(DATA_DIR)

    # get rid of weird prolific_pids
    ratings_crowd_source = ratings_crowd_source[ratings_crowd_source.prolific_pid.str.len() == ID_LENGTH]

    # compute quality of aggregated ratings per figure
    rating_stdevs = _compute_ratings_quality(ratings_crowd_source)
    ground_truth = _compute_ground_truth(ratings_crowd_source)

    # compute number of ratrings per figure
    ratings_counts = _get_rating_counts_per_figure(ratings_crowd_source)
    ratings_counts = ratings_counts[ratings_counts.num_ratings < 10]
    ratings_counts.to_csv('./data/ratings_counts.csv')

    # compute average performance of raters
    raters_errors = _compute_rater_errors(ratings_crowd_source, ground_truth)
    raters_subset = raters_errors[raters_errors.FILE.isin(ratings_clinicians.FILE)]
    ratings_counts = raters_errors.groupby('FILE').agg(counts=('prolific_pid', 'count'))
    raters_errors = raters_errors.sort_values(by=['FILE'])
    raters_errors['squared_error'] = raters_errors['error'] ** 2
    raters_errors['absolute_error'] = raters_errors['error'].abs()
    raters_mses = raters_errors.groupby('prolific_pid').agg(mse=('squared_error', 'mean'))
    raters_maes = raters_errors.groupby('prolific_pid').agg(mae=('absolute_error', 'mean'))

    # compute average performance of clinicians
    clinicians_errors = _compute_clinicians_errors(ratings_clinicians, ground_truth)
    clinicians_errors = clinicians_errors.sort_values(by=['FILE'])
    clinicians_errors['squared_error'] = clinicians_errors['error'] ** 2
    clinicians_errors['absolute_error'] = clinicians_errors['error'].abs()
    clinicians_mses = clinicians_errors.groupby('clinician_id').agg(mse=('squared_error', 'mean'))
    clinicians_maes = clinicians_errors.groupby('clinician_id').agg(mae=('absolute_error', 'mean'))

    # print statistics for clinicians
    print('\n======== Clinicians')
    print(f'# figures rated by clinicians:\t{len(ratings_clinicians.FILE.unique())}')
    print(f'Average Clinician Score: {ratings_clinicians.clinician_total_score.mean()}')
    print(f'Avg. Clinicians MSE:\t{clinicians_mses.mse.mean()}')
    print(f'Avg. Clinicians MAE:\t{clinicians_maes.mae.mean()}')

    # print statistics for human raters
    print('\n======== Human Raters')
    print(f'# figures rated by human annotators:\t{len(raters_errors.FILE.unique())}')
    print(f'Average Raters Score: {raters_subset.rater_total_score.mean()}')
    print(f'Average Standard Deviation of Ratings:\t{rating_stdevs.stdev.mean()}')
    print(f'min num ratings per figure:\t{ratings_counts.counts.min()}')
    print(f'mean num ratings per figure:\t{ratings_counts.counts.mean()}')
    print(f'max num ratings per figure:\t{ratings_counts.counts.max()}')
    print(f'Avg. Human Raters MSE: {raters_mses.mse.mean()}')
    print(f'Avg. Human Raters MAE:\t{raters_maes.mae.mean()}')


def _compute_rater_errors(ratings, ground_truths):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(rater_total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = pd.merge(ratings_aggregated, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['error'] = (
            ratings_aggregated['rater_total_score'] - ratings_aggregated['total_score']
    )
    return ratings_aggregated


def _compute_clinicians_errors(ratings, ground_truths):
    ratings_aggregated = pd.merge(ratings, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['error'] = (
            ratings_aggregated['clinician_total_score'] - ratings_aggregated['total_score']
    )
    return ratings_aggregated


def _compute_ground_truth(ratings: pd.DataFrame) -> pd.DataFrame:
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
    ratings_aggregated['total_score'] = ratings_aggregated.loc[:, SCORE_COLUMNS].sum(axis=1)

    return ratings_aggregated


def _compute_ratings_quality(ratings: pd.DataFrame):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(stdev=('total_score', lambda rows: rows.std()))
    return ratings_aggregated.reset_index()


def _get_rating_counts_per_figure(ratings: pd.DataFrame):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()

    ratings_counts = ratings_aggregated.groupby(['FILE']).agg(num_ratings=('prolific_pid', 'count'))
    ratings_counts = ratings_counts.reset_index()
    ratings_counts = ratings_counts.sort_values('num_ratings', ascending=True)
    return ratings_counts


if __name__ == '__main__':
    main()
