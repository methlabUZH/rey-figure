import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import map_to_score_grid

from constants import N_ITEMS

_DATA_DIR = '/Users/maurice/phd/src/rey-figure/data/'
_RATING_DATA_DIR = os.path.join(_DATA_DIR, 'UserRatingData')
_COLUMNS = ['assessment_id', 'prolific_pid', 'drawing_id', 'FILE', 'part_id', 'part_points']
_MERGED_RATINGS_FILES = os.path.join(_DATA_DIR, 'merged_ratings.feather')
_MERGED_RATINGS_FILES_CSV = os.path.join(_DATA_DIR, 'merged_ratings.csv')
_MODEL_PREDICTIONS_CSV = '/Users/maurice/phd/src/rey-figure/code-main/results/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier/test_predictions.csv'

_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]

unique_columns = []


def main():
    if not os.path.isfile(_MERGED_RATINGS_FILES):
        merge_rating_files()

    ratings = pd.read_feather(_MERGED_RATINGS_FILES)
    ratings = ratings.sort_values(by=['FILE', 'prolific_pid', 'part_id'])

    model_predictions = pd.read_csv(_MODEL_PREDICTIONS_CSV)

    rating_qualities = compute_ratings_quality(ratings, mode='variance')
    ground_truth = compute_ground_truth(ratings)

    # compute average performance of raters
    rater_errors = compute_rater_errors(ratings, ground_truth)
    # rater_errors = pd.merge(rater_errors, rating_qualities, on='FILE')

    # compute performance of model
    model_errors = compute_model_errors(model_predictions, ground_truth)
    errors = pd.merge(model_errors, rating_qualities, on='FILE')
    errors = pd.merge(errors, rater_errors, on='FILE')

    sns.scatterplot(x=errors['quality'], y=errors['model_absolute_error'], label='CNN', alpha=0.25, marker='X', s=20)
    sns.scatterplot(x=errors['quality'], y=errors['rater_absolute_error'], label='Raters', alpha=0.25, marker='X', s=20)
    plt.show()
    plt.close()

    sns.histplot(data=rating_qualities, x='quality')
    plt.show()
    plt.close()

    # compute total score absolute error for each rater per figure


def compute_model_errors(predictions, ground_truths):
    predictions['FILE'] = predictions.loc[:, ['image_file']].applymap(lambda s: os.path.split(s)[-1])
    predictions = predictions.rename(columns={'total_score': 'model_total_score'})
    predictions = pd.merge(predictions, ground_truths, on='FILE')
    predictions = predictions[['FILE', 'model_total_score', 'total_score']]
    predictions['model_absolute_error'] = (predictions['model_total_score'] - predictions['total_score']).abs()
    return predictions


def compute_rater_errors(ratings, ground_truths):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(rater_total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = pd.merge(ratings_aggregated, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['rater_absolute_error'] = (
            ratings_aggregated['rater_total_score'] - ratings_aggregated['total_score']
    ).abs()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(rater_absolute_error=('rater_absolute_error', 'mean'))
    ratings_aggregated = ratings_aggregated.reset_index()
    return ratings_aggregated


def compute_ratings_quality(ratings: pd.DataFrame, mode='agree_fraction'):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    if mode == 'agree_fraction':
        ratings_aggregated = ratings_aggregated.groupby('FILE').agg(
            quality=('total_score', lambda rows: len(rows.unique()) / len(rows)))
    elif mode == 'variance':
        ratings_aggregated = ratings_aggregated.groupby('FILE').agg(
            quality=('total_score', lambda rows: rows.std()))

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
    ratings_aggregated['total_score'] = ratings_aggregated.loc[:, _SCORE_COLS].sum(axis=1)

    return ratings_aggregated


def merge_rating_files():
    dataframes = []
    for f in (pbar := tqdm(os.listdir(_RATING_DATA_DIR))):
        pbar.set_description(f'processing {f}')

        if str(f).startswith('.'):
            continue

        df = pd.read_csv(os.path.join(_RATING_DATA_DIR, f))[_COLUMNS]
        dataframes.append(df)

    # merge dataframes
    ratings = pd.concat(dataframes, ignore_index=True, sort=False)

    # drop duplicates
    ratings = ratings.drop_duplicates()

    num_ratings = len(ratings)
    ratings = filter_ratings(ratings)
    print(f'# invalid ratings removed: {num_ratings - len(ratings)}')

    ratings = ratings.reset_index()

    ratings.to_feather(_MERGED_RATINGS_FILES)
    print(f'saved dataframe as {_MERGED_RATINGS_FILES}')


def filter_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    # keep only figures with valid ratings (i.e. no nan values)
    ratings = ratings.dropna(axis=0, how='any')

    # aggregate df by number of figures with unique rated item_ids
    grouped_df = ratings.groupby(['FILE', 'prolific_pid']).agg(num_ratings=('part_id', lambda rows: len(rows.unique())))
    grouped_df = grouped_df.reset_index()

    # merge
    ratings = pd.merge(ratings, grouped_df)
    ratings = ratings[ratings.num_ratings == N_ITEMS]

    return ratings


if __name__ == '__main__':
    main()
