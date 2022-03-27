import pandas as pd
import os
from tqdm import tqdm
from typing import Tuple

from constants import N_ITEMS, USER_RATING_DATA_DIR, MAIN_LABEL_FILENAME
from src.utils import map_to_score_grid

__all__ = ['create_label_files']

_COLUMNS = ['assessment_id', 'prolific_pid', 'drawing_id', 'FILE', 'part_id', 'part_points']
_SCORE_COLS = [f'score_item_{i + 1}' for i in range(N_ITEMS)]


def create_label_files(data_dir: str, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_labels_fp = os.path.join(data_dir, MAIN_LABEL_FILENAME.format(split='test'))
    train_labels_fp = os.path.join(data_dir, MAIN_LABEL_FILENAME.format(split='train'))

    if os.path.isfile(train_labels_fp) and os.path.isfile(test_labels_fp):
        test_labels = pd.read_csv(test_labels_fp)
        train_labels = pd.read_csv(train_labels_fp)
        return test_labels, train_labels

    labels = _merge_rating_files(data_dir)

    # compute median score per item
    labels = labels.groupby(['FILE', 'part_id']).agg(item_score=('part_points', 'median'))
    labels = labels.reset_index()

    # map item score to {0, 0.5, 1, 2} score grid
    labels.item_score = labels.loc[:, ['item_score']].applymap(lambda s: map_to_score_grid(s))

    # rearrange table so that each row corresponds to a single figure
    labels = labels.sort_values(by=['FILE', 'part_id'])
    labels = labels.pivot_table(values='item_score', index=['FILE'], columns='part_id')
    labels = labels.reset_index()

    # rename columns for readability
    labels = labels.rename(columns={c: f'score_item_{c}' for c in labels.columns if isinstance(c, int)})

    # compute total score
    labels['total_score'] = labels.loc[:, _SCORE_COLS].sum(axis=1)

    # split into test and training data
    test_labels = labels.sample(frac=test_fraction, replace=False, axis=0)
    train_labels = labels[~labels.index.isin(test_labels.index)]

    assert set(test_labels.index).isdisjoint(train_labels.index)

    test_labels = test_labels.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)

    # save dataframes

    test_labels.to_csv(train_labels_fp)
    train_labels.to_csv(test_labels_fp)

    print(f'--> save test and train labels to {data_dir}')

    return test_labels, train_labels


def _merge_rating_files(data_dir) -> pd.DataFrame:
    rating_data_dir = os.path.join(data_dir, USER_RATING_DATA_DIR)
    dataframes = []
    for f in (pbar := tqdm(os.listdir(rating_data_dir))):
        pbar.set_description(f'processing {f}')

        if str(f).startswith('.'):
            continue

        df = pd.read_csv(os.path.join(rating_data_dir, f))[_COLUMNS]
        dataframes.append(df)

    # merge dataframes
    ratings = pd.concat(dataframes, ignore_index=True, sort=False)

    # drop duplicates
    ratings = ratings.drop_duplicates()

    num_ratings = len(ratings)
    ratings = _filter_ratings(ratings)
    print(f'--> # invalid ratings removed: {num_ratings - len(ratings)}')

    ratings = ratings.reset_index()

    save_as = os.path.join(rating_data_dir, 'merged_user_ratings.feather')
    ratings.to_feather(save_as)

    print('--> finished merging user rating data files')
    print(f'--> saved data as {save_as}')

    return ratings


def _filter_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    # keep only figures with valid ratings (i.e. no nan values)
    ratings = ratings.dropna(axis=0, how='any')

    # aggregate df by number of figures with unique rated item_ids
    grouped_df = ratings.groupby(['FILE', 'prolific_pid']).agg(num_ratings=('part_id', lambda rows: len(rows.unique())))
    grouped_df = grouped_df.reset_index()

    # merge
    ratings = pd.merge(ratings, grouped_df)
    ratings = ratings[ratings.num_ratings == N_ITEMS]

    return ratings
