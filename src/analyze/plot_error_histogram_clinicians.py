import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.analyze.utils import init_mpl
from src.utils import map_to_score_grid
from src.preprocessing2.process_ratings import merge_rating_files
from constants import DATA_DIR, SCORE_COLUMNS

ID_LENGTH = 24

colors = init_mpl()


def make_plot(num_scores=4, save_as=None):
    # load ratings of clinicians
    clinicians_ratings = pd.read_csv(os.path.join(DATA_DIR, 'raters_clinicians_merged.csv'))
    clinicians_ratings = clinicians_ratings.loc[:, ['ID', 'Name', 'Score', 'FigureID', 'drawing_id']]
    clinicians_ratings = clinicians_ratings.dropna(axis=0, subset=['Score'])
    clinicians_ratings = clinicians_ratings[clinicians_ratings.Score <= 36.0]
    clinicians_ratings = clinicians_ratings.rename(
        columns={'ID': 'clinician_id', 'Name': 'FILE', 'Score': 'clinician_total_score'}
    )

    # load prolific ratings
    mt_ratings = merge_rating_files(DATA_DIR)
    mt_ratings = mt_ratings[mt_ratings.prolific_pid.str.len() == ID_LENGTH]

    # compute ground truth from raters
    ground_truth = compute_ground_truth(mt_ratings, num_scores=num_scores)

    # compute total scores by clinicians
    clinicians_ratings = pd.merge(clinicians_ratings, ground_truth[['FILE', 'total_score']], on='FILE')
    clinicians_ratings['error'] = (clinicians_ratings['clinician_total_score'] - clinicians_ratings['total_score'])

    # setup figure
    plt.figure()
    ax = plt.gca()

    # histogram
    sns.histplot(x=clinicians_ratings['error'], bins=np.arange(-16, 16), ax=ax)

    # format axes
    ax.set_ylabel("# Samples")
    ax.set_xlabel(r"Total Score Error ($\hat{y} - y$)")
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=4))
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=4))
    sns.despine(offset=10, trim=True, ax=ax)
    sns.despine(offset=10, trim=True, ax=ax)
    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')


def compute_ground_truth(ratings: pd.DataFrame, num_scores: int = 4) -> pd.DataFrame:
    # compute median score per item
    ratings_aggregated = ratings.groupby(['FILE', 'part_id']).agg(item_score=('part_points', 'median'))
    ratings_aggregated = ratings_aggregated.reset_index()

    # map item score to {0, 0.5, 1, 2} score grid
    ratings_aggregated.item_score = ratings_aggregated.loc[:, ['item_score']].applymap(
        lambda s: map_to_score_grid(s, num_scores=num_scores))

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
