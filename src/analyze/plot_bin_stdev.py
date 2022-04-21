import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from src.analyze.utils import init_mpl
from src.utils import map_to_score_grid
from src.preprocessing2.process_ratings import merge_rating_files
from constants import DATA_DIR, SCORE_COLUMNS, CI_CONFIDENCE

FIG_SIZE = (8, 4)
ID_LENGTH = 24

colors = init_mpl()


def make_plot(num_scores=4, bin_granularity=4, save_as=None):
    mt_ratings = merge_rating_files(DATA_DIR)

    # get rid of weird prolific_pids
    mt_ratings = mt_ratings[mt_ratings.prolific_pid.str.len() == ID_LENGTH]

    # compute stdev of aggregated ratings per figure
    rating_qualities = compute_ratings_quality(mt_ratings)
    ground_truth = compute_ground_truth(mt_ratings, num_scores=num_scores)
    bins = np.arange(0, 36, step=bin_granularity)
    ground_truth['bin'] = np.digitize(ground_truth['total_score'], bins=bins, right=False)
    merged_data = pd.merge(ground_truth, rating_qualities, on='FILE')

    # aggregate to bins and compute confidence intervals
    def _compute_ci(data: pd.Series):
        ci = stats.bootstrap(data=(data,), statistic=np.mean, confidence_level=CI_CONFIDENCE, method='BCa', axis=0)
        ci = ci.confidence_interval
        return [ci.low, ci.high]

    # aggregate
    merged_data = merged_data.groupby('bin').agg(MEAN_STD=('std', 'mean'), STD_CI=('std', lambda arr: _compute_ci(arr)))
    merged_data['scores'] = [
                                r"${}-{}$".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)
                            ] + [
                                fr'${bins[-1]}-36$'
                            ]

    print(tabulate(merged_data, headers=merged_data.columns))

    y_values = merged_data['MEAN_STD']
    y_cis = merged_data['STD_CI']

    plt.figure(figsize=FIG_SIZE)
    x_labels = merged_data['scores']
    y_values = np.array(y_values).reshape(-1, 1)
    y_errs = np.array([np.array(ci) for ci in y_cis])
    y_errs = np.abs(y_errs - np.repeat(y_values, repeats=2, axis=1))

    plt.errorbar(range(len(x_labels)), y_values, yerr=y_errs.T, elinewidth=1.0, capsize=5, capthick=2,
                 ls='-.', marker='d', color=colors[0])

    plt.ylabel('Normalized Ratings StDev')
    plt.xlabel('Total Scores')
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(True)
    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()
    print(f'saved figure as {save_as}')


def compute_ratings_quality(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(
        std=('total_score', lambda rows: rows.std() / rows.mean() if rows.mean() > 0 else 0)
    )
    return ratings_aggregated.reset_index()


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
