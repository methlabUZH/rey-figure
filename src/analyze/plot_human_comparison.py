import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from tabulate import tabulate

from constants import DATA_DIR, SCORE_COLUMNS, CI_CONFIDENCE
from src.analyze.utils import init_mpl
from src.preprocessing2.process_ratings import merge_rating_files
from src.utils import map_to_score_grid

__all__ = ['make_plot']

ID_LENGTH = 24
NUM_BINS = 10
BINS = [-0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0]
FIG_SIZE = (7, 4)

colors = init_mpl()


def make_plot(results_dir, include_clinicians=False, save_as=None):
    # compute lines
    xticks, model_lines, raters_lines, clinicians_lines, bins = get_lines(results_dir)

    # setup plotting
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()

    # raters
    y_values = np.array([v if v is not None else (np.nan, np.nan, np.nan) for v in raters_lines['mean_ci']])
    y_errs = np.abs(y_values[:, np.array([0, 2])] - np.repeat(y_values[:, 1].reshape(-1, 1), repeats=2, axis=1))
    plt.errorbar(xticks, y_values[:, 1], yerr=y_errs.T, label='Human Avg. Performance', elinewidth=1.0, capsize=5,
                 capthick=2, ls='--', marker='o')

    # cnn
    y_values = np.array([v if v is not None else (np.nan, np.nan, np.nan) for v in model_lines['mean_ci']])
    y_errs = np.abs(y_values[:, np.array([0, 2])] - np.repeat(y_values[:, 1].reshape(-1, 1), repeats=2, axis=1))
    plt.errorbar(xticks, y_values[:, 1], yerr=y_errs.T, label='CNN Performance', elinewidth=1.0, capsize=5, capthick=2,
                 ls='-.', marker='d')

    # clinicians
    if include_clinicians:
        y_values = np.array([v if v is not None else (np.nan, np.nan, np.nan) for v in clinicians_lines['mean_ci']])
        y_errs = np.abs(y_values[:, np.array([0, 2])] - np.repeat(y_values[:, 1].reshape(-1, 1), repeats=2, axis=1))
        plt.errorbar(xticks, y_values[:, 1], yerr=y_errs.T, label='Avg. Clinician Performance', elinewidth=1.0,
                     capsize=5,
                     capthick=2, ls='-', marker='x')
        fn = os.path.splitext(os.path.split(save_as)[-1])[0]
        save_as = save_as.replace(fn, fn + '-with_clinicians')

    ax.set_ylabel('Total Score MAE')
    ax.set_xlabel('Ratings Standard Deviation')
    ax.set_xticks([max(0, b) for b in bins])
    ax.set_xticklabels([f"{max(0, b):.1f}" for b in bins])
    plt.legend(fancybox=False, fontsize=12)
    plt.grid(True)
    fig.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
        print(f'saved figure as {save_as}')
        plt.close()
        return

    plt.show()
    plt.close()


def get_lines(results_dir):
    ratings_clinicians = pd.read_csv(os.path.join(DATA_DIR, 'raters_clinicians_merged.csv'))
    ratings_clinicians = ratings_clinicians.loc[:, ['ID', 'Name', 'Score', 'FigureID', 'drawing_id']]
    ratings_clinicians = ratings_clinicians.dropna(axis=0, subset=['Score'])
    ratings_clinicians = ratings_clinicians[ratings_clinicians.Score <= 36.0]
    ratings_clinicians = ratings_clinicians.rename(
        columns={'ID': 'clinician_id', 'Name': 'FILE', 'Score': 'clinician_total_score'}
    )

    ratings_crowd_source = merge_rating_files(DATA_DIR)
    ratings_model = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))

    # get rid of weird prolific_pids
    ratings_crowd_source = ratings_crowd_source[ratings_crowd_source.prolific_pid.str.len() == ID_LENGTH]

    # compute quality of aggregated ratings per figure
    rating_qualities = _compute_ratings_quality(ratings_crowd_source)
    ground_truth = _compute_ground_truth(ratings_crowd_source)

    # compute performance of model
    model_errors = _compute_model_errors(ratings_model, ground_truth)
    model_errors = pd.merge(model_errors, rating_qualities, on='FILE')

    # compute average performance of raters
    raters_errors = _compute_rater_errors(ratings_crowd_source, ground_truth)
    raters_errors = pd.merge(raters_errors, rating_qualities, on='FILE')
    raters_errors = raters_errors.sort_values(by=['FILE'])

    # compute average performance of clinicians
    clinicians_errors = _compute_clinicians_errors(ratings_clinicians, ground_truth)
    clinicians_errors = pd.merge(clinicians_errors, rating_qualities, on='FILE')
    clinicians_errors = clinicians_errors.sort_values(by=['FILE'])

    # # only use figures from model test set
    # raters_errors = raters_errors[raters_errors['FILE'].isin(model_errors['FILE'])]
    # clinicians_errors = clinicians_errors[clinicians_errors['FILE'].isin(model_errors['FILE'])]

    # assign bins and compute mean
    model_errors['binned_quality'], bins = pd.cut(model_errors['quality'], bins=BINS, retbins=True)
    model_lines = model_errors.groupby('binned_quality').agg(mean_ci=(
        'absolute_error', lambda x: _compute_mean_with_confidence_interval(x, np.mean)))
    model_lines = model_lines.reset_index()

    raters_errors['binned_quality'] = pd.cut(raters_errors['quality'], bins=bins)
    raters_lines = raters_errors.groupby('binned_quality').agg(mean_ci=(
        'absolute_error', lambda x: _compute_mean_with_confidence_interval(x, np.mean)))
    raters_lines = raters_lines.reset_index()

    clinicians_errors['binned_quality'] = pd.cut(clinicians_errors['quality'], bins=bins)
    clinicians_lines = clinicians_errors.groupby('binned_quality').agg(mean_ci=(
        'absolute_error', lambda x: _compute_mean_with_confidence_interval(x, np.mean)))
    clinicians_lines = clinicians_lines.reset_index()

    # xtick labels = mean of bin boundaries
    xticks = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]

    return xticks, model_lines, raters_lines, clinicians_lines, bins


def _compute_ratings_quality(ratings: pd.DataFrame):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(quality=('total_score', lambda rows: rows.std()))
    return ratings_aggregated.reset_index()


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


def _compute_model_errors(predictions, ground_truths):
    predictions['FILE'] = predictions.loc[:, ['image_file']].applymap(lambda s: os.path.split(s)[-1])
    predictions = predictions.rename(columns={'total_score': 'model_total_score'})
    predictions = pd.merge(predictions, ground_truths, on='FILE')
    predictions = predictions[['FILE', 'model_total_score', 'total_score']]
    predictions['absolute_error'] = (predictions['model_total_score'] - predictions['total_score']).abs()
    return predictions


def _compute_rater_errors(ratings, ground_truths):
    ratings_aggregated = ratings.groupby(['FILE', 'prolific_pid']).agg(rater_total_score=('part_points', 'sum'))
    ratings_aggregated = ratings_aggregated.reset_index()
    ratings_aggregated = pd.merge(ratings_aggregated, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['absolute_error'] = (
            ratings_aggregated['rater_total_score'] - ratings_aggregated['total_score']
    ).abs()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(absolute_error=('absolute_error', 'mean'))
    ratings_aggregated = ratings_aggregated.reset_index()
    return ratings_aggregated


def _compute_clinicians_errors(ratings, ground_truths):
    ratings_aggregated = pd.merge(ratings, ground_truths[['FILE', 'total_score']], on='FILE')
    ratings_aggregated['absolute_error'] = (
            ratings_aggregated['clinician_total_score'] - ratings_aggregated['total_score']
    ).abs()
    ratings_aggregated = ratings_aggregated.groupby('FILE').agg(absolute_error=('absolute_error', 'mean'))
    ratings_aggregated = ratings_aggregated.reset_index()
    return ratings_aggregated


def _compute_mean_with_confidence_interval(data: pd.DataFrame, statistic_callable=np.mean):
    if len(data) < 2:
        return np.nan, statistic_callable(data), np.nan

    # compute confidence interval (need > 1 datapoints)
    ci = stats.bootstrap(
        data=(data,), statistic=statistic_callable, confidence_level=CI_CONFIDENCE, method='BCa', axis=0
    ).confidence_interval
    statistic = statistic_callable(data, axis=0)

    return ci.low, statistic, ci.high
