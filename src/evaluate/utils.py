import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import scipy.stats as stats
import sklearn.metrics as metrics

from constants import N_ITEMS

__all__ = ["compute_total_score_error", "compute_accuracy_scores", "compute_class_conditional_acc_scores",
           "compute_bin_mse_scores", "compute_multilabel_f1_score"]


def compute_total_score_error(predictions_df: pd.DataFrame,
                              ground_truth_df: pd.DataFrame,
                              columns: List[str],
                              which='mse',
                              return_ci: bool = False,
                              alpha_ci: float = None) -> Union[List[float],
                                                               Tuple[List[float], List[Tuple[float, float]]]]:
    """this function computes the mse for each column in columns

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        columns: List of column names for which MSE scores are calculated
        which: str, must be one of "mae" or "mse"
        return_ci: whether or not to return confidence intervals for each bin-MSE score
        alpha_ci: confidence level

    Returns:
        either a list of mse scores, or a tuple of (mse_scores, confidence_intervals)
    """
    if which == 'mse':
        error_scores = [(ground_truth_df.loc[:, col] - predictions_df.loc[:, col]) ** 2 for col in columns]
    elif which == 'mae':
        error_scores = [np.abs(ground_truth_df.loc[:, col] - predictions_df.loc[:, col]) for col in columns]
    else:
        raise ValueError('param "which" must be one of "mae" or "mse"')

    n_samples = [len(m) for m in error_scores]
    error_scores = [float(np.mean(m)) for m in error_scores]

    if not return_ci or which == 'mae':
        return error_scores

    ci = [(float(n * m / stats.chi2.ppf(1 - alpha_ci / 2, df=n - 1)),
           float(n * m / stats.chi2.ppf(alpha_ci / 2, df=n - 1))) for m, n in zip(error_scores, n_samples)]

    return error_scores, ci


def compute_accuracy_scores(predictions_df: pd.DataFrame,
                            ground_truth_df: pd.DataFrame,
                            columns: List[str]) -> List[float]:
    """This function computes accuracy for each column in columns

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        columns: List of column names for which Accuracy scores are calculated

    Returns:
        list of Accuracy scores
    """
    return [float(np.mean(ground_truth_df.loc[:, col] == predictions_df.loc[:, col])) for col in columns]


def compute_class_conditional_acc_scores(predictions_df: pd.DataFrame,
                                         ground_truth_df: pd.DataFrame,
                                         columns: List[str],
                                         num_classes: int) -> List[List[float]]:
    """This function computes class specific accuracy score for each column in columns

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        columns: List of column names for which Accuracy scores are calculated
        num_classes: number of classes per category / column

    Returns:
        list of (List of Accuracy scores) for each column
    """
    return [[float(np.mean(predictions_df.loc[ground_truth_df[col] == k, col] == k))
             for k in range(num_classes)] for col in columns]


def compute_bin_mse_scores(
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        bins: List[int],
        return_ci: bool = False,
        alpha_ci: float = None,
        aggregte_column: str = 'total_score') -> Union[List[float], Tuple[List[float], List[Tuple[float, float]]]]:
    """This function computes bin specific mse scores

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        bins: list of bins (ints)
        return_ci: whether or not to return confidence intervals for each bin-MSE score
        alpha_ci: confidence level
        aggregte_column: column in df over which to aggregate

    Returns:
        either a list of bin specific mse scores, or a tuple of (mse_scores, confidence_intervals)
    """
    # get list of per-sample MSE scores for each bin
    mse_scores = [(predictions_df.loc[ground_truth_df.bin == b, aggregte_column]
                   - ground_truth_df.loc[ground_truth_df.bin == b, aggregte_column]) ** 2 for b in bins]

    n_samples = [len(m) for m in mse_scores]
    mse_scores = [float(np.mean(m)) for m in mse_scores]

    if not return_ci:
        return mse_scores

    ci = [(float(n * m / stats.chi2.ppf(1 - alpha_ci / 2, df=n - 1)),
           float(n * m / stats.chi2.ppf(alpha_ci / 2, df=n - 1))) for m, n in zip(mse_scores, n_samples)]

    return mse_scores, ci


def compute_multilabel_f1_score(predictions_df: pd.DataFrame,
                                ground_truth_df: pd.DataFrame,
                                bins: List[int] = None,
                                num_classes: int = 4,
                                average='micro') -> List[float]:
    columns = [f"class_item_{i + 1}" for i in range(N_ITEMS)]

    preds = predictions_df.loc[:, columns]
    preds = np.array(preds.values)
    preds = np.array(list(list(map(lambda l: one_hot(l, num_classes), p)) for p in preds))
    preds = np.reshape(preds, newshape=(np.shape(preds)[0], -1))

    gts = ground_truth_df.loc[:, [f"class_item_{i + 1}" for i in range(N_ITEMS)]]
    gts = np.array(gts.values)
    gts = np.array(list(list(map(lambda l: one_hot(l, num_classes), p)) for p in gts))
    gts = np.reshape(gts, newshape=(np.shape(gts)[0], -1))

    if bins is None:
        return [metrics.f1_score(gts, preds, average=average)]

    return [metrics.f1_score(
        gts[ground_truth_df.bin == b], preds[ground_truth_df.bin == b], average=average) for b in bins]

    # bin_indices = ground_truth_df.bin == 1
    # print(bin_indices)
    # print(np.shape(preds[bin_indices]))
    #
    # # if bins is not None:
    # #     f1_scores = []
    # #     for b in bins:
    # #         preds_df = predictions_df.loc[ground_truth_df.bin == b, :].copy(deep=True)
    # #         gt_df = ground_truth_df.loc[ground_truth_df.bin == b, :].copy(deep=True)
    # #         f1_scores.append(compute_multilabel_f1_score(
    # #             preds_df, gt_df, bins=None, num_classes=num_classes, average=average))
    # #     return f1_scores
    #
    # f1_score = metrics.f1_score(gts, preds, average=average)
    # return f1_score


def one_hot(label, num_classes=4):
    onehot = np.zeros(shape=num_classes)
    onehot[label] = 1
    return onehot

# if __name__ == '__main__':
#     import os
#     from tabulate import tabulate
#     import sklearn.metrics as metrics
#     from constants import BIN_LOCATIONS3_V2
#     from src.utils import assign_bin
#
#     results_root = '../../results/euler-results/data-2018-2021-116x150-pp0/final/'
#     multilabel_root = results_root + 'rey-multilabel-classifier'
#     # multilabel_root = results_root + 'rey-regressor'
#     multilabel_preds = pd.read_csv(os.path.join(multilabel_root, 'test_predictions.csv'))
#     multilabel_gt = pd.read_csv(os.path.join(multilabel_root, 'test_ground_truths.csv'))
#
#     multilabel_gt[['bin']] = multilabel_gt[['total_score']].applymap(lambda x: assign_bin(x, BIN_LOCATIONS3_V2))
#
#     print(compute_multilabel_f1_score(multilabel_preds, multilabel_gt,
#                                       bins=[i for i in range(1, len(BIN_LOCATIONS3_V2))]))
