import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import scipy.stats as stats

__all__ = ["compute_mse_scores", "compute_accuracy_scores", "compute_class_conditional_accuracy_scores",
           "compute_bin_mse_scores"]


def compute_mse_scores(predictions_df: pd.DataFrame,
                       ground_truth_df: pd.DataFrame,
                       columns: List[str]) -> List[float]:
    """this function computes the mse for each column in columns

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        columns: List of column names for which MSE scores are calculated

    Returns:
        List of MSE scores
    """
    return [float(np.mean((ground_truth_df.loc[:, col] - predictions_df.loc[:, col]) ** 2)) for col in columns]


def compute_accuracy_scores(predictions_df: pd.DataFrame,
                            ground_truth_df: pd.DataFrame,
                            columns: List[str]) -> List[float]:
    """This function computes accuracy for each column in columns

    Args:
        predictions_df: dataframe containing the predictions for each sample
        ground_truth_df: dataframe containing the ground truths for each sample
        columns: List of column names for which Accuracy scores are calculated

    Returns:
        List of Accuracy scores
    """
    return [float(np.mean(ground_truth_df.loc[:, col] == predictions_df.loc[:, col])) for col in columns]


def compute_class_conditional_accuracy_scores(predictions_df: pd.DataFrame,
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
        List of (List of Accuracy scores) for each column
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
