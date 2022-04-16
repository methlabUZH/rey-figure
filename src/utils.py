from datetime import datetime as dt
import numpy as np
import scipy.stats
from tabulate import tabulate


def timestamp_human():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S')


def timestamp_dir():
    return dt.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]


def print_dataframe(df, n=-1, n_digits=3):
    print(tabulate(df if n < 0 else df.head(n), headers='keys', tablefmt='presto', floatfmt=f".{n_digits}f"))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def map_to_score_grid(s: float, num_scores=4) -> float:
    if num_scores == 4:
        if s < 0.25:
            return 0.0
        if 0.25 <= s < 0.75:
            return 0.5
        if 0.75 <= s < 1.5:
            return 1.0
        return 2.0
    elif num_scores == 3:
        if s < 0.25:
            return 0.0
        if 0.25 <= s < 1.5:
            return 1.0
        return 2.0
    else:
        raise ValueError(f'`num_classes` must be 3 or 4; got {num_scores}')


def class_to_score(s: int, num_classes=4) -> float:
    if num_classes == 4:
        return {0: 0, 1: 0.5, 2: 1.0, 3: 2.0}[s]
    elif num_classes == 3:
        return {0: 0, 1: 1.0, 2: 2.0}[s]
    else:
        raise ValueError(f'`num_classes` must be 3 or 4; got {num_classes}')


def score_to_class(s: float, num_classes=4) -> int:
    if num_classes == 4:
        return {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}[float(s)]
    elif num_classes == 3:
        return {0.0: 0, 1.0: 1, 2.0: 2}[float(s)]
    else:
        raise ValueError(f'`num_classes` must be 3 or 4; got {num_classes}')


def assign_bin(x, bin_locations):
    return np.digitize(x, bin_locations, right=True)
