from datetime import datetime as dt
import numpy as np
import scipy.stats
from tabulate import tabulate

from constants import BIN_LOCATIONS1, BIN_LOCATIONS2


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


def map_to_score_grid(s):
    if s < 0.25:
        return 0
    if 0.25 <= s < 0.75:
        return 0.5
    if 0.75 <= s < 1.5:
        return 1
    return 2


def class_to_score(s) -> float:
    return {0.0: 0,
            1.0: 0.5,
            2.0: 1.0,
            3.0: 2.0}[float(s)]


def score_to_class(s) -> int:
    return {0.0: 0,
            0.5: 1,
            1.0: 2,
            2.0: 3}[float(s)]


def assign_bin(x, bin_locations):
    return np.digitize(x, bin_locations, right=True)
