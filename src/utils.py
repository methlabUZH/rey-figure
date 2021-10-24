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
