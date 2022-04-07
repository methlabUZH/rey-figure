import argparse
import os
import pandas as pd

from src.analyze.performance_measures import PerformanceMeasures

_DEFAULT_RESULTS_DIR = 'results/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results-dir', type=str, default=_DEFAULT_RESULTS_DIR)
args = parser.parse_args()


def main():
    preds = pd.read_csv(os.path.join(args.results_dir, 'test_predictions.csv'))
    gts = pd.read_csv(os.path.join(args.results_dir, 'test_ground_truths.csv'))
    perf_meas = PerformanceMeasures(gts, preds)
    perf_meas.generate_report()


if __name__ == '__main__':
    main()
