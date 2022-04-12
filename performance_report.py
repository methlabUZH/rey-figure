import argparse
import os
import pandas as pd

from src.analyze.performance_measures import PerformanceMeasures

_DEFAULT_RESULTS_DIR = 'results/spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results-dir', type=str, default=_DEFAULT_RESULTS_DIR)
args = parser.parse_args()


def main():
    preds = pd.read_csv(os.path.join(args.results_dir, 'test_predictions.csv'))
    gts = pd.read_csv(os.path.join(args.results_dir, 'test_ground_truths.csv'))

    perf_meas = PerformanceMeasures(gts, preds)
    perf_meas.generate_report(save_dir=args.results_dir)
    perf_meas.create_figures(save_dir=args.results_dir)


if __name__ == '__main__':
    main()
