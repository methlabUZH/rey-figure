import argparse
import json
import os
import pandas as pd

from src.analyze.performance_measures import PerformanceMeasures

# _DEBUG_RESULTS_DIR = 'results/spaceml-results/data_232x300-seed_1/final-3_scores/rey-multilabel-classifier'
_DEBUG_RESULTS_DIR = 'results/spaceml-results/data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier'
# _DEBUG_RESULTS_DIR = 'results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier'
# _DEBUG_RESULTS_DIR = 'results/spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier'

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results-dir', type=str, default=_DEBUG_RESULTS_DIR)
args = parser.parse_args()


def main():
    with open(os.path.join(args.results_dir, 'args.json'), 'r') as f:
        num_classes = json.load(f).get('n_classes', 4)

    preds = pd.read_csv(os.path.join(args.results_dir, 'test_predictions.csv'))
    gts = pd.read_csv(os.path.join(args.results_dir, 'test_ground_truths.csv'))

    perf_meas = PerformanceMeasures(gts, preds, num_classes=num_classes)
    perf_meas.generate_report(save_dir=os.path.join(args.results_dir, 'performance-report'))
    perf_meas.create_figures(save_dir=os.path.join(args.results_dir, 'performance-report'))


if __name__ == '__main__':
    main()
