import argparse
import json
import os
import pandas as pd

from src.analyze.performance_measures import PerformanceMeasures

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results-dir', type=str, default=None)
args = parser.parse_args()


def main(results_dir):
    with open(os.path.join(results_dir, 'args.json'), 'r') as f:
        num_classes = json.load(f).get('n_classes', 4)

    preds = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'))
    gts = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))

    perf_meas = PerformanceMeasures(gts, preds, num_classes=num_classes)
    perf_meas.generate_report(save_dir=os.path.join(results_dir, 'performance-report'))
    perf_meas.create_figures(save_dir=os.path.join(results_dir, 'performance-report'))


if __name__ == '__main__':
    import glob

    dir0 = 'results/spaceml-results-light/**/rey-multilabel-classifier'
    for d in glob.glob(dir0, recursive=True):
        main(d)
    # main(args.results_dir)
