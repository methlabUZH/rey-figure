import os
import argparse
import json
import pandas as pd
import sys

from config_train import config as cfg_train
from config_eval import config as cfg_eval
import hyperparameters_multilabel

from constants import REYMULTICLASSIFIER, DATA_DIR
from src.analyze.performance_measures import PerformanceMeasures
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import MultilabelEvaluator

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default=None)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()


def main():
    # load args from .json
    with open(os.path.join(args.results_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)

    num_classes = train_args['n_classes']
    image_size_str = " ".join(str(s) for s in train_args['image_size'])

    data_dir = os.path.join(DATA_DIR, cfg_train['data_root'][image_size_str])

    print(f'--> evaluating model from {args.results_dir}')
    print(f'--> using data from {data_dir}')

    # Read parameters from hyperparameters_multilabel.py
    hyperparams = hyperparameters_multilabel.train_params[image_size_str]

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval_out.txt'))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=num_classes)
    evaluator = MultilabelEvaluator(model=model, image_size=hyperparams['image_size'],
                                    results_dir=args.results_dir,
                                    data_dir=data_dir, batch_size=args.batch_size,
                                    tta=args.tta, validation=args.validation, workers=hyperparams['workers'],
                                    angles=cfg_eval[REYMULTICLASSIFIER]['angles'], num_classes=num_classes)
    evaluator.run_eval(save=True)

    predictions = pd.read_csv(os.path.join(args.results_dir, 'test_predictions.csv'))
    ground_truths = pd.read_csv(os.path.join(args.results_dir, 'test_ground_truths.csv'))

    # run performance report
    perf_meas = PerformanceMeasures(ground_truths=ground_truths, predictions=predictions, num_classes=num_classes)
    perf_meas.generate_report(save_dir=os.path.join(args.results_dir, 'performance-report'))
    perf_meas.create_figures(save_dir=os.path.join(args.results_dir, 'performance-report'))


if __name__ == '__main__':
    main()
