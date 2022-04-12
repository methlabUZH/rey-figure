import os
import argparse
import numpy as np
import pandas as pd
import sys

from tabulate import tabulate

from config_train import config as cfg_train
from config_eval import config as cfg_eval
import hyperparameters_multilabel

from constants import REYMULTICLASSIFIER, N_ITEMS, DATA_DIR, CLASS_COLUMNS, SCORE_COLUMNS
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import MultilabelEvaluator
from src.evaluate.utils import *

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--image-size', type=str, help='height and width', default='116 150',
                    choices=['78 100', '116 150', '232 300', '348 450'])
parser.add_argument('--augmented', type=int, choices=[0, 1], default=0)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--tta', action='store_true')
parser.add_argument('--n_train', type=int, default=-1, help='number of training data points')
args = parser.parse_args()

NUM_CLASSES = 4


def main():
    results_dir = cfg_eval[REYMULTICLASSIFIER]['aug' if args.augmented else 'non-aug'][args.image_size]

    if args.n_train > 0:
        results_dir = results_dir.replace('/data_', f'/{args.n_train}-data_')

    print(f'--> evaluating model from {results_dir}')

    data_dir = os.path.join(DATA_DIR, cfg_train['data_root'][args.image_size])

    # Read parameters from hyperparameters_multilabel.py
    hyperparams = hyperparameters_multilabel.train_params[args.image_size]

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'eval_out.txt'))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=NUM_CLASSES)
    evaluator = MultilabelEvaluator(model=model, image_size=hyperparams['image_size'],
                                    results_dir=results_dir,
                                    data_dir=data_dir, batch_size=args.batch_size,
                                    tta=args.tta, validation=args.validation, workers=hyperparams['workers'],
                                    angles=cfg_eval[REYMULTICLASSIFIER]['angles'])
    evaluator.run_eval(save=True)

    predictions = evaluator.predictions
    ground_truths = evaluator.ground_truths

    # ------- item specific scores -------
    item_accuracy_scores = compute_accuracy_scores(
        predictions, ground_truths, columns=CLASS_COLUMNS)

    item_mse_scores = compute_total_score_error(predictions, ground_truths, columns=SCORE_COLUMNS, which='mse')

    # ------- toal score mse -------
    total_score_mse = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mse')[0]

    # ------- toal score mae -------
    total_score_mae = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mae')[0]

    print('---------- Item Scores ----------')
    print_df = pd.DataFrame(data=np.stack([item_accuracy_scores, item_mse_scores], axis=0),
                            columns=[f'item-{i + 1}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print(f'\nOverall Score MSE: {total_score_mse}')
    print(f'\nOverall Score MAE: {total_score_mae}')


if __name__ == '__main__':
    main()
