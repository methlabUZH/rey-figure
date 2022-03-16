import argparse
import numpy as np
import os
import pandas as pd
import sys

from tabulate import tabulate

from constants import *
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import SemanticMultilabelEvaluator
from src.evaluate.utils import *

_RES_DIR = './results/euler-results/data-2018-2021-116x150-pp0/final/rey-multilabel-classifier'

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default=DEBUG_DATADIR_SMALL)
parser.add_argument('--results-dir', type=str, default=_RES_DIR)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--binary', default=0, type=int, choices=[0, 1])
parser.add_argument('--ensemble', default=0, type=int, choices=[0, 1])
parser.add_argument('--workers', default=8, type=int)

parser.add_argument('--rot', type=float, default=None)
parser.add_argument('--tilt', type=float, default=None)

args = parser.parse_args()

_CLASS_LABEL_COLS = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
_CLASS_PRED_COLS = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]

_SCORE_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]


def main():
    num_classes = 2 if args.binary else 4

    assert (args.rot is None or args.tilt is None), "only one of rot and tilt can be not None"

    # save terminal output to file
    if args.rot is not None:
        prefix = f'rot={args.rot}'
    else:
        prefix = f"perspective={args.tilt}"

    log_file = "semantic_eval_out_" + prefix + ".txt"

    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, log_file))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=num_classes)
    evaluator = SemanticMultilabelEvaluator(model=model, results_dir=args.results_dir, data_dir=args.data_root,
                                            is_ensemble=args.ensemble, is_binary=args.binary, rotation_angle=args.rot,
                                            perspective_change=args.tilt, batch_size=args.batch_size)
    evaluator.run_eval(save=True, prefix=prefix)

    if args.binary:
        return

    predictions = evaluator.predictions
    ground_truths = evaluator.ground_truths

    # ------- item specific scores -------
    item_accuracy_scores = compute_accuracy_scores(
        predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])

    item_mse_scores = compute_total_score_error(
        predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])
    item_mae_scores = compute_total_score_error(
        predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)], which='mae')

    # ------- toal score mse -------
    total_score_mse = compute_total_score_error(predictions, ground_truths, ["total_score"])[0]

    # ------- toal score mae -------
    total_score_mae = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mae')[0]

    print('---------- Item Scores ----------')
    print_df = pd.DataFrame(data=np.stack([item_accuracy_scores, item_mse_scores, item_mae_scores], axis=0),
                            columns=[f'item-{i}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE', 'MAE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print(f'\nOverall Score MSE: {total_score_mse}')
    print(f'\nOverall Score MAE: {total_score_mae}')


if __name__ == '__main__':
    main()
