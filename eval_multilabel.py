import argparse
import numpy as np
import os
import pandas as pd
import sys

from tabulate import tabulate

from constants import N_ITEMS, BIN_LOCATIONS2_V2, REYMULTICLASSIFIER
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import MultilabelEvaluator
from src.utils import assign_bin

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='')
parser.add_argument('--results-dir', type=str, default='')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--binary', default=0, type=int, choices=[0, 1])
parser.add_argument('--ensemble', default=0, type=int, choices=[0, 1])
parser.add_argument('--workers', default=8, type=int)
args = parser.parse_args()

_CLASS_LABEL_COLS = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
_CLASS_PRED_COLS = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]

_SCORE_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]


def main():
    num_classes = 2 if args.binary else 4
    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval_out.txt'))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=num_classes)
    evaluator = MultilabelEvaluator(model=model, results_dir=args.results_dir, data_dir=args.data_root,
                                    is_ensemble=args.ensemble, is_binary=args.binary, batch_size=args.batch_size)
    evaluator.run_eval()
    evaluator.save_predictions(save_as=None)

    if args.binary:
        raise NotImplementedError

    predictions = evaluator.predictions

    # ------- item specific scores -------
    # compute accuracy for each item
    item_accuracy_scores = [np.mean(
        predictions.loc[:, lab_col] == predictions.loc[:, pred_col])
        for lab_col, pred_col in zip(_CLASS_LABEL_COLS, _CLASS_PRED_COLS)
    ]

    # compute class specific acc for each item
    class_conditional_accuracy_scores = [[np.mean(
        predictions.loc[predictions[lab_col] == k, pred_col] == k)
        for k in range(num_classes)] for lab_col, pred_col in zip(_CLASS_LABEL_COLS, _CLASS_PRED_COLS)]

    # compute MSE for each item
    item_mse_scores = [np.mean(
        (predictions.loc[:, lab_col] - predictions.loc[:, pred_col]) ** 2)
        for lab_col, pred_col in zip(_SCORE_LABEL_COLS, _SCORE_PRED_COLS)
    ]

    # ------- bin specific scores -------
    # assign bin to each sample
    predictions[['bin']] = predictions[['true_total_score']].applymap(lambda x: assign_bin(x, BIN_LOCATIONS2_V2))

    # compute mse for each bin
    bin_mse_scores = [np.mean(
        (predictions.loc[predictions['bin'] == b, 'pred_total_score']
         - predictions.loc[predictions['bin'] == b, 'true_total_score']) ** 2)
        for b in range(1, len(BIN_LOCATIONS2_V2))]

    # ------- global scores -------
    # compute overall MSE
    pred_total_scores = predictions.loc[:, 'pred_total_score']
    true_total_scores = predictions.loc[:, 'true_total_score']
    score_mse = np.mean((pred_total_scores - true_total_scores) ** 2)

    print('---------- Item Scores ----------')
    print_df = pd.DataFrame(data=np.stack([item_accuracy_scores, item_mse_scores], axis=0),
                            columns=[f'item-{i}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print('---------- Class Conditional Item Accuracy Scores ----------')
    print_df = pd.DataFrame(data=np.stack(class_conditional_accuracy_scores, axis=0),
                            columns=[f'Class {k}' for k in range(4)],
                            index=[f"Item {i}" for i in range(N_ITEMS)])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print('---------- BIN MSE Scores ----------')
    print_df = pd.DataFrame(data=np.expand_dims(bin_mse_scores, axis=0),
                            columns=[f'Bin-{i}' for i in range(1, len(BIN_LOCATIONS2_V2))],
                            index=['MSE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print('---------- Global Scores ----------')
    print(f'Overall Score MSE: {score_mse}')


if __name__ == '__main__':
    main()
