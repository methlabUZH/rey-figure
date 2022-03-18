import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import numpy as np
import pandas as pd
import sys

from tabulate import tabulate

from constants import *
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import MultilabelEvaluator
from src.evaluate.utils import *

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str)
parser.add_argument('--results-dir', type=str)
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--workers', default=8, type=int)
args = parser.parse_args()

_CLASS_LABEL_COLS = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
_CLASS_PRED_COLS = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]

_SCORE_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]

_NUM_CLASSES = 4


def main():
    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval_out.txt'))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=_NUM_CLASSES)
    evaluator = MultilabelEvaluator(model=model, image_size=args.image_size, results_dir=args.results_dir,
                                    data_dir=args.data_root, batch_size=args.batch_size)
    evaluator.run_eval(save=True)

    predictions = evaluator.predictions
    ground_truths = evaluator.ground_truths

    # ------- item specific scores -------
    item_accuracy_scores = compute_accuracy_scores(
        predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])

    item_mse_scores = compute_total_score_error(
        predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])

    # ------- toal score mse -------
    total_score_mse = compute_total_score_error(predictions, ground_truths, ["total_score"])[0]

    # ------- toal score mae -------
    total_score_mae = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mae')[0]

    print('---------- Item Scores ----------')
    print_df = pd.DataFrame(data=np.stack([item_accuracy_scores, item_mse_scores], axis=0),
                            columns=[f'item-{i+1}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print(f'\nOverall Score MSE: {total_score_mse}')
    print(f'\nOverall Score MAE: {total_score_mae}')


if __name__ == '__main__':
    main()
