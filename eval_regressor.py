import argparse
import numpy as np
import os
import pandas as pd
import sys

from tabulate import tabulate

from constants import *
from src.training.train_utils import Logger
from src.models import get_regressor, get_regressor_v2
from src.evaluate import RegressionEvaluator
from src.evaluate.utils import *

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='')
parser.add_argument('--arch', type=str, default='v2', required=False)
parser.add_argument('--results-dir', type=str, default='')
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--tta', action='store_true')
args = parser.parse_args()

_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]


def main():
    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval_out.txt'))

    if args.arch == 'v1':
        model = get_regressor()
    elif args.arch == 'v2':
        model = get_regressor_v2()
    else:
        raise ValueError(f'unknown arch version {args.arch}')

    evaluator = RegressionEvaluator(model=model, image_size=args.image_size, 
                                    results_dir=args.results_dir,
                                    data_dir=args.data_root, 
                                    tta=args.tta,
                                    validation=args.validation,
                                    batch_size=args.batch_size, 
                                    workers=args.workers)
    evaluator.run_eval(save=True)

    predictions = evaluator.predictions
    ground_truths = evaluator.ground_truths

    # ------- item specific scores -------
    item_mse_scores = compute_total_score_error(
        predictions, ground_truths, columns=[f"score_item_{i + 1}" for i in range(N_ITEMS)])

    # ------- toal score mse -------
    total_score_mse = compute_total_score_error(predictions, ground_truths, ["total_score"])[0]
    total_score_mae = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mae')[0]

    print('---------- Item Scores ----------')
    print_df = pd.DataFrame(data=np.stack([item_mse_scores], axis=0),
                            columns=[f'item-{i}' for i in range(N_ITEMS)],
                            index=['MSE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print(f'\nOverall Score MSE: {total_score_mse}')
    print(f'Overall Score MAE: {total_score_mae}')


if __name__ == '__main__':
    main()
