import argparse
import numpy as np
import os
import pandas as pd
import sys

from tabulate import tabulate

from constants import *
from config_eval import config as eval_config
from config_train import config as train_config
import hyperparameters_multilabel

from src.training.train_utils import Logger
from src.models import get_classifier
from src.dataloaders.semantic_transforms_dataset import TF_BRIGHTNESS, TF_PERSPECTIVE, TF_CONTRAST, TF_ROTATION
from src.evaluate import SemanticMultilabelEvaluator
from src.evaluate.utils import *

_RES_DIR = './results/euler-results/data-2018-2021-116x150-pp0/final/rey-multilabel-classifier'

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--image-size', type=str, help='height and width', default='116 150',
                    choices=['78 100', '116 150', '232 300', '348 450'])
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--augmented', default=0, type=int, choices=[0, 1])

# transformations
parser.add_argument('--transform', type=str, default=TF_ROTATION,
                    choices=[TF_BRIGHTNESS, TF_PERSPECTIVE, TF_CONTRAST, TF_ROTATION])
parser.add_argument('--angles', nargs='+', type=float, default=[0, 5], help='absolute value (min, max) rotation angles')
parser.add_argument('--distortion', type=float, help='amount of distortion; ranges from 0 to 1')
parser.add_argument('--brightness', type=float, help='0 = black image, 1 = original image, 2 increases the brightness')
parser.add_argument('--contrast', type=float, help='0 = gray image, 1 = original image, 2 increases the contrast')

args = parser.parse_args()

_CLASS_LABEL_COLS = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
_CLASS_PRED_COLS = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]

_SCORE_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]

_NUM_CLASSES = 4


def main():
    results_dir = eval_config[REYMULTICLASSIFIER]['aug' if args.augmented else 'non-aug'][args.image_size]
    data_dir = os.path.join(DATA_DIR, train_config['data_root'][args.image_size])
    hyperparams = hyperparameters_multilabel.train_params[args.image_size]

    # save terminal output to file
    if args.transform == TF_ROTATION:
        prefix = f'rotation_{args.angles}'
    elif args.transform == TF_CONTRAST:
        prefix = f'contrast_{args.contrast}'
    elif args.transform == TF_BRIGHTNESS:
        prefix = f'brightness_{args.brightness}'
    elif args.transform == TF_PERSPECTIVE:
        prefix = f'perspective_{args.distortion}'
    else:
        raise ValueError

    log_file = "semantic_eval_out_" + prefix + ".txt"

    sys.stdout = Logger(print_fp=os.path.join(results_dir, log_file))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=_NUM_CLASSES)
    evaluator = SemanticMultilabelEvaluator(model=model, image_size=hyperparams['image_size'],
                                            results_dir=results_dir,
                                            data_dir=data_dir,
                                            batch_size=args.batch_size,
                                            workers=hyperparams['workers'],
                                            transform=args.transform,
                                            rotation_angles=args.angles,
                                            distortion_scale=args.distortion,
                                            brightness_factor=args.brightness,
                                            contrast_factor=args.contrast)

    evaluator.run_eval(save=True, prefix=prefix)

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
                            columns=[f'item-{i + 1}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE', 'MAE'])
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    print(f'\nOverall Score MSE: {total_score_mse}')
    print(f'\nOverall Score MAE: {total_score_mae}')


if __name__ == '__main__':
    main()
