import argparse
import json
import os
import pandas as pd
import sys

from constants import DATA_DIR, REYMULTICLASSIFIER
from config_train import config as train_config
import hyperparameters_multilabel
from src.analyze.performance_measures import PerformanceMeasures
from src.training.train_utils import Logger
from src.models import get_classifier
from src.dataloaders.semantic_transforms_dataset import TF_BRIGHTNESS, TF_PERSPECTIVE, TF_CONTRAST, TF_ROTATION
from src.evaluate import SemanticMultilabelEvaluator

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default=None)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--workers', default=8, type=int)

# transformations
parser.add_argument('--transform', type=str, default=TF_ROTATION,
                    choices=[TF_BRIGHTNESS, TF_PERSPECTIVE, TF_CONTRAST, TF_ROTATION])
parser.add_argument('--angles', nargs='+', type=float, default=[0, 5], help='absolute value (min, max) rotation angles')
parser.add_argument('--distortion', type=float, help='amount of distortion; ranges from 0 to 1')
parser.add_argument('--brightness', type=float, help='0 = black image, 1 = original image, 2 increases the brightness')
parser.add_argument('--contrast', type=float, help='0 = gray image, 1 = original image, 2 increases the contrast')

args = parser.parse_args()


def main():
    # load args from .json
    with open(os.path.join(args.results_dir, 'args.json'), 'r') as f:
        train_args = json.load(f)

    num_classes = train_args['n_classes']
    image_size_str = " ".join(str(s) for s in train_args['image_size'])

    data_dir = os.path.join(DATA_DIR, train_config['data_root'][image_size_str])

    print(f'--> evaluating model from {args.results_dir}')
    print(f'--> using data from {data_dir}')

    # Read parameters from hyperparameters_multilabel.py
    hyperparams = hyperparameters_multilabel.train_params[image_size_str]

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

    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, log_file))

    model = get_classifier(arch=REYMULTICLASSIFIER, num_classes=num_classes)
    evaluator = SemanticMultilabelEvaluator(model=model, image_size=hyperparams['image_size'],
                                            results_dir=args.results_dir, data_dir=data_dir, batch_size=args.batch_size,
                                            workers=hyperparams['workers'],
                                            transform=args.transform,
                                            rotation_angles=args.angles,
                                            distortion_scale=args.distortion,
                                            brightness_factor=args.brightness,
                                            contrast_factor=args.contrast,
                                            num_classes=num_classes)

    evaluator.run_eval(save=True, prefix=prefix)


if __name__ == '__main__':
    main()
