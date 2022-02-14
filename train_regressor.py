import argparse
import random
import sys
import numpy as np
import os
import pandas as pd
import json

import torch

from constants import *
from src.dataloaders.dataloader_regression import get_regression_dataloader
from src.training.train_utils import directory_setup, Logger, train_val_split
from src.models import get_regressor
from src.training.regression_trainer import RegressionTrainer

_VAL_FRACTION = 0.2
_SEED = 7

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default=DEBUG_DATADIR, required=False)
parser.add_argument('--results-dir', type=str, default='./temp', required=False)
parser.add_argument('--simulated-data', type=str, default=None, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--eval-test', action='store_true')
parser.add_argument('--id', default='debug', type=str)
parser.add_argument('--epochs', default=75, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--beta', type=float, default=0.0, help='weight of the score mse for total loss')
parser.add_argument('--gamma', type=float, default=1.0, help='learning rate decay factor')
parser.add_argument('--wd', '--weight-decay', type=float, default=0)
parser.add_argument('--weighted-sampling', default=1, type=int, choices=[0, 1])
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
np.random.seed(_SEED)
random.seed(_SEED)
torch.manual_seed(_SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(_SEED)


def main():
    # setup dirs
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=REYREGRESSOR,
                                                   dataset=dataset_name,
                                                   results_dir=args.results_dir,
                                                   train_id=args.id)

    # dump args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'out.txt'))

    # read and split labels into train and val
    labels_csv = os.path.join(args.data_root, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)

    # split df into validation and train parts
    train_labels, val_labels = train_val_split(labels_df, fraction=_VAL_FRACTION)

    # include simulated data
    if args.simulated_data is not None:
        sim_df = pd.read_csv(args.simulated_data)
        train_labels = pd.concat([train_labels, sim_df], ignore_index=True)

    # get train dataloader
    train_loader = get_regression_dataloader(args.data_root, labels_df=train_labels, batch_size=args.batch_size,
                                             num_workers=args.workers, shuffle=True,
                                             weighted_sampling=args.weighted_sampling)

    # get val dataloader
    val_loader = get_regression_dataloader(args.data_root, labels_df=val_labels, batch_size=args.batch_size,
                                           num_workers=args.workers, shuffle=False)

    model = get_regressor()
    criterion = torch.nn.MSELoss(reduction="mean")
    trainer = RegressionTrainer(model, criterion, train_loader, val_loader, args, results_dir)
    trainer.train()

    if args.eval_test:
        eval_test(trainer, results_dir)


def eval_test(trainer, results_dir):
    # load best checkpoint
    ckpt = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')
    ckpt = torch.load(ckpt, map_location=torch.device('cuda' if USE_CUDA else 'cpu'))
    trainer.model.load_state_dict(ckpt['state_dict'], strict=True)

    # get dataloader
    test_labels = pd.read_csv(os.path.join(args.data_root, 'test_labels.csv'))
    test_dataloader = get_regression_dataloader(args.data_root, labels_df=test_labels,
                                                batch_size=args.batch_size, num_workers=args.workers,
                                                shuffle=False)
    test_stats = trainer.run_epoch(test_dataloader, is_train=False)

    print('\n-------eval on test set with best model-------')
    for k, v in test_stats.items():
        print(f'{k.replace("val", "test")}: {v:.5f}')


if __name__ == '__main__':
    main()
