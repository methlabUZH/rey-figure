import argparse
import random
import sys
import numpy as np
import os
import pandas as pd
import json

import torch

from constants import *
from old_code.dataloaders.dataloader_item_classification import get_item_classification_dataloader
from src.training.train_utils import directory_setup, Logger, train_val_split
from src.models import get_classifier

from src.training.classification_trainer import ClassificationTrainer

_VAL_FRACTION = 0.2
_SEED = 7

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default=DATADIR_SMALL, required=False)
parser.add_argument('--results-dir', type=str, default='./temp', required=False)
parser.add_argument('--simulated-data', type=str, default=None, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--arch', type=str, default=WIDE_RESNET50_2, required=False)
parser.add_argument('--is_binary', type=int, default=0, choices=[0, 1])
parser.add_argument('--eval-test', action='store_true')
parser.add_argument('--item', type=int, default=1)
parser.add_argument('--id', default='debug', type=str)
parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
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
    num_classes = 2 if args.is_binary else 4

    # setup dirs
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=f'{num_classes}-way-item-classifier/item-{args.item}',
                                                   dataset=dataset_name, results_dir=args.results_dir, train_id=args.id)

    # dump args
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'out.txt'))

    # read and split labels into train and val
    labels_csv = os.path.join(args.data_root, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)

    # split df into validation and train parts
    train_labels, val_labels = train_val_split(labels_df, val_fraction=_VAL_FRACTION)

    # include simulated data
    if args.simulated_data is not None:
        sim_df = pd.read_csv(args.simulated_data)
        train_labels = pd.concat([train_labels, sim_df], ignore_index=True)

    # get train dataloader
    train_loader = get_item_classification_dataloader(args.item, args.data_root, labels_df=train_labels,
                                                      batch_size=args.batch_size, num_workers=args.workers,
                                                      shuffle=True, weighted_sampling=args.weighted_sampling,
                                                      is_binary=args.is_binary)
    # get val dataloader
    val_loader = get_item_classification_dataloader(args.item, args.data_root, labels_df=val_labels,
                                                    batch_size=args.batch_size, num_workers=args.workers,
                                                    shuffle=False, is_binary=args.is_binary)

    model = get_classifier(args.arch, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = ClassificationTrainer(model, criterion, train_loader, val_loader, args, results_dir, args.is_binary)
    trainer.train()

    if args.eval_test:
        eval_test(trainer)


def eval_test(trainer):
    # dataloader
    test_labels = pd.read_csv(os.path.join(args.data_root, 'test_labels.csv'))
    test_dataloader = get_item_classification_dataloader(args.item, args.data_root, labels_df=test_labels,
                                                         batch_size=args.batch_size, num_workers=args.workers,
                                                         shuffle=False, is_binary=args.is_binary)
    test_stats = trainer.run_epoch(test_dataloader, is_train=False)

    print('-------eval test-------')
    for k, v in test_stats.items():
        print(f'{k}: {v:.5f}')


if __name__ == '__main__':
    main()
