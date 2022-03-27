import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import random

from constants import *
from src.models.tdrg import get_model
from src.dataloaders.tdrg_dataloader import get_dataloader
from src.training.tdrg_trainer import Trainer
from src.training.train_utils import directory_setup, Logger, train_val_split

_VAL_FRACTION = 0.2
_SEED = 3923
_DEBUG_DATADIR = '/Users/maurice/phd/src/rey-figure/data/serialized-data/debug-116x150-pp0'

USE_CUDA = torch.cuda.is_available()
np.random.seed(_SEED)
random.seed(_SEED)
torch.manual_seed(_SEED)

if USE_CUDA:
    torch.cuda.manual_seed_all(_SEED)

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data-root', default=_DEBUG_DATADIR, type=str, help='save path')
parser.add_argument('--results-dir', type=str, default='./temp', required=False)
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--augment', default=0, type=int, choices=[0, 1])
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+', help='number of epochs to change learning rate')
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='INT',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=800, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP',
                    help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')

''' Train setting '''
parser.add_argument('--model_name', type=str, default='TDRG')
parser.add_argument('--image-size', nargs='+', default=(232, 232), help='height and width', type=int)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()


def main():
    is_train = True if not args.evaluate else False

    num_classes = 4 * N_ITEMS

    # setup dirs
    model_name = "TDRG"
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=model_name, dataset=dataset_name,
                                                   results_dir=args.results_dir, train_id=args.id)

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

    # get train dataloader
    train_loader = get_dataloader(data_root=args.data_root, labels=train_labels, batch_size=args.batch_size,
                                  num_workers=args.workers, shuffle=True, weighted_sampling=True,
                                  augment=args.augment, image_size=args.image_size)
    # get val dataloader
    val_loader = get_dataloader(data_root=args.data_root, labels=val_labels, batch_size=args.batch_size,
                                num_workers=args.workers, shuffle=False, augment=False, image_size=args.image_size)

    model = get_model(num_classes)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    trainer = Trainer(model, criterion, train_loader, val_loader, args, save_dir=results_dir)

    if is_train:
        trainer.train()
    else:
        trainer.validate()


if __name__ == "__main__":
    main()
