import argparse
import sys
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import json
import random

import torch

from constants import *
from src.dataloaders.rocf_dataloader import get_dataloader
from src.models import get_classifier
from src.training.train_utils import directory_setup, Logger, train_val_split

from src.training.multilabel_trainer import MultilabelTrainer

_DEBUG_DATADIR = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default=_DEBUG_DATADIR, required=False)
parser.add_argument('--results-dir', type=str, default='./temp', required=False)
parser.add_argument('--simulated-data', type=str, default=None, required=False)
parser.add_argument('--max-simulated', type=int, default=-1, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--is_binary', type=int, default=0, choices=[0, 1])
parser.add_argument('--eval-test', action='store_true')
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--epochs', default=75, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='learning rate decay factor')
parser.add_argument('--wd', '--weight-decay', type=float, default=0)
parser.add_argument('--weighted-sampling', default=1, type=int, choices=[0, 1])
parser.add_argument('--augment', default=0, type=int, choices=[0, 1])
parser.add_argument('--image-size', nargs='+', default=DEFAULT_CANVAS_SIZE, help='height and width', type=int)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
VAL_FRACTION = 0.2

if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)


def main():
    num_classes = 2 if args.is_binary else 4

    # setup dirs
    model_name = "binary-" + REYMULTICLASSIFIER if num_classes == 2 else REYMULTICLASSIFIER
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
    train_labels, val_labels = train_val_split(labels_df, val_fraction=VAL_FRACTION)

    # include simulated data
    if args.simulated_data is not None:
        sim_df = pd.read_csv(args.simulated_data)

        # subsamble simulated data
        if args.max_simulated > 0:
            sim_df = sim_df.sample(n=args.max_simulated)

        train_labels = pd.concat([train_labels, sim_df], ignore_index=True)

    # get train dataloader
    train_loader = get_dataloader(data_root=args.data_root, labels=train_labels, label_type=CLASSIFICATION_LABELS,
                                  batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                  weighted_sampling=args.weighted_sampling, augment=args.augment,
                                  image_size=args.image_size)
    # get val dataloader
    val_loader = get_dataloader(data_root=args.data_root, labels=val_labels, label_type=CLASSIFICATION_LABELS,
                                batch_size=args.batch_size, num_workers=args.workers, shuffle=False, augment=False,
                                image_size=args.image_size)

    print(f'# train images:\t{len(train_labels)}')
    print(f'# val images:\t{len(val_labels)}')

    model = get_classifier(REYMULTICLASSIFIER, num_classes=num_classes)
    loss_func = torch.nn.CrossEntropyLoss()
    trainer = MultilabelTrainer(model, loss_func, train_loader, val_loader, args, results_dir, args.is_binary)
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
    test_dataloader = get_dataloader(args.data_root, labels=test_labels, label_type=CLASSIFICATION_LABELS,
                                     batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                                     image_size=args.image_size)
    test_stats = trainer.run_epoch(test_dataloader, is_train=False)

    print('\n-------eval test-------')
    # build train table
    accuracies = test_stats['val-accuracies']
    losses = test_stats['val-losses']
    data = np.stack([accuracies, losses], axis=0)
    indices = ['test-acc', 'test-loss']

    if args.is_binary:
        specificities = test_stats['val-specificities']
        sensitivities = test_stats['val-sensitivities']
        gmeans = test_stats['val-gmeans']
        data = np.concatenate([data, np.stack([specificities, sensitivities, gmeans], axis=0)], axis=0)
        indices += ['test-specificity', 'test-sensitivity', 'test-g-mean']

    df = pd.DataFrame(data, columns=[f'item_{i + 1}' for i in range(N_ITEMS)])
    df['index'] = indices
    df = df.set_index('index')

    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".3f"))


if __name__ == '__main__':
    main()
