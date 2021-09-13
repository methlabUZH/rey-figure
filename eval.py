import argparse
import random
import sys
import os

import numpy as np
import pandas as pd

import torch

from constants import BIN_LOCATIONS
from src.training.data_loader import get_dataloader
from src.training.helpers import timestamp_human, count_parameters, assign_bins
from src.training.helpers import AverageMeter, Logger
from src.models.model_factory import get_architecture

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224'
default_results_dir = '/Users/maurice/phd/src/psychology/results/sum-score/scans-2018-2021-224x224-augmented/resnet18/2021-09-12_20-03-15.155'

# setup arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=default_results_dir, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# architecture
parser.add_argument('--arch', type=str, default='resnet18', required=False)
parser.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
parser.add_argument('--norm-layer', type=str, default='batch_norm', choices=['batch_norm', 'group_norm'])

# misc
parser.add_argument('--seed', default=8, type=int)
parser.add_argument('--score-type', default='sum', type=str, choices=['sum', 'median'])

args = parser.parse_args()

if DEBUG:
    args.batch_size = 2

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


def main():
    if DEBUG:
        print('==> debugging on!')

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval.txt'))

    # data
    print(f'==> data from {args.data_root}')
    labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_dataloader(args.data_root, labels_df=labels, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, score_type=args.score_type)

    # setup model
    model = get_architecture(arch=args.arch, num_outputs=18, dropout=None, norm_layer=args.norm_layer,
                             image_size=args.image_size)

    # load checkpoint
    checkpoint_file = os.path.join(args.results_dir, 'checkpoints/model_best.pth.tar')
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint found!'
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('gpu' if use_cuda else 'cpu'))
    if not use_cuda:
        checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'==> loaded checkpoint {checkpoint_file}')

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction="mean")

    print('{0:10}: {1}'.format('num-test', len(dataloader.dataset)))
    print('{0:10}: {1}'.format('#params', count_parameters(model)))

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start eval')

    # eval on test set
    test_loss, test_score_mse, test_bin_mse, single_bin_mses = eval_model(model, dataloader, criterion)

    print(f'[{timestamp_human()}] finish eval')
    print('\n----------------------\n')

    print('total test scores:')
    print('----------------------')
    print('{0:10}: {1}'.format('loss', test_loss))
    print('{0:10}: {1}'.format('score mse', test_score_mse))
    print('{0:10}: {1}'.format('bin mse', test_bin_mse))
    print('')
    print('bin mse scores:')
    print('----------------------')

    for i, _ in enumerate(BIN_LOCATIONS):
        bin_mse = single_bin_mses.get(i)
        print('{0:10}: {1}'.format(f'mse bin {i}', bin_mse if bin_mse is not None else np.nan))



def eval_model(model, dataloader, criterion):
    model.eval()

    loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    total_bin_mse_meter = AverageMeter()
    single_bin_score_mse_meters = {i: AverageMeter() for i, _ in enumerate(BIN_LOCATIONS)}

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        outputs[:, -1] = torch.clip(outputs[:, -1], 0, 36)

        loss = criterion(outputs, labels)
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        total_bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(labels[:, -1]))
        single_bin_score_mses = mse_per_bin(outputs[:, -1], labels[:, -1], criterion)

        # record loss
        loss_meter.update(loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        total_bin_mse_meter.update(total_bin_mse.data, images.size()[0])

        for b, (bin_mse, n) in single_bin_score_mses.items():
            single_bin_score_mse_meters[b].update(bin_mse.data, n)

        if DEBUG:
            print(f'val batch={batch_idx}')
            if batch_idx >= 5:
                break

    # compute_averages
    single_bin_score_mses = {i: m.avg for i, m in single_bin_score_mse_meters.items()}

    return loss_meter.avg, score_mse_meter.avg, total_bin_mse_meter.avg, single_bin_score_mses


def mse_per_bin(predictions, labels, criterion):
    binned_labels = torch.squeeze(assign_bins(labels))

    bin_mses = {}
    for i, _ in enumerate(BIN_LOCATIONS):
        n_labels_in_bin = int(sum(binned_labels == i))
        if n_labels_in_bin == 0:
            continue
        with torch.no_grad():
            bin_mses[i] = (criterion(predictions[binned_labels == i], labels[binned_labels == i]), n_labels_in_bin)

    return bin_mses


if __name__ == '__main__':
    main()
