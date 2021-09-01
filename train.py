import argparse
import os.path
import json
import random
import sys
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from constants import LABEL_FORMATS, RESULTS_DIR
from src.training.data_loader import get_dataloader
from src.training.helpers import directory_setup, timestamp_human, plot_scores_preds, count_parameters, assign_bins
from src.training.helpers import AverageMeter, Logger, TrainingLogger
from src.models.model_factory import get_architecture
from src.training.hyperparams import get_train_setup

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224'

# setup arg parser
parser = argparse.ArgumentParser(description='Process some integers.')

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=RESULTS_DIR, required=False)
parser.add_argument('--workers', default=8, type=int)

# architecture
parser.add_argument('--arch', type=str, default='efficientnet-l2', required=False)
parser.add_argument('--label-format', type=str, default='items-median-scores', choices=LABEL_FORMATS)
parser.add_argument('--image-size', nargs='+', type=int, default=[224, 224])

# misc
parser.add_argument('--seed', default=7, type=int)

args = parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


def main():
    # setup dirs for trained model and log data
    results_dir, checkpoints_dir = directory_setup(model_name=args.arch, label_format=args.label_format,
                                                   results_dir=args.results_dir)

    # logging
    logger = TrainingLogger(fpath=os.path.join(results_dir, 'log.txt'))
    # logger.set_names(['Learning Rate', 'Train Loss', 'Train MSE Median-Score', 'Train MSE Median-Score Bin',
    #                   'Valid Loss', 'Valid MSE Median-Score', 'Valid MSE Sum-Score', 'Valid MSE Sum-to-Sum-Score'])
    logger.set_names(['Learning Rate', 'Train Loss', 'Train MSE Median-Score', 'Train MSE Median-Score Bin',
                      'Valid Loss', 'Valid MSE Median-Score', 'Valid MSE Median-Score Bin'])

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'out.txt'))

    print(f'seed:\t{args.seed}')

    # train setup
    train_setup = get_train_setup(args.arch)
    print('\n----------------------\n')
    for k, v in train_setup.__dict__.items():
        print(f'-{k}: {v}')
    print(f'-arch: {args.arch}')
    print(f'-data-root: {args.data_root}')

    if DEBUG:
        train_setup.batch_size = 2

    # read train set mean and std for normalization
    with open(os.path.join(args.data_root, 'trainset-stats.json'), 'r') as f:
        train_set_statistics = json.load(f)
        mean = train_set_statistics['mean']
        std = train_set_statistics['std']

    # get dataloaders
    train_labels_csv = os.path.join(args.data_root, 'train_labels.csv')
    train_data_loader = get_dataloader(data_root=args.data_root, labels_csv=train_labels_csv,
                                       batch_size=train_setup.batch_size, num_workers=args.workers, shuffle=True,
                                       mean=mean, std=std)

    test_labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    test_data_loader = get_dataloader(data_root=args.data_root, labels_csv=test_labels_csv,
                                      batch_size=train_setup.batch_size, num_workers=args.workers, shuffle=True,
                                      mean=mean, std=std)

    # setup model
    model = get_architecture(arch=args.arch, num_outputs=19, dropout=train_setup.dropout,
                             track_running_stats=train_setup.track_running_stats, image_size=args.image_size)

    print(f'-#params:\t{count_parameters(model)}')

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction="mean")

    # setup optimizer and learning rate schedule
    optimizer = train_setup.optimizer(model.parameters())
    lr_scheduler = train_setup.get_lr_scheduler(optimizer)

    # tensorboard
    writer = SummaryWriter(results_dir)

    if DEBUG:
        print('* debugging on!')

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start training')

    best_test_loss = np.inf
    best_mse_score = np.inf

    for epoch in range(train_setup.epochs):

        # train and eval step
        train_loss, train_mse_score1, train_bin_mse1, epoch_time = train(train_data_loader, model, criterion, optimizer,
                                                                         writer)
        test_loss, test_mse_score1, test_bin_mse1, images, labels = test(test_data_loader, model, criterion, writer)

        lr_scheduler.step()
        train_setup.learning_rate = lr_scheduler.get_last_lr()[0]

        # log progress
        logger.append([train_setup.learning_rate, train_loss, train_mse_score1, train_bin_mse1, test_loss,
                       test_mse_score1, test_bin_mse1])
        print_stats(epoch, epoch_time, train_loss, train_mse_score1, train_bin_mse1, test_loss,
                    test_mse_score1, test_bin_mse1, train_setup.learning_rate)

        # add tensorboard summaries
        writer.add_scalars('items-mse', {'train': train_loss, 'test': test_loss}, global_step=epoch)
        writer.add_scalars('mse-median-score',
                           {'train': train_mse_score1, 'test': test_mse_score1},
                           global_step=epoch)
        writer.add_scalars('mse-bin-score',
                           {'train': train_bin_mse1, 'test': test_bin_mse1},
                           global_step=epoch)
        writer.add_figure('predictions', plot_scores_preds(model, images, labels, use_cuda), global_step=epoch)
        writer.flush()

        # save model
        is_best = test_loss < best_test_loss
        best_test_loss = min(best_test_loss, test_loss)
        best_mse_score = min(test_mse_score1, best_mse_score)
        save_checkpoint(training_state={'epoch': epoch + 1,
                                        'state_dict': model.state_dict(),
                                        'loss': test_loss,
                                        'mse score': test_mse_score1,
                                        'best mse items': best_test_loss,
                                        'optimizer': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint_dir=checkpoints_dir)

        if DEBUG:
            break

    logger.close()
    writer.flush()
    writer.close()

    print('\n-----------------------')
    print(f'Best Test MSE Items:\t{best_test_loss}')
    print(f'Best Test MSE SCore:\t{best_mse_score}')


def train(trainloader, model, criterion, optimizer, summary_writer):
    model.train()

    items_loss_meter = AverageMeter()
    score_loss_meter = AverageMeter()
    bin_mse_meter = AverageMeter()

    t0 = time.time()

    for batch_idx, (images, labels, median_scores, sum_scores) in enumerate(trainloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
            median_scores, sum_scores = median_scores.cuda(), sum_scores.cuda()

        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
        median_scores, sum_scores = torch.autograd.Variable(median_scores), torch.autograd.Variable(sum_scores)

        # compute output
        outputs = model(images.float())
        loss_items = criterion(outputs, labels)
        mse_score = criterion(outputs[:, -1], median_scores)
        bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(median_scores))

        # record loss
        items_loss_meter.update(loss_items.data, images.size()[0])
        score_loss_meter.update(mse_score.data, images.size()[0])
        bin_mse_meter.update(bin_mse.data, images.size()[0])

        # set grads to zero; this is fater than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # compute grads and make train step
        loss_items.backward()
        optimizer.step()

        if batch_idx % 15 == 0:
            record_images(images, summary_writer)  # noqa

        if DEBUG:
            print(f'test batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    epoch_time = time.time() - t0

    return items_loss_meter.avg, score_loss_meter.avg, bin_mse_meter.avg, epoch_time


def test(testloader, model, criterion, summary_writer):
    model.eval()

    items_loss_meter = AverageMeter()
    score1_loss_meter = AverageMeter()
    bin_mse_meter = AverageMeter()
    # score2_loss_meter = AverageMeter()
    # score3_loss_meter = AverageMeter()

    images, labels = None, None

    for batch_idx, (images, labels, median_scores, sum_scores) in enumerate(testloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
            median_scores, sum_scores = median_scores.cuda(), sum_scores.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        loss_items = criterion(outputs, labels)
        loss_score1 = criterion(outputs[:, -1], median_scores)
        bin_mse1 = criterion(assign_bins(outputs[:, -1]), assign_bins(median_scores))
        # loss_score2 = criterion(outputs[:, -1], sum_scores)
        # loss_score3 = criterion(torch.sum(outputs[:, :-1], dim=1), sum_scores)

        # record loss
        items_loss_meter.update(loss_items.data, images.size()[0])
        score1_loss_meter.update(loss_score1.data, images.size()[0])
        bin_mse_meter.update(bin_mse1.data, images.size()[0])
        # score2_loss_meter.update(loss_score2.data, images.size()[0])
        # score3_loss_meter.update(loss_score3.data, images.size()[0])

        if batch_idx % 15 == 0:
            record_images(images, summary_writer, name='validation-images')  # noqa

        if DEBUG:
            print(f'val batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    return items_loss_meter.avg, score1_loss_meter.avg, bin_mse_meter.avg, images, labels


def record_images(images, writer, name='training-images'):
    images = images.cpu()
    height, width = images.size()[-2:]
    images = images.view(images.size(0), -1)
    images -= images.min(1, keepdim=True)[0]
    images /= images.max(1, keepdim=True)[0]
    images = images.view(images.size(0), 1, height, width)
    writer.add_images(name, images)


def print_stats(epoch, epoch_time, train_loss_items, train_loss_score, train_bin_mse1, val_loss_items, val_loss_score1,
                val_bin_mse1, learning_rate):
    print_str = f'[{timestamp_human()}]\tepoch {epoch + 1} | epoch time: {epoch_time:.2f} '
    print_str += f'| train loss items: {train_loss_items:.4f} | train mse median-score: {train_loss_score:.4f} '
    print_str += f'| train mse bin-score: {train_bin_mse1:.4f}  | val loss items: {val_loss_items:.4f} '
    print_str += f'| val mse median-score: {val_loss_score1:.4f} | val mse bin-score: {val_bin_mse1:.4f} '
    print_str += f'| learning rate: {learning_rate:.4f}'
    print(print_str)


def save_checkpoint(training_state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(training_state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
