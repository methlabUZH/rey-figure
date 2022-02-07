import argparse
import copy
import random
import sys
import shutil
import time
import numpy as np
import os
import pandas as pd
from typing import Tuple

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from constants import RESULTS_DIR, BIN_LOCATIONS1, BIN_LOCATIONS2
from src.data_preprocessing.augmentation import AugmentParameters
from src.dataloaders.dataloader_regression import get_regression_dataloader_train
from src.train_utils import directory_setup, plot_scores_preds, count_parameters, AverageMeter, Logger
from src.utils import timestamp_human
from src.inference.utils import assign_bins
from src.models import get_reyregressor

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'

# setup arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=RESULTS_DIR, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--val-fraction', default=0.2, type=float)
parser.add_argument('--eval-test', action='store_true')
parser.add_argument('--id', default=None, type=str)

# architecture
parser.add_argument('--image-size', nargs='+', type=int, default=[116, 150])
parser.add_argument('--norm-layer', type=str, default=None, choices=[None, 'batch_norm', 'group_norm'])

# optimization
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--beta', type=float, default=0.1, help='weight of the score mse for total loss')
parser.add_argument('--gamma', type=float, default=1.0, help='learning rate decay factor')
parser.add_argument('--wd', '--weight-decay', type=float, default=0)
parser.add_argument('--dropout', nargs='+', type=float, default=[0.3, 0.5])
parser.add_argument('--bn-momentum', type=float, default=0.01)
parser.add_argument('--resume', default='', type=str, help='path to results-dir')

# misc
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--score-type', default='sum', type=str, choices=['sum', 'median'])

args = parser.parse_args()

if DEBUG:
    args.epochs = 1
    args.batch_size = 2

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

MODEL_ARCH = 'rey-regressor'


def main():
    # setup dirs for trained model and log data
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=MODEL_ARCH,
                                                   dataset=dataset_name,
                                                   results_dir=args.results_dir,
                                                   args=args, resume=args.resume,
                                                   train_id=args.id)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'out.txt'))

    if DEBUG:
        print('==> debugging on!')

    # read and split labels into train and val
    labels_csv = os.path.join(args.data_root, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)

    if args.val_fraction > 0.0:
        train_labels, val_labels = split_df(labels_df, fraction=args.val_fraction)
    else:
        val_labels = pd.read_csv(os.path.join(args.data_root, 'test_labels.csv'))
        train_labels = labels_df
        print('==> eval on test set')

    train_dataloader = get_regression_dataloader_train(args.data_root, labels_df=train_labels,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers, shuffle=True,
                                                       score_type=args.score_type)
    val_dataloader = get_regression_dataloader_train(args.data_root, labels_df=val_labels, batch_size=args.batch_size,
                                                     num_workers=args.workers, shuffle=False,
                                                     score_type=args.score_type)

    # setup model
    model = get_reyregressor(n_outputs=18, dropout=args.dropout, bn_momentum=args.bn_momentum,
                             norm_layer_type=args.norm_layer)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction="mean")

    # setup optimizer and lr scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    if args.resume:
        print('==> resuming from checkpoint...')
        checkpoint_file = os.path.join(checkpoints_dir, 'checkpoint.pth.tar')
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint found!'
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_epoch = 0
        best_val_loss = np.inf

    # tensorboard
    summary_writer = SummaryWriter(results_dir)

    # print setup
    print('\n----------------------\n')
    for k, v in args.__dict__.items():
        print('{0:20}: {1}'.format(k, v))

    print('{0:20}: {1}'.format('num-train', len(train_dataloader.dataset)))
    print('{0:20}: {1}'.format('num-val', len(val_dataloader.dataset)))
    print('{0:20}: {1}'.format('#params', count_parameters(model)))

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start training')

    best_val_score_mse, best_val_items_loss = np.inf, np.inf
    epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        # train
        t0 = time.time()
        train_total_loss, train_items_loss, train_score_mse = train_epoch(
            train_dataloader, model, criterion, optimizer, summary_writer, epoch + 1)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # validation
        val_total_loss, val_items_loss, val_score_mse = eval_model(
            val_dataloader, model, criterion, summary_writer, epoch + 1)

        print_stats(epoch, args.epochs, epoch_time, lr_scheduler.get_last_lr()[0],
                    train_total_loss, train_items_loss, train_score_mse, val_total_loss, val_items_loss, val_score_mse)

        # add tensorboard summaries
        summary_writer.add_scalars('total-loss', {'train': train_total_loss, 'val': val_total_loss}, global_step=epoch)
        summary_writer.add_scalars('items-loss', {'train': train_items_loss, 'val': val_items_loss}, global_step=epoch)
        summary_writer.add_scalars('score-mse', {'train': train_score_mse, 'val': val_score_mse}, global_step=epoch)
        summary_writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_step=epoch)

        summary_writer.flush()

        # save model
        is_best = val_total_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_total_loss)

        if is_best:
            best_epoch = epoch + 1
            best_val_items_loss = val_items_loss
            best_val_score_mse = val_score_mse

        save_checkpoint({'epoch': epoch + 1,
                         'best_epoch': best_epoch,
                         'val_loss': val_total_loss,
                         'best_val_loss': best_val_loss,
                         'state_dict': copy.deepcopy(model.state_dict()),
                         'optimizer': copy.deepcopy(optimizer.state_dict())
                         }, is_best=is_best, checkpoint_dir=checkpoints_dir)

        # decay learning rate
        lr_scheduler.step()

        if DEBUG:
            break

    summary_writer.flush()
    summary_writer.close()

    print(f'\ntraining finished; average epoch time: {np.average(epoch_times):.4f}s')

    ####################################################################################################################
    # Evaluation
    ####################################################################################################################

    print('\n-----------------------')
    print('** validation stats **')
    print('best {0:25}: {1:.4f}'.format('epoch', best_epoch))
    print('best {0:25}: {1:.4f}'.format('total loss', best_val_loss))
    print('best {0:25}: {1:.4f}'.format('items loss', best_val_items_loss))
    print('best {0:25}: {1:.4f}'.format('score mse', best_val_score_mse))

    if args.eval_test:
        print('\n==> evaluating best model on test set...')
        if args.val_fraction > 0:
            ckpt = os.path.join(checkpoints_dir, 'model_best.pth.tar')
        else:
            ckpt = os.path.join(checkpoints_dir, 'checkpoint.pth.tar')
        print(f'==> checkpoint: {ckpt}')

        total_loss, item_loss, score_mse, bin1_mse, bin2_mse, n_test = eval_test(model, criterion, args.data_root, ckpt)

        print('\n-----------------------')
        print('** test stats **')
        print('{0:25}: {1:.4f}'.format('# test samples', n_test))
        print('{0:25}: {1:.4f}'.format('total loss', total_loss))
        print('{0:25}: {1:.4f}'.format('items loss', item_loss))
        print('{0:25}: {1:.4f}'.format('score mse', score_mse))
        print('{0:25}: {1:.4f}'.format('bin1 mse', bin1_mse))
        print('{0:25}: {1:.4f}'.format('bin2 mse', bin2_mse))


def train_epoch(dataloader, model, criterion, optimizer, summary_writer, epoch):
    model.train()

    total_loss_meter = AverageMeter()
    items_loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

        # compute output
        outputs = model(images.float())
        items_loss = criterion(outputs[:, :-1], labels[:, :-1])
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        total_loss = items_loss + args.beta * score_mse

        # record loss
        total_loss_meter.update(float(total_loss.data), images.size()[0])
        items_loss_meter.update(float(items_loss.data), images.size()[0])
        score_mse_meter.update(float(score_mse.data), images.size()[0])

        # set grads to zero; this is fater than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # compute grads and make train step
        total_loss.backward()
        optimizer.step()

        if batch_idx == 0 and epoch == 1:
            record_images(images, summary_writer)  # noqa

        if DEBUG:
            print(f'train batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    return total_loss_meter.avg, items_loss_meter.avg, score_mse_meter.avg


def eval_model(dataloader, model, criterion, summary_writer, epoch):
    model.eval()

    total_loss_meter = AverageMeter()
    items_loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        items_loss = criterion(outputs[:, :-1], labels[:, :-1])
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        total_loss = items_loss + args.beta * score_mse

        # record loss
        total_loss_meter.update(total_loss.data, images.size()[0])
        items_loss_meter.update(items_loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])

        if batch_idx == 0 and epoch == 0:
            record_images(images, summary_writer, name='validation-images')
            summary_writer.add_figure('predictions', plot_scores_preds(model, images, labels))

        if DEBUG:
            print(f'val batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    return total_loss_meter.avg, items_loss_meter.avg, score_mse_meter.avg


def eval_test(model, criterion, data_root, checkpoint):
    # data
    labels_csv = os.path.join(data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_regression_dataloader_train(args.data_root, labels_df=labels, batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False, score_type=args.score_type)

    num_test_samples = len(dataloader.dataset)

    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    total_loss_meter = AverageMeter()
    items_loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    bin1_mse_meter = AverageMeter()
    bin2_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        items_loss = criterion(outputs[:, :-1], labels[:, :-1])
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        total_loss = items_loss + args.beta * score_mse
        bin1_mse = criterion(assign_bins(scores=outputs[:, -1], bin_locations=BIN_LOCATIONS1),
                             assign_bins(scores=labels[:, -1], bin_locations=BIN_LOCATIONS1))
        bin2_mse = criterion(assign_bins(scores=outputs[:, -1], bin_locations=BIN_LOCATIONS2),
                             assign_bins(scores=labels[:, -1], bin_locations=BIN_LOCATIONS2))

        # record loss
        total_loss_meter.update(total_loss.data, images.size()[0])
        items_loss_meter.update(items_loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        bin1_mse_meter.update(bin1_mse.data, images.size()[0])
        bin2_mse_meter.update(bin2_mse.data, images.size()[0])

        if DEBUG:
            print(f'val batch={batch_idx}')
            if batch_idx >= 5:
                break

    return (total_loss_meter.avg, items_loss_meter.avg, score_mse_meter.avg, bin1_mse_meter.avg, bin2_mse_meter.avg,
            num_test_samples)


def split_df(labels_df, fraction) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels_df_original = labels_df[labels_df.augmented == False]  # noqa
    labels_df_augmented = labels_df[labels_df.augmented == True]  # noqa
    num_original_datapoints = len(labels_df_original)

    train_indices = random.sample(list(range(num_original_datapoints)), k=int(num_original_datapoints * (1 - fraction)))
    val_indices = [i for i in range(num_original_datapoints) if i not in train_indices]
    assert set(train_indices).isdisjoint(val_indices)

    train_df_original = labels_df_original.iloc[train_indices]
    train_figure_ids = []

    for fid in train_df_original.figure_id.to_list():
        train_figure_ids.append(fid)
        for i in range(AugmentParameters.num_augment):
            train_figure_ids.append(fid + f'_augm{i + 1}')

    train_df = labels_df[labels_df.figure_id.isin(train_figure_ids)]
    val_df = labels_df_original.iloc[val_indices]

    return train_df, val_df


def record_images(images, writer, name='training-images'):
    images = images.cpu()
    height, width = images.size()[-2:]
    images = images.view(images.size(0), -1)
    images -= images.min(1, keepdim=True)[0]
    images /= images.max(1, keepdim=True)[0]
    images = images.view(images.size(0), 1, height, width)
    writer.add_images(name, images)


def print_stats(epoch, total_epochs, epoch_time, lr, train_total_loss, train_items_loss, train_score_mse,
                val_total_loss, val_items_loss, val_score_mse):
    def _print_str(mode, total_loss, items_loss, score_mse):
        return f'|| {mode} total loss={total_loss:.6f}, items loss={items_loss:.6f}, score-mse={score_mse:.4f} '

    print_str = f'[{timestamp_human()} | {epoch + 1}/{total_epochs}] epoch time: {epoch_time:.2f}, lr: {lr:.6f} '
    print_str += _print_str('train', train_total_loss, train_items_loss, train_score_mse)
    print_str += _print_str('val', val_total_loss, val_items_loss, val_score_mse)
    print(print_str)


def save_checkpoint(training_state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(training_state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
