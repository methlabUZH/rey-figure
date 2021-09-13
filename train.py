import argparse
import copy
import pandas as pd
from filelock import FileLock
import random
import sys
import shutil
import time
import os

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from constants import RESULTS_DIR
from src.training.data_loader import get_dataloader
from src.training.helpers import directory_setup, timestamp_human, plot_scores_preds, count_parameters, assign_bins
from src.training.helpers import AverageMeter, Logger, TrainingLogger
from src.models.model_factory import get_architecture

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224'

"""
hyperparams:
    resnet18 / resnext29 with lr=0.01, adam, gamma=0.99, 250 epochs, wd=5e-4, batch_norm, bs=256/128
"""

# setup arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=RESULTS_DIR, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--val-fraction', default=0.0, type=float)

# architecture
parser.add_argument('--arch', type=str, default='resnet18', required=False)
parser.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
parser.add_argument('--norm-layer', type=str, default='batch_norm', choices=['batch_norm', 'group_norm'])

# optimization
parser.add_argument('--epochs', default=250, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, help='train batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='learning rate decay factor')
parser.add_argument('--wd', '--weight-decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--resume', default='', type=str, help='path to results-dir')

# misc
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--score-type', default='sum', type=str, choices=['sum', 'median'])
parser.add_argument('--finetune-file', default=None, type=str)

args = parser.parse_args()

if DEBUG:
    args.epochs = 1
    args.batch_size = 2
    args.finetune_file = './finetune_stats.txt'

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


def main():
    # setup dirs for trained model and log data
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=args.arch,
                                                   score_type=args.score_type,
                                                   dataset=dataset_name,
                                                   results_dir=args.results_dir,
                                                   resume=args.resume)

    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(results_dir, 'out.txt'))

    if DEBUG:
        print('==> debugging on!')

    # read and split labels into train and val
    labels_csv = os.path.join(args.data_root, 'train_labels.csv')
    labels_df = pd.read_csv(labels_csv)

    if args.val_fraction > 0.0:
        val_labels = labels_df.sample(frac=args.val_fraction)
        train_labels = labels_df.drop(val_labels.index)
    else:
        val_labels = pd.read_csv(os.path.join(args.data_root, 'test_labels.csv'))
        train_labels = labels_df
        print('==> eval on test set')

    train_dataloader = get_dataloader(args.data_root, labels_df=train_labels, batch_size=args.batch_size,
                                      num_workers=args.workers, shuffle=True, score_type=args.score_type)
    val_dataloader = get_dataloader(args.data_root, labels_df=val_labels, batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False, score_type=args.score_type)

    # setup model
    model = get_architecture(arch=args.arch, num_outputs=18, dropout=args.dropout, norm_layer=args.norm_layer,
                             image_size=args.image_size)
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
        logger = TrainingLogger(fpath=os.path.join(results_dir, 'log.txt'), resume=True)
    else:
        start_epoch = 0
        best_epoch = 0
        best_val_loss = np.inf
        logger = TrainingLogger(fpath=os.path.join(results_dir, 'log.txt'), resume=False)
        logger.set_names(['learning rate', 'train Loss', 'train score-mse', 'train bin-mse', 'valid loss',
                          'valid score-mse', 'val bin-mse'])

    # tensorboard
    writer = SummaryWriter(results_dir)

    # print setup
    print('\n----------------------\n')
    for k, v in args.__dict__.items():
        print('{0:20}: {1}'.format(k, v))

    print('{0:20}: {1}'.format('num-train', len(train_dataloader.dataset)))
    print('{0:20}: {1}'.format('num-val', len(val_dataloader.dataset)))
    print('{0:20}: {1}'.format('#params', count_parameters(model)))

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start training')

    best_val_score_mse, best_val_bin_mse = np.inf, np.inf
    best_train_loss, best_train_score_mse, best_train_bin_mse = np.inf, np.inf, np.inf

    for epoch in range(start_epoch, args.epochs):

        # train
        t0 = time.time()
        train_loss, train_score_mse, train_bin_mse = train(train_dataloader, model, criterion, optimizer, writer,
                                                           epoch + 1)
        epoch_time = time.time() - t0

        # validation
        val_loss, val_score_mse, val_bin_mse = val(val_dataloader, model, criterion, writer, epoch + 1)

        # log progress
        logger.append([lr_scheduler.get_last_lr()[0], train_loss, train_score_mse, train_bin_mse, val_loss,
                       val_score_mse, val_bin_mse])

        print_stats(epoch, args.epochs, epoch_time, lr_scheduler.get_last_lr()[0],
                    train_loss, train_score_mse, train_bin_mse,
                    val_loss, val_score_mse, val_bin_mse)

        # add tensorboard summaries
        writer.add_scalars('items-mse', {'train': train_loss, 'val': val_loss}, global_step=epoch)
        writer.add_scalars('score-mse', {'train': train_score_mse, 'val': val_score_mse}, global_step=epoch)
        writer.add_scalars('bin-mse', {'train': train_bin_mse, 'val': val_bin_mse}, global_step=epoch)
        writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_step=epoch)

        writer.flush()

        # save model
        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        if is_best:
            best_epoch = epoch + 1
            best_val_score_mse = val_score_mse
            best_val_bin_mse = val_bin_mse
            best_train_loss = train_loss
            best_train_score_mse = train_score_mse
            best_train_bin_mse = train_bin_mse

        save_checkpoint({'epoch': epoch + 1,
                         'best_epoch': best_epoch,
                         'val_loss': val_loss,
                         'best_val_loss': best_val_loss,
                         'state_dict': copy.deepcopy(model.state_dict()),
                         'optimizer': copy.deepcopy(optimizer.state_dict())
                         }, is_best=is_best, checkpoint_dir=checkpoints_dir)

        # decay learning rate
        lr_scheduler.step()

        if DEBUG:
            break

    logger.close()
    writer.flush()
    writer.close()

    print('\ntraining finished.')

    print('\n-----------------------')
    print('** validation stats **')
    print('best {0:25}: {1:.4f}'.format('epoch', best_epoch))
    print('best {0:25}: {1:.4f}'.format('loss', best_val_loss))
    print('best {0:25}: {1:.4f}'.format('bin-mse', best_val_bin_mse))
    print('best {0:25}: {1:.4f}'.format('score-mse', best_val_score_mse))

    print('\n==> evaluating best model on test set...')

    ckpt = os.path.join(checkpoints_dir, 'model_best.pth.tar')
    test_loss, test_score_mse, test_bin_mse = eval_test(model, criterion, args.data_root, ckpt)

    print('\n-----------------------')
    print('** test stats **')
    print('{0:25}: {1:.4f}'.format('loss', test_loss))
    print('{0:25}: {1:.4f}'.format('bin-mse', test_bin_mse))
    print('{0:25}: {1:.4f}'.format('score-mse', test_score_mse))

    # write results to file
    if args.finetune_file is not None:
        store_stats(best_train_loss, best_val_loss, test_loss, best_train_score_mse, best_val_score_mse, test_score_mse,
                    best_train_bin_mse, best_val_bin_mse, test_bin_mse, best_epoch)


def train(dataloader, model, criterion, optimizer, summary_writer, epoch):
    model.train()

    loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    bin_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

        # compute output
        outputs = model(images.float())
        loss = criterion(outputs, labels)
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(labels[:, -1]))

        # record loss
        loss_meter.update(float(loss.data), images.size()[0])
        score_mse_meter.update(float(score_mse.data), images.size()[0])
        bin_mse_meter.update(float(bin_mse.data), images.size()[0])

        # set grads to zero; this is fater than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # compute grads and make train step
        loss.backward()
        optimizer.step()

        if batch_idx == 0 and epoch == 1:
            record_images(images, summary_writer)  # noqa

        if DEBUG:
            print(f'train batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    return loss_meter.avg, score_mse_meter.avg, bin_mse_meter.avg


def val(dataloader, model, criterion, summary_writer, epoch):
    model.eval()

    loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    bin_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        loss = criterion(outputs, labels)
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(labels[:, -1]))

        # record loss
        loss_meter.update(loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        bin_mse_meter.update(bin_mse.data, images.size()[0])

        if batch_idx == 0 and epoch == 0:
            record_images(images, summary_writer, name='validation-images')
            summary_writer.add_figure('predictions', plot_scores_preds(model, images, labels, use_cuda))

        if DEBUG:
            print(f'val batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    return loss_meter.avg, score_mse_meter.avg, bin_mse_meter.avg


def eval_test(model, criterion, data_root, checkpoint):
    # data
    labels_csv = os.path.join(data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_dataloader(args.data_root, labels_df=labels, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=False, score_type=args.score_type)

    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    # checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    bin_mse_meter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float())

        loss = criterion(outputs, labels)
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(labels[:, -1]))

        # record loss
        loss_meter.update(loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        bin_mse_meter.update(bin_mse.data, images.size()[0])

        if DEBUG:
            print(f'val batch={batch_idx}')
            if batch_idx >= 5:
                break

    return loss_meter.avg, score_mse_meter.avg, bin_mse_meter.avg


def store_stats(train_loss, val_loss, test_loss, train_score_mse, val_score_mse, test_score_mse, train_bin_mse,
                val_bin_mse, test_bin_mse, best_epoch):
    with FileLock(args.finetune_file + '.lock'):
        if not os.path.isfile(args.finetune_file):
            with open(args.finetune_file, 'a') as f:
                # write header
                cols = [k for k, _ in sorted(args.__dict__.items())]
                cols += ['train-loss', 'val-loss', 'test-loss', 'train-score-mse', 'val-score-mse', 'test-score-mse',
                         'train-bin-mse', 'val-bin-mse', 'test-bin-mse', 'best epoch']
                f.write(','.join(cols) + '\n')

        with open(args.finetune_file, 'a') as f:
            # write data
            data = [v for _, v in sorted(args.__dict__.items())]
            data += [train_loss, val_loss, test_loss, train_score_mse, val_score_mse, test_score_mse, train_bin_mse,
                     val_bin_mse, test_bin_mse, best_epoch]
            f.write(','.join([str(v) for v in data]) + '\n')


def record_images(images, writer, name='training-images'):
    images = images.cpu()
    height, width = images.size()[-2:]
    images = images.view(images.size(0), -1)
    images -= images.min(1, keepdim=True)[0]
    images /= images.max(1, keepdim=True)[0]
    images = images.view(images.size(0), 1, height, width)
    writer.add_images(name, images)


def print_stats(epoch, total_epochs, epoch_time, lr, train_loss_items, train_score_mse, train_bin_mse, val_loss_items,
                val_score_mse, val_bin_mse):
    def _print_str(mode, loss, score_mse, bin_mse):
        return f'|| {mode} loss={loss:.6f}, score-mse={score_mse:.4f}, bin_mse={bin_mse:.4f} '

    print_str = f'[{timestamp_human()} | {epoch + 1}/{total_epochs}] epoch time: {epoch_time:.2f}, lr: {lr:.6f} '
    print_str += _print_str('train', train_loss_items, train_score_mse, train_bin_mse)
    print_str += _print_str('val', val_loss_items, val_score_mse, val_bin_mse)
    print(print_str)


def save_checkpoint(training_state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(training_state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
