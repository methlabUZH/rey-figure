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

from src.preprocessing.augmentation import AugmentParameters
from old_code.dataloaders.dataloader_item_classification import get_item_classification_dataloader
from src.training.train_utils import directory_setup, plot_scores_preds, count_parameters, AverageMeter, Logger, accuracy
from src.utils import timestamp_human
from src.models import get_reyclassifier

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/rey-figure/data/serialized-data/debug-116x150-pp0'
default_data_dir = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'

# setup arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default='./temp', required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--val-fraction', default=0.2, type=float)
parser.add_argument('--eval-test', action='store_true')
parser.add_argument('--item', type=int, default=1)
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--is-binary', default=0, type=int, choices=[0, 1])

# architecture
parser.add_argument('--image-size', nargs='+', type=int, default=[116, 150])
parser.add_argument('--norm-layer', type=str, default='batch_norm', choices=[None, 'batch_norm', 'group_norm'])

# optimization
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=2, type=int, help='train batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--gamma', type=float, default=1.0, help='learning rate decay factor')
parser.add_argument('--wd', '--weight-decay', type=float, default=0)
parser.add_argument('--dropout', nargs='+', type=float, default=[0.3, 0.5])
parser.add_argument('--bn-momentum', type=float, default=0.01)
parser.add_argument('--resume', default='', type=str, help='path to results-dir')
parser.add_argument('--weighted-sampling', default=1, type=int, choices=[0, 1])

# misc
parser.add_argument('--seed', default=7, type=int)

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


def main():
    num_classes = 2 if args.is_binary else 4

    # setup dirs for trained model and log data
    dataset_name = os.path.split(os.path.normpath(args.data_root))[-1]
    results_dir, checkpoints_dir = directory_setup(model_name=f'{num_classes}-way-item-classifier/item-{args.item}',
                                                   dataset=dataset_name,
                                                   results_dir=args.results_dir,
                                                   args=args,
                                                   train_id=args.id,
                                                   resume=args.resume)

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

    train_dataloader = get_item_classification_dataloader(args.item, args.data_root, labels_df=train_labels,
                                                          batch_size=args.batch_size, num_workers=args.workers,
                                                          shuffle=True, weighted_sampling=args.weighted_sampling,
                                                          is_train=True, is_binary=args.is_binary)

    val_dataloader = get_item_classification_dataloader(args.item, args.data_root, labels_df=val_labels,
                                                        batch_size=args.batch_size, num_workers=args.workers,
                                                        shuffle=False, weighted_sampling=False, is_train=False,
                                                        is_binary=args.is_binary)

    train_class_counts = train_dataloader.dataset.get_class_counts()
    val_class_counts = val_dataloader.dataset.get_class_counts()

    # setup model
    model = get_reyclassifier(dropout=args.dropout, num_clases=num_classes, bn_momentum=args.bn_momentum,
                              norm_layer_type=args.norm_layer)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

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
        best_val_acc = checkpoint['best_val_acc']
        best_val_sensitivity = checkpoint['best_val_sensitivity']
        best_val_specificity = checkpoint['best_val_specificity']
        best_val_g_mean = checkpoint['best_val_g_mean']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_epoch = 0
        best_val_loss = np.inf
        best_val_acc = -np.inf
        best_val_specificity = -np.inf
        best_val_sensitivity = -np.inf
        best_val_g_mean = -np.inf

    # tensorboard
    summary_writer = SummaryWriter(results_dir)

    # print setup
    print('\n----------------------\n')
    for k, v in args.__dict__.items():
        print('{0:27}: {1}'.format(k, v))

    n_train = sum(train_class_counts)
    n_val = sum(val_class_counts)
    print('{}: {}'.format('train class distribution', [c / n_train for c in train_class_counts]))
    print('{}: {}'.format('validation class distribution', [vc / n_val for vc in val_class_counts]))
    print('{0:27}: {1}'.format('num-train', len(train_dataloader.dataset)))
    print('{0:27}: {1}'.format('num-val', len(val_dataloader.dataset)))
    print('{0:27}: {1}'.format('#params', count_parameters(model)))

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start training')

    epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        # train
        t0 = time.time()
        train_loss, train_acc, train_sensitivity, train_specificity, train_g_mean = train_epoch(
            train_dataloader, model, criterion, optimizer, summary_writer, epoch + 1)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # validation
        val_loss, val_acc, val_sensitivity, val_specificity, val_g_mean = eval_model(
            val_dataloader, model, criterion, summary_writer, epoch + 1)

        print_stats(epoch, args.epochs, epoch_time, lr_scheduler.get_last_lr()[0],
                    train_loss, train_acc, train_sensitivity, train_specificity, train_g_mean,
                    val_loss, val_acc, val_sensitivity, val_specificity, val_g_mean)

        # add tensorboard summaries
        summary_writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, global_step=epoch)
        summary_writer.add_scalars('accuracy', {'train': train_acc, 'val': val_acc}, global_step=epoch)
        summary_writer.add_scalars('sensitivity',
                                   {'train': train_sensitivity, 'val': val_sensitivity}, global_step=epoch)
        summary_writer.add_scalars('specificity',
                                   {'train': train_specificity, 'val': val_specificity}, global_step=epoch)
        summary_writer.add_scalars('g-mean', {'train': train_g_mean, 'val': val_g_mean}, global_step=epoch)
        summary_writer.add_scalar('learning-rate', lr_scheduler.get_last_lr()[0], global_step=epoch)
        summary_writer.flush()

        # save model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_val_sensitivity = val_sensitivity
            best_val_specificity = val_specificity
            best_val_g_mean = val_g_mean

        save_checkpoint({'epoch': epoch + 1,
                         'best_epoch': best_epoch,
                         'val_acc': val_acc,
                         'best_val_acc': best_val_acc,
                         'best_val_specificity': best_val_specificity,
                         'best_val_sensitivity': best_val_sensitivity,
                         'best_val_g_mean': best_val_g_mean,
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

    print('\n-----------------------')
    print('** early stop validation stats **')
    print('{0:25}: {1:.4f}'.format('epoch', best_epoch))
    print('{0:25}: {1:.4f}'.format('loss', best_val_loss))
    print('{0:25}: {1:.4f}%'.format('accuracy', best_val_acc))
    print('{0:25}: {1:.4f}'.format('sensitivity', best_val_sensitivity))
    print('{0:25}: {1:.4f}'.format('specificity', best_val_specificity))
    print('{0:25}: {1:.4f}'.format('g-mean', best_val_g_mean))

    if args.eval_test:
        print('\n==> evaluating best model on test set...')
        if args.val_fraction > 0:
            ckpt = os.path.join(checkpoints_dir, 'model_best.pth.tar')
        else:
            ckpt = os.path.join(checkpoints_dir, 'checkpoint.pth.tar')
        print(f'==> checkpoint: {ckpt}')

        test_loss, test_acc, n_test, sensitivity, specificity, g_mean = eval_test(model, criterion, args.data_root,
                                                                                  ckpt)

        print('\n-----------------------')
        print('** test stats **')
        print('{0:25}: {1:.4f}'.format('# test samples', n_test))
        print('{0:25}: {1:.4f}'.format('loss', test_loss))
        print('{0:25}: {1:.4f}%'.format('accuracy', test_acc))
        print('{0:25}: {1:.4f}'.format('sensitivity', sensitivity))
        print('{0:25}: {1:.4f}'.format('specificity', specificity))
        print('{0:25}: {1:.4f}'.format('g-mean', g_mean))


def train_epoch(dataloader, model, criterion, optimizer, summary_writer, epoch):
    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    true_positives = true_negatives = false_positives = false_negatives = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)

        # compute output
        class_probs = model(images.float())
        loss = criterion(class_probs, labels)

        # confusion matrix
        _, predicted_classes = torch.topk(class_probs, 1, 1, True, True)
        predicted_classes = torch.squeeze(predicted_classes)
        true_positives += sum(predicted_classes * labels)
        false_positives += sum(predicted_classes * (1 - labels))
        false_negatives += sum((1 - predicted_classes) * labels)
        true_negatives += sum((1 - predicted_classes) * (1 - labels))

        # measure accuracy and record loss
        acc = accuracy(class_probs.data, labels.data, topk=(1,))
        loss_meter.update(loss.data, images.size()[0])
        accuracy_meter.update(acc[0], images.size()[0])

        # set grads to zero; this is faster than optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        # compute grads and make train step
        loss.backward()
        optimizer.step()

        if batch_idx == 0 and epoch == 1:
            record_images(images, summary_writer)

        if DEBUG:
            print(f'train batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    g_mean = np.sqrt(sensitivity.cpu() * specificity.cpu())

    return loss_meter.avg, accuracy_meter.avg, sensitivity, specificity, g_mean


def eval_model(dataloader, model, criterion, summary_writer, epoch):
    model.run_eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    true_positives = true_negatives = false_positives = false_negatives = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            class_probs = model(images.float())

        loss = criterion(class_probs, labels)

        # confusion matrix
        _, predicted_classes = torch.topk(class_probs, 1, 1, True, True)
        predicted_classes = torch.squeeze(predicted_classes)
        true_positives += sum(predicted_classes * labels)
        false_positives += sum(predicted_classes * (1 - labels))
        false_negatives += sum((1 - predicted_classes) * labels)
        true_negatives += sum((1 - predicted_classes) * (1 - labels))

        # measure accuracy and record loss
        acc = accuracy(class_probs.data, labels.data, topk=(1,))
        loss_meter.update(loss.data, images.size()[0])
        accuracy_meter.update(acc[0], images.size()[0])

        if batch_idx == 0 and epoch == 0:
            record_images(images, summary_writer, name='validation-images')
            summary_writer.add_figure('predictions', plot_scores_preds(model, images, labels))

        if DEBUG:
            print(f'val batch={batch_idx}')

        if batch_idx >= 5 and DEBUG:
            break

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    g_mean = np.sqrt(sensitivity.cpu() * specificity.cpu())

    return loss_meter.avg, accuracy_meter.avg, sensitivity, specificity, g_mean


def eval_test(model, criterion, data_root, checkpoint):
    # data
    labels_csv = os.path.join(data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_item_classification_dataloader(args.item, args.data_root, labels_df=labels,
                                                    batch_size=args.batch_size, num_workers=args.workers,
                                                    shuffle=False)

    num_test_samples = len(dataloader.dataset)

    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.run_eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    true_positives = true_negatives = false_positives = false_negatives = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            class_probs = model(images.float())

        loss = criterion(class_probs, labels)

        # confusion matrix
        _, predicted_classes = torch.topk(class_probs, 1, 1, True, True)
        predicted_classes = torch.squeeze(predicted_classes)
        true_positives += sum(predicted_classes * labels)
        false_positives += sum(predicted_classes * (1 - labels))
        false_negatives += sum((1 - predicted_classes) * labels)
        true_negatives += sum((1 - predicted_classes) * (1 - labels))

        # measure accuracy and record loss
        acc = accuracy(class_probs.data, labels.data, topk=(1,))
        loss_meter.update(loss.data, images.size()[0])
        accuracy_meter.update(acc[0], images.size()[0])

        if DEBUG:
            print(f'val batch={batch_idx}')
            if batch_idx >= 5:
                break

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    g_mean = np.sqrt(sensitivity.cpu() * specificity.cpu())

    return loss_meter.avg, accuracy_meter.avg, num_test_samples, sensitivity, specificity, g_mean


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


def print_stats(epoch, total_epochs, epoch_time, lr, train_loss, train_acc, train_sensitivity, train_specificity,
                train_g_mean, val_loss, val_acc, val_sensitivity, val_specificity, val_g_mean):
    def _print_str(mode, loss, acc, sens, spec, gm):
        s = f'|| {mode} loss={loss:.6f}, acc={acc:.4f}%, sensitivity={sens:.4f}, '
        s += f'specificity={spec:.4f}, g-mean={gm:.4f}'
        return s

    print_str = f'[{timestamp_human()} | {epoch + 1}/{total_epochs}] epoch time: {epoch_time:.2f}, lr: {lr:.6f} '
    print_str += _print_str('train', train_loss, train_acc, train_sensitivity, train_specificity, train_g_mean)
    print_str += _print_str('val', val_loss, val_acc, val_sensitivity, val_specificity, val_g_mean)
    print(print_str)


def save_checkpoint(training_state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(training_state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
