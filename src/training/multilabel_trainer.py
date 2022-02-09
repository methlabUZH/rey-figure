import copy
import shutil
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate

import torch
import torch.backends.cudnn as cudnn
from torch import optim, Tensor
from torch.utils.tensorboard import SummaryWriter

from constants import *
from src.training.train_utils import AverageMeter, accuracy
from src.utils import timestamp_human


class MultilabelTrainer:
    def __init__(self, model, loss_func, train_loader, val_loader, args, save_dir, is_binary):
        self.model = model
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.save_dir = save_dir
        self.is_binary = is_binary

        self.summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        # print setup
        print('--------args----------')
        for k, v in args.__dict__.items():
            print('{0:27}: {1}'.format(k, v))
        print('--------args----------\n')

    def initialize(self, is_train):
        if is_train:
            self._init_optimizer_and_scheduler()

        self._init_meters()
        self._init_confusion_matrix()

        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.loss_func = self.loss_func.cuda()

    def _init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)

    def _init_meters(self):
        self.total_loss_meter = AverageMeter()
        self.loss_meters = [AverageMeter() for _ in range(N_ITEMS)]
        self.accuracy_meters = [AverageMeter() for _ in range(N_ITEMS)]
        self.epoch_time_meter = AverageMeter()

    def _reset_meters(self):
        self.total_loss_meter.reset()
        for acc_met, loss_met in zip(self.accuracy_meters, self.loss_meters):
            acc_met.reset()
            loss_met.reset()

    def _update_loss_meters(self, losses, n):
        for i, loss in enumerate(losses):
            self.loss_meters[i].update(loss.data, n)

    def _update_accuracy_meters(self, accuracies, n):
        for i, acc in enumerate(accuracies):
            self.accuracy_meters[i].update(acc.data, n)

    def _init_confusion_matrix(self):
        self.confusion_matrices = [
            {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0}
            for _ in range(N_ITEMS)
        ]

    def train(self):
        self.initialize(is_train=True)

        start_epoch = 0
        best_epoch = 0
        best_val_loss = np.inf

        print(f'[{timestamp_human()}] start training')

        for epoch in range(start_epoch, self.args.epochs):
            # train for one epoch
            train_stats = self.run_epoch(self.train_loader, is_train=True)

            # run validation
            val_stats = self.run_epoch(self.val_loader, is_train=False)

            # save model
            is_best = val_stats['val-total-loss'] > best_val_loss
            if is_best:
                best_val_loss = val_stats['val-total-loss']
                best_epoch = epoch + 1
            self.save_checkpoint(epoch, best_epoch, val_stats['val-total-loss'], best_val_loss, is_best)

            # print stats
            self.print_stats(train_stats, val_stats, epoch)

            # add tensorboard summaries
            self.summary_writer.add_scalars('total-loss',
                                            {'train': train_stats["train-total-loss"],
                                             'val': val_stats["val-total-loss"]}, global_step=epoch)
            self.summary_writer.add_scalars('item-loss',
                                            {**{f'train-{i}': l for i, l in enumerate(train_stats["train-losses"])},
                                             **{f'val-{i}': l for i, l in enumerate(val_stats["val-losses"])}},
                                            global_step=epoch)
            self.summary_writer.add_scalars('item-accuracy',
                                            {**{f'train-{i}': l for i, l in enumerate(train_stats["train-accuracies"])},
                                             **{f'val-{i}': l for i, l in enumerate(val_stats["val-accuracies"])}},
                                            global_step=epoch)
            self.summary_writer.add_scalar('learning-rate', self.lr_scheduler.get_last_lr()[0], global_step=epoch)
            self.summary_writer.flush()

            # decay learning rate
            self.lr_scheduler.step()

        self.summary_writer.flush()
        self.summary_writer.close()

        print(f'\ntraining finished; average epoch time: {self.epoch_time_meter.average():.4f}s\n')

    def print_stats(self, train_stats, val_stats, epoch):
        # build train table
        train_accuracies = train_stats['train-accuracies']
        val_accuracies = val_stats['val-accuracies']
        train_losses = train_stats['train-losses']
        val_losses = val_stats['val-losses']
        data = np.stack([train_accuracies, val_accuracies, train_losses, val_losses], axis=0)
        indices = ['train-acc', 'val-acc', 'train-loss', 'val-loss']

        if self.is_binary:
            train_specificities = train_stats['train-specificities']
            val_specificities = val_stats['val-specificities']
            train_sensitivities = train_stats['train-sensitivities']
            val_sensitivities = val_stats['val-sensitivities']
            train_gmeans = train_stats['train-gmeans']
            val_gmeans = val_stats['val-gmeans']
            data = np.stack([data, train_specificities, val_specificities, train_sensitivities, val_sensitivities,
                             train_gmeans, val_gmeans], axis=0)
            indices += ['train-specificity', 'val-specificity', 'train-sensitivity', 'val-sensitivity', 'train-g-mean',
                        'val-g-mean']

        df = pd.DataFrame(data, columns=[f'item_{i + 1}' for i in range(N_ITEMS)])
        df['index'] = indices
        df = df.set_index('index')

        # print
        learning_rate = self.lr_scheduler.get_last_lr()[0]
        timestamp = timestamp_human()
        print_str = f'\n-- [{timestamp} | {epoch + 1}/{self.args.epochs}] epoch time: {train_stats["epoch_time"]:.2f}, '
        print_str += f'lr: {learning_rate:.6f} --'
        print(print_str)
        print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    def run_epoch(self, dataloader, is_train):
        self._reset_meters()
        self._init_confusion_matrix()

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        epoch_start = time.time()

        for i, (inputs_batch, targets_batch) in enumerate(dataloader):
            if self.use_cuda:
                inputs_batch = inputs_batch.cuda()
                targets_batch = targets_batch.cuda()

            outputs = self.forward_step(inputs_batch, targets_batch, is_train=is_train)

            if self.is_binary:
                for ii, outs in enumerate(outputs):
                    pred_classes = torch.argmax(outs, dim=1)
                    self.confusion_matrices[ii]['true_positives'] += sum(pred_classes * targets_batch)
                    self.confusion_matrices[ii]['false_positives'] += sum(pred_classes * (1 - targets_batch))
                    self.confusion_matrices[ii]['false_negatives'] += sum((1 - pred_classes) * targets_batch)
                    self.confusion_matrices[ii]['true_negatives'] += sum((1 - pred_classes) * (1 - targets_batch))

        epoch_time = time.time() - epoch_start
        self.epoch_time_meter.update(epoch_time, n=1)

        return self.on_end_epoch(is_train=is_train)

    def forward_step(self, inputs, targets, is_train):
        inputs = torch.autograd.Variable(inputs).float()
        targets = torch.autograd.Variable(targets)

        if is_train:
            outputs = self.model(inputs)
        else:
            with torch.no_grad():
                outputs = self.model(inputs)

        losses, total_loss = self.criterion(outputs, targets)
        accuracies = [accuracy(out.data, targets[:, i].data) for i, out in enumerate(outputs)]

        # update meters
        n = inputs.size()[0]
        self.total_loss_meter.update(total_loss, n)
        self._update_loss_meters(losses, n)
        self._update_accuracy_meters(accuracies, n)

        if is_train:
            for param in self.model.parameters():
                param.grad = None
            total_loss.backward()
            self.optimizer.step()

        return outputs

    def on_end_epoch(self, is_train):
        prefix = 'train' if is_train else 'val'
        stats = {f'{prefix}-total-loss': self.total_loss_meter.average(),
                 f'{prefix}-losses': [m.average() for m in self.loss_meters],
                 f'{prefix}-accuracies': [m.average() for m in self.accuracy_meters],
                 'epoch_time': self.epoch_time_meter.average()}

        if self.is_binary:
            sensitivities = [cm["true_positives"] / (cm["true_positives"] + cm["false_negatives"])
                             for cm in self.confusion_matrices]
            specificities = [cm["true_negatives"] / (cm["true_negatives"] + cm["false_positives"])
                             for cm in self.confusion_matrices]
            gmeans = [np.sqrt(sens * spec) for sens, spec in zip(sensitivities, specificities)]
            stats[f'{prefix}-specificities'] = specificities
            stats[f'{prefix}-sensitivities'] = sensitivities
            stats[f'{prefix}-gmeans'] = gmeans

        return stats

    def criterion(self, outputs, targets) -> Tuple[List[Tensor], Tensor]:
        """ sum of cross entropy losses per category/item"""
        losses = [self.loss_func(pred, targets[:, i]) for i, pred in enumerate(outputs)]
        return losses, torch.stack(losses).sum()

    def save_checkpoint(self, epoch, best_epoch, val_loss, best_val_loss, is_best):
        checkpoint = {'epoch': epoch + 1,
                      'best_epoch': best_epoch,
                      'val_loss': val_loss,
                      'best_val_loss': best_val_loss,
                      'state_dict': copy.deepcopy(self.model.state_dict()),
                      'optimizer': copy.deepcopy(self.optimizer.state_dict())}

        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints/')
        checkpint_fp = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(checkpoint, checkpint_fp)

        if is_best:
            shutil.copyfile(checkpint_fp, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
