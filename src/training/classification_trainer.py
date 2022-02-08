import copy
import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from constants import *
from src.training.train_utils import AverageMeter, accuracy
from src.utils import timestamp_human


class ClassificationTrainer:
    def __init__(self, model, criterion, train_loader, val_loader, args, save_dir, is_binary):
        self.model = model
        self.criterion = criterion
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
            self.criterion = self.criterion.cuda()

    def _init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)

    def _init_meters(self):
        self.loss_meter = AverageMeter()
        self.accuracy_meter = AverageMeter()
        self.epoch_time_meter = AverageMeter()

    def _reset_meters(self):
        self.accuracy_meter.reset()
        self.loss_meter.reset()

    def _init_confusion_matrix(self):
        self.confusion_matrix = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0}

    def train(self):
        self.initialize(is_train=True)

        start_epoch = 0
        best_epoch = 0
        best_val_acc = -np.inf

        print(f'[{timestamp_human()}] start training')

        for epoch in range(start_epoch, self.args.epochs):
            # train for one epoch
            train_stats = self.run_epoch(self.train_loader, is_train=True)

            # run validation
            val_stats = self.run_epoch(self.val_loader, is_train=False)

            # save model
            is_best = val_stats['val-acc'] > best_val_acc
            if is_best:
                best_val_acc = val_stats['val-acc']
                best_epoch = epoch + 1
            self.save_checkpoint(epoch, best_epoch, val_stats['val-acc'], best_val_acc, is_best)

            # print stats
            learning_rate = self.lr_scheduler.get_last_lr()[0]
            timestamp = timestamp_human()
            print_str = f'[{timestamp} | {epoch + 1}/{self.args.epochs}] epoch time: {train_stats["epoch_time"]:.2f}, '
            print_str += f'lr: {learning_rate:.6f} ||'
            for k, v in train_stats.items():
                print_str += f' {k}: {v:.5f} '
            print_str += '||'
            for k, v in val_stats.items():
                print_str += f' {k}: {v:.5f} '
            print(print_str)

            # add tensorboard summaries
            self.summary_writer.add_scalars('loss',
                                            {'train': train_stats["train_loss"], 'val': val_stats["val_loss"]},
                                            global_step=epoch)
            self.summary_writer.add_scalars('accuracy',
                                            {'train': train_stats["train_acc"], 'val': val_stats["val_acc"]},
                                            global_step=epoch)
            self.summary_writer.add_scalar('learning-rate', learning_rate, global_step=epoch)
            self.summary_writer.flush()

            # decay learning rate
            self.lr_scheduler.step()

        self.summary_writer.flush()
        self.summary_writer.close()

        print(f'\ntraining finished; average epoch time: {self.epoch_time_meter.average():.4f}s')
        print('\n-----------------------')
        print('** early stop validation stats **')
        print('{0:25}: {1:.4f}'.format('epoch', best_epoch))
        print('{0:25}: {1:.4f}%'.format('accuracy', best_val_acc))

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
                predicted_classes = torch.argmax(outputs, dim=1)
                self.confusion_matrix['true_positives'] += sum(predicted_classes * targets_batch)
                self.confusion_matrix['false_positives'] += sum(predicted_classes * (1 - targets_batch))
                self.confusion_matrix['false_negatives'] += sum((1 - predicted_classes) * targets_batch)
                self.confusion_matrix['true_negatives'] += sum((1 - predicted_classes) * (1 - targets_batch))

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

        loss = self.criterion(outputs, targets)

        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data)
        self.loss_meter.update(loss.data, inputs.size()[0])
        self.accuracy_meter.update(acc[0], inputs.size()[0])

        if is_train:
            # set grads to zero
            for param in self.model.parameters():
                param.grad = None

            # compute grads and make train step
            loss.backward()
            self.optimizer.step()

        return outputs

    def on_end_epoch(self, is_train):
        prefix = 'train-' if is_train else 'val-'
        stats = {f'{prefix}loss': self.loss_meter.average(),
                 f'{prefix}acc': self.accuracy_meter.average(),
                 'epoch_time': self.epoch_time_meter.average()}

        if self.is_binary:
            sensitivity = self.confusion_matrix["true_positives"] / (
                    self.confusion_matrix["true_positives"] + self.confusion_matrix["false_negatives"])
            specificity = self.confusion_matrix["true_negatives"] / (
                    self.confusion_matrix["true_negatives"] + self.confusion_matrix["false_positives"])
            gmean = np.sqrt(sensitivity.cpu() * specificity.cpu())
            stats[f'{prefix}specificity'] = specificity
            stats[f'{prefix}sensitivity'] = sensitivity
            stats[f'{prefix}g_mean'] = gmean

        return stats

    def save_checkpoint(self, epoch, best_epoch, val_acc, best_val_acc, is_best):
        checkpoint = {'epoch': epoch + 1,
                      'best_epoch': best_epoch,
                      'val_acc': val_acc,
                      'best_val_acc': best_val_acc,
                      'state_dict': copy.deepcopy(self.model.state_dict()),
                      'optimizer': copy.deepcopy(self.optimizer.state_dict())}

        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints/')
        checkpint_fp = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(checkpoint, checkpint_fp)

        if is_best:
            shutil.copyfile(checkpint_fp, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
