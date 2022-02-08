import copy
import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from constants import *
from src.training.train_utils import AverageMeter
from src.utils import timestamp_human


class RegressionTrainer:
    def __init__(self, model, criterion, train_loader, val_loader, args, save_dir):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.save_dir = save_dir

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

        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()

    def _init_optimizer_and_scheduler(self):
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)

    def _init_meters(self):
        self.total_loss_meter = AverageMeter()
        self.items_loss_meter = AverageMeter()
        self.score_loss_meter = AverageMeter()
        self.epoch_time_meter = AverageMeter()

    def _reset_meters(self):
        self.total_loss_meter.reset()
        self.items_loss_meter.reset()
        self.score_loss_meter.reset()

    def train(self):
        self.initialize(is_train=True)

        start_epoch = 0
        best_epoch = 0
        best_val_score_mse, best_val_items_loss, best_val_loss = np.inf, np.inf, np.inf

        print(f'[{timestamp_human()}] start training')

        for epoch in range(start_epoch, self.args.epochs):
            # train for one epoch
            train_stats = self.run_epoch(self.train_loader, is_train=True)

            # run validation
            val_stats = self.run_epoch(self.val_loader, is_train=False)

            # save model
            is_best = val_stats['val-total-loss'] < best_val_loss
            if is_best:
                best_val_loss = val_stats['val-total-loss']
                best_val_items_loss = val_stats['val-items-loss']
                best_val_score_mse = val_stats['val-score-loss']
                best_epoch = epoch + 1
            self.save_checkpoint(epoch, best_epoch, val_stats['val-total-loss'], best_val_loss, is_best)

            # print stats
            learning_rate = self.lr_scheduler.get_last_lr()[0]
            timestamp = timestamp_human()
            print_str = f'[{timestamp} | {epoch + 1}/{self.args.epochs}] epoch time: {train_stats["epoch_time"]:.2f}, '
            print_str += f'lr: {learning_rate:.6f} ||'
            for k, v in train_stats.items():
                if k == 'epoch_time':
                    continue
                print_str += f' {k}: {v:.5f} '
            print_str += '||'
            for k, v in val_stats.items():
                if k == 'epoch_time':
                    continue
                print_str += f' {k}: {v:.5f} '
            print(print_str)

            # add tensorboard summaries
            self.summary_writer.add_scalars('total-loss',
                                            {'train': train_stats["train-total-loss"],
                                             'val': val_stats["val-total-loss"]},
                                            global_step=epoch)
            self.summary_writer.add_scalars('items-loss',
                                            {'train': train_stats["train-items-loss"],
                                             'val': val_stats["val-items-loss"]},
                                            global_step=epoch)
            self.summary_writer.add_scalars('score-loss',
                                            {'train': train_stats["train-score-loss"],
                                             'val': val_stats["val-score-loss"]},
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
        print('best {0:25}: {1:.4f}'.format('total loss', best_val_loss))
        print('best {0:25}: {1:.4f}'.format('items loss', best_val_items_loss))
        print('best {0:25}: {1:.4f}'.format('score loss', best_val_score_mse))

    def run_epoch(self, dataloader, is_train):
        self._reset_meters()

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        epoch_start = time.time()

        for i, (inputs_batch, targets_batch) in enumerate(dataloader):
            if self.use_cuda:
                inputs_batch = inputs_batch.cuda()
                targets_batch = targets_batch.cuda()

            self.forward_step(inputs_batch, targets_batch, is_train=is_train)

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

        # measure losses
        items_loss = self.criterion(outputs[:, :-1], targets[:, :-1])
        score_loss = self.criterion(outputs[:, -1], targets[:, -1])
        total_loss = items_loss + self.args.beta * score_loss

        self.items_loss_meter.update(float(items_loss.data), inputs.size()[0])
        self.score_loss_meter.update(float(score_loss.data), inputs.size()[0])
        self.total_loss_meter.update(float(total_loss.data), inputs.size()[0])

        if is_train:
            # set grads to zero
            for param in self.model.parameters():
                param.grad = None

            # compute grads and make train step
            total_loss.backward()
            self.optimizer.step()

    def on_end_epoch(self, is_train):
        prefix = 'train' if is_train else 'val'
        return {f'{prefix}-total-loss': self.total_loss_meter.average(),
                f'{prefix}-items-loss': self.items_loss_meter.average(),
                f'{prefix}-score-loss': self.score_loss_meter.average(),
                'epoch_time': self.epoch_time_meter.average()}

    def save_checkpoint(self, epoch, best_epoch, val_total_loss, best_val_loss, is_best):
        checkpoint = {'epoch': epoch + 1,
                      'best_epoch': best_epoch,
                      'val_loss': val_total_loss,
                      'best_val_loss': best_val_loss,
                      'state_dict': copy.deepcopy(self.model.state_dict()),
                      'optimizer': copy.deepcopy(self.optimizer.state_dict())}

        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints/')
        checkpint_fp = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
        torch.save(checkpoint, checkpint_fp)

        if is_best:
            shutil.copyfile(checkpint_fp, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
