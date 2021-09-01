import json
import torch
from torch import optim

from constants import (
    RESNET18_CONFIG,
    RESNET50_CONFIG,
    RESNET101_CONFIG,
    RESNEXT50_CONFIG,
    EFFICIENTNET_B0,
    EFFICIENTNET_L2
)

__all__ = ['get_train_setup']


class TrainSetup:
    def __init__(self, epochs, learning_rate, lr_decay, lr_schedule, batch_size, weight_decay, dropout, momentum,
                 optimizer_name, track_running_stats):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.momentum = momentum
        self.optimizer_name = optimizer_name
        self.track_running_stats = track_running_stats

    @classmethod
    def from_config(cls, config_file):
        with open(config_file) as f:
            config = json.load(f)
        return cls(**config)

    def get_lr_scheduler(self, optimizer) -> optim.lr_scheduler:
        if self.optimizer_name == 'sgd':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_schedule, gamma=self.lr_decay)

        if self.optimizer_name == 'adam':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)

    def optimizer(self, params) -> torch.optim.Optimizer:
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params=params, lr=self.learning_rate, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)


def get_train_setup(arch: str) -> TrainSetup:
    return TrainSetup.from_config(config_file={'resnet18': RESNET18_CONFIG,
                                               'resnet50': RESNET50_CONFIG,
                                               'resnet101': RESNET101_CONFIG,
                                               'resnext50-32x4d': RESNEXT50_CONFIG,
                                               'efficientnet-b0': EFFICIENTNET_B0,
                                               'efficientnet-l2': EFFICIENTNET_L2,
                                               }[arch])
