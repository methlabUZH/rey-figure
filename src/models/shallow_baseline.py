import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ['shallow_baseline']


class ShallowBaseline(nn.Module):

    def __init__(self, dropout, n_outputs):
        super(ShallowBaseline, self).__init__()

        # setup layers
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self._conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5))
        self._conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self._conv4 = nn.Conv2d(64, 64, kernel_size=(5, 5))

        self._pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self._fc1 = nn.Linear(in_features=512, out_features=1024)
        self._fc2 = nn.Linear(1024, n_outputs)

        self._dropout = nn.Dropout(dropout)

        self._leaky_relu = nn.LeakyReLU(0.01)
        self._relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        # convolutional layers
        conv1 = self._pool(self._leaky_relu(self._conv1(inputs)))
        conv2 = self._pool(self._leaky_relu(self._conv2(conv1)))
        conv3 = self._pool(self._leaky_relu(self._conv3(conv2)))
        conv4 = self._pool(self._leaky_relu(self._conv4(conv3)))
        flattened = conv4.view(inputs.size()[0], -1)

        # dense layers
        dense1 = self._dropout(self._leaky_relu(self._fc1(flattened)))
        dense2 = self._relu(self._fc2(dense1))

        return dense2


def shallow_baseline(dropout_rate, n_outputs):
    return ShallowBaseline(dropout=dropout_rate, n_outputs=n_outputs)

# class Regressor1(nn.Module):
#
#     def __init__(self, dropout_rate, num_conv=4):
#         super(Regressor1, self).__init__()
#
#         self._dropout_rate = dropout_rate
#         self._num_conv = num_conv
#
#         # setup layers
#         self._conv1 = nn.Conv2d(1, 32, (5, 5))
#         self._pool = nn.MaxPool2d(2, 2)
#         self._conv2 = nn.Conv2d(32, 32, (5, 5))
#
#         if num_conv == 2:
#             self._fc1 = nn.Linear(32 * 26 * 34, 120)
#
#         if num_conv > 2:
#             self._conv3 = nn.Conv2d(32, 64, 5)
#             self._fc1 = nn.Linear(64 * 11 * 15, 120)
#
#         if num_conv == 4:
#             self._conv4 = nn.Conv2d(64, 64, 5)
#             self._fc1 = nn.Linear(64 * 3 * 5, 120)
#             self._bn1 = nn.BatchNorm1d(num_features=120)
#
#         self._fc2 = nn.Linear(120, 84)
#         self._bn2 = nn.BatchNorm1d(num_features=84)
#
#         self._fc3 = nn.Linear(84, 19)
#
#         self._dropout = nn.Dropout(self._dropout_rate)
#
#         self._activation = nn.LeakyReLU(0.1)
#
#     def forward(self, inputs):
#
#         # convolutional layers
#         conv1 = self._pool(self._activation(self._conv1(inputs)))
#         conv2 = self._pool(self._activation(self._conv2(conv1)))
#
#         if self._num_conv == 2:
#             flattened = conv2.view(inputs.size()[0], -1)
#
#         else:
#             conv3 = self._pool(self._activation(self._conv3(conv2)))
#             if self._num_conv == 3:
#                 flattened = conv3.view(inputs.size()[0], -1)
#             elif self._num_conv == 4:
#                 conv4 = self._pool(self._activation(self._conv4(conv3)))
#                 flattened = conv4.view(inputs.size()[0], -1)
#             else:
#                 raise ValueError
#
#         # dense layers
#         dense1 = self._activation(self._fc1(flattened))
#         dense2 = self._activation(self._fc2(dense1))
#         dense3 = self._fc3(dense2)
#
#         return dense3
#
#     @property
#     def criterion(self):
#         return nn.MSELoss(reduction="mean")
