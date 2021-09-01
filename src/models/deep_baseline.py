import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ['deep_baseline']


class DeepBaseline(nn.Module):

    def __init__(self, dropout, n_outputs):
        super(DeepBaseline, self).__init__()

        # block 1
        self._conv11 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self._conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self._conv21 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self._conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3
        self._conv31 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self._conv32 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self._pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4
        self._conv41 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)

        self._fc1 = nn.Linear(in_features=512, out_features=4096)
        self._fc2 = nn.Linear(in_features=4096, out_features=2048)
        self._fc3 = nn.Linear(in_features=2048, out_features=n_outputs)

        self._dropout = nn.Dropout(dropout)
        self._relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        # block 1
        conv11 = self._relu(self._conv11(inputs))
        conv12 = self._relu(self._conv12(conv11))
        pool1 = self._pool1(conv12)

        # block 2
        conv21 = self._relu(self._conv21(pool1))
        conv22 = self._relu(self._conv22(conv21))
        pool2 = self._pool2(conv22)

        # block 3
        conv31 = self._relu(self._conv31(pool2))
        conv32 = self._relu(self._conv32(conv31))
        pool3 = self._pool3(conv32)

        # block 4
        conv41 = self._relu(self._conv41(pool3))
        pool4 = F.max_pool2d(conv41, kernel_size=conv41.size()[2:])
        pool4 = pool4.view(inputs.size()[0], -1)

        # fc layers
        fc1 = self._relu(self._fc1(pool4))
        fc2 = self._relu(self._fc2(fc1))
        predictions = self._fc3(fc2)

        return predictions


def deep_baseline(dropout_rate, n_outputs):
    return DeepBaseline(dropout=dropout_rate, n_outputs=n_outputs)

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
