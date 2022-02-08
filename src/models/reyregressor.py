from typing import Optional, Callable, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ['reyregressor']

_DROPOUT_RATES = (0.3, 0.5)
_BN_MOMENTUM = 0.01
_NUM_ITEMS = 18


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, downsample, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = norm_layer(num_features=out_channels, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = norm_layer(num_features=out_channels, momentum=_BN_MOMENTUM)

        self.downsample = downsample
        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.downsample(out)
        out = self.dropout(out)

        return out


class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, downsample, dropout):
        super(SingleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = norm_layer(num_features=out_channels, momentum=_BN_MOMENTUM)

        self.downsample = downsample
        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.downsample(out)
        out = self.dropout(out)

        return out


class RegressionHead(nn.Module):
    def __init__(self, in_channels, norm_layer, dropout_layer):
        super(RegressionHead, self).__init__()

        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels * 8)
        self.fc2 = nn.Linear(in_features=in_channels * 8, out_features=in_channels * 4)
        self.fc3 = nn.Linear(in_features=in_channels * 4, out_features=_NUM_ITEMS)

        self.bn1 = norm_layer(num_features=in_channels * 8, momentum=_BN_MOMENTUM)
        self.bn2 = norm_layer(num_features=in_channels * 4, momentum=_BN_MOMENTUM)

        self.score_layer = nn.Linear(in_features=_NUM_ITEMS, out_features=1, bias=False)
        self.score_layer.weight.requires_grad = False
        self.score_layer.weight.copy_(torch.ones(size=(_NUM_ITEMS,)))

        self.dropout = dropout_layer
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # flatten
        x = x.view(x.size()[0], -1)

        # fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # compute and append final score = sum
        score = self.score_layer(x)
        outputs = torch.cat([x, score], dim=1)

        return outputs


class ReyRegressor(nn.Module):

    def __init__(self,
                 dropout_rates: Tuple[float, float],
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyRegressor, self).__init__()

        if norm_layer_2d is None:
            norm_layer_2d = nn.BatchNorm2d

        if norm_layer_1d is None:
            norm_layer_1d = nn.BatchNorm1d

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.global_maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.droput_conv = nn.Dropout(dropout_rates[0])
        self.droput_fc = nn.Dropout(dropout_rates[1])

        # conv layers
        self.block1 = ConvBlock(1, 64, norm_layer_2d, self.maxpool, self.droput_conv)
        self.block2 = ConvBlock(64, 128, norm_layer_2d, self.maxpool, self.droput_conv)
        self.block3 = ConvBlock(128, 256, norm_layer_2d, self.maxpool, self.droput_conv)
        # self.block4 = ConvBlock(256, 512, norm_layer_2d, self.global_maxpool, self.droput_conv)
        self.block4 = SingleConvBlock(256, 512, norm_layer_2d, self.global_maxpool, nn.Identity())

        # regressor head
        self.regressor_head = RegressionHead(512, norm_layer_1d, self.droput_fc)

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.regressor_head(out)
        return out


def reyregressor():
    return ReyRegressor(dropout_rates=_DROPOUT_RATES)


# class ReyRegressor(nn.Module):
#
#     def __init__(self, n_outputs,
#                  dropout_rates: Tuple[float, float],
#                  norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
#                  norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
#         super(ReyRegressor, self).__init__()
#
#         if norm_layer_2d is None:
#             norm_layer_2d = lambda x: torch.nn.Identity()  # noqa
#
#         if norm_layer_1d is None:
#             norm_layer_1d = lambda x: torch.nn.Identity()  # noqa
#
#         self._conv_dropout = nn.Dropout(dropout_rates[0])
#         self._dense_dropout = nn.Dropout(dropout_rates[1])
#
#         # block 1
#         self._conv11 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
#         self._bn11 = norm_layer_2d(64)
#         self._conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
#         self._bn12 = norm_layer_2d(64)
#         self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#
#         # block 2
#         self._conv21 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
#         self._bn21 = norm_layer_2d(128)
#         self._conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
#         self._bn22 = norm_layer_2d(128)
#         self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#
#         # block 3
#         self._conv31 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
#         self._bn31 = norm_layer_2d(256)
#         self._conv32 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
#         self._bn32 = norm_layer_2d(256)
#         self._pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
#
#         # block 4
#         self._conv41 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
#         self._bn41 = norm_layer_2d(512)
#         self._pool4 = nn.AdaptiveMaxPool2d(output_size=(1, 1))
#
#         self._fc1 = nn.Linear(in_features=512, out_features=4096)
#         self._bn_fc1 = norm_layer_1d(4096)
#         self._fc2 = nn.Linear(in_features=4096, out_features=2048)
#         self._bn_fc2 = norm_layer_1d(2048)
#         self._fc3 = nn.Linear(in_features=2048, out_features=n_outputs)
#
#         self.score_layer = nn.Linear(in_features=18, out_features=1, bias=False)
#         self.score_layer.weight.requires_grad = False
#         self.score_layer.weight.copy_(torch.ones(size=(18,)))
#
#         self._relu = nn.ReLU()
#
#         self.init_weights()
#
#     def init_weights(self):
#         def _init(m):
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight.data)
#                 nn.init.zeros_(m.bias)
#
#         self.apply(_init)
#
#     def forward(self, inputs):
#         # block 1
#         x = self._conv11(inputs)
#         x = self._bn11(x)
#         x = self._relu(x)
#
#         x = self._conv12(x)
#         x = self._bn12(x)
#         x = self._relu(x)
#
#         x = self._pool1(x)
#         x = self._conv_dropout(x)
#
#         # block 2
#         x = self._conv21(x)
#         x = self._bn21(x)
#         x = self._relu(x)
#
#         x = self._conv22(x)
#         x = self._bn22(x)
#         x = self._relu(x)
#
#         x = self._pool2(x)
#         x = self._conv_dropout(x)
#
#         # block 3
#         x = self._conv31(x)
#         x = self._bn31(x)
#         x = self._relu(x)
#
#         x = self._conv32(x)
#         x = self._bn32(x)
#         x = self._relu(x)
#
#         x = self._pool3(x)
#         x = self._conv_dropout(x)
#
#         # block 4
#         x = self._conv41(x)
#         x = self._bn41(x)
#         x = self._relu(x)
#
#         x = self._pool4(x)
#         x = x.view(inputs.size()[0], -1)
#
#         # fc layers
#         x = self._fc1(x)
#         x = self._bn_fc1(x)
#         x = self._relu(x)
#         x = self._dense_dropout(x)
#
#         x = self._fc2(x)
#         x = self._bn_fc2(x)
#         x = self._relu(x)
#         x = self._dense_dropout(x)
#
#         x = self._fc3(x)
#
#         # compute and append final score = sum
#         score = self.score_layer(x)
#         outputs = torch.cat([x, score], dim=1)
#
#         return outputs
#
#
# def reyregressor():
#     def norm_layer_1d(x): return nn.BatchNorm1d(num_features=x, momentum=_BN_MOMENTUM)
#     def norm_layer_2d(x): return nn.BatchNorm2d(num_features=x, momentum=_BN_MOMENTUM)
#     return ReyRegressor(n_outputs=_NUM_ITEMS, dropout_rates=_DROPOUT_RATES, norm_layer_2d=norm_layer_2d,
#                         norm_layer_1d=norm_layer_1d)
#
#
# # def get_reyregressor(n_outputs: int,
# #                      dropout: Tuple[float, float] = (0.0, 0.0),
# #                      bn_momentum: float = 0.1,
# #                      norm_layer_type: str = 'batch_norm'):
# #     if norm_layer_type == 'batch_norm':
# #         def norm_layer_1d(x): return nn.BatchNorm1d(num_features=x, momentum=bn_momentum)
# #         def norm_layer_2d(x): return nn.BatchNorm2d(num_features=x, momentum=bn_momentum)
# #     elif norm_layer_type == 'group_norm':
# #         def norm_layer_1d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
# #         def norm_layer_2d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
# #     else:
# #         norm_layer_1d = norm_layer_2d = None
# #
# #     return ReyRegressor(n_outputs=n_outputs, dropout_rates=dropout,
# #                         norm_layer_2d=norm_layer_2d, norm_layer_1d=norm_layer_1d)
#
# if __name__ == '__main__':
#     import numpy as np
#
#     inputs = torch.from_numpy(np.random.normal(size=(1, 1, 116, 150)))
#     model = reyregressor()
#     model.eval()
#     outputs = model(inputs.float())
#     print(outputs)
