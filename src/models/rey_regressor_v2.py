from typing import Optional, Callable, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

from constants import *

__all__ = ['reyregressor_v2']

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


class ItemRegressor(nn.Module):
    def __init__(self, in_channels, norm_layer_2d, norm_layer_1d, dropout_rates):
        super(ItemRegressor, self).__init__()

        # conv layer
        self.block1 = ConvBlock(in_channels, out_channels=in_channels * 2,
                                norm_layer=norm_layer_2d,
                                downsample=nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                                dropout=nn.Dropout(dropout_rates[0]))

        # fc layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels * 2, out_features=in_channels * 4)
        self.bn1 = norm_layer_1d(num_features=in_channels * 4, momentum=_BN_MOMENTUM)
        self.fc2 = nn.Linear(in_features=in_channels * 4, out_features=1)

        self.dropout_fc = nn.Dropout(dropout_rates[1])
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


class ReyRegressorV2(nn.Module):

    def __init__(self, dropout_rates: Tuple[float, float],
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyRegressorV2, self).__init__()

        if norm_layer_2d is None:
            norm_layer_2d = nn.BatchNorm2d

        if norm_layer_1d is None:
            norm_layer_1d = nn.BatchNorm1d

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.global_maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.droput_conv = nn.Dropout(dropout_rates[0])
        self.droput_fc = nn.Dropout(dropout_rates[1])

        # shared conv layers
        self.block1 = ConvBlock(1, 64, norm_layer_2d, self.maxpool, self.droput_conv)
        self.block2 = ConvBlock(64, 128, norm_layer_2d, self.maxpool, self.droput_conv)
        self.block3 = ConvBlock(128, 256, norm_layer_2d, self.maxpool, self.droput_conv)

        # set item regressors
        item_classifer_kwargs = dict(in_channels=256, norm_layer_2d=norm_layer_2d, norm_layer_1d=norm_layer_1d,
                                     dropout_rates=dropout_rates)
        for i in range(N_ITEMS):
            setattr(self, f"item-{i + 1}", ItemRegressor(**item_classifer_kwargs))

        self.score_layer = nn.Linear(in_features=_NUM_ITEMS, out_features=1, bias=False)
        self.score_layer.weight.requires_grad = False
        self.score_layer.weight.copy_(torch.ones(size=(_NUM_ITEMS,)))

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
        shared_features = self.block3(out)
        item_scores = [getattr(self, f"item-{i + 1}")(shared_features) for i in range(N_ITEMS)]
        item_scores = torch.squeeze(torch.stack(item_scores, dim=1))
        total_score = self.score_layer(item_scores)
        outputs = torch.cat([item_scores, total_score], dim=1)
        return outputs


def reyregressor_v2():
    return ReyRegressorV2(dropout_rates=_DROPOUT_RATES)


# if __name__ == '__main__':
#     import numpy as np
#
#     inputs = torch.from_numpy(np.random.normal(size=(3, 1, 116, 150)))
#     model = reyregressor_v2()
#     model.eval()
#     outputs = model(inputs.float())
#     # print(outputs)
#     # print(np.shape(outputs))
