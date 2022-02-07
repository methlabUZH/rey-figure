from typing import Optional, Callable, Tuple

import torch
from torch import Tensor
import torch.nn as nn

# from src.models.utils import conv_block, fc_layer

__all__ = ['get_reyclassifier']


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, downsample, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = norm_layer(out_channels)

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


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer, dropout_layer):
        super(ClassificationHead, self).__init__()

        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels * 2)
        self.bn1 = norm_layer(in_channels * 2)
        self.fc2 = nn.Linear(in_features=in_channels * 2, out_features=in_channels * 2)
        self.bn2 = norm_layer(in_channels * 2)
        self.fc3 = nn.Linear(in_features=in_channels * 2, out_features=num_classes)
        self.dropout = dropout_layer
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out


class ReyClassifierV2(nn.Module):
    """
        model to classify each item separately;
            if num_classes = 2, the classifier only predicts presence / absence of the item
            if num_classes = 4, the classifier predicts the score for this item.
        """
    def __init__(self,
                 dropout_rates: Tuple[float, float],
                 num_classes: int = 2,
                 num_blocks: int = 3,
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyClassifierV2, self).__init__()

        if norm_layer_2d is None:
            def norm_layer_2d(*args): return torch.nn.Identity()

        if norm_layer_1d is None:
            def norm_layer_1d(*args): return torch.nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.global_maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.droput_conv = nn.Dropout(dropout_rates[0])
        self.droput_fc = nn.Dropout(dropout_rates[1])

        # conv layers
        self.block1 = ConvBlock(1, 64, norm_layer_2d, self.maxpool, self.droput_conv)
        self.block2 = ConvBlock(64, 128, norm_layer_2d, self.maxpool, self.droput_conv)

        if num_blocks == 3:
            self.block3 = ConvBlock(128, 256, norm_layer_2d, self.global_maxpool, self.droput_conv)
            self.block4 = nn.Identity()
            n_final_planes = 256
        else:
            self.block3 = ConvBlock(128, 256, norm_layer_2d, self.maxpool, self.droput_conv)
            self.block4 = ConvBlock(256, 512, norm_layer_2d, self.global_maxpool, self.droput_conv)
            n_final_planes = 512

        # classification head
        self.classification_head = ClassificationHead(n_final_planes, num_classes, norm_layer_1d, self.droput_fc)

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size()[0], -1)
        out = self.classification_head(out)
        return out


class ReyClassifier(nn.Module):
    """
    model to classify each item separately;
        if num_classes = 2, the classifier only predicts presence / absence of the item
        if num_classes = 4, the classifier predicts the score for this item.
    """
    def __init__(self,
                 dropout_rates: Tuple[float, float],
                 num_classes: int = 2,
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyClassifier, self).__init__()

        if norm_layer_2d is None:
            def norm_layer_2d(*args): return torch.nn.Identity()

        if norm_layer_1d is None:
            def norm_layer_1d(*args): return torch.nn.Identity()

        # block 1
        self._conv11 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self._bn11 = norm_layer_2d(64)
        self._conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self._bn12 = norm_layer_2d(64)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # block 2
        self._conv21 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self._bn21 = norm_layer_2d(128)
        self._conv22 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self._bn22 = norm_layer_2d(128)
        self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # block 3
        self._conv31 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self._bn31 = norm_layer_2d(256)
        self._conv32 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self._bn32 = norm_layer_2d(256)
        self._pool3 = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # fc layers
        self._fc1 = nn.Linear(in_features=256, out_features=512)
        self._bn_fc1 = norm_layer_1d(512)
        self._fc2 = nn.Linear(in_features=512, out_features=512)
        self._bn_fc2 = norm_layer_1d(512)
        self._fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self._out_activation = nn.Softmax(dim=1)
        self._relu = nn.ReLU()
        self._conv_dropout = nn.Dropout(dropout_rates[0])
        self._dense_dropout = nn.Dropout(dropout_rates[1])

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        # block 1
        x = self._conv11(inputs)
        x = self._bn11(x)
        x = self._relu(x)

        x = self._conv12(x)
        x = self._bn12(x)
        x = self._relu(x)

        x = self._pool1(x)
        x = self._conv_dropout(x)

        # block 2
        x = self._conv21(x)
        x = self._bn21(x)
        x = self._relu(x)

        x = self._conv22(x)
        x = self._bn22(x)
        x = self._relu(x)

        x = self._pool2(x)
        x = self._conv_dropout(x)

        # block 3
        x = self._conv31(x)
        x = self._bn31(x)
        x = self._relu(x)

        x = self._conv32(x)
        x = self._bn32(x)
        x = self._relu(x)

        x = self._pool3(x)
        x = self._conv_dropout(x)

        x = x.view(inputs.size()[0], -1)

        # fc layers
        x = self._fc1(x)
        x = self._bn_fc1(x)
        x = self._relu(x)
        x = self._dense_dropout(x)

        x = self._fc2(x)
        x = self._bn_fc2(x)
        x = self._relu(x)
        x = self._dense_dropout(x)

        x = self._fc3(x)
        prediction = self._out_activation(x)

        return prediction


def get_reyclassifier(num_clases: int = 2,
                      num_blocks: int = 3,
                      dropout: Tuple[float, float] = (.0, .0),
                      bn_momentum: float = 0.1,
                      norm_layer_type: str = 'batch_norm'):
    if norm_layer_type == 'batch_norm':
        def norm_layer_1d(x): return nn.BatchNorm1d(num_features=x, momentum=bn_momentum)
        def norm_layer_2d(x): return nn.BatchNorm2d(num_features=x, momentum=bn_momentum)
    elif norm_layer_type == 'group_norm':
        def norm_layer_1d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
        def norm_layer_2d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
    else:
        norm_layer_1d = norm_layer_2d = None

    return ReyClassifierV2(dropout_rates=dropout, num_classes=num_clases, norm_layer_2d=norm_layer_2d,
                           norm_layer_1d=norm_layer_1d, num_blocks=num_blocks)


if __name__ == '__main__':
    import numpy as np

    inputs = torch.from_numpy(np.random.normal(size=(1, 1, 116, 150)))
    model = ReyClassifierV2(dropout_rates=(0.0, 0.0), num_classes=4, num_blocks=4)
    model.eval()
    print(model)
    outputs = model(inputs.float())
    print(outputs)
#
