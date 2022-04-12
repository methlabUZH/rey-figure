from typing import Optional, Callable, Tuple, List

import torch
from torch import Tensor
import torch.nn as nn

from torchvision.models.vgg import VGG
from torchvision.models.resnet import ResNet

from constants import N_ITEMS

__all__ = ['rey_multiclassifier']

_DROPOUT_RATES = (0.3, 0.5)
_BN_MOMENTUM = 0.1


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


class ItemClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer_2d, norm_layer_1d, dropout_rates):
        super(ItemClassifier, self).__init__()

        # conv layer
        self.block1 = ConvBlock(in_channels, out_channels=in_channels * 2,
                                norm_layer=norm_layer_2d,
                                downsample=nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                                dropout=nn.Dropout(dropout_rates[0]))

        # fc layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels * 2, out_features=in_channels * 4)
        self.bn1 = norm_layer_1d(num_features=in_channels * 4, momentum=_BN_MOMENTUM)
        self.fc2 = nn.Linear(in_features=in_channels * 4, out_features=num_classes)

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


class ReyMultiClassifier(nn.Module):
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
        super(ReyMultiClassifier, self).__init__()

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

        # set item classifiers
        item_classifer_kwargs = dict(in_channels=256, num_classes=num_classes, norm_layer_2d=norm_layer_2d,
                                     norm_layer_1d=norm_layer_1d, dropout_rates=dropout_rates)
        for i in range(N_ITEMS):
            setattr(self, f"item-{i + 1}", ItemClassifier(**item_classifer_kwargs))

        # self.output_layers = nn.ModuleList([ItemClassifier(**item_classifer_kwargs) for _ in range(N_ITEMS)])
        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x: Tensor):
        out = self.block1(x)
        out = self.block2(out)
        shared_features = self.block3(out)
        return [getattr(self, f"item-{i + 1}")(shared_features) for i in range(N_ITEMS)]

    # def forward0(self, x: Tensor) -> Tensor:
    #     out = self.block1(x)
    #     out = self.block2(out)
    #     shared_features = self.block3(out)
    #     # return [getattr(self, f"item-{i + 1}")(shared_features) for i in range(N_ITEMS)]
    #     return self.output_layers(shared_features)
    #
    # def forward1(self, x: Tensor) -> Tensor:
    #     out = self.block1(x)
    #     out = self.block2(out)
    #     shared_features = self.block3(out)
    #     return getattr(self, f"item-{self.item}")(shared_features)


def rey_multiclassifier(num_classes):
    return ReyMultiClassifier(dropout_rates=_DROPOUT_RATES, num_classes=num_classes)


if __name__ == '__main__':
    model = rey_multiclassifier(4, None)
    input_tensor = torch.rand(size=[8, 1, 116, 150])
    preds = model(input_tensor)
    print(preds.size())
    for p in preds:
        print(p.size())
