from typing import Optional, Callable, Tuple

from torch import Tensor
import torch.nn as nn

__all__ = ['rey_classifier_3', 'rey_classifier_4']

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


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer, dropout_layer):
        super(ClassificationHead, self).__init__()

        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels * 2)
        self.bn1 = norm_layer(num_features=in_channels * 2, momentum=_BN_MOMENTUM)
        self.fc2 = nn.Linear(in_features=in_channels * 2, out_features=in_channels * 2)
        self.bn2 = norm_layer(num_features=in_channels * 2, momentum=_BN_MOMENTUM)
        self.fc3 = nn.Linear(in_features=in_channels * 2, out_features=num_classes)
        self.dropout = dropout_layer
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # flatten
        out = x.view(x.size()[0], -1)

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

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
                 num_blocks: int = 3,
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyClassifier, self).__init__()

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
        out = self.classification_head(out)
        return out


def rey_classifier_3(num_classes):
    return ReyClassifier(dropout_rates=_DROPOUT_RATES, num_classes=num_classes, num_blocks=3)


def rey_classifier_4(num_classes):
    return ReyClassifier(dropout_rates=_DROPOUT_RATES, num_classes=num_classes, num_blocks=4)


# def get_reyclassifier(num_clases: int = 2,
#                       num_blocks: int = 3,
#                       dropout: Tuple[float, float] = (.0, .0),
#                       bn_momentum: float = 0.1,
#                       norm_layer_type: str = 'batch_norm'):
#     if norm_layer_type == 'batch_norm':
#         def norm_layer_1d(x): return nn.BatchNorm1d(num_features=x, momentum=bn_momentum)
#         def norm_layer_2d(x): return nn.BatchNorm2d(num_features=x, momentum=bn_momentum)
#     elif norm_layer_type == 'group_norm':
#         def norm_layer_1d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
#         def norm_layer_2d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
#     else:
#         norm_layer_1d = norm_layer_2d = None
#
#     return ReyClassifier(dropout_rates=dropout, num_classes=num_clases, norm_layer_2d=norm_layer_2d,
#                          norm_layer_1d=norm_layer_1d, num_blocks=num_blocks)
#
#

if __name__ == '__main__':
    import torch
    import numpy as np

    inputs = torch.from_numpy(np.random.normal(size=(1, 1, 116, 150)))
    model = rey_classifier_3(num_classes=4)
    model.eval()
    print(model)
    outputs = model(inputs.float())
    print(outputs)
