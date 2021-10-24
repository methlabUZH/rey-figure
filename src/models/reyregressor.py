from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ['get_reyregressor']


class ReyRegressor(nn.Module):

    def __init__(self, n_outputs,
                 dropout_rates: Tuple[float, float],
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyRegressor, self).__init__()

        if norm_layer_2d is None:
            norm_layer_2d = lambda x: torch.nn.Identity()  # noqa

        if norm_layer_1d is None:
            norm_layer_1d = lambda x: torch.nn.Identity()  # noqa

        self._conv_dropout = nn.Dropout(dropout_rates[0])
        self._dense_dropout = nn.Dropout(dropout_rates[1])

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
        self._pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # block 4
        self._conv41 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self._bn41 = norm_layer_2d(512)
        self._pool4 = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self._fc1 = nn.Linear(in_features=512, out_features=4096)
        self._bn_fc1 = norm_layer_1d(4096)
        self._fc2 = nn.Linear(in_features=4096, out_features=2048)
        self._bn_fc2 = norm_layer_1d(2048)
        self._fc3 = nn.Linear(in_features=2048, out_features=n_outputs)

        self.score_layer = nn.Linear(in_features=18, out_features=1, bias=False)
        self.score_layer.weight.requires_grad = False
        self.score_layer.weight.copy_(torch.ones(size=(18,)))

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

        # block 4
        x = self._conv41(x)
        x = self._bn41(x)
        x = self._relu(x)

        x = self._pool4(x)
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

        # compute and append final score = sum
        score = self.score_layer(x)
        outputs = torch.cat([x, score], dim=1)

        return outputs


def get_reyregressor(n_outputs: int,
                     dropout: Tuple[float, float] = (0.0, 0.0),
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

    return ReyRegressor(n_outputs=n_outputs, dropout_rates=dropout,
                        norm_layer_2d=norm_layer_2d, norm_layer_1d=norm_layer_1d)

# if __name__ == '__main__':
#     import numpy as np
#
#     inputs = torch.from_numpy(np.random.normal(size=(1, 1, 116, 150)))
#     model = ReyVGG(n_outputs=18, dropout_rates=(0.0, 0.0))
#     model.eval()
#     outputs = model(inputs.float())
