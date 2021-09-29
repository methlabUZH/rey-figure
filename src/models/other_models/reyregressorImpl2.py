from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn

from src.models.utils import conv_block, fc_layer

__all__ = ['get_reyregressor']


class ReyRegressor(nn.Module):
    """
    regression model, designed to predict item scores and final score for a rey figure
    """

    def __init__(self, n_outputs,
                 dropout_rates: Tuple[float, float],
                 norm_layer_2d: Optional[Callable[..., nn.Module]] = None,
                 norm_layer_1d: Optional[Callable[..., nn.Module]] = None):
        super(ReyRegressor, self).__init__()

        if norm_layer_2d is None:
            def norm_layer_2d(*args): return torch.nn.Identity()

        if norm_layer_1d is None:
            def norm_layer_1d(*args): return torch.nn.Identity()

        # conv blocks
        self._block1 = conv_block([1, 64], [64, 64], [1, 1], (3, 3), norm_layer_2d, 2, 1, dropout_rates[0])
        self._block2 = conv_block([64, 128], [128, 128], [1, 1], (3, 3), norm_layer_2d, 2, 1, dropout_rates[0])
        self._block3 = conv_block([128, 256], [256, 256], [1, 1], (3, 3), norm_layer_2d, 2, 1, dropout_rates[0])
        self._block4 = conv_block([256], [512], [1], (3, 3), norm_layer_2d, None, None, dropout_rates[0],
                                  pooling_layer=nn.AdaptiveMaxPool2d(output_size=(1, 1)))

        # fc layers
        self._dense_layer1 = fc_layer(512, 4096, norm_layer=norm_layer_1d, dropout_rate=dropout_rates[1])
        self._dense_layer2 = fc_layer(4096, 2048, norm_layer=norm_layer_1d, dropout_rate=dropout_rates[1])

        # final output layers
        self._output_layer = nn.Linear(in_features=2048, out_features=n_outputs)

        # sum of item scores
        self._score_layer = nn.Linear(in_features=18, out_features=1, bias=False)
        self._score_layer.weight.requires_grad = False
        self._score_layer.weight.copy_(torch.ones(size=(18,)))

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        x = self._block1(inputs)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)

        x = x.view(x.size()[0], -1)
        x = self._dense_layer1(x)
        x = self._dense_layer2(x)

        item_scores = self._output_layer(x)
        score = self._score_layer(item_scores)
        outputs = torch.cat([item_scores, score], dim=1)

        return outputs


def get_reyregressor(n_outputs: int,
                     dropout: Tuple[float, float],
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

    return ReyRegressor(n_outputs=n_outputs, dropout_rates=dropout, norm_layer_2d=norm_layer_2d,
                        norm_layer_1d=norm_layer_1d)

# if __name__ == '__main__':
#     import numpy as np
#
#     inputs = torch.from_numpy(np.random.normal(size=(1, 1, 116, 150)))
#     model = ReyNet(n_outputs=18, dropout_rates=(0.0, 0.0))
#     model.eval()
#     outputs = model(inputs.float())
