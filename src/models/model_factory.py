from .resnet import *
from .shallow_baseline import *
from .deep_baseline import *
from .efficientnet import *

from typing import *
import torch.nn as nn


def get_architecture(arch: str,
                     num_outputs: int,
                     dropout: Union[float, None],
                     norm_layer: str,
                     image_size: Union[Tuple[int], List[int]]):

    if norm_layer == 'batch_norm':
        norm_layer = nn.BatchNorm2d
    elif norm_layer == 'group_norm':
        norm_layer = lambda x: nn.GroupNorm(num_groups=32, num_channels=x)  # noqa
    else:
        raise ValueError

    if arch == 'shallow-baseline':
        return shallow_baseline(dropout_rate=dropout, n_outputs=num_outputs)
    elif arch == 'deep-baseline':
        return deep_baseline(dropout_rate=dropout, n_outputs=num_outputs)
    elif arch == 'resnet18':
        return resnet18(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnet34':
        return resnet34(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnet50':
        return resnet50(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnet101':
        return resnet101(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnet152':
        return resnet152(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnext29_16x64d':
        return resnext29_16x64d(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnext50-32x4d':
        return resnext50_32x4d(num_outputs, norm_layer=norm_layer)
    elif arch == 'resnext101-32x8d':
        return resnext101_32x8d(num_outputs, norm_layer=norm_layer)
    elif arch == 'wide_resnet50_2':
        return wide_resnet50_2(num_outputs, norm_layer=norm_layer)
    elif arch == 'wide_resnet101_2':
        return wide_resnet101_2(num_outputs, norm_layer=norm_layer)
    elif arch == 'efficientnet-b0':
        return efficientnet_b0(num_outputs, image_size=image_size, dropout=dropout)
    elif arch == 'efficientnet-l2':
        return efficientnet_l2(num_outputs, image_size=image_size, dropout=dropout)
    else:
        raise ValueError(f'architecture {arch} not found!')
