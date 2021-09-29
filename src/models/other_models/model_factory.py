from src.models.other_models.resnet import *
from src.models.other_models.resnetV2 import resnet18V2
from src.models.other_models.shallow_cnn import *
from src.models.reyregressor import *
from src.models.other_models.efficientnet import *

from typing import *
import torch.nn as nn


def get_architecture(arch: str,
                     num_outputs: int,
                     dropout: Union[float, Tuple[float, float]],
                     norm_layer_type: str,
                     image_size: Union[Tuple[int], List[int]],
                     bn_momentum: float = 0.1):
    if norm_layer_type == 'batch_norm':
        def norm_layer_1d(x): return nn.BatchNorm1d(num_features=x, momentum=bn_momentum)
        def norm_layer_2d(x): return nn.BatchNorm2d(num_features=x, momentum=bn_momentum)

    elif norm_layer_type == 'group_norm':
        def norm_layer_1d(x): return nn.GroupNorm(num_groups=32, num_channels=x)
        def norm_layer_2d(x): return nn.GroupNorm(num_groups=32, num_channels=x)

    else:
        if arch not in ['deep-cnn', 'shallow-cnn']:
            raise ValueError('norm layer needed for resnets! must be one of ["batch_norm", "group_norm"]}')
        norm_layer_1d = norm_layer_2d = None

    if arch == 'shallow-cnn':
        return shallow_cnn(dropout_rate=dropout, n_outputs=num_outputs)
    elif arch == 'deep-cnn':
        dropout = (0.0, 0.0) if dropout == 0.0 else dropout
        return get_reyregressor(n_outputs=num_outputs, dropout=dropout, norm_layer_2d=norm_layer_2d,
                                norm_layer_1d=norm_layer_1d)
    elif arch == 'resnet18':
        return resnet18(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnet18-V2':
        return resnet18V2(num_outputs, norm_layer=norm_layer_2d, dropout=dropout)
    elif arch == 'resnet34':
        return resnet34(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnet50':
        return resnet50(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnet101':
        return resnet101(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnet152':
        return resnet152(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnext29_16x64d':
        return resnext29_16x64d(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnext50-32x4d':
        return resnext50_32x4d(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'resnext101-32x8d':
        return resnext101_32x8d(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'wide_resnet50_2':
        return wide_resnet50_2(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'wide_resnet101_2':
        return wide_resnet101_2(num_outputs, norm_layer=norm_layer_2d)
    elif arch == 'efficientnet-b0':
        return efficientnet_b0(num_outputs, image_size=image_size, dropout=dropout)
    elif arch == 'efficientnet-l2':
        return efficientnet_l2(num_outputs, image_size=image_size, dropout=dropout)
    else:
        raise ValueError(f'architecture {arch} not found!')
