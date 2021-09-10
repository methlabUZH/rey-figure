from .resnet import *
from .shallow_baseline import *
from .deep_baseline import *
from .efficientnet import *

from typing import Union


def get_architecture(arch: str, num_outputs: int, dropout: Union[float, None], track_running_stats: bool, image_size):
    if arch == 'shallow-baseline':
        return shallow_baseline(dropout_rate=dropout, n_outputs=num_outputs)
    elif arch == 'deep-baseline':
        return deep_baseline(dropout_rate=dropout, n_outputs=num_outputs)
    elif arch == 'resnet18':
        return resnet18(num_outputs, track_running_stats)
    elif arch == 'resnet34':
        return resnet34(num_outputs, track_running_stats)
    elif arch == 'resnet50':
        return resnet50(num_outputs, track_running_stats)
    elif arch == 'resnet101':
        return resnet101(num_outputs, track_running_stats)
    elif arch == 'resnet152':
        return resnet152(num_outputs, track_running_stats)
    elif arch == 'resnext29_16x64d':
        return resnext29_16x64d(num_outputs, track_running_stats)
    elif arch == 'resnext50-32x4d':
        return resnext50_32x4d(num_outputs, track_running_stats)
    elif arch == 'resnext101-32x8d':
        return resnext101_32x8d(num_outputs, track_running_stats)
    elif arch == 'wide_resnet50_2':
        return wide_resnet50_2(num_outputs, track_running_stats)
    elif arch == 'wide_resnet101_2':
        return wide_resnet101_2(num_outputs, track_running_stats)
    elif arch == 'efficientnet-b0':
        return efficientnet_b0(num_outputs, track_running_stats, image_size, dropout=dropout)
    elif arch == 'efficientnet-l2':
        return efficientnet_l2(num_outputs, track_running_stats, image_size, dropout=dropout)
    else:
        raise ValueError(f'architecture {arch} not found!')
