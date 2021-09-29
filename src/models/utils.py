from torch import nn as nn


def conv_layer(in_channels, out_channels, kernel_size, norm_layer, padding):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                         norm_layer(out_channels),
                         nn.ReLU())


def conv_block(in_list, out_list, p_list, kernel_size, norm_layer, pooling_kernel, pooling_stride,
               dropout_rate: float = 0.0, pooling_layer: nn.Module = None):
    layers = [conv_layer(in_c, out_c, kernel_size, norm_layer, p) for in_c, out_c, p in zip(in_list, out_list, p_list)]

    if pooling_layer is None:
        layers += [nn.MaxPool2d(kernel_size=pooling_kernel, stride=pooling_stride)]
    else:
        layers += [pooling_layer]

    if dropout_rate > 0:
        layers += [nn.Dropout(p=dropout_rate)]

    return nn.Sequential(*layers)


def fc_layer(in_features, out_features, norm_layer, dropout_rate):
    layers = [nn.Linear(in_features, out_features),
              norm_layer(out_features),
              nn.ReLU()]

    if dropout_rate > 0:
        layers += [nn.Dropout(dropout_rate)]

    return nn.Sequential(*layers)
