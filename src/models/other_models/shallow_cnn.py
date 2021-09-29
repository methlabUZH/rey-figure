import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

__all__ = ['shallow_cnn']


class ShallowCNN(nn.Module):

    def __init__(self, dropout, n_outputs):
        super(ShallowCNN, self).__init__()

        # setup layers
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self._conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5))
        self._conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self._conv4 = nn.Conv2d(64, 64, kernel_size=(5, 5))

        self._pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self._fc1 = nn.Linear(in_features=512, out_features=1024)
        self._fc2 = nn.Linear(1024, n_outputs)

        self.score_layer = nn.Linear(in_features=18, out_features=1, bias=False)
        self.score_layer.weight.requires_grad = False
        self.score_layer.weight.copy_(torch.ones(size=(18,)))

        self._dropout = nn.Dropout(dropout)

        self._leaky_relu = nn.LeakyReLU(0.01)
        self._relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, inputs):
        # convolutional layers
        conv1 = self._pool(self._leaky_relu(self._conv1(inputs)))
        conv2 = self._pool(self._leaky_relu(self._conv2(conv1)))
        conv3 = self._pool(self._leaky_relu(self._conv3(conv2)))
        conv4 = self._pool(self._leaky_relu(self._conv4(conv3)))
        flattened = conv4.view(inputs.size()[0], -1)

        # dense layers
        dense1 = self._dropout(self._leaky_relu(self._fc1(flattened)))
        dense2 = self._relu(self._fc2(dense1))

        score = self.score_layer(dense2)
        outputs = torch.cat([dense2, score], dim=1)

        return outputs


def shallow_cnn(dropout_rate, n_outputs):
    return ShallowCNN(dropout=dropout_rate, n_outputs=n_outputs)
