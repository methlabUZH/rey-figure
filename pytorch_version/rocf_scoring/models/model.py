import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONV_LAYERS, DROPOUT_MC_RATE, DROPOUT_MC
from rocf_scoring.data_preprocessing.loading_data import preprocess_dataset
from rocf_scoring.data_preprocessing.preprocess import BIN_LOCATIONS
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)

        if CONV_LAYERS == 2:
            self.fc1 = nn.Linear(32 * 26 * 34, 120)

        if CONV_LAYERS > 2:
            self.conv3 = nn.Conv2d(32, 64, 5)
            self.fc1 = nn.Linear(64 * 11 * 15, 120)

        if CONV_LAYERS == 4:
            self.conv4 = nn.Conv2d(64, 64, 5)
            self.fc1 = nn.Linear(64 * 3 * 5, 120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(BIN_LOCATIONS))

        self.dropout = nn.Dropout(DROPOUT_MC_RATE)


    def forward(self, input):

        x = self.pool(F.relu(self.conv1(input)))
        if DROPOUT_MC:
            x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        if DROPOUT_MC:
            x = self.dropout(x)


        if CONV_LAYERS > 2:
            x = self.pool(F.relu(self.conv3(x)))
            if DROPOUT_MC:
                x = self.dropout(x)

        if CONV_LAYERS == 4:
            x = self.pool(F.relu(self.conv4(x)))
            if DROPOUT_MC:
                x = self.dropout(x)


        x = x.view(input.size()[0],-1)

        x = F.relu(self.fc1(x))
        if DROPOUT_MC:
            x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if DROPOUT_MC:
            x = self.dropout(x)

        x = self.fc3(x)
        if DROPOUT_MC:
            x = self.dropout(x)

        return x

def weights_init(m):

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


if __name__=="__main__":
    preprocessed_images, preprocessed_labels, files = preprocess_dataset()
    cnn = CNN()
    input_nn = torch.from_numpy(preprocessed_images[0][np.newaxis, np.newaxis, :])
    print(input_nn.size())
    output_nn = cnn.forward(input_nn.float())
    print(output_nn.size())

