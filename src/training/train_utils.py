import os
import pandas as pd
import random
import sys
from typing import Tuple
import uuid

import numpy as np
from matplotlib import pyplot as plt

import torch

from src.preprocessing.augmentation import AugmentParameters


def directory_setup(model_name, dataset, results_dir, train_id: int = None, resume: str = ''):
    """setup dir for training results and model checkpoints"""
    if resume:
        checkpoints_dir = os.path.join(resume, 'checkpoints/')
        if not os.path.exists(checkpoints_dir):
            raise NotADirectoryError(f'no checkpoints in {checkpoints_dir}')
        return resume, checkpoints_dir

    if train_id is None:
        train_id = 'id-' + str(uuid.uuid4())
        print(f'==> generated random uuid {train_id}')

    results_dir = os.path.join(results_dir, dataset, train_id, model_name)

    try:
        os.makedirs(results_dir)
    except OSError:
        raise OSError(f'results dir already exists! {results_dir}')

    # create directory to save trained model
    checkpoints_dir = os.path.join(results_dir, "checkpoints/")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    return results_dir, checkpoints_dir


def train_val_split(labels_df, val_fraction) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split dataframe into train and validation parts"""
    val_labels = labels_df.sample(frac=val_fraction, replace=False, axis=0, random_state=42)
    train_labels = labels_df[~labels_df.index.isin(val_labels.index)]
    assert set(val_labels.index).isdisjoint(train_labels.index)
    return train_labels, val_labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def average(self):
        return self.avg

    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


def accuracy(predictions, targets):
    batch_size = targets.size(0)
    predictions = torch.argmax(predictions, dim=1)
    num_correct = torch.eq(predictions, targets).sum()
    return num_correct / batch_size


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_scores_preds(model, images, labels):
    labels = np.squeeze(labels.cpu().detach().numpy()[:, -1])
    outputs = model(images.float())
    pred_scores = np.squeeze(outputs.cpu().detach().numpy()[:, -1])  # total score is last

    # move to cpu
    images = images.cpu()

    # scale to 0-1
    height, width = images.size()[-2:]
    images = images.view(images.size(0), -1)
    images -= images.min(1, keepdim=True)[0]
    images /= images.max(1, keepdim=True)[0]
    images = images.view(images.size(0), 1, height, width)

    fig = plt.figure(figsize=(12, 48))

    labels = [labels] if isinstance(labels, float) else labels
    preds = [pred_scores] if isinstance(pred_scores, float) else pred_scores

    for idx in np.arange(min(4, images.size()[0])):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(f"prediction: {preds[idx]:.2f}\nactual: {labels[idx]:.2f}")

    return fig
