from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

from constants import BIN_LOCATIONS


def directory_setup(model_name, label_format, results_dir, debug=False):
    """
    setup dir for training results and model checkpoints
    """
    if debug:
        results_dir = os.path.join(results_dir, 'debugging')
    else:
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(results_dir, label_format, model_name, timestamp)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create directory to save trained model
    checkpoints_dir = os.path.join(results_dir, "checkpoints/")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    return results_dir, checkpoints_dir


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_scores(model, images):
    outputs = model(images.float())
    scores = np.squeeze(outputs.cpu().detach().numpy()[:, -1])  # total score is last
    return scores


def plot_scores_preds(model, images, labels, use_cuda):
    labels = np.squeeze(labels.cpu().detach().numpy()[:, -1])
    pred_scores = images_to_scores(model, images.cuda() if use_cuda else images)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timestamp_human():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S')


def assign_bins(scores, out_type=torch.float):
    def assign_bin(x):
        for i in range(len(BIN_LOCATIONS)):
            if BIN_LOCATIONS[i][0] <= x < BIN_LOCATIONS[i][1]:
                return i

        return len(BIN_LOCATIONS) - 1

    binned = list(map(assign_bin, torch.unbind(scores, 0)))
    binned = torch.unsqueeze(torch.tensor(binned, dtype=out_type), 1)

    return binned


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


class TrainingLogger:
    """
    Save training process to log file with simple plot function.

    """

    def __init__(self, fpath, resume=False):
        self.file = None
        self.resume = resume

        if fpath is not None:

            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}

                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])

                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass
