from datetime import datetime as dt
from typing import *
from filelock import FileLock
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch


def timestamp_human():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S')


def round_item_scores(item_scores: torch.Tensor):
    def assign_score(x):
        if x < 0.25:
            return 0
        if 0.25 <= x < 0.75:
            return 0.5
        if 0.75 <= x < 1.5:
            return 1
        if 1.5 <= x:
            return 2

    rounded_scores = list(
        [list(map(assign_score, list_of_scores)) for list_of_scores in torch.unbind(item_scores, dim=0)])
    rounded_scores = torch.tensor(rounded_scores, dtype=torch.float)

    return rounded_scores


def assign_bins(scores: Union[torch.Tensor, np.ndarray, int],
                bin_locations: List[Tuple[float, float]],
                out_type=torch.float) -> Union[torch.Tensor, np.ndarray, int]:
    def assign_bin(x) -> int:
        for i in range(len(bin_locations)):
            if bin_locations[i][0] <= x < bin_locations[i][1]:
                return i

        return len(bin_locations) - 1

    if isinstance(scores, float):
        return assign_bin(scores)

    if isinstance(scores, np.ndarray):
        binned = np.array(list(map(assign_bin, scores)))
        return binned

    binned = list(map(assign_bin, torch.unbind(scores, 0)))
    binned = torch.unsqueeze(torch.tensor(binned, dtype=out_type), 1)

    return binned


def directory_setup(model_name, dataset, results_dir, args, resume: str = ''):
    """
    setup dir for training results and model checkpoints
    """

    if resume:
        checkpoints_dir = os.path.join(resume, 'checkpoints/')
        if not os.path.exists(checkpoints_dir):
            raise NotADirectoryError(f'no checkpoints in {checkpoints_dir}')
        return resume, checkpoints_dir

    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
    hyperparam_str = f'epochs-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}_gamma-{args.gamma}_wd-{args.wd}'
    hyperparam_str += f'_dropout-{args.dropout}_bn-momentum-{args.bn_momentum}'

    try:
        hyperparam_str += f'_beta={args.beta}'
    except AttributeError:
        pass

    hyperparam_str = hyperparam_str.replace('[', '').replace(']', '').replace(' ', '_').replace(',', '')
    results_dir = os.path.join(results_dir, dataset, model_name, hyperparam_str, timestamp)

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
    Save training process to log file

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
            self.file.write("{0:.6f}".format(num) if num is not None else "nan")
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

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


def store_stats(train_loss, val_loss, test_loss, train_score_mse, val_score_mse, test_score_mse, train_bin_mse,
                val_bin_mse, test_bin_mse, best_epoch, args):
    with FileLock(args.paramtuning_file + '.lock'):
        if not os.path.isfile(args.paramtuning_file):
            with open(args.paramtuning_file, 'a') as f:
                # write header
                cols = [k for k, _ in sorted(args.__dict__.items())]
                cols += ['train-loss', 'val-loss', 'test-loss', 'train-score-mse', 'val-score-mse', 'test-score-mse',
                         'train-bin-mse', 'val-bin-mse', 'test-bin-mse', 'best epoch']
                f.write(','.join(cols) + '\n')

        with open(args.paramtuning_file, 'a') as f:
            # write data
            data = [v for _, v in sorted(args.__dict__.items())]
            data += [train_loss, val_loss, test_loss, train_score_mse, val_score_mse, test_score_mse, train_bin_mse,
                     val_bin_mse, test_bin_mse, best_epoch]
            f.write(','.join([str(v) for v in data]) + '\n')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max((1,))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
