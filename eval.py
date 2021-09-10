import argparse
# import json
import random
import sys
import os

import torch

from src.training.data_loader import get_dataloader
from src.training.helpers import timestamp_human, count_parameters, assign_bins
from src.training.helpers import AverageMeter, Logger
from src.models.model_factory import get_architecture

DEBUG = True
default_data_dir = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224'
default_results_dir = '/Users/maurice/phd/src/psychology/results/resnet18'

# setup arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=default_results_dir, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# architecture
parser.add_argument('--arch', type=str, default='resnet18', required=False)
parser.add_argument('--image-size', nargs='+', type=int, default=[224, 224])

# misc
parser.add_argument('--seed', default=8, type=int)
parser.add_argument('--val-split', default=0.0, type=float)
parser.add_argument('--score-type', default='sum', type=str, choices=['sum', 'median'])

args = parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


def main():
    # save terminal output to file
    sys.stdout = Logger(print_fp=os.path.join(args.results_dir, 'eval.txt'))
    mean = std = None

    if DEBUG:
        args.batch_size = 2

    test_labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    test_data_loader, _ = get_dataloader(data_root=args.data_root, labels_csv=test_labels_csv,
                                         batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                         score_type=args.score_type, fraction=1.0, mean=mean, std=std)

    print('{0:25}: {1}'.format('num-test', len(test_data_loader.dataset)))

    # setup model
    model = get_architecture(arch=args.arch, num_outputs=19, dropout=None,
                             track_running_stats=False, image_size=args.image_size)

    print('{0:25}: {1}'.format('# params', count_parameters(model)))

    # load checkpoint
    ckpt = os.path.join(args.results_dir, 'checkpoints/model_best.pth.tar')
    checkpoint = torch.load(ckpt, map_location=torch.device('gpu' if use_cuda else 'cpu'))
    checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction="mean")

    if DEBUG:
        print(' --> debugging on!')

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start eval')

    # eval on test set
    test_loss, test_score_mse, test_bin_mse = test(test_data_loader, model, criterion)

    print(f'loss={test_loss}, score_mse={test_score_mse}, bin_mse={test_bin_mse}')


def test(testloader, model, criterion):
    model.eval()

    loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    bin_mse_meter = AverageMeter()

    for batch_idx, (images, labels, files) in enumerate(testloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(images.float().contiguous())

        predicted_scores = outputs[:, -1]
        true_scores = labels[:, -1]
        for pred, actual, f in zip(predicted_scores, true_scores, files):
            print(f'prediction: {pred:.1f}, actual: {actual:.1f}, file: {f}')

        loss = criterion(outputs, labels)
        score_mse = criterion(outputs[:, -1], labels[:, -1])
        bin_mse = criterion(assign_bins(outputs[:, -1]), assign_bins(labels[:, -1]))

        # record loss
        loss_meter.update(loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        bin_mse_meter.update(bin_mse.data, images.size()[0])

        if batch_idx >= 10 and DEBUG:
            break

    return loss_meter.avg, score_mse_meter.avg, bin_mse_meter.avg


if __name__ == '__main__':
    main()
