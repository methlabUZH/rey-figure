import argparse
import random
import sys
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import torch

from constants import BIN_LOCATIONS1, BIN_LOCATIONS2
from old_code.dataloaders import get_regression_dataloader_eval
from src.training.train_utils import count_parameters, AverageMeter, Logger
from src.utils import timestamp_human
from src.inference.utils import assign_bins
from src.models.other_models.model_factory import get_architecture

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'
default_results_dir = '/Users/maurice/phd/src/rey-figure/results/sum-score/scans-2018-2021-116x150-augmented/deep-cnn/2021-09-22_18-10-51.221'

# setup arg parser
parser = argparse.ArgumentParser()
# setup
parser.add_argument('--data_preprocessing-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=default_results_dir, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# architecture
parser.add_argument('--arch', type=str, default='deep-cnn', required=False)
parser.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
parser.add_argument('--norm-layer', type=str, default='batch_norm', choices=['batch_norm', 'group_norm'])

# misc
parser.add_argument('--seed', default=8, type=int)
parser.add_argument('--score-type', default='sum', type=str, choices=['sum', 'median'])

args = parser.parse_args()

if DEBUG:
    args.batch_size = 2

# Use CUDA
use_cuda = torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


def main():
    if DEBUG:
        print('==> debugging on!')

    # save terminal output to file
    out_file = os.path.join(args.results_dir, 'eval.txt' if not DEBUG else 'debug_eval.txt')
    print(f'==> results will be saved to {out_file}')
    sys.stdout = Logger(print_fp=out_file)

    # data_preprocessing
    print(f'==> data_preprocessing from {args.data_root}')
    labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_regression_dataloader_eval(args.data_root, labels_df=labels, batch_size=args.batch_size,
                                                num_workers=args.workers, score_type=args.score_type)

    # setup model
    model = get_architecture(arch=args.arch, num_outputs=18, dropout=0.0, norm_layer_type=args.norm_layer,
                             image_size=args.image_size)

    # load checkpoint
    checkpoint_file = os.path.join(args.results_dir, 'checkpoints/model_best.pth.tar')
    assert os.path.isfile(checkpoint_file), 'Error: no checkpoint found!'
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    # if not use_cuda:
    checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'==> loaded checkpoint {checkpoint_file}')

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(reduction="mean")

    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start eval')

    # eval on test set
    (test_loss, test_score_mse, test_bin1_mse, single_bin1_mses,
     test_bin2_mse, single_bin2_mses, results_df) = eval_model(model, dataloader, criterion)

    print(f'[{timestamp_human()}] finish eval')

    print('\n----------------------\n')
    print(tabulate(results_df.head(20), headers='keys', tablefmt='presto', floatfmt=".9f"))
    csv_file = os.path.join(args.results_dir, 'test-predictions.csv')
    results_df.to_csv(csv_file)  # noqa
    print(f'\n==> saved predictions as {csv_file}')
    print('----------------------')

    print('\ntotal test scores:')
    print('----------------------')
    print('{0:15}: {1}'.format('num-test', len(dataloader.dataset)))
    print('{0:15}: {1}'.format('# model-params', count_parameters(model)))
    print('{0:15}: {1}'.format('loss', test_loss))
    print('{0:15}: {1}'.format('score mse', test_score_mse))
    print('{0:15}: {1}'.format('bin-v1 mse', test_bin1_mse))
    print('{0:15}: {1}'.format('bin-v2 mse', test_bin2_mse))

    print('\nbin-v1 mse scores:')
    print('----------------------')
    for i, _ in enumerate(BIN_LOCATIONS1):
        bin_mse = single_bin1_mses.get(i)
        print('{0:15}: {1}'.format(f'mse bin {i}', bin_mse if bin_mse is not None else np.nan))

    print('\nbin-v2 mse scores:')
    print('----------------------')
    for i, _ in enumerate(BIN_LOCATIONS2):
        bin_mse = single_bin2_mses.get(i)
        print('{0:15}: {1}'.format(f'mse bin {i}', bin_mse if bin_mse is not None else np.nan))


def eval_model(model, dataloader, criterion):
    model.run_eval()

    total_loss_meter = AverageMeter()
    score_mse_meter = AverageMeter()
    total_bin1_mse_meter = AverageMeter()
    total_bin2_mse_meter = AverageMeter()
    single_bin1_score_mse_meters = {i: AverageMeter() for i, _ in enumerate(BIN_LOCATIONS1)}
    single_bin2_score_mse_meters = {i: AverageMeter() for i, _ in enumerate(BIN_LOCATIONS2)}

    num_batches = len(dataloader.dataset) // dataloader.batch_size

    columns = ['figure_id'] + [f'true_score_item_{i + 1}' for i in range(18)]
    columns += ['true_score_sum', 'true_bin_1', 'true_bin_2']
    columns += [f'pred_score_item_{i + 1}' for i in range(18)] + ['pred_total_score', 'pred_bin_1', 'pred_bin_2']
    columns += ['image_file', 'serialized_file']
    results_df = pd.DataFrame(columns=columns)

    for batch_idx, (images, labels, image_fp_npy, image_jpeg, image_id) in tqdm(enumerate(dataloader),
                                                                                total=num_batches):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            predictions = model(images.float())

        # clip outputs to ranges
        predictions[:, -1] = torch.clip(predictions[:, -1], 0, 36)
        predictions[:, :-1] = torch.clip(predictions[:, :-1], 0, 2)

        true_scores = labels[:, -1]
        true_bins1 = assign_bins(scores=true_scores, bin_locations=BIN_LOCATIONS1)
        true_bins2 = assign_bins(scores=true_scores, bin_locations=BIN_LOCATIONS2)

        pred_scores = predictions[:, -1]
        pred_bins1 = assign_bins(scores=pred_scores, bin_locations=BIN_LOCATIONS1)
        pred_bins2 = assign_bins(scores=pred_scores, bin_locations=BIN_LOCATIONS2)

        total_loss = criterion(predictions, labels)
        score_mse = criterion(pred_scores, true_scores)
        total_bin1_mse = criterion(pred_bins1, true_bins1)
        total_bin2_mse = criterion(pred_bins2, true_bins2)

        single_bin1_score_mses = mse_per_bin(pred_scores, true_scores, criterion, BIN_LOCATIONS1)
        single_bin2_score_mses = mse_per_bin(pred_scores, true_scores, criterion, BIN_LOCATIONS2)

        # write to df
        df_data = np.concatenate(
            [np.expand_dims(image_id, -1), labels.cpu(), true_bins1.cpu(), true_bins2.cpu(), predictions.cpu(),
             pred_bins1.cpu(), pred_bins2.cpu(), np.expand_dims(image_jpeg, -1),
             np.expand_dims(image_fp_npy, -1)], axis=1)
        results_df = pd.concat([results_df, pd.DataFrame(columns=columns, data=df_data)], ignore_index=True)

        # record loss
        total_loss_meter.update(total_loss.data, images.size()[0])
        score_mse_meter.update(score_mse.data, images.size()[0])
        total_bin1_mse_meter.update(total_bin1_mse.data, images.size()[0])
        total_bin2_mse_meter.update(total_bin2_mse.data, images.size()[0])

        for b, (bin_mse, n) in single_bin1_score_mses.items():
            single_bin1_score_mse_meters[b].update(bin_mse.data, n)

        for b, (bin_mse, n) in single_bin2_score_mses.items():
            single_bin2_score_mse_meters[b].update(bin_mse.data, n)

        if DEBUG:
            print(f'processed batch {batch_idx}')
            if batch_idx >= 5:
                break

    # compute_averages
    single_bin1_score_mses = {i: m.avg for i, m in single_bin1_score_mse_meters.items()}
    single_bin2_score_mses = {i: m.avg for i, m in single_bin2_score_mse_meters.items()}

    return (total_loss_meter.avg, score_mse_meter.avg, total_bin1_mse_meter.avg, single_bin1_score_mses,
            total_bin2_mse_meter.avg, single_bin2_score_mses, results_df)


def mse_per_bin(predictions, labels, criterion, bin_locations):
    binned_labels = torch.squeeze(assign_bins(labels, bin_locations=bin_locations))

    bin_mses = {}
    for i, _ in enumerate(bin_locations):
        n_labels_in_bin = int(sum(binned_labels == i))
        if n_labels_in_bin == 0:
            continue
        with torch.no_grad():
            bin_mses[i] = (criterion(predictions[binned_labels == i], labels[binned_labels == i]), n_labels_in_bin)

    return bin_mses


if __name__ == '__main__':
    main()
