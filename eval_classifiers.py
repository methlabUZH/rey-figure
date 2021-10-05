import argparse
from datetime import datetime
import numpy as np
import os
import pandas as pd
import random
import sys
from tabulate import tabulate
from typing import Tuple, List

import torch

from src.utils.eval_item_classification_dataloader import get_eval_item_classiciation_dataloader
from src.utils.helpers import timestamp_human, Logger
from src.models import get_reyclassifier

DEBUG = False
default_data_dir = '/Users/maurice/phd/src/rey-figure/data/serialized-data/scans-2018-116x150'
default_results_dir = '/Users/maurice/phd/src/rey-figure/results/classifiers/scans-2018-116x150-augmented'

# setup arg parser
parser = argparse.ArgumentParser()
# setup
parser.add_argument('--data-root', type=str, default=default_data_dir, required=False)
parser.add_argument('--results-dir', type=str, default=default_results_dir, required=False)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# architecture
parser.add_argument('--image-size', nargs='+', type=int, default=[116, 150])
parser.add_argument('--norm-layer', type=str, default='batch_norm', choices=['batch_norm', 'group_norm'])

# misc
parser.add_argument('--seed', default=8, type=int)

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
    print('==> debugging on!' if DEBUG else '')

    # save terminal output to file
    out_file = os.path.join(args.results_dir, 'eval_classifier.txt' if not DEBUG else 'debug_eval_classifier.txt')
    print(f'==> results will be saved to {out_file}')
    sys.stdout = Logger(print_fp=out_file)

    # data
    print(f'==> data from {args.data_root}')
    labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)

    # setup model
    model = get_reyclassifier(dropout=(0., 0.), norm_layer_type=args.norm_layer)

    # get checkpoint files
    items_and_checkpoint_files = get_checkpoints(args.results_dir)
    avail_items = np.array(items_and_checkpoint_files)[:, 0].tolist()

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    print('\n----------------------\n')
    predictions = pd.DataFrame()
    for item, ckpt in items_and_checkpoint_files:
        print(f'[{timestamp_human()}] start eval item {item}, ckpt: {ckpt}')

        dataloader = get_eval_item_classiciation_dataloader(item, args.data_root, labels_df=labels,
                                                            batch_size=args.batch_size, num_workers=args.workers,
                                                            max_samples=12 if DEBUG else -1)
        item_predictions = eval_model(model, dataloader, ckpt, item)

        if item == 1:
            predictions = item_predictions
            continue

        predictions = pd.merge(predictions, item_predictions, on=['figure_id', 'image_file', 'serialized_file'])

    print(f'[{timestamp_human()}] finished eval')
    print('\n---------------------- predictions\n')
    print(tabulate(predictions.head(10), headers='keys', tablefmt='presto', floatfmt=".3f"))
    csv_file = os.path.join(args.results_dir, 'test-classification-predictions.csv')
    predictions.to_csv(csv_file)  # noqa
    print(f'\n==> saved predictions as {csv_file}')

    # convert to numpy
    ground_truth_item_cols = [f'true_score_item_{i}' for i in avail_items]
    ground_truth_values = np.array(predictions[ground_truth_item_cols].values.tolist())

    predictions_item_cols = [f'pred_score_item_{i}' for i in avail_items]
    predictions_values = np.array(predictions[predictions_item_cols].values.tolist())

    # compute confusion matrices
    true_positives = np.sum(ground_truth_values * predictions_values, axis=0)
    false_positives = np.sum((1 - ground_truth_values) * predictions_values, axis=0)
    false_negatives = np.sum(ground_truth_values * (1 - predictions_values), axis=0)
    true_negatives = np.sum((1 - ground_truth_values) * (1 - predictions_values), axis=0)

    # compote sensitivity, specificity and g mean
    sensitivity_scores = true_positives / (true_positives + false_negatives)
    specificity_scores = true_negatives / (true_negatives + false_positives)
    g_mean_scores = np.sqrt(specificity_scores * sensitivity_scores)

    # print scores
    data = np.stack([sensitivity_scores, specificity_scores, g_mean_scores], axis=0)
    scores = pd.DataFrame(data, columns=[f'item {i}' for i in avail_items])
    scores['index'] = ['sensitivity', 'specificity', 'g-mean']
    scores = scores.set_index('index')
    print('\n---------------------- scores\n')
    print(tabulate(scores, headers='keys', tablefmt='presto', floatfmt=".3f"))


def eval_model(model, dataloader, checkpoint_fp, item):
    # load checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    if not use_cuda:
        checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    columns = ['figure_id', 'image_file', 'serialized_file', f'true_score_item_{item}', f'pred_score_item_{item}']
    results_df = pd.DataFrame(columns=columns)

    for batch_idx, (images, labels, image_fp_npy, image_jpeg, image_id) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            class_probs = model(images.float())

        # clip outputs to ranges
        _, predicted_classes = torch.topk(class_probs, 1, 1, True, True)

        # write to df
        df_data = np.concatenate([
            np.expand_dims(image_id, axis=-1),
            np.expand_dims(image_jpeg, axis=-1),
            np.expand_dims(image_fp_npy, axis=-1),
            np.expand_dims(labels.cpu().numpy(), axis=-1),
            predicted_classes.cpu().numpy()
        ], axis=1)

        results_df = pd.concat([results_df, pd.DataFrame(columns=columns, data=df_data)], ignore_index=True)

        if DEBUG:
            print(f'processed batch {batch_idx}')
            if batch_idx >= 5:
                break

    results_df[columns[3:]] = results_df[columns[3:]].astype(int)

    return results_df


def get_checkpoints(results_root, max_ckpts=-1) -> List[Tuple[int, str]]:
    checkpoints = []

    for i in range(1, 19):
        classifier_root = os.path.join(results_root, f'aux_classifier_item_{i}')

        if not os.path.exists(classifier_root):
            print(f'no checkpoints found for item {i}!')
            continue

        hyperparam_dir = os.listdir(classifier_root)[0]
        timestamps = os.listdir(os.path.join(classifier_root, hyperparam_dir))
        max_timestamp = max([datetime.strptime(ts, '%Y-%m-%d_%H-%M-%S.%f') for ts in timestamps])
        max_timestamp = max_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
        checkpoint = os.path.join(classifier_root, hyperparam_dir, max_timestamp, 'checkpoints/model_best.pth.tar')

        if not os.path.isfile(checkpoint):
            print(f'no checkpoints found for item {i}!')
            continue

        checkpoints.append((i, checkpoint))

    return checkpoints if max_ckpts == -1 else checkpoints[:max_ckpts]


if __name__ == '__main__':
    main()
