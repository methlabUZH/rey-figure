import os
import argparse
import numpy as np
import pandas as pd
import random
import sys
from tabulate import tabulate

import torch

from src.dataloaders.dataloader_item_classification import get_item_classification_dataloader
from src.inference.model_initialization import get_classifiers_checkpoints
from src.training.train_utils import Logger
from src.utils import timestamp_human, class_to_score
from src.models import get_classifier

# setup arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='')
parser.add_argument('--results-dir', type=str, default='')
parser.add_argument('--arch', type=str, default=None, required=False)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--workers', default=8, type=int)

args = parser.parse_args()

_SEED = 7
_NUM_CLASSES = 4

USE_CUDA = torch.cuda.is_available()
np.random.seed(_SEED)
random.seed(_SEED)
torch.manual_seed(_SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(_SEED)


def main():
    # save terminal output to file
    out_file = os.path.join(args.results_dir, 'eval_classifiers.txt')
    print(f'==> results will be saved to {out_file}')
    sys.stdout = Logger(print_fp=out_file)

    # data
    print(f'==> data from {args.data_root}')
    labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)

    # setup model
    model = get_classifier(arch=args.arch, num_classes=_NUM_CLASSES)

    # get checkpoint files
    items_and_checkpoint_files = get_classifiers_checkpoints(args.results_dir)
    avail_items = np.array(items_and_checkpoint_files)[:, 0].tolist()

    if USE_CUDA:
        model = torch.nn.DataParallel(model).cuda()

    print('\n----------------------\n')
    predictions = pd.DataFrame()
    for i, (item, ckpt) in enumerate(items_and_checkpoint_files):
        print(f'[{timestamp_human()}] start eval item {item}, ckpt: {ckpt}')

        dataloader = get_item_classification_dataloader(item, args.data_root, labels_df=labels,
                                                        batch_size=args.batch_size, num_workers=args.workers,
                                                        shuffle=False, is_binary=False)
        item_predictions = eval_model(model, dataloader, ckpt, item)

        if i == 0:
            predictions = item_predictions
            continue

        predictions = pd.merge(predictions, item_predictions, on=['figure_id', 'image_file', 'serialized_file'])

    print(f'[{timestamp_human()}] finished eval')
    print('\n---------------------- predictions\n')
    print(tabulate(predictions.head(10), headers='keys', tablefmt='presto', floatfmt=".3f"))
    csv_file = os.path.join(args.results_dir, 'test-classification-predictions.csv')
    predictions.to_csv(csv_file)  # noqa
    print(f'\n==> saved predictions as {csv_file}')

    # compute accuracy for each item
    item_accuracy_scores = [np.mean(
        predictions.loc[:, f'true_class_item_{i}'] == predictions.loc[:, f'pred_class_item_{i}'])
        for i in avail_items
    ]

    # compute class specific acc for each item
    class_conditional_accuracy_scores = [[np.mean(
        predictions.loc[predictions[f'true_class_item_{i}'] == k, f'pred_class_item_{i}'] == k)
        for k in range(_NUM_CLASSES)] for i in avail_items]

    # compute MSE for each item
    item_mse_scores = [np.mean(
        (predictions.loc[:, f'true_score_item_{i}'] - predictions.loc[:, f'pred_score_item_{i}']) ** 2)
        for i in avail_items
    ]

    # compute overall MSE
    pred_total_scores = predictions.loc[:, [f'pred_score_item_{i}' for i in avail_items]].sum(axis=1)
    true_total_scores = predictions.loc[:, [f'true_score_item_{i}' for i in avail_items]].sum(axis=1)
    score_mse = np.mean((pred_total_scores - true_total_scores) ** 2)

    # print class conditianl accuracy score
    print('\n*** Class conditional Accuracy per item\n')
    print_df = pd.DataFrame(np.stack(class_conditional_accuracy_scores, axis=0),
                            columns=[f'Class {k}' for k in range(_NUM_CLASSES)],
                            index=[f"Item {i}" for i in avail_items])

    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    # print scores
    print('\n*** Accuracy and MSE per item\n')
    print_df = pd.DataFrame(np.stack([item_accuracy_scores, item_mse_scores], axis=0),
                            columns=[f'item {i}' for i in avail_items])
    print_df['total'] = np.array([np.nan, score_mse])
    print_df['index'] = ['Accuracy', 'MSE']
    print_df = print_df.set_index('index')
    print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))


def eval_model(model, dataloader, checkpoint_fp, item):
    # load checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=torch.device('cuda' if USE_CUDA else 'cpu'))
    if not USE_CUDA:
        checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.run_eval()

    # loop through data
    predictions = np.empty(shape=(0, 2), dtype=float)
    for batch_idx, (images, labels) in enumerate(dataloader):
        if USE_CUDA:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            class_probs = model(images.float())

        # clip outputs to ranges
        predicted_classes = torch.argmax(class_probs, dim=1)

        # concatenate data
        batch_preds = np.concatenate([np.expand_dims(labels.cpu().numpy(), -1),
                                      np.expand_dims(predicted_classes.cpu().numpy(), -1)], axis=1)

        predictions = np.concatenate([predictions, batch_preds], axis=0)

    # setup df with results
    columns = ['figure_id', 'image_file', 'serialized_file', f'true_class_item_{item}', f'pred_class_item_{item}',
               f'true_score_item_{item}', f'pred_score_item_{item}']
    results_df = pd.DataFrame(columns=columns)

    # add filepaths
    results_df['figure_id'] = dataloader.dataset.image_ids[:len(predictions)]
    results_df['image_file'] = dataloader.dataset.jpeg_filepaths[:len(predictions)]
    results_df['serialized_file'] = dataloader.dataset.npy_filepaths[:len(predictions)]

    # add class predictions
    results_df[[f'true_class_item_{item}', f'pred_class_item_{item}']] = predictions

    # convert classes to scores
    results_df[f'true_score_item_{item}'] = results_df[f'true_class_item_{item}'].apply(class_to_score)
    results_df[f'pred_score_item_{item}'] = results_df[f'pred_class_item_{item}'].apply(class_to_score)

    return results_df


if __name__ == '__main__':
    main()
