import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
from tabulate import tabulate
from typing import Tuple
import torch

from constants import BIN_LOCATIONS1, BIN_LOCATIONS2, N_ITEMS
from src.models import get_reyclassifier, get_reyregressor
from src.dataloaders.dataloader_item_classification import get_item_classification_dataloader_eval
from src.dataloaders.dataloader_regression import get_regression_dataloader_eval
from src.training.train_utils import Logger
from src.utils import timestamp_human, timestamp_dir
from src.inference.utils import assign_bins
from src.inference.model_initialization import get_classifiers_checkpoints

DEBUG = False

# arg parser
parser = argparse.ArgumentParser()

# setup
parser.add_argument('--data-root', type=str, default=None)
parser.add_argument('--results-dir', type=str, default=None,
                    help='dir containing item-classifier and rey-regressor subdir')
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)

# architecture
parser.add_argument('--image-size', nargs='+', type=int, default=[116, 150])
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

    # weights for regressor
    reg_ckpt_fp = os.path.join(args.results_dir, 'rey-regressor/checkpoints/model_best.pth.tar')
    assert os.path.isfile(reg_ckpt_fp), 'no checkpoint for regressor found!'

    # weights for classifiers
    items_and_cls_ckpt_files = get_classifiers_checkpoints(os.path.join(args.results_dir, 'item-classifier'))
    avail_items = np.array(items_and_cls_ckpt_files)[:, 0].tolist()
    assert len(avail_items) == 18, 'classifier checkpoints missing!'

    # save terminal output to file
    eval_dir = os.path.join(args.results_dir, 'eval-hybrid/', timestamp_dir())
    os.makedirs(eval_dir)
    eval_file = os.path.join(eval_dir, 'eval_hybrid.txt' if not DEBUG else 'debug_eval_hybrid.txt')
    print(f'==> results will be saved to {eval_dir}')
    sys.stdout = Logger(print_fp=eval_file)

    # data
    print(f'==> data from {args.data_root}')
    labels_csv = os.path.join(args.data_root, 'test_labels.csv')
    labels = pd.read_csv(labels_csv)
    dataloader = get_regression_dataloader_eval(args.data_root, labels_df=labels, batch_size=args.batch_size,
                                                num_workers=args.workers, score_type=args.score_type)

    # setup models
    regressor = get_reyregressor(n_outputs=N_ITEMS, dropout=(0.0, 0.0), norm_layer_type=args.norm_layer)
    item_classifiers = {i + 1: get_reyclassifier(dropout=(.0, .0), norm_layer_type=args.norm_layer) for i in
                        range(N_ITEMS)}

    if use_cuda:
        regressor = torch.nn.DataParallel(regressor).cuda()
        item_classifiers = {k: torch.nn.DataParallel(m).cuda() for k, m in item_classifiers.items()}

    # load model weights for regressor
    reg_ckpt = torch.load(reg_ckpt_fp, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    if not use_cuda:
        reg_ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in reg_ckpt['state_dict'].items()}
    regressor.load_state_dict(reg_ckpt['state_dict'], strict=True)
    print(f'==> loaded checkpoint for regresor {reg_ckpt_fp}')

    # eval regressor
    print('\n----------------------\n')
    print(f'[{timestamp_human()}] start eval regressor')
    regressor_predictions = eval_regression_model(regressor, dataloader)

    # eval classifiers
    classifiers_predictions = pd.DataFrame()
    for item, ckpt_fp in items_and_cls_ckpt_files:
        print(f'[{timestamp_human()}] start eval item classifier {item}, ckpt: {ckpt_fp}')
        dataloader = get_item_classification_dataloader_eval(item,
                                                             args.data_root,
                                                             labels_df=labels,
                                                             batch_size=args.batch_size,
                                                             num_workers=args.workers,
                                                             max_samples=12 if DEBUG else -1)
        classifier = item_classifiers[item]
        item_predictions = eval_classification_model(classifier, dataloader, ckpt_fp, item)

        if item == 1:
            classifiers_predictions = item_predictions
            continue

        classifiers_predictions = pd.merge(classifiers_predictions, item_predictions,
                                           on=['figure_id', 'image_file', 'serialized_file'])

    print(tabulate(regressor_predictions.head(10), headers='keys', tablefmt='presto', floatfmt=".3f"))
    print(tabulate(classifiers_predictions.head(10), headers='keys', tablefmt='presto', floatfmt=".3f"))

    # save single predictions dfs
    save_as = os.path.join(eval_dir, 'regressor_predictions.csv')
    regressor_predictions.to_csv(save_as)  # noqa
    print(f'\n==> saved regressor predictions as {save_as}')
    save_as = os.path.join(eval_dir, 'classifiers_predictions.csv')
    classifiers_predictions.to_csv(save_as)  # noqa
    print(f'\n==> saved classifiers predictions as {save_as}')

    # compute predictions for hybrid model and save
    reg_cols = [f'pred_score_item_{i}' for i in range(1, N_ITEMS + 1)]
    cls_cols = [f'pred_item_{i}_present' for i in range(1, N_ITEMS + 1)]
    hybrid_predictions = regressor_predictions.copy()
    hybrid_predictions[reg_cols] = hybrid_predictions[reg_cols].values * classifiers_predictions[cls_cols].values
    save_preds_as = os.path.join(eval_dir, 'hybrid_predictions.csv')
    hybrid_predictions.to_csv(save_preds_as)  # noqa
    print(f'\n==> saved combined predictions as {save_preds_as}')

    # compute metrics for individual models
    classifiers_metrics = process_classifier_predictions(classifiers_predictions, verbose=True)
    reg_total_scores_metrics, reg_items_metrics = process_regression_predictions(regressor_predictions, verbose=True)
    hybrid_score_metrics, hybrid_items_metrics = process_regression_predictions(hybrid_predictions, verbose=True,
                                                                                title='hybrid metrics')

    # save metrics as csvs
    classifiers_metrics.to_csv(os.path.join(eval_dir, 'classifiers_metrics.csv'))
    reg_total_scores_metrics.to_csv(os.path.join(eval_dir, 'regressor_score_metrics.csv'))  # noqa
    reg_items_metrics.to_csv(os.path.join(eval_dir, 'regressor_items_metrics.csv'))  # noqa
    hybrid_score_metrics.to_csv(os.path.join(eval_dir, 'hybrid_score_metrics.csv'))  # noqa
    hybrid_items_metrics.to_csv(os.path.join(eval_dir, 'hybrid_items_metrics.csv'))  # noqa


def process_classifier_predictions(predictions: pd.DataFrame, verbose=False):
    # convert to numpy
    ground_truth_item_cols = [f'item_{i}_present' for i in range(1, N_ITEMS + 1)]
    ground_truth_values = np.array(predictions[ground_truth_item_cols].values.tolist())

    predictions_item_cols = [f'pred_item_{i}_present' for i in range(1, N_ITEMS + 1)]
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
    scores = pd.DataFrame(data, columns=[f'item_{i}' for i in range(1, N_ITEMS + 1)])
    scores['index'] = ['sensitivity', 'specificity', 'g-mean']
    scores = scores.set_index('index')

    if verbose:
        print('\n// item classifiers metrics //\n')
        print(tabulate(scores, headers='keys', tablefmt='presto', floatfmt=".3f"))

    return scores


def process_regression_predictions(preds_df: pd.DataFrame,
                                   verbose=False,
                                   title='regressor metrics') -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_item_cols = [f'pred_score_item_{i}' for i in range(1, N_ITEMS + 1)]
    ground_truth_item_cols = [f'true_score_item_{i}' for i in range(1, N_ITEMS + 1)]

    items_mse = (preds_df[pred_item_cols].subtract(preds_df[ground_truth_item_cols].values) ** 2).mean().mean()
    score_mse = ((preds_df['true_score_sum'] - preds_df['pred_total_score']) ** 2).mean()
    bin1_mse = ((preds_df['true_bin_1'] - preds_df['pred_bin_1']) ** 2).mean()
    bin2_mse = ((preds_df['true_bin_2'] - preds_df['pred_bin_2']) ** 2).mean()

    # compute mse for each item
    item_mses = [((preds_df[f'pred_score_item_{i}'] -
                   preds_df[f'true_score_item_{i}']) ** 2).mean() for i in range(1, N_ITEMS + 1)]

    # compute score mse for each bin with V1 binning
    bin_v1_mses = {}
    for i in range(len(BIN_LOCATIONS1)):
        bin_df = preds_df[preds_df['true_bin_1'] == i]
        bin_mse = ((bin_df['true_score_sum'] - bin_df['pred_total_score']) ** 2).mean()
        bin_v1_mses[i] = bin_mse

    # compute score mse for each bin with V2 binning
    bin_v2_mses = {}
    for i in range(len(BIN_LOCATIONS2)):
        bin_df = preds_df[preds_df['true_bin_2'] == i]
        bin_mse = ((bin_df['true_score_sum'] - bin_df['pred_total_score']) ** 2).mean()
        bin_v2_mses[i] = bin_mse

    # df with total score metrics
    total_score_metrics = pd.DataFrame(columns=['metric', 'mse'])
    total_score_metrics = total_score_metrics.set_index('metric')
    total_score_metrics.loc['items', 'mse'] = items_mse
    total_score_metrics.loc['score', 'mse'] = score_mse
    total_score_metrics.loc['score-binV1', 'mse'] = bin1_mse
    total_score_metrics.loc['score-binV2', 'mse'] = bin2_mse
    for i, _ in enumerate(BIN_LOCATIONS1):
        total_score_metrics.loc[f'score-binV1-bin-{i}', 'mse'] = bin_v1_mses.get(i)
    for i, _ in enumerate(BIN_LOCATIONS2):
        total_score_metrics.loc[f'score-binV2-bin-{i}', 'mse'] = bin_v2_mses.get(i)

    # df with per item score metrics
    item_scores_metrics = pd.DataFrame(columns=['metric'] + [f'item_{i}' for i in range(1, N_ITEMS + 1)])
    item_scores_metrics = item_scores_metrics.set_index('metric')
    item_scores_metrics.loc['mse', :] = item_mses
    classification_scores = compute_confusion_matrix(preds_df, pred_item_cols, ground_truth_item_cols)
    item_scores_metrics = pd.concat([item_scores_metrics, classification_scores])

    if verbose:
        print('\n// ' + title + ' //\n')
        print(tabulate(item_scores_metrics, headers='keys', tablefmt='presto', floatfmt=".3f"))
        print(tabulate(total_score_metrics, headers='keys', tablefmt='presto', floatfmt=".3f"))

    return total_score_metrics, item_scores_metrics


def compute_confusion_matrix(predictions: pd.DataFrame, pred_item_cols, ground_truth_item_cols):
    # turn item scores into binary "item present" / "item not present" labels
    ground_truth_values = np.array(predictions[ground_truth_item_cols].values.tolist())
    ground_truth_values = np.where(ground_truth_values > 0, 1, ground_truth_values)

    predictions_values = np.array(predictions[pred_item_cols].values.tolist())
    predictions_values = np.where(predictions_values >= 0.25, 1, predictions_values)
    predictions_values = np.where(predictions_values < 0.25, 0, predictions_values)

    # compute confusion matrices
    true_positives = np.sum(ground_truth_values * predictions_values, axis=0)
    false_positives = np.sum((1 - ground_truth_values) * predictions_values, axis=0)
    false_negatives = np.sum(ground_truth_values * (1 - predictions_values), axis=0)
    true_negatives = np.sum((1 - ground_truth_values) * (1 - predictions_values), axis=0)

    # num neg / pos samples
    num_pos = np.sum(ground_truth_values, axis=0)
    num_neg = np.sum(1 - ground_truth_values, axis=0)

    # compute sensitivity, specificity and g mean
    sensitivity_scores = true_positives / (true_positives + false_negatives)
    specificity_scores = true_negatives / (true_negatives + false_positives)
    g_mean_scores = np.sqrt(specificity_scores * sensitivity_scores)

    data = np.stack([sensitivity_scores, specificity_scores, g_mean_scores, num_neg, num_pos], axis=0)
    scores = pd.DataFrame(data, columns=[f'item_{i}' for i in range(1, 19)])
    scores['metric'] = ['sensitivity', 'specificity', 'g-mean', 'num-neg', 'num-pos']
    scores = scores.set_index('metric')

    return scores


def eval_regression_model(model, dataloader):
    model.eval()

    columns = ['figure_id'] + [f'true_score_item_{i + 1}' for i in range(18)]
    columns += ['true_score_sum', 'true_bin_1', 'true_bin_2']
    columns += [f'pred_score_item_{i + 1}' for i in range(18)] + ['pred_total_score', 'pred_bin_1', 'pred_bin_2']
    columns += ['image_file', 'serialized_file']
    results_df = pd.DataFrame(columns=columns)

    for batch_idx, (images, labels, image_fp_npy, image_jpeg, image_id) in enumerate(dataloader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # compute output
        with torch.no_grad():
            predictions = model(images.float())

        # clip outputs to ranges
        predictions[:, :-1] = torch.clip(predictions[:, :-1], 0, 2)
        predictions[:, -1] = torch.clip(predictions[:, -1], 0, 36)

        # assign bins
        true_bins1 = assign_bins(scores=labels[:, -1], bin_locations=BIN_LOCATIONS1)
        true_bins2 = assign_bins(scores=labels[:, -1], bin_locations=BIN_LOCATIONS2)
        pred_bins1 = assign_bins(scores=predictions[:, -1], bin_locations=BIN_LOCATIONS1)
        pred_bins2 = assign_bins(scores=predictions[:, -1], bin_locations=BIN_LOCATIONS2)

        # write to df
        df_data = np.concatenate(
            [np.expand_dims(image_id, -1), labels.cpu().float(), true_bins1.cpu(), true_bins2.cpu(),
             predictions.cpu().float(), pred_bins1.cpu(), pred_bins2.cpu(), np.expand_dims(image_jpeg, -1),
             np.expand_dims(image_fp_npy, -1)], axis=1)
        results_df = pd.concat([results_df, pd.DataFrame(columns=columns, data=df_data)], ignore_index=True)

        if DEBUG:
            print(f'processed batch {batch_idx}')
            if batch_idx >= 5:
                break

    results_df[columns[1:-2]] = results_df[columns[1:-2]].astype(float)

    return results_df


def eval_classification_model(model, dataloader, checkpoint_fp, item):
    # load checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    if not use_cuda:
        checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    columns = ['figure_id', 'image_file', 'serialized_file', f'item_{item}_present', f'pred_item_{item}_present']
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


if __name__ == '__main__':
    main()
