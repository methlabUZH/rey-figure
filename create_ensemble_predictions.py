import argparse
import os
import pandas as pd

from constants import SCORE_COLUMNS

from tabulate import tabulate

_BASE_DIR = './results/lukas-final/'
_ML_PREDICTIONS_ROOT = _BASE_DIR + 'final-aug_01/rey-multilabel-classifier/'
_RG_PREDICTIONS_ROOT = _BASE_DIR + 'final-reg_01/rey-regressor-v2/'

_PRED_CSV_PATTERN = '{}test_predictions.csv'
_GT_CSV_PATTERN = '{}test_ground_truths.csv'

_REG_ITEMS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 13, 16]
_ML_ITEMS = [3, 8, 11, 14, 15, 17, 18]


def main(prefix_list):
    for prefix in prefix_list:
        csv_reg = os.path.join(_RG_PREDICTIONS_ROOT, _PRED_CSV_PATTERN.format(prefix))
        csv_ml = os.path.join(_ML_PREDICTIONS_ROOT, _PRED_CSV_PATTERN.format(prefix))
        csv_gt = os.path.join(_ML_PREDICTIONS_ROOT, _GT_CSV_PATTERN.format(prefix))
        create_ensemble(csv_reg, csv_ml, csv_gt, prefix=prefix)


def create_ensemble(csv_reg, csv_ml, csv_gt, prefix=""):
    preds_reg = pd.read_csv(csv_reg)
    preds_ml = pd.read_csv(csv_ml)
    gt = pd.read_csv(csv_gt)

    # take columns for regressor
    reg_columns = ['figure_id', 'image_file', 'serialized_file']
    reg_columns = reg_columns + [f'score_item_{i}' for i in _REG_ITEMS]
    reg_columns = reg_columns + [f'class_item_{i}' for i in _REG_ITEMS]
    preds_reg = preds_reg[reg_columns]

    # take columns for multilabel classifier
    ml_columns = ['figure_id', 'image_file', 'serialized_file']
    ml_columns = ml_columns + [f'score_item_{i}' for i in _ML_ITEMS]
    ml_columns = ml_columns + [f'class_item_{i}' for i in _ML_ITEMS]
    preds_ml = preds_ml[ml_columns]

    joined_dfs = pd.merge(left=preds_ml, right=preds_reg, how='left',
                          left_on=['figure_id', 'image_file', 'serialized_file'],
                          right_on=['figure_id', 'image_file', 'serialized_file'])
    joined_dfs['total_score'] = joined_dfs[SCORE_COLUMNS].sum(axis=1)

    # save joinded df
    prefix = prefix + 'ensemble-'
    save_preds_as = os.path.join(_BASE_DIR, 'ensemble_predictions', _PRED_CSV_PATTERN.format(prefix))
    save_gt_as = os.path.join(_BASE_DIR, 'ensemble_predictions', 'test_ground_truths.csv')
    joined_dfs.to_csv(save_preds_as)
    gt.to_csv(save_gt_as)
    print(f'saved predictions as {save_preds_as}')


if __name__ == '__main__':
    angles = [(float(a1), float(a2)) for a1, a2 in zip(range(0, 45, 5), range(5, 50, 5))]
    distortions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contrast = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    brightness = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    br_prefixes = [f'brightness_{b}-' for b in brightness]
    rt_prefixes = ['rotation_[{}, {}]-'.format(*a) for a in angles]
    ds_prefixes = [f'perspective_{d}-' for d in distortions]
    ct_prefixes = [f'contrast_{c}-' for c in contrast]

    main(prefix_list=br_prefixes + rt_prefixes + ds_prefixes + ct_prefixes)
