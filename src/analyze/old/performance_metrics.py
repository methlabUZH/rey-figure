from itertools import chain
import numpy as np
import os
import pandas as pd
from typing import List, Tuple

from src.utils import print_dataframe, mean_confidence_interval
from src.inference.utils import assign_bin_single
from constants import RESULTS_DIR, N_ITEMS, BIN_LOCATIONS1, BIN_LOCATIONS2, FOTO_FOLDERS

RESULTS_DIR = os.path.join(RESULTS_DIR, 'data-2018-2021-116x150-pp0-augmented/id-e3a767c4-9f39-47c8-bf83-294e0ec8e49e/')
N_DIGITS = 3

FILTER_FOTOS = False

# column names
ground_truth_columns_item_scores = [f'true_score_item_{i}' for i in range(1, N_ITEMS + 1)]
predictions_columns_item_scores = [f'pred_score_item_{i}' for i in range(1, N_ITEMS + 1)]

ground_truth_columns_score = ['true_score_sum']
predictions_columns_score = ['pred_total_score']

ground_truth_columns_bin1_score = ['true_bin_1']
predictions_columns_bin1_score = ['pred_bin_1']

ground_truth_columns_bin2_score = ['true_bin_2']
predictions_columns_bin2_score = ['pred_bin_2']

ground_truth_columns_classification = [f'true_score_item_{i}' for i in range(1, N_ITEMS + 1)]
predictions_columns_classification = [f'pred_score_item_{i}' for i in range(1, N_ITEMS + 1)]


def get_num_equal_per_row(row, pred_cols, gt_cols):
    predictions = row[pred_cols]
    ground_truths = row[gt_cols]
    n_correct = np.sum(predictions.values == ground_truths.values)
    return n_correct


def calc_per_item_metrics(predictions: pd.DataFrame, ground_truth_cols: List[str], predictions_cols: List[str]):
    predictions_copy = predictions.copy()

    # turn item scores into binary "item present" / "item not present" labels
    ground_truth_values = np.array(predictions_copy[ground_truth_cols].values.tolist())
    ground_truth_values = np.where(ground_truth_values > 0, 1, ground_truth_values)
    predictions_values = np.array(predictions_copy[predictions_cols].values.tolist())
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
    scores = pd.DataFrame(data, columns=[f'item {i}' for i in range(1, 19)])
    scores['index'] = ['sensitivity', 'specificity', 'g-mean', 'num-neg', 'num-pos']
    scores = scores.set_index('index')
    return scores


def calc_item_detection_ratio(predictions: pd.DataFrame, ground_truth_cols: List[str], predictions_cols: List[str]):
    predictions_copy = predictions.copy()

    # turn predictions and labels into binary
    ground_truth_values = np.array(predictions_copy[ground_truth_cols].values.tolist())
    ground_truth_values = np.where(ground_truth_values > 0, 1, ground_truth_values)
    predictions_copy[ground_truth_cols] = ground_truth_values

    predictions_values = np.array(predictions_copy[predictions_cols].values.tolist())
    predictions_values = np.where(predictions_values >= 0.25, 1, predictions_values)
    predictions_values = np.where(predictions_values < 0.25, 0, predictions_values)
    predictions_copy[predictions_cols] = predictions_values

    predictions_copy['correct-presence'] = predictions_copy.apply(
        lambda x: get_num_equal_per_row(x, predictions_cols, ground_truth_cols), axis=1)

    mean, ci_err = mean_confidence_interval(data=predictions_copy['correct-presence'].values)

    return mean, ci_err


def calc_mse(predictions: pd.DataFrame, ground_truth_cols: List[str], predictions_cols: List[str]):
    return np.mean((predictions[ground_truth_cols].values - predictions[predictions_cols].values) ** 2)


def calc_bin_metrics(predictions: pd.DataFrame, binning_version: int = 1) -> pd.DataFrame:
    assert binning_version in [1, 2]
    bins = BIN_LOCATIONS1 if binning_version == 1 else BIN_LOCATIONS2

    metrics_df = pd.DataFrame(columns=['bin-num', 'bin-boundaries', 'avg-detected', 'item-mse', 'score-mse'])
    ground_truth_bin_col = 'true_bin_1' if binning_version == 1 else 'true_bin_2'

    for i, b in enumerate(bins):
        predictions_filtered = predictions[predictions[ground_truth_bin_col] == i].copy()
        bin_total_score_mse = calc_mse(
            predictions_filtered, ground_truth_columns_score, predictions_columns_score)
        bin_item_score_mse = calc_mse(
            predictions_filtered, ground_truth_columns_item_scores, predictions_columns_item_scores)
        avg_detected = calc_item_detection_ratio(
            predictions_filtered, ground_truth_columns_item_scores, predictions_columns_item_scores)
        metrics_df.loc[i, :] = [i, b, avg_detected, bin_item_score_mse, bin_total_score_mse]

    return metrics_df


def get_predictions_hybrid(regression_predictions: pd.DataFrame,
                           classifiers_predictions: pd.DataFrame,
                           # regression_pred_cols,
                           # classification_pred_cols
                           ) -> pd.DataFrame:
    pred_classification_items = classifiers_predictions[predictions_columns_classification].values
    pred_regression_item_scores = regression_predictions[predictions_columns_item_scores].values
    pred_hybrid_values = pred_classification_items * pred_regression_item_scores
    pred_hybrid_total_scores = np.expand_dims(np.sum(pred_hybrid_values, axis=1), -1)
    pred_hybrid_bin1_scores = np.array(map(lambda x: assign_bin_single(x, BIN_LOCATIONS1), pred_hybrid_total_scores))
    pred_hybrid_bin2_scores = np.array(map(lambda x: assign_bin_single(x, BIN_LOCATIONS2), pred_hybrid_total_scores))

    hybrid_predictions = regression_predictions.copy()
    hybrid_predictions[predictions_columns_item_scores] = pred_hybrid_values
    hybrid_predictions[predictions_columns_score] = pred_hybrid_total_scores
    hybrid_predictions[predictions_columns_bin1_score] = pred_hybrid_bin1_scores
    hybrid_predictions[predictions_columns_bin2_score] = pred_hybrid_bin2_scores

    return hybrid_predictions


def filter_for_fotos(reg_preds_df: pd.DataFrame, cls_preds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = reg_preds_df.apply(func=lambda row: any([folder in row['image_file'] for folder in FOTO_FOLDERS]), axis=1)
    reg_preds_df = reg_preds_df[m]

    m = cls_preds_df.apply(func=lambda row: any([folder in row['image_file'] for folder in FOTO_FOLDERS]), axis=1)
    cls_preds_df = cls_preds_df[m]

    return reg_preds_df, cls_preds_df


def main(regressor_predictions_csv: str, classifiers_predictions_csv: str):
    regression_predictions = pd.read_csv(regressor_predictions_csv, index_col=0)
    classifiers_predictions = pd.read_csv(classifiers_predictions_csv, index_col=0)

    if FILTER_FOTOS:
        print(f'!!! ==> FILTERED DATA! ONLY CONTAINS FOTO IMAGE FROM FOLDERS {FOTO_FOLDERS} !!!')
        regression_predictions, classifiers_predictions = filter_for_fotos(regression_predictions,
                                                                           classifiers_predictions)

    print(f'==> # figures: {len(regression_predictions)}')
    hybrid_predictions = get_predictions_hybrid(regression_predictions, classifiers_predictions)

    # sensitivity, specificity, g-mean per item
    classif_item_metrics = calc_per_item_metrics(
        classifiers_predictions, ground_truth_columns_classification, predictions_columns_classification)
    regress_item_metrics = calc_per_item_metrics(
        regression_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)
    hybrid_item_metrics = calc_per_item_metrics(
        hybrid_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)

    # item mse
    regress_item_score_mse = calc_mse(
        regression_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)
    hybrid_item_score_mse = calc_mse(
        hybrid_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)

    # total score mse
    regress_total_score_mse = calc_mse(
        regression_predictions, ground_truth_columns_score, predictions_columns_score)
    hybrid_total_score_mse = calc_mse(
        hybrid_predictions, ground_truth_columns_score, predictions_columns_score)

    # total score mse bin 1
    regress_total_bin1_score_mse = calc_mse(
        regression_predictions, ground_truth_columns_bin1_score, predictions_columns_bin1_score)
    hybrid_total_bin1_score_mse = calc_mse(
        hybrid_predictions, ground_truth_columns_bin1_score, predictions_columns_bin1_score)

    # total score mse bin 2
    regress_total_bin2_score_mse = calc_mse(
        regression_predictions, ground_truth_columns_bin2_score, predictions_columns_bin2_score)
    hybrid_total_bin2_score_mse = calc_mse(
        hybrid_predictions, ground_truth_columns_bin2_score, predictions_columns_bin2_score)

    # avg number of items correctly detected
    classif_item_detection_ratio = calc_item_detection_ratio(
        classifiers_predictions, ground_truth_columns_classification, predictions_columns_classification)
    regress_item_detection_ratio = calc_item_detection_ratio(
        regression_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)
    hybrid_item_detection_ratio = calc_item_detection_ratio(
        hybrid_predictions, ground_truth_columns_item_scores, predictions_columns_item_scores)

    # global metrics
    data = [[classif_item_detection_ratio, np.nan, np.nan, np.nan, np.nan],
            [regress_item_detection_ratio, regress_item_score_mse, regress_total_score_mse,
             regress_total_bin1_score_mse, regress_total_bin2_score_mse],
            [hybrid_item_detection_ratio, hybrid_item_score_mse, hybrid_total_score_mse, hybrid_total_bin1_score_mse,
             hybrid_total_bin2_score_mse]]

    # bin metrics
    print('\n**** bin specific metrics ****\n')
    regress_per_bin_metrics = calc_bin_metrics(regression_predictions)
    hybrid_per_bin_metrics = calc_bin_metrics(hybrid_predictions)
    bin_table = generate_bin_metric_latex_table(regress_per_bin_metrics, hybrid_per_bin_metrics)
    print(bin_table)

    # print global metrics
    print('\n**** global metrics ****\n')
    global_metrics = pd.DataFrame(data, columns=[
        'avg. detected items', 'item mse', 'score mse', 'score-bin1 mse', 'score-bin2 mse'])
    global_metrics.index = ['classifiers', 'regressor', 'hybrid']
    print_dataframe(global_metrics, n=-1, n_digits=4)

    # generate table with item specific metrics
    print('\n**** item-specific metrics ****\n')
    print('\n// item-classifiers //\n')
    print_dataframe(classif_item_metrics, n=-1)
    latex_table_classifiers = generate_item_metrics_latex_table_classifiers(classif_item_metrics)
    print('latex table:')
    print(latex_table_classifiers)

    print('\n// regressor //\n')
    print_dataframe(regress_item_metrics, n=-1)

    print('\n// hybrid (regressor + item-classifiers) //\n')
    print_dataframe(hybrid_item_metrics, n=-1)

    # merged item metrics regressor + hybrid
    print('latex table regressor + hybrid')
    latex_table_regressors = generate_item_metrics_latex_table_regressors(regress_item_metrics, hybrid_item_metrics)
    print(latex_table_regressors)


def generate_item_metrics_latex_table_regressors(regress_metrics: pd.DataFrame, hybrid_metrics: pd.DataFrame,
                                                 verbose=False):
    regress_metrics = regress_metrics.transpose()
    hybrid_metrics = hybrid_metrics.transpose()

    sample_stats = regress_metrics.loc[:, ['num-neg', 'num-pos']]
    regress_metrics = regress_metrics.drop(columns=['num-neg', 'num-pos'])
    hybrid_metrics = hybrid_metrics.drop(columns=['num-neg', 'num-pos'])

    regress_metrics.columns = [f'regressor-{col}' for col in regress_metrics.columns]
    hybrid_metrics.columns = [f'hybrid-{col}' for col in hybrid_metrics.columns]

    all_metrics = pd.merge(regress_metrics, hybrid_metrics, left_index=True, right_index=True)
    reindex_cols = list(chain.from_iterable([[reg_col, hyb_col] for reg_col, hyb_col in zip(
        regress_metrics.columns, hybrid_metrics.columns)]))
    all_metrics = all_metrics.reindex(columns=reindex_cols)
    all_metrics = pd.merge(sample_stats, all_metrics, left_index=True, right_index=True)

    if verbose:
        print_dataframe(all_metrics)

    table = ""
    for idx, row in all_metrics.iterrows():
        neg_samples_ratio = row[0] / (row[0] + row[1])
        table += str(idx).replace("item ", "") + "& $" + str(round(100 * float(neg_samples_ratio), 1)) + "$"
        for j in np.arange(start=2, stop=len(row), step=2):
            max_idx = j + np.argmax(row[j:j + 2])
            k = j
            table += " &"
            while k < j + 2:
                if k == max_idx:
                    table += " & $\\mathbf{" + str(round(row[k], N_DIGITS)) + "}$ "
                else:
                    table += " & $" + str(round(row[k], N_DIGITS)) + "$ "
                k += 1
        table += "\\\\\n"

    return table


def generate_bin_metric_latex_table(regress_metrics: pd.DataFrame,
                                    hybrid_metrics: pd.DataFrame):
    merge_columns = ['bin-num', 'bin-boundaries']
    regress_metrics_rename_cols = [f'regressor-{col}' for col in regress_metrics.columns if col not in merge_columns]
    hybrid_metrics_rename_cols = [f'hybrid-{col}' for col in hybrid_metrics.columns if col not in merge_columns]
    regress_metrics.columns = merge_columns + regress_metrics_rename_cols
    hybrid_metrics.columns = merge_columns + hybrid_metrics_rename_cols

    all_metrics = pd.merge(regress_metrics, hybrid_metrics, on=merge_columns)
    reindex_cols = list(chain.from_iterable([[reg_col, hyb_col] for reg_col, hyb_col in zip(
        regress_metrics_rename_cols, hybrid_metrics_rename_cols)]))
    all_metrics = all_metrics.reindex(columns=merge_columns + reindex_cols)
    all_metrics = all_metrics.set_index('bin-num')

    print_dataframe(all_metrics)

    table = ""
    for bin_num, row in all_metrics.iterrows():
        bin_boundaries = row['bin-boundaries']
        reg_det_avg, reg_det_err = row['regressor-avg-detected']
        hyb_det_avg, hyb_det_err = row['hybrid-avg-detected']
        reg_item_mse = row['regressor-item-mse']
        hyb_item_mse = row['hybrid-item-mse']
        reg_score_mse = row['regressor-score-mse']
        hyb_score_mse = row['hybrid-score-mse']

        table += f'$[{bin_boundaries[0]}, \\ {bin_boundaries[1]})$ & '

        # detection rates
        if reg_det_avg < hyb_det_avg:
            table += f'${round(reg_det_avg, N_DIGITS)} \\pm {round(reg_det_err, N_DIGITS)}$ & '
            table += '$\\mathbf{' + f'{round(hyb_det_avg, N_DIGITS)} \\pm {round(hyb_det_err, N_DIGITS)}' + '}$ & '
        else:
            table += '$\\mathbf{' + f'{round(reg_det_avg, N_DIGITS)} \\pm {round(reg_det_err, N_DIGITS)}' + '}$ & '
            table += f'${round(hyb_det_avg, N_DIGITS)} \\pm {round(hyb_det_err, N_DIGITS)}$ & '

        if reg_item_mse < hyb_item_mse:
            table += '$\\mathbf{' + f'{round(reg_item_mse, N_DIGITS)}' + '}$ & '
            table += f'${round(hyb_item_mse, N_DIGITS)}$ & '
        else:
            table += f'${round(reg_item_mse, N_DIGITS)}$ & '
            table += '$\\mathbf{' + f'{round(hyb_item_mse, N_DIGITS)}' + '}$ & '

        if reg_score_mse < hyb_score_mse:
            table += '$\\mathbf{' + f'{round(reg_score_mse, N_DIGITS)}' + '}$ & '
            table += f'${round(hyb_score_mse, N_DIGITS)}$ '
        else:
            table += f'${round(reg_score_mse, N_DIGITS)}$ & '
            table += '$\\mathbf{' + f'{round(hyb_score_mse, N_DIGITS)}' + '}$ '

        table += '\\\\\n'

    return table


def generate_item_metrics_latex_table_classifiers(classif_metrics: pd.DataFrame, verbose=False):
    classif_metrics = classif_metrics.transpose()
    sample_stats = classif_metrics.loc[:, ['num-neg', 'num-pos']]
    classif_metrics = classif_metrics.drop(columns=['num-neg', 'num-pos'])
    all_metrics = pd.merge(sample_stats, classif_metrics, left_index=True, right_index=True)

    if verbose:
        print_dataframe(classif_metrics)

    table = ""
    for idx, row in all_metrics.iterrows():
        neg_samples_ratio = row[0] / (row[0] + row[1])
        table += str(idx).replace("item ", "") + "& $" + str(round(100 * float(neg_samples_ratio), 1)) + "$"
        for j in np.arange(start=2, stop=len(row), step=3):
            k = j
            while k < j + 3:
                table += " & $" + str(round(row[k], 3)) + "$ "
                k += 1
        table += "\\\\\n"

    return table


if __name__ == '__main__':
    main(os.path.join(RESULTS_DIR, 'rey-regressor/test-predictions.csv'),
         os.path.join(RESULTS_DIR, 'item-classifier/test-classification-predictions.csv'))
