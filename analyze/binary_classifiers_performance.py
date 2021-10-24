"""
Script to evaluate the confusion matrix based on prediction made by the neural net regressor.
"""
from itertools import chain
import numpy as np
import pandas as pd
from tabulate import tabulate


def pandas_to_latex(df: pd.DataFrame, n_digits: int = 4):
    table = ""
    for idx, row in df.iterrows():
        neg_samples_ratio = row[0] / (row[0] + row[1])
        table += str(idx).replace("item ", "") + "& $" + str(round(100*float(neg_samples_ratio), 1)) + "$"
        for j in np.arange(start=2, stop=len(row), step=2):
            if row[j] > row[j + 1]:
                table += "& $\\mathbf{" + str(round(row[j], n_digits)) + "}$ & $" + str(
                    round(row[j + 1], n_digits)) + "$"
            elif row[j + 1] > row[j]:
                table += "& $" + str(round(row[j], n_digits)) + "$ & $\\mathbf{" + str(
                    round(row[j + 1], n_digits)) + "}$"
            else:
                table += "& $" + str(round(row[j], n_digits)) + "$ & $" + str(round(row[j + 1], n_digits)) + "$"

        table += "\\\\\n"

    return table


def compute_confusion_matrix(predictions_csv):
    predictions = pd.read_csv(predictions_csv)

    # turn item scores into binary "item present" / "item not present" labels
    ground_truth_item_cols = [f'true_score_item_{i}' for i in range(1, 19)]
    ground_truth_values = np.array(predictions[ground_truth_item_cols].values.tolist())
    ground_truth_values = np.where(ground_truth_values > 0, 1, ground_truth_values)

    predictions_item_cols = [f'pred_score_item_{i}' for i in range(1, 19)]
    predictions_values = np.array(predictions[predictions_item_cols].values.tolist())
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
    print(tabulate(scores, headers='keys', tablefmt='presto', floatfmt=".4f"))

    return scores


def main():
    results_dir = '/Users/maurice/phd/src/rey-figure/results/'
    reg_csv = results_dir + 'sum-score/scans-2018-116x150-augmented/deep-cnn/'
    reg_csv += 'epochs=500_bs=64_lr=0.0001_gamma=1.0_wd=0.0_dropout=[0.3, 0.5]_bn-momentum=0.01_beta=0.0/'
    reg_csv += '2021-09-28_12-48-32.913/test-predictions.csv'
    cls_csv = results_dir + 'scans-2018-116x150-augmented/test-classification-predictions.csv'

    regression_score_df = compute_confusion_matrix(reg_csv)
    classifier_score_df = compute_confusion_matrix(cls_csv)

    # merge dfs into one
    regression_score_df = regression_score_df.transpose()
    classifier_score_df = classifier_score_df.transpose()
    sample_stats = regression_score_df.loc[:, ['num-neg', 'num-pos']]
    regression_score_df = regression_score_df.drop(columns=['num-neg', 'num-pos'])
    classifier_score_df = classifier_score_df.drop(columns=['num-neg', 'num-pos'])

    regression_score_df.columns = [f'regressor-{col}' for col in regression_score_df.columns]
    classifier_score_df.columns = [f'classifier-{col}' for col in classifier_score_df.columns]
    all_scores = pd.merge(regression_score_df, classifier_score_df, left_index=True, right_index=True)
    reindex_cols = list(chain.from_iterable([[reg_col, cls_col] for reg_col, cls_col in zip(
        regression_score_df.columns, classifier_score_df.columns)]))
    all_scores = all_scores.reindex(columns=reindex_cols)
    all_scores = pd.merge(sample_stats, all_scores, left_index=True, right_index=True)

    print(tabulate(all_scores, headers='keys', tablefmt='presto', floatfmt=".3f"))

    latex_table = pandas_to_latex(all_scores)
    print(latex_table)


if __name__ == '__main__':
    main()
