#!/usr/bin/env python
# coding: utf-8

import os
from pydoc import classname
import numpy as np
import pandas as pd
import argparse
import numpy as np
import pandas as pd
import sys
from tabulate import tabulate
from constants import *
from src.training.train_utils import Logger
from src.models import get_classifier
from src.evaluate import MultilabelEvaluator
from src.evaluate.utils import *
from collections import Counter
import time
import copy

# Set models_root to the top directory from which the script starts collecting all models below into an ensemble 
models_root = '/home/ubuntu/projects/rey-figure/ensembles/new_runs'
test_label_path = '/home/ubuntu/projects/rey-figure/ensembles/new_runs/final-aug/rey-multilabel-classifier/test_ground_truths.csv'
ground_truths = pd.read_csv(test_label_path)
result_dir = './ensembles'
# save terminal output to file
sys.stdout = Logger(print_fp=os.path.join(result_dir, f'{time.time()}_ensemble_class_eval_out.txt'))
print(f"Loading ensemble models from {models_root}")

_CLASS_LABEL_COLS = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
_CLASS_PRED_COLS = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_LABEL_COLS = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)]
_SCORE_PRED_COLS = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)]
_NUM_CLASSES = 4


def class_to_score(class_nb):
    if class_nb == 0: return 0
    if class_nb == 1: return 0.5
    if class_nb == 2: return 1
    if class_nb == 3: return 2
    raise Exception("not a valid class")


def find_files(filename, search_path):
    result = []
    # Wlaking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result


# create the ensemble dataframes
filename = 'test_predictions.csv'
files = find_files(filename, models_root)
print(f"Building an ensemble from {len(files)} models:")
print(*files, sep='\n')

dataframes = []
for file in files:
    dataframes.append(pd.read_csv(file))

ensemble_df = pd.DataFrame().reindex_like(dataframes[0])  # just for initialization
# ensemble_df.to_csv('ensemble_start.csv')

test_df = copy.deepcopy(dataframes[1])

# Write the ensemble classes by majority vote 
class_names = [f'class_item_{i}' for i in range(1, 19)]
for i, class_name in enumerate(class_names):
    # if i > 3: break
    for j in range(len(ensemble_df[class_name])):
        # if j > 5: break
        class_votes = []
        for dataframe in dataframes:
            class_votes.append(dataframe[class_name][j])
        majority_vote = Counter(class_votes).most_common(1)[0][0]
        old = ensemble_df[class_name][j]
        ensemble_df[class_name][j] = majority_vote
        if i % 100 == 0:  # old != ensemble_df[class_name][j]:
            # print(f"changed class at {class_name, j}")
            # print("votes: ", class_votes, "majority: ", majority_vote)
            pass

        # Now write the item scores for the majority votes computed above
score_names = [f'score_item_{i}' for i in range(1, 19)]
for i, score_name in enumerate(score_names):
    for j in range(len(ensemble_df[score_name])):
        old = ensemble_df[score_name][j]
        # Option 1: translate majority vote to score 
        ensemble_df[score_name][j] = class_to_score(ensemble_df[class_names[i]][j])

        # Option 2: average the raw scores to new scores 
        # scores = []
        # for dataframe in dataframes:
        #    scores.append(dataframe[score_name][j])
        # ensemble_df[score_name][j] = np.mean(scores)
        # if ensemble_df[score_name][j] != old:
        # print(f"changed score at {score_name, j}")
        # print(scores, np.mean(scores), ensemble_df[score_name][j])

# Now compute the total score for each column
for j in range(len(ensemble_df['score_item_1'])):  # iterate row wise
    total_score = 0
    for i, score_name in enumerate(score_names):
        total_score += ensemble_df[score_name][j]
    # print(f"total score", total_score)
    ensemble_df['total_score'][j] = total_score
    # print(ensemble_df['total_score'][j])

# print(test_df == ensemble_df)
# print(ensemble_df.compare(test_df))
# print(f"dataframe equals original one: {ensemble_df.equals(test_df)}")

# ensemble_df.to_csv('ensemble_end.csv')

# Compute the metrics of the ensemble 
predictions = ensemble_df

# ------- item specific scores -------
item_accuracy_scores = compute_accuracy_scores(
    predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])

item_mse_scores = compute_total_score_error(
    predictions, ground_truths, columns=[f"class_item_{i + 1}" for i in range(N_ITEMS)])

# ------- toal score mse -------
total_score_mse = compute_total_score_error(predictions, ground_truths, ["total_score"])[0]

# ------- toal score mae -------
total_score_mae = compute_total_score_error(predictions, ground_truths, ["total_score"], which='mae')[0]

print('---------- Item Scores ----------')
print_df = pd.DataFrame(data=np.stack([item_accuracy_scores, item_mse_scores], axis=0),
                        columns=[f'item-{i + 1}' for i in range(N_ITEMS)],
                        index=['Accuracy', 'MSE'])
print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

print(f'\nOverall Score MSE: {total_score_mse}')
print(f'\nOverall Score MAE: {total_score_mae}')
