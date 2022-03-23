#!/usr/bin/env python
# coding: utf-8

import os 
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


models_root = '/home/ubuntu/projects/rey-figure/results/'
result_dir = './ensemble'
# save terminal output to file
sys.stdout = Logger(print_fp=os.path.join(result_dir, f'{time.time()}_ensemble_eval_out.txt'))
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

test_label_path = '/home/ubuntu/projects/rey-figure/results/data-2018-2021-116x150-pp0/id-312124a4-a223-41a3-8b0c-115ea9cbda53/rey-multilabel-classifier/test_ground_truths.csv'
ground_truths = pd.read_csv(test_label_path)

# create the ensemble dataframes
filename = 'test_predictions.csv'
files = find_files(filename, models_root)
print(f"Building an ensemble from {len(files)} models:")
print(*files, sep='\n')

dataframes = []
for file in files:
    dataframes.append(pd.read_csv(file))
ensemble_df = dataframes[0].copy() # just for initialization 

# Write the ensemble classes by majority vote 
class_names = [f'class_item_{i}' for i in range(1,19)]
for i, class_name in enumerate(class_names):
    #if i > 3: break 
    for j in range(len(ensemble_df[class_name])):
        #if j > 5: break 
        class_votes = []
        for dataframe in dataframes:
            class_votes.append(dataframe[class_name][j])
        majority_vote = Counter(class_votes).most_common(1)[0][0]
        ensemble_df[class_name][j] = majority_vote
        

# Now write the item scores for the majority votes computed above 
score_names = [f'score_item_{i}' for i in range(1,19)]
for i, score_name in enumerate(score_names):
    for j in range(len(ensemble_df[score_name])):
        # translate majority vote to score 
        #ensemble_df[score_name][j] = class_to_score(ensemble_df[class_names[i]][j])
        scores = []
        for dataframe in dataframes:
            scores.append(dataframe[score_name][j])
        ensemble_df[score_name][j] = np.mean(scores)


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
                            columns=[f'item-{i+1}' for i in range(N_ITEMS)],
                            index=['Accuracy', 'MSE'])
print(tabulate(print_df, headers='keys', tablefmt='presto', floatfmt=".3f"))

print(f'\nOverall Score MSE: {total_score_mse}')
print(f'\nOverall Score MAE: {total_score_mae}')