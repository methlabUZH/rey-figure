import pandas as pd
import numpy as np
import os
from tabulate import tabulate

from constants import *

SCORE_COLUMNS = [f'score_item_{i + 1}' for i in range(N_ITEMS)] + ['total_score']
CLASS_COLUMNS = [f'class_item_{i + 1}' for i in range(N_ITEMS)]

ABSOLUTE_ERROR = 'absolute_error'
SQUARED_ERROR = 'squared_error'
NUM_MISCLASS = 'num_misclassified'


def compute_errors(predictions, ground_truths):
    results_df = pd.DataFrame(columns=['figure_id', ABSOLUTE_ERROR, SQUARED_ERROR, NUM_MISCLASS])
    results_df['figure_id'] = predictions['figure_id']

    results_df[ABSOLUTE_ERROR] = np.abs(predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score'])
    results_df[SQUARED_ERROR] = (predictions.loc[:, 'total_score'] - ground_truths.loc[:, 'total_score']) ** 2
    results_df[NUM_MISCLASS] = np.sum(predictions.loc[:, CLASS_COLUMNS] != ground_truths.loc[:, CLASS_COLUMNS], axis=1)

    results_df = results_df.set_index('figure_id')

    return results_df

# if __name__ == '__main__':
#     res_dir = '/Users/maurice/Desktop/temp-results/aug/'
#     prefix = 'rotation_[0.0, 5.0]'
#     predictions = pd.read_csv(os.path.join(res_dir, f'{prefix}-test_predictions.csv'))
#     ground_truths = pd.read_csv(os.path.join(res_dir, f'{prefix}-test_ground_truths.csv'))
#     compute_errors(predictions, ground_truths)
