import numpy as np
import os
import pandas as pd

import torch

from constants import *
from src.utils import map_to_score_grid, score_to_class
from src.dataloaders.dataloader_regression import get_regression_dataloader


class RegressionEvaluator:
    def __init__(self, model, results_dir, data_dir, batch_size=128, workers=8):
        self.model = model
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers

        self.predictions = None
        self.ground_truths = None
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

        # fetch checkpoint
        self.checkpoint = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')

        # init dataloader
        test_labels = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))
        self.dataloader = get_regression_dataloader(self.data_dir, labels_df=test_labels, batch_size=self.batch_size,
                                                    num_workers=self.workers, shuffle=False)

    def run_eval(self, save=True):
        # load checkpoint
        ckpt = torch.load(self.checkpoint, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

        ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(ckpt['state_dict'], strict=True)

        # get predictions
        predictions, ground_truths = self._run_inference()

        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        column_names = [f'score_item_{item + 1}' for item in range(N_ITEMS)]
        predictions_df = pd.DataFrame(columns=id_columns + column_names)
        ground_truths_df = pd.DataFrame(columns=id_columns + column_names)

        predictions_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        predictions_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(predictions)]
        predictions_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        ground_truths_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        ground_truths_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(predictions)]
        ground_truths_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        predictions_df[column_names] = predictions
        ground_truths_df[column_names] = ground_truths

        # convert continuous scores to discrete
        predictions_df[column_names] = predictions_df[column_names].applymap(map_to_score_grid)

        # turn scores into classes
        class_cols = [str(c).replace('score_', 'class_') for c in column_names]
        predictions_df[class_cols] = predictions_df[column_names].applymap(score_to_class)
        ground_truths_df[class_cols] = ground_truths_df[column_names].applymap(score_to_class)

        # compute total score
        predictions_df['total_score'] = predictions_df[column_names].sum(axis=1)
        ground_truths_df['total_score'] = ground_truths_df[column_names].sum(axis=1)

        self.predictions = predictions_df
        self.ground_truths = ground_truths_df

        if save:
            self.save_predictions()

    def _run_inference(self):
        self.model.eval()
        predictions, ground_truths = None, None

        for inputs, targets in self.dataloader:
            targets = targets.numpy()[:, :-1]

            if self.use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = self.model(inputs.float())
                outputs = outputs.cpu().numpy()[:, :-1]

            predictions = outputs if predictions is None else np.concatenate([predictions, outputs], axis=0)
            ground_truths = targets if ground_truths is None else np.concatenate([ground_truths, targets], axis=0)

        return predictions, ground_truths

    def save_predictions(self):
        if self.predictions is None or self.ground_truths is None:
            raise ValueError('predictions is None!')

        self.predictions.to_csv(os.path.join(self.results_dir, 'test_predictions.csv'))
        self.ground_truths.to_csv(os.path.join(self.results_dir, 'test_ground_truths.csv'))
