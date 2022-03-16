import numpy as np
import os
import pandas as pd

import torch

from constants import *
from src.dataloaders.rocf_dataloader import get_dataloader
from src.utils import class_to_score


class MultilabelEvaluator:
    def __init__(self, model, image_size, results_dir, data_dir, batch_size=128, workers=8):
        self.model = model
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.workers = workers

        self.predictions = None
        self.ground_truths = None
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

        self.checkpoints = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')
        # self.items = None

        # init dataloader
        test_labels = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))
        self.dataloader = get_dataloader(data_root=self.data_dir, labels=test_labels, label_type=CLASSIFICATION_LABELS,
                                         batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
                                         augment=False, image_size=self.image_size)

    def run_eval(self, save=True):
        self.predictions, self.ground_truths = self._make_predictions_single_model()

        if save:
            self.save_predictions()

    def save_predictions(self):
        if self.predictions is None or self.ground_truths is None:
            raise ValueError('predictions is None!')

        self.predictions.to_csv(os.path.join(self.results_dir, 'test_predictions.csv'))
        self.ground_truths.to_csv(os.path.join(self.results_dir, 'test_ground_truths.csv'))

    def _make_predictions_single_model(self):
        # load checkpoint
        ckpt = torch.load(self.checkpoints, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

        # if not self.use_cuda:
        ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

        self.model.load_state_dict(ckpt['state_dict'], strict=True)

        # get predictions
        predictions, ground_truths = self._run_inference(item=None)

        column_names = [f'class_item_{item + 1}' for item in range(N_ITEMS)]

        return self._make_dataframes(predictions, ground_truths, column_names)

    def _make_dataframes(self, predictions, ground_truths, column_names):
        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        predictions_df = pd.DataFrame(columns=id_columns + column_names)
        ground_truths_df = pd.DataFrame(columns=id_columns + column_names)

        predictions_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        predictions_df['image_file'] = self.dataloader.dataset.image_files[:len(predictions)]
        predictions_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        ground_truths_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        ground_truths_df['image_file'] = self.dataloader.dataset.image_files[:len(predictions)]
        ground_truths_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        predictions_df[column_names] = predictions
        ground_truths_df[column_names] = ground_truths

        # turn classes into scores
        score_cols = [str(c).replace('class_', 'score_') for c in column_names]
        predictions_df[score_cols] = predictions_df[column_names].applymap(class_to_score)
        ground_truths_df[score_cols] = ground_truths_df[column_names].applymap(class_to_score)

        # compute total score
        predictions_df['total_score'] = predictions_df[score_cols].sum(axis=1)
        ground_truths_df['total_score'] = ground_truths_df[score_cols].sum(axis=1)

        return predictions_df, ground_truths_df

    def _run_inference(self, item=None):
        self.model.eval()
        predictions, ground_truths = None, None

        for inputs, targets in self.dataloader:
            targets = targets.numpy()

            if self.use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                logits = self.model(inputs.float())

            if isinstance(logits, list):
                outputs = [torch.argmax(lgts, dim=1).cpu().numpy() for lgts in logits]
            else:
                outputs = [torch.argmax(logits, dim=1).cpu().numpy()]
                targets = np.expand_dims(targets, -1)

            targets = targets[:, item - 1] if item is not None else targets
            outputs = np.concatenate([np.expand_dims(out, -1) for out in outputs], axis=1)

            predictions = outputs if predictions is None else np.concatenate([predictions, outputs], axis=0)
            ground_truths = targets if ground_truths is None else np.concatenate([ground_truths, targets], axis=0)

        return predictions, ground_truths
