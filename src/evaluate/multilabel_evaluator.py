import numpy as np
import os
import pandas as pd

import torch

from constants import *
from src.dataloaders.dataloader_multilabel import get_multilabel_dataloader
from src.inference.model_initialization import get_classifiers_checkpoints
from src.utils import timestamp_human, class_to_score


class MultilabelEvaluator:
    def __init__(self, model, results_dir, data_dir, is_ensemble, is_binary, batch_size=128, workers=8):
        self.model = model
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.is_ensemble = is_ensemble
        self.batch_size = batch_size
        self.workers = workers

        self.predictions = None
        self.ground_truths = None
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

        # fetch checkpoints
        if is_ensemble:
            self.items, self.checkpoints = get_classifiers_checkpoints(results_dir)
        else:
            self.checkpoints = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')
            self.items = None

        # init dataloader
        test_labels = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))
        self.dataloader = get_multilabel_dataloader(self.data_dir, labels_df=test_labels, batch_size=self.batch_size,
                                                    num_workers=self.workers, shuffle=False, is_binary=is_binary)

    def run_eval(self, save=True):
        if self.is_ensemble:
            self.predictions, self.ground_truths = self._get_predictions_ensemble()
        else:
            self.predictions, self.ground_truths = self._make_predictions_single_model()

        if save:
            self.save_predictions()

    def save_predictions(self):
        if self.predictions is None or self.ground_truths is None:
            raise ValueError('predictions is None!')

        self.predictions.to_csv(os.path.join(self.results_dir, 'test_predictions.csv'))
        self.ground_truths.to_csv(os.path.join(self.results_dir, 'test_ground_truths.csv'))

    def _get_predictions_ensemble(self):
        column_names = []
        predictions, ground_truths = None, None
        for item, ckpt in zip(self.items, self.checkpoints):
            print(f'[{timestamp_human()}] start eval item {item}, ckpt: {ckpt}')
            ckpt = torch.load(ckpt, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

            if not self.use_cuda:
                ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

            self.model.load_state_dict(ckpt['state_dict'], strict=True)

            # get predictions
            item_predictions, item_ground_truths = self._run_inference(item=item)

            # merge with previous data
            predictions = item_predictions if predictions is None else np.concate(
                [predictions, item_predictions], axis=1)
            ground_truths = item_ground_truths if ground_truths is None else np.concate(
                [ground_truths, item_ground_truths], axis=1)

            column_names += [f'class_item_{item}']

        return self._make_dataframes(predictions, ground_truths, column_names)

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
        predictions_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(predictions)]
        predictions_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        ground_truths_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        ground_truths_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(predictions)]
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
