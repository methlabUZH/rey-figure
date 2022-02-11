import numpy as np
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

    def run_eval(self):
        if self.is_ensemble:
            self.predictions = self._get_predictions_ensemble()
        else:
            self.predictions = self._make_predictions_single_model()

    def save_predictions(self, save_as=None):
        if self.predictions is None:
            raise ValueError('predictions is None!')

        if save_as is None:
            self.predictions.to_csv(os.path.join(self.results_dir, 'test_predictions.csv'))
            return

        self.predictions.to_csv(save_as)

    def _get_predictions_ensemble(self):
        data_cols = []
        data = None
        for item, ckpt in zip(self.items, self.checkpoints):
            print(f'[{timestamp_human()}] start eval item {item}, ckpt: {ckpt}')
            ckpt = torch.load(ckpt, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

            if not self.use_cuda:
                ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

            self.model.load_state_dict(ckpt['state_dict'], strict=True)

            # get predictions
            item_data = self._run_inference(item=item)

            # merge with previous data
            data = item_data if data is None else np.concatenate([data, item_data], axis=1)
            data_cols += [f'true_class_item_{item}', f'pred_class_item_{item}']

        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        outputs_df = pd.DataFrame(columns=id_columns + data_cols)

        outputs_df['figure_id'] = self.dataloader.dataset.image_ids[:len(data)]
        outputs_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(data)]
        outputs_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(data)]

        outputs_df[data_cols] = data

        # turn classes into scores
        score_cols = [str(c).replace('_class_', '_score_') for c in data_cols]
        outputs_df[score_cols] = outputs_df[data_cols].applymap(class_to_score)

        # compute total score
        outputs_df['true_total_score'] = outputs_df[[c for c in score_cols if str(c).startswith('true_')]].sum(axis=1)
        outputs_df['pred_total_score'] = outputs_df[[c for c in score_cols if str(c).startswith('pred_')]].sum(axis=1)

        return outputs_df

    def _make_predictions_single_model(self):
        # load checkpoint
        ckpt = torch.load(self.checkpoints, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

        # if not self.use_cuda:
        ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

        self.model.load_state_dict(ckpt['state_dict'], strict=True)

        # get predictions
        data = self._run_inference(item=None)

        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        label_columns = [f'true_class_item_{item + 1}' for item in range(N_ITEMS)]
        pred_columns = [f'pred_class_item_{item + 1}' for item in range(N_ITEMS)]
        outputs_df = pd.DataFrame(columns=id_columns + label_columns + pred_columns)

        outputs_df['figure_id'] = self.dataloader.dataset.image_ids[:len(data)]
        outputs_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(data)]
        outputs_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(data)]

        outputs_df[label_columns + pred_columns] = data

        # turn classes into scores
        score_cols = [str(c).replace('_class_', '_score_') for c in label_columns + pred_columns]
        outputs_df[score_cols] = outputs_df[label_columns + pred_columns].applymap(class_to_score)

        # compute total score
        outputs_df['true_total_score'] = outputs_df[[c for c in score_cols if str(c).startswith('true_')]].sum(axis=1)
        outputs_df['pred_total_score'] = outputs_df[[c for c in score_cols if str(c).startswith('pred_')]].sum(axis=1)

        return outputs_df

    def _run_inference(self, item=None):
        self.model.eval()
        data = None

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

            batch_data = np.concatenate([targets, outputs], axis=1)
            data = batch_data if data is None else np.concatenate([data, batch_data], axis=0)

        return data
