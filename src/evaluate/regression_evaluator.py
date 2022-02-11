import numpy as np
import pandas as pd

import torch

from constants import *
from src.dataloaders.dataloader_regression import get_regression_dataloader
from src.utils import timestamp_human, class_to_score


class RegressionEvaluator:
    def __init__(self, model, results_dir, data_dir, batch_size=128, workers=8):
        self.model = model
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers

        self.predictions = None
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

        # fetch checkpoint
        self.checkpoint = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')

        # init dataloader
        test_labels = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))
        self.dataloader = get_regression_dataloader(self.data_dir, labels_df=test_labels, batch_size=self.batch_size,
                                                    num_workers=self.workers, shuffle=False)

    def run_eval(self):
        # load checkpoint
        ckpt = torch.load(self.checkpoint, map_location=torch.device('cuda' if self.use_cuda else 'cpu'))

        ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(ckpt['state_dict'], strict=True)

        # get predictions
        data = self._run_inference()

        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        label_columns = [f'true_score_item_{item + 1}' for item in range(N_ITEMS)] + ['true_total_score']
        pred_columns = [f'pred_score_item_{item + 1}' for item in range(N_ITEMS)] + ['pred_total_score']
        outputs_df = pd.DataFrame(columns=id_columns + label_columns + pred_columns)

        outputs_df['figure_id'] = self.dataloader.dataset.image_ids[:len(data)]
        outputs_df['image_file'] = self.dataloader.dataset.jpeg_filepaths[:len(data)]
        outputs_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(data)]

        outputs_df[label_columns + pred_columns] = data

        self.predictions = outputs_df

    def save_predictions(self, save_as=None):
        if self.predictions is None:
            raise ValueError('predictions is None!')

        if save_as is None:
            self.predictions.to_csv(os.path.join(self.results_dir, 'test_predictions.csv'))
            return

        self.predictions.to_csv(save_as)

    def _run_inference(self):
        self.model.eval()
        data = None

        for inputs, targets in self.dataloader:
            targets = targets.numpy()
            if self.use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = self.model(inputs.float())

            # outputs = np.concatenate([np.expand_dims(out, -1) for out in outputs], axis=1)

            batch_data = np.concatenate([targets, outputs.cpu().numpy()], axis=1)
            data = batch_data if data is None else np.concatenate([data, batch_data], axis=0)

        return data
