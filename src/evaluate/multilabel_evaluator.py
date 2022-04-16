import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
import torchvision

from constants import *
from src.dataloaders.rocf_dataloader import get_dataloader
from src.utils import class_to_score


class MultilabelEvaluator:
    def __init__(self, model, image_size, results_dir, data_dir, batch_size=128, workers=8,
                 tta=False, validation=False, angles=[-2.5, -1.5, 0, 1.5, 2.5], num_classes=4):
        self.model = model
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.workers = workers
        self.tta = tta
        self.validation = validation
        self.angles = angles
        self.num_classes = num_classes

        self.predictions = None
        self.ground_truths = None
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            print(f"Using gpu acceleration")
            self.model.cuda()

        self.checkpoints = os.path.join(results_dir, 'checkpoints/model_best.pth.tar')

        # init dataloader
        if self.validation:  # use validation set for evaluation
            test_labels = pd.read_csv(os.path.join(data_dir, 'val_labels.csv'))
        else:  # use test set for evaluation
            test_labels = pd.read_csv(os.path.join(self.data_dir, 'test_labels.csv'))

        self.dataloader = get_dataloader(labels=test_labels, label_type=CLASSIFICATION_LABELS,
                                         batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
                                         augment=False, image_size=self.image_size, num_classes=num_classes)

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
        predictions, ground_truths, prediction_logits = self._run_inference(item=None)

        column_names = [f'class_item_{item + 1}' for item in range(N_ITEMS)]

        return self._make_dataframes(predictions, ground_truths, prediction_logits, column_names)

    def _make_dataframes(self, predictions, ground_truths, prediction_logits, column_names):
        # write to df
        id_columns = ['figure_id', 'image_file', 'serialized_file']
        logit_column_names = [f'logit_item_{item + 1}' for item in range(N_ITEMS)]
        predictions_df = pd.DataFrame(columns=id_columns + column_names + logit_column_names)
        ground_truths_df = pd.DataFrame(columns=id_columns + column_names)

        predictions_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        predictions_df['image_file'] = self.dataloader.dataset.image_files[:len(predictions)]
        predictions_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        ground_truths_df['figure_id'] = self.dataloader.dataset.image_ids[:len(predictions)]
        ground_truths_df['image_file'] = self.dataloader.dataset.image_files[:len(predictions)]
        ground_truths_df['serialized_file'] = self.dataloader.dataset.npy_filepaths[:len(predictions)]

        predictions_df[column_names] = predictions
        ground_truths_df[column_names] = ground_truths
        # ugly hack to write the logits to the csv
        for i, _ in predictions_df.iterrows():
            for item in range(N_ITEMS):
                predictions_df.at[i, f'logit_item_{item + 1}'] = prediction_logits[i, item]

        # turn classes into scores
        score_cols = [str(c).replace('class_', 'score_') for c in column_names]
        predictions_df[score_cols] = predictions_df[column_names].applymap(
            lambda x: class_to_score(x, num_classes=self.num_classes)
        )
        ground_truths_df[score_cols] = ground_truths_df[column_names].applymap(
            lambda x: class_to_score(x, num_classes=self.num_classes)
        )

        # compute total score
        predictions_df['total_score'] = predictions_df[score_cols].sum(axis=1)
        ground_truths_df['total_score'] = ground_truths_df[score_cols].sum(axis=1)

        return predictions_df, ground_truths_df  # , prediction_logits_df

    def _run_inference(self, item=None):
        self.model.eval()
        predictions, ground_truths, prediction_logits = None, None, None

        n = len(self.dataloader)

        for inputs, targets in tqdm(self.dataloader, total=n):
            targets = targets.numpy()

            if self.use_cuda:
                inputs = inputs.cuda()

            with torch.no_grad():
                if self.tta:
                    # test time augmentation (TTA) with angle rotations 
                    logits = None
                    for angle in self.angles:
                        inputs = torchvision.transforms.functional.rotate(inputs.float(), angle)
                        outputs = self.model(inputs.float())  # list of 18 elements of shape (bs, 4) each
                        # add up the logits for all 18 items 
                        if logits is None:
                            logits = outputs
                        else:
                            for i in range(len(outputs)):
                                logits[i] += outputs[i]
                    # divide by nb of angles to get the average prediction 
                    for i in range(len(logits)):
                        logits[i] /= len(self.angles)
                else:
                    # normal inference without Test-Time-Augmentation
                    logits = self.model(inputs.float())  # list of 18 elements of shape (bs, 4) each

                # Apply softmax since no loss here 
                logits = list(map(torch.nn.Softmax(dim=1), logits))

            if isinstance(logits, list):
                outputs = [torch.argmax(lgts, dim=1).cpu().numpy() for lgts in logits]
                output_logits = [lgts.cpu().numpy() for lgts in logits]
            else:
                outputs = [torch.argmax(logits, dim=1).cpu().numpy()]
                output_logits = [logits.cpu().numpy()]
                targets = np.expand_dims(targets, -1)

            targets = targets[:, item - 1] if item is not None else targets
            outputs = np.concatenate([np.expand_dims(out, -1) for out in outputs], axis=1)
            output_logits = np.array(output_logits).transpose((1, 0, 2))

            predictions = outputs if predictions is None else np.concatenate([predictions, outputs], axis=0)
            ground_truths = targets if ground_truths is None else np.concatenate([ground_truths, targets], axis=0)
            prediction_logits = output_logits if prediction_logits is None else np.concatenate(
                [prediction_logits, output_logits], axis=0)

        return predictions, ground_truths, prediction_logits
