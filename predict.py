import cv2
import numpy as np
import os
import pandas as pd
import torch

from constants import *
from src.utils import class_to_score, map_to_score_grid
from src.inference.preprocess import preprocess_image_v0
from src.models import get_classifier, get_regressor


def _postprocess_logits(logits):
    item_classes = np.argmax(np.concatenate([lgt.cpu().numpy() for lgt in logits]), axis=1)
    item_scores = {f'item{i + 1}': s for i, s, in enumerate(map(class_to_score, item_classes))}
    total_score = sum(item_scores.values())
    return item_scores, total_score


class PredictorBase:

    def __init__(self, results_dir):
        self.results_dir = results_dir

        # setup model
        self.model = self._get_model()
        self._init_model(ckpt=os.path.join(results_dir, 'checkpoints/model_best.pth.tar'))

        # fetch ground truths
        self.test_labels = pd.read_csv(os.path.join(results_dir, 'test_ground_truths.csv'))
        self.test_labels = self.test_labels.set_index('figure_id')

    def _get_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _postprocess_outputs(self, outputs):
        raise NotImplementedError

    def _init_model(self, ckpt):
        ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
        ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        self.model.eval()

    def predict(self, image_filepath):
        image = cv2.imread(image_filepath)
        image_preprocessed = preprocess_image_v0(image, simulate_augment=False)
        image_tensor = torch.from_numpy(image_preprocessed)

        with torch.no_grad():
            outputs = self.model(image_tensor.float())

        pred_scores = self._postprocess_outputs(outputs)
        true_scores = self._get_ground_truth(image_filepath)

        return pred_scores, true_scores

    def _get_ground_truth(self, image_fp):
        figure_id = os.path.split(os.path.splitext(image_fp)[0])[-1]
        try:
            figure_data = self.test_labels.loc[figure_id, [f'score_item_{i + 1}' for i in range(N_ITEMS)]]
        except KeyError:
            print(f'warning! figure {figure_id} not found in test data.')
            return None

        scores = {f'item{i + 1}': s for i, s in enumerate(figure_data.values)}
        scores['total_score'] = sum(scores.values())

        return scores


class MultilabelClassifier(PredictorBase):
    def __init__(self, results_dir):
        super(MultilabelClassifier, self).__init__(results_dir)

    def _get_model(self) -> torch.nn.Module:
        return get_classifier(arch=REYMULTICLASSIFIER, num_classes=4, item=None)

    def _postprocess_outputs(self, outputs):
        item_classes = np.argmax(np.concatenate([lgt.cpu().numpy() for lgt in outputs]), axis=1)
        scores = {f'item{i + 1}': s for i, s, in enumerate(map(class_to_score, item_classes))}
        scores['total_score'] = sum(scores.values())
        return scores


class Regressor(PredictorBase):
    def __init__(self, results_dir):
        super(Regressor, self).__init__(results_dir)

    def _get_model(self) -> torch.nn.Module:
        return get_regressor()

    def _postprocess_outputs(self, outputs):
        item_scores = outputs.cpu().numpy()[0][:-1]
        scores = {f'item{i + 1}': s for i, s, in enumerate(map(map_to_score_grid, item_scores))}
        scores['total_score'] = sum(scores.values())
        return scores


if __name__ == '__main__':
    image_fp = "./sample_data/REF_147_NaN_ROCF_Puerto_Rico_Children-104.jpg"
    ckpt_root = 'results/euler-results/data-2018-2021-116x150-pp0/final/'
    ml_predictor = MultilabelClassifier(results_dir=ckpt_root + 'rey-multilabel-classifier')
    ml_scores, true_scores = ml_predictor.predict(image_filepath=image_fp)
    reg_predictor = Regressor(results_dir=ckpt_root + 'rey-regressor')
    reg_scores, _ = reg_predictor.predict(image_filepath=image_fp)
    print(f'Ground Truth Scores: {true_scores}')
    print(f'Multilabel Scores: {ml_scores}')
    print(f'Regression Scores: {reg_scores}')
