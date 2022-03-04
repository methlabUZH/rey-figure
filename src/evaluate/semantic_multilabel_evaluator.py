import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from constants import *
from src.data_preprocessing.helpers import resize_padded, cutdown
from src.dataloaders.dataloader_multilabel import ROCFDatasetMultiLabelClassification
from src.inference.model_initialization import get_classifiers_checkpoints
from src.utils import timestamp_human, class_to_score


class ChangePerspective:

    def __init__(self, scale=0.0):
        self._scale = min(0.5, scale)

    def __call__(self, img):
        image_size = np.shape(img)[::-1]
        tilt_pixels = int(image_size[0] * self._scale)
        pts1 = np.array([[0, image_size[1]], [0, 0], [image_size[0], 0], image_size], np.float32)
        pts2 = np.array([[0, image_size[1]], [tilt_pixels, 0], [image_size[0] - tilt_pixels, 0], image_size],
                        np.float32)
        tf_mat = cv2.getPerspectiveTransform(pts1, pts2)
        image_tilted = cv2.warpPerspective(img, tf_mat, np.shape(img)[::-1], borderValue=1.0)
        return image_tilted


class Normalize:

    def __call__(self, img):
        return (img - np.mean(img)) / np.std(img)


class RotateImage:

    def __init__(self, angle):
        self._angle = angle

        if angle == 0.0:
            self._call_impl = lambda x: x
        else:
            self._call_impl = self._call_impl_0

    def __call__(self, img):
        return self._call_impl(img)

    def _call_impl_0(self, img):
        rot_angle = np.random.choice([-1, 1]) * self._angle

        image_shape = np.shape(img)
        x_pad = image_shape[0] // 20
        y_pad = image_shape[1] // 20
        image_numpy = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad)), constant_values=1.0)
        image_center = tuple(np.array(image_numpy.shape) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot_angle, 1.0)
        image_rotated = cv2.warpAffine(image_numpy, rot_mat, image_numpy.shape[::-1], flags=cv2.INTER_LINEAR,
                                       borderValue=1.0)

        return image_rotated

        # image_shape = np.shape(img)
        # y_pad = (AUGM_CANVAS_SIZE[0] - image_shape[0]) // 2
        # x_pad = (AUGM_CANVAS_SIZE[1] - image_shape[1]) // 2
        # image_numpy = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad)), constant_values=1.0)
        # image_center = tuple(np.array(image_numpy.shape[1::-1]) / 2)
        # rot_mat = cv2.getRotationMatrix2D(image_center, rot_angle, 1.0)
        # image_rotated = cv2.warpAffine(image_numpy, rot_mat, image_numpy.shape[1::-1], flags=cv2.INTER_LINEAR,
        #                                borderValue=1.0)
        # image_rotated = cutdown(image_rotated, pad=5)
        # image_rotated = resize_padded(image_rotated, image_shape)
        # return image_rotated


class ResizePadded:

    def __init__(self, target_shape):
        self._target_shape = target_shape

    def __call__(self, img):
        return resize_padded(img, self._target_shape)


def get_semantic_multilabel_dataloader(data_root: str, labels_df: pd.DataFrame, batch_size: int, num_workers: int,
                                       shuffle: bool, prefectch_factor: int = 16, rotation_angle=None,
                                       perspective_change=None, is_binary=False, pin_memory: bool = True):
    transforms_list = []
    if rotation_angle is not None:
        transforms_list += [RotateImage(rotation_angle)]

    if perspective_change is not None:
        transforms_list += [ChangePerspective(perspective_change)]
        raise NotImplementedError

    # resize
    transforms_list += [ResizePadded(target_shape=DEFAULT_CANVAS_SIZE)]

    dataset = ROCFDatasetMultiLabelClassification(data_root, labels_df, transforms_list=transforms_list,
                                                  is_binary=is_binary, load_numpy=False)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefectch_factor)


class SemanticMultilabelEvaluator:
    def __init__(self, model, results_dir, data_dir, is_ensemble, is_binary, rotation_angle, perspective_change,
                 batch_size=128, workers=8):
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
        self.dataloader = get_semantic_multilabel_dataloader(
            self.data_dir, labels_df=test_labels, batch_size=self.batch_size, num_workers=self.workers, shuffle=False,
            rotation_angle=rotation_angle, perspective_change=perspective_change, is_binary=is_binary)

    def run_eval(self, save=True, prefix=None):
        if self.is_ensemble:
            self.predictions, self.ground_truths = self._get_predictions_ensemble()
        else:
            self.predictions, self.ground_truths = self._make_predictions_single_model()

        if save:
            self.save_predictions(prefix=prefix)

    def save_predictions(self, prefix):
        if self.predictions is None or self.ground_truths is None:
            raise ValueError('predictions is None!')

        if prefix is not None:
            pred_fp = os.path.join(self.results_dir, prefix + "-test_predictions.csv")
            gt_fp = os.path.join(self.results_dir, prefix + "-test_ground_truths.csv")
        else:
            pred_fp = os.path.join(self.results_dir, "test_predictions.csv")
            gt_fp = os.path.join(self.results_dir, "test_ground_truths.csv")

        self.predictions.to_csv(pred_fp)
        self.ground_truths.to_csv(gt_fp)

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
