from typing import Dict, Tuple

import numpy as np
import torch

from constants import N_ITEMS
from src.inference.utils import assign_score


def predict_item_presence(input_tensor: torch.Tensor, classifiers: Dict[int, torch.nn.Module]) -> Dict[int, int]:
    results = {}

    with torch.no_grad():
        for i in range(1, N_ITEMS + 1):
            class_probs = classifiers[i](input_tensor.float())
            _, predicted_class = torch.topk(class_probs, 1, 1, True, True)
            predicted_class = np.squeeze(predicted_class.numpy())
            results[i] = int(predicted_class)

    return results


def predict_item_scores(input_tensor: torch.Tensor, regressor: torch.nn.Module) -> Dict[int, float]:
    with torch.no_grad():
        predicted_scores = regressor(input_tensor.float()).numpy()

    predicted_scores = np.squeeze(predicted_scores)
    predicted_scores = np.clip(predicted_scores, 0, 2)
    predicted_scores = list(predicted_scores)
    predicted_scores = list(map(assign_score, predicted_scores))

    return {i: score for i, score in zip(range(1, N_ITEMS + 1), predicted_scores)}


def do_score_image(
        input_tensor: torch.Tensor,
        regressor: torch.nn.Module,
        classifiers: Dict[int, torch.nn.Module]) -> Dict[int, Tuple[bool, int]]:
    items_present = predict_item_presence(input_tensor, classifiers)
    items_scores = predict_item_scores(input_tensor, regressor)
    final_scores = {k: (v1, items_scores[k]) for k, v1 in items_present.items()}
    return final_scores
