import os
from typing import List, Tuple

import torch

from constants import N_ITEMS
from src.models import get_reyregressor, get_reyclassifier


def get_classifiers_checkpoints(trained_classifiers_root, max_ckpts=-1) -> List[Tuple[int, str]]:
    checkpoints = []

    for i in range(1, N_ITEMS + 1):
        classifier_root = os.path.join(trained_classifiers_root, f'item-{i}')

        if not os.path.exists(classifier_root):
            print(f'no checkpoints found for item {i}!')
            continue

        checkpoint = os.path.join(classifier_root, 'checkpoints/model_best.pth.tar')

        if not os.path.isfile(checkpoint):
            print(f'no checkpoints found for item {i}!')
            continue

        checkpoints.append((i, checkpoint))

    return checkpoints if max_ckpts == -1 else checkpoints[:max_ckpts]


def init_model_weights(model: torch.nn.Module, ckpt_fp: str) -> torch.nn.Module:
    assert os.path.isfile(ckpt_fp), f'no checkpoint found: {ckpt_fp}'
    ckpt = torch.load(ckpt_fp, map_location=torch.device('cpu'))
    ckpt['state_dict'] = {str(k).replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(ckpt['state_dict'], strict=True)
    print(f'==> loaded checkpoint {ckpt_fp}')
    return model


def init_regressor(ckpt_fp: str, norm_layer: str) -> torch.nn.Module:
    regressor = get_reyregressor(n_outputs=N_ITEMS, norm_layer_type=norm_layer)
    regressor = init_model_weights(regressor, ckpt_fp)
    regressor.eval()
    return regressor


def init_classifier(ckpt_fp: str, norm_layer: str) -> torch.nn.Module:
    classifier = get_reyclassifier(norm_layer_type=norm_layer)
    classifier = init_model_weights(classifier, ckpt_fp)
    classifier.eval()
    return classifier

# def get_classifiers_checkpoints_OLD(results_root, max_ckpts=-1) -> List[Tuple[int, str]]:
#     checkpoints = []
#
#     for i in range(1, N_ITEMS + 1):
#         classifier_root = os.path.join(results_root, f'item-{i}')
#
#         if not os.path.exists(classifier_root):
#             print(f'no checkpoints found for item {i}!')
#             continue
#
#         hyperparam_dir = os.listdir(classifier_root)[0]
#         timestamps = os.listdir(os.path.join(classifier_root, hyperparam_dir))
#         max_timestamp = max([datetime.strptime(ts, '%Y-%m-%d_%H-%M-%S.%f') for ts in timestamps])
#         max_timestamp = max_timestamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
#         checkpoint = os.path.join(classifier_root, hyperparam_dir, max_timestamp, 'checkpoints/model_best.pth.tar')
#
#         if not os.path.isfile(checkpoint):
#             print(f'no checkpoints found for item {i}!')
#             continue
#
#         checkpoints.append((i, checkpoint))
#
#     return checkpoints if max_ckpts == -1 else checkpoints[:max_ckpts]
