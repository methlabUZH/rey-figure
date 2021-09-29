import numpy as np
import os
import torch
from typing import *

from src.utils.helpers import assign_bins


def load_model(checkpoint_fp, model: torch.nn.Module) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_fp):
        raise FileNotFoundError(f'no checkpoint in {checkpoint_fp} found!')

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    checkpoint = torch.load(checkpoint_fp, map_location=device)
    checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    return model


def round_to_item_scores(score: float) -> float:
    if score < 0.25:
        return 0
    if 0.25 <= score < 0.75:
        return 0.5
    if 0.75 <= score < 1.5:
        return 1
    if 1.5 <= score:
        return 2


def inference(model: torch.nn.Module, images_numpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    while len(images_numpy.shape) < 4:
        images_numpy = np.expand_dims(images_numpy, axis=0)

    # normalize image and convert to pytorch tensor
    images_numpy = (images_numpy - np.mean(images_numpy, axis=(1, 2, 3))) / np.std(images_numpy, axis=(1, 2, 3))
    images_tensor = torch.from_numpy(images_numpy).float()

    model.eval()

    with torch.no_grad():
        predictions = model(images_tensor)

    # post process scores
    item_scores = predictions[:, :-1].numpy()
    for i, score_array in enumerate(item_scores):
        item_scores[i] = [round_to_item_scores(x) for x in score_array]

    total_scores = np.sum(item_scores, axis=-1)
    predicted_bin = assign_bins(total_scores)

    return np.squeeze(item_scores), np.squeeze(total_scores), np.squeeze(predicted_bin)

# if __name__ == '__main__':
#     from src.models import get_architecture
#
#     m = get_architecture('resnet18', num_outputs=18, dropout=None, norm_layer='batch_norm', image_size=[224, 224])
#     ckpt = '/Users/maurice/phd/src/psychology/results/sum-score/scans-2018-2021-224x224-augmented/resnet18/2021-09-12_20-03-15.155/checkpoints/model_best.pth.tar'
#     checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
#     checkpoint['state_dict'] = {str(k).replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
#     m.load_state_dict(checkpoint['state_dict'], strict=True)
#     m.eval()
#
#     image_fp = '/Users/maurice/phd/src/data/psychology/serialized-data/scans-2018-224x224/data2018/newupload_15_11_2018/B5882_07B_NaN_img169.npy'
#     image = np.load(image_fp)
#
#     item_scores, total_score, binned_score = inference(m, image)
#
#     print(item_scores, total_score, binned_score)
