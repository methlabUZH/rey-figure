import numpy as np
import os
import torch


def load_model(checkpoint_fp, model: torch.nn.Module) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_fp):
        raise FileNotFoundError(f'no checkpoint in {checkpoint_fp} found!')

    checkpoint = torch.load(checkpoint_fp)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def inference(model: torch.nn.Module, images_numpy: np.ndarray) -> np.ndarray:
    while len(images_numpy.shape) < 4:
        images_numpy = np.expand_dims(images_numpy, axis=0)

    images_tensor = torch.from_numpy(images_numpy).float()
    item_scores, _ = model(images_tensor)

    return item_scores.cpu().numpy()
