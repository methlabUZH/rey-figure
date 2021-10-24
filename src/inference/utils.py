import numpy as np
import torch
from typing import *


def assign_score(x):
    if x < 0.25:
        return 0
    if 0.25 <= x < 0.75:
        return 0.5
    if 0.75 <= x < 1.5:
        return 1
    if 1.5 <= x:
        return 2


def assign_bin_single(x, bin_locations) -> int:
    for i in range(len(bin_locations)):
        if bin_locations[i][0] <= x < bin_locations[i][1]:
            return i

    return len(bin_locations) - 1


def assign_bins(scores: Union[torch.Tensor, np.ndarray, int],
                bin_locations: List[Tuple[float, float]],
                out_type=torch.float) -> Union[torch.Tensor, np.ndarray, int]:
    if isinstance(scores, float):
        return assign_bin_single(scores, bin_locations)

    if isinstance(scores, np.ndarray):
        binned = np.array(list(map(lambda x: assign_bin_single(x, bin_locations), scores)))
        return binned

    binned = list(map(lambda x: assign_bin_single(x, bin_locations), torch.unbind(scores, 0)))
    binned = torch.unsqueeze(torch.tensor(binned, dtype=out_type), 1)

    return binned
