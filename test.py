import torch

from constants import BIN_LOCATIONS1


def assign_bin(x):
    for i in range(len(BIN_LOCATIONS1)):
        if BIN_LOCATIONS1[i][0] <= x < BIN_LOCATIONS1[i][1]:
            return i

    return len(BIN_LOCATIONS1) - 1


scores = torch.randint(0, 38, size=(10, 1))
binned = list(map(assign_bin, torch.unbind(scores, 0)))
binned = torch.unsqueeze(torch.tensor(binned), 1)
print(scores)
print(binned)
