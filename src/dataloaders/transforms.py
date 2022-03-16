import torch


class NormalizeImage:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - torch.mean(img)) / torch.std(img)
