import torch as _torch
import torchvision.transforms.functional as _F


class NormalizeImage:
    def __call__(self, img: _torch.Tensor) -> _torch.Tensor:
        return (img - _torch.mean(img)) / _torch.std(img)


class ResizePadded:

    def __init__(self, size, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: _torch.Tensor) -> _torch.Tensor:
        _, height, width = img.size()
        new_height, new_width = self.size
        ratio = min([new_height / height, new_width / width])
        resize_shape = [int(height * ratio // 2) * 2, int(width * ratio // 2) * 2]
        padding = [(new_width - resize_shape[1]) // 2, (new_height - resize_shape[0]) // 2]

        img_resized = _F.resize(img, resize_shape)
        img_resized = _F.pad(img_resized, padding=padding, fill=self.fill)
        return img_resized


class RandomRotation:

    def __init__(self, angles, fill):
        self.angles = angles
        if isinstance(fill, (int, float)):
            self.fill = [float(fill)]
        else:
            self.fill = [float(f) for f in fill]

    def __call__(self, img: _torch.Tensor):
        sign = 2 * _torch.randint(0, 2, (1,)) - 1
        degrees = sorted([x * sign for x in self.angles])
        angle = float(_torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return _F.rotate(img, angle, expand=True, fill=self.fill)


class AdjustBrightness:

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img: _torch.Tensor):
        return _F.adjust_brightness(img, brightness_factor=self.brightness_factor)


class AdjustContrast:

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img: _torch.Tensor):
        return _F.adjust_contrast(img, contrast_factor=self.contrast_factor)
