import numbers
from typing import *

import torch as _torch
import torchvision.transforms.functional as _F


class Identity:
    def __call__(self, img: _torch.Tensor) -> _torch.Tensor:
        return img


class NormalizeImage:
    def __call__(self, img: _torch.Tensor) -> _torch.Tensor:
        mean = _torch.mean(img, dim=[1, 2], keepdim=True)
        std = _torch.std(img, dim=[1, 2], keepdim=True)
        return (img - mean) / std


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


class ColorJitter(_torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @_torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[_torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = _torch.randperm(4)

        b = None if brightness is None else float(_torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(_torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(_torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(_torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = _F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = _F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = _F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = _F.adjust_hue(img, hue_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
