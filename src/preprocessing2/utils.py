import numpy as np
from skimage.transform import resize

__all__ = ['resize_padded']


def resize_padded(img, new_shape, fill_cval=None, order=1, anti_alias=True):
    if fill_cval is None:
        fill_cval = np.max(img)

    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio / 2 for s in img.shape]).astype(np.int) * 2
    interm_img = resize(img, interm_shape, order=order, cval=fill_cval, anti_aliasing=anti_alias,
                        mode='constant')

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[tuple([slice(p, -p, None) if 0 != p else slice(None, None, None) for p in pad])] = interm_img

    return new_img
