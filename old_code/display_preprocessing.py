import argparse
from cv2 import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray as skimage_rgb2gray
from skimage.morphology import erosion as skimage_erosion
from skimage.exposure import adjust_gamma as skimage_adjust_gamma

from constants import DEFAULT_CANVAS_SIZE
from src.preprocessing.helpers import cutdown, resize_padded

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--size', nargs='+', default=DEFAULT_CANVAS_SIZE, type=int, help='height and width')
args = parser.parse_args()


def main(image_file, gamma, cutdown_thresh, whiten_thresh, save_as=None):
    image_raw = imread(image_file)

    image_grayscale = skimage_rgb2gray(image_raw)
    image_erosion = skimage_erosion(image_grayscale)
    image_gamma_adjust = skimage_adjust_gamma(image_erosion, gamma=gamma)

    # cutdown
    thresh_cut = np.percentile(image_gamma_adjust, cutdown_thresh)
    image_cutdown = cutdown(img=image_gamma_adjust, threshold=thresh_cut)

    # whiten background
    thresh_white = np.percentile(image_cutdown, whiten_thresh)
    image_cutdown[image_cutdown > thresh_white] = 1.0

    # fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes[0].imshow(image_grayscale, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('grayscale')

    axes[1].imshow(image_erosion, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('erosion')

    axes[2].imshow(image_gamma_adjust, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('gamma adjust')

    axes[3].imshow(image_cutdown, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('cutdown')

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    if save_as is None:
        plt.show()
    else:
        print(f'saved image as {save_as}')
        plt.savefig(save_as, dpi=250, bbox_inches='tight', pad_inches=0.05)

    plt.close()


def main2(image_file):
    image_raw = imread(image_file)

    image = skimage_rgb2gray(image_raw)
    # image = skimage_adjust_gamma(image, gamma=-3)
    # image = skimage_erosion(image)

    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # data_root = '/Users/maurice/phd/src/rey-figure/preprocessing/ReyFigures/data2021/'
    # scan = data_root + 'USZ_scans/14802C_NaN_dava_120190308092244_Seite_01.jpg'
    # # photo = data_root + 'USZ_fotos/C9_none_foto_20190307_123539.jpg'
    # photo = data_root + 'USZ_fotos/C5694C1K_none_foto_20190306_173942.jpg'
    #
    # save_dir = '/Users/maurice/Desktop/rey-figures-preprocessing/'
    #
    # main(image_file=photo, gamma=3, cutdown_thresh=4, whiten_thresh=8)
    # # main(image_file=photo, gamma=10, cutdown_thresh=2, whiten_thresh=12)

    # combination0 = [3, 4, 8]
    # combination1 = [8, 2, 12]
    # combination2 = [10, 1, 12]
    # combination3 = [10, 2, 12]
    #
    # for (g, c, w) in [combination0, combination1, combination2, combination3]:
    #     img_save_as = save_dir + f'photo_gamma={g}_cutdown={c}_whiten={w}.pdf'
    #     main(image_file=photo, gamma=g, cutdown_thresh=c, whiten_thresh=w, save_as=img_save_as)
    #
    #     img_save_as = save_dir + f'scan_gamma={g}_cutdown={c}_whiten={w}.pdf'
    #     main(image_file=scan, gamma=g, cutdown_thresh=c, whiten_thresh=w, save_as=img_save_as)

    photo = '/Users/maurice/phd/src/rey-figure/data/ReyFigures/data2021/USZ_fotos/C623003K_NaN_foto_20190315_111131.jpg'
    main2(photo)