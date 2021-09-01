import argparse
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from constants import DEFAULT_CANVAS_SIZE, AUGM_CANVAS_SIZE
from src.data.preprocess import preprocess_image
from src.data.augmentation import augment_image, AugmentParameters

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--size', nargs='+', default=DEFAULT_CANVAS_SIZE, type=int, help='height and width')
args = parser.parse_args()


def main(filepath, image_size, datadir=None):
    max_reps = 100

    prep_time = 0

    for i, f in enumerate(os.listdir(datadir)):
        image_raw = imread(os.path.join(datadir, f))
        img = preprocess_image(image_raw, target_size=image_size)

        print(np.min(image_raw), np.max(image_raw))
        print(np.min(img), np.max(img))

        # if i + 1 == max_reps:
        break

    print(f'total elapsed time:\t\t{prep_time:.4f}')
    print(f'avg preprocessing time:\t {prep_time / max_reps :.4f}')

    # for _ in range(10):
    #     img = augment_image(preprocess_image(image_raw, target_size=AUGM_CANVAS_SIZE),
    #                         alpha_elastic_transform=AugmentParameters.alpha_elastic_transform,
    #                         sigma_elastic_transform=AugmentParameters.sigma_elastic_transform,
    #                         max_factor_skew=AugmentParameters.max_factor_skew,
    #                         max_angle_rotate=AugmentParameters.max_angle_rotate,
    #                         target_size=image_size)
    #
    #     plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    #     plt.show()
    #     plt.close()


if __name__ == '__main__':
    # main(filepath=None, image_size=[116, 150],
    #      datadir='/Users/maurice/phd/src/data/psychology/ReyFigures/data2018/uploadFinal')
    # print()
    main(filepath=None, image_size=[224, 224],
         datadir='/Users/maurice/phd/src/data/psychology/ReyFigures/data2018/uploadFinal')

    # main(filepath=None, image_size=[300, 300],
    #      datadir='/Users/maurice/phd/src/data/psychology/ReyFigures/data2018/uploadFinal')
