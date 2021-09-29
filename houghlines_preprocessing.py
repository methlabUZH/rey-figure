import argparse
from cv2 import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray as skimage_rgb2gray
from skimage.morphology import erosion as skimage_erosion
from skimage.exposure import adjust_gamma as skimage_adjust_gamma

from constants import DEFAULT_CANVAS_SIZE
from src.data.helpers import cutdown, resize_padded

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--size', nargs='+', default=DEFAULT_CANVAS_SIZE, type=int, help='height and width')
args = parser.parse_args()


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_image_hough(image):
    """
    see: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    """
    image = cv2.imread(image)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_erosion = cv2.erode(image_grayscale, kernel=(5, 5))
    image_gamma_adjust = adjust_gamma(image_erosion, gamma=3)

    # cutdown
    thresh_cut = np.percentile(image_gamma_adjust, 4)
    image_cutdown = cutdown(img=image_gamma_adjust, threshold=thresh_cut)

    # whiten background
    thresh_white = np.percentile(image_cutdown, 8)
    image_cutdown[image_cutdown > thresh_white] = 255

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(image_cutdown, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    threshold = 10
    min_line_length = 10
    max_line_gap = 10
    line_image = np.copy(cv2.cvtColor(image_cutdown, cv2.COLOR_GRAY2BGR)) * 0

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    if lines is None:
        print('No lines found!')
        return image_grayscale, image_cutdown, image

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    lines_edges = cv2.addWeighted(cv2.cvtColor(image_cutdown, cv2.COLOR_GRAY2BGR), 0.5, line_image, 1, 0)

    return image_grayscale, image_cutdown, lines_edges


def main(image_file):
    image_grayscale, image_binary, lines_edges = preprocess_image_hough(image_file)
    lines_edges = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    axs[0].imshow(image_grayscale, cmap='gray')
    axs[1].imshow(image_binary, cmap='gray')
    axs[2].imshow(lines_edges)

    plt.show()
    plt.close()


if __name__ == '__main__':
    data_root = '/Users/maurice/phd/src/rey-figure/data/ReyFigures/data2021/'
    scan = data_root + 'USZ_scans/14802C_NaN_dava_120190308092244_Seite_01.jpg'
    # photo = data_root + 'USZ_fotos/C65xx01R1_NaN_foto_20190321_114206.jpg'
    photo = data_root + 'USZ_fotos/C9_none_foto_20190307_123539.jpg'
    # photo = data_root + 'USZ_fotos/C5694C1K_none_foto_20190306_173942.jpg'

    main(photo)
