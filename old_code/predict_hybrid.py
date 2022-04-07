import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tabulate import tabulate

from src.inference.model_initialization import get_classifiers_checkpoints, init_regressor, init_classifier
from src.inference.predict import do_score_image
from constants import RESULTS_DIR
from src.inference.preprocess import load_and_normalize_image

RESULTS_DIR = os.path.join(RESULTS_DIR, 'scans-2018-116x150-augmented/id-1/')
CLASSIFIERS_DIR = os.path.join(RESULTS_DIR, 'item-classifier')
REGRESSOR_DIR = os.path.join(RESULTS_DIR, 'rey-regressor')

# preprocessing
DEFAULT_IMG = './sample_data/B9483_02C_NaN_img421.npy'
NORM_LAYER = 'batch_norm'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='path to npy-image file', default=DEFAULT_IMG)
parser.add_argument('--show-figure', type=int, default=1, choices=[0, 1])
args = parser.parse_args()


def display_predictions(predictions, image, show_figure=True):
    # print results
    data = [[item, present, score] for item, (present, score) in predictions.items()]

    # write scores to df
    df = pd.DataFrame(data, columns=['item', 'classifier', 'regressor'])
    df = df.set_index('item')
    hybrid_score = (df['regressor'][:-1] * df['classifier'][:-1]).sum()
    regressor_score = df['regressor'].sum()
    df.loc['total regressor', :] = ['-', regressor_score]
    df.loc['total hybrid', :] = ['-', hybrid_score]

    print(tabulate(df, headers='keys', tablefmt='presto', floatfmt=".3f"))

    # plot figure and scores
    if show_figure:
        image = np.squeeze(image)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.show()
        plt.close()


def predict_single(regressor, classifiers, image_fp):
    input_tensor = load_and_normalize_image(image_fp)
    scores = do_score_image(input_tensor, regressor, classifiers)
    return input_tensor, scores


def main():
    reg_ckpt_fp = os.path.join(REGRESSOR_DIR, 'checkpoints/model_best.pth.tar')
    items_and_cls_ckpt_files = get_classifiers_checkpoints(CLASSIFIERS_DIR)

    # init regressor
    regressor = init_regressor(reg_ckpt_fp, norm_layer=NORM_LAYER)

    # init item classifiers
    classifiers = {i: init_classifier(ckpt_fp, norm_layer=NORM_LAYER) for i, ckpt_fp in items_and_cls_ckpt_files}

    # make prediction
    image_tensor, scores = predict_single(regressor, classifiers, args.image)
    display_predictions(scores, image_tensor.numpy(), show_figure=args.show_figure)


if __name__ == '__main__':
    main()
