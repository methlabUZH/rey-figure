import matplotlib.pyplot as plt
import os

from constants import ABSOLUTE_ERROR

from src.analyze.utils import init_mpl
from src.dataloaders.semantic_transforms_dataset import TF_BRIGHTNESS, TF_PERSPECTIVE, TF_CONTRAST, TF_ROTATION

colors = init_mpl(sns_style='ticks', colorpalette='muted')

_CSV_PATTERN_ROT = "rotation_[{}, {}]-{}.csv"
_CSV_PATTERN_PERSPECTIVE = "perspective_{}-{}.csv"
_CSV_PATTERN_BRIGHTNESS = "brightness_{}-{}.csv"
_CSV_PATTERN_CONTRAST = "contrast_{}-{}.csv"

_TF_BRIGHTNESS_INCREASE = TF_BRIGHTNESS + '_increase'
_TF_BRIGHTNESS_DECREASE = TF_BRIGHTNESS + '_decrease'

_XLABELS = {
    TF_ROTATION: 'Rotation Angle',
    TF_PERSPECTIVE: 'Perspective Distortion',
    TF_CONTRAST: 'Contrast Change',
    _TF_BRIGHTNESS_INCREASE: 'Brightness Increase',
    _TF_BRIGHTNESS_DECREASE: 'Brightness Decrease'
}

_FIG_SIZE = (7, 4)


def main(res_dir_aug, res_dir_non_aug, parameters, transformation=TF_ROTATION, quantity=ABSOLUTE_ERROR, save_as=None):
    # calculate errors with transformation
    if transformation == TF_ROTATION:
        xticks, xlabels, aug_errors, non_aug_errors = compute_lines_rotation(
            res_dir_aug, res_dir_non_aug, parameters, quantity)
    elif transformation == TF_PERSPECTIVE:
        xticks, xlabels, aug_errors, non_aug_errors = compute_lines(
            res_dir_aug, res_dir_non_aug, parameters, _CSV_PATTERN_PERSPECTIVE, quantity)
    elif transformation == TF_CONTRAST:
        xticks, xlabels, aug_errors, non_aug_errors = compute_lines(
            res_dir_aug, res_dir_non_aug, parameters, _CSV_PATTERN_CONTRAST, quantity)
    elif transformation in [_TF_BRIGHTNESS_INCREASE, _TF_BRIGHTNESS_DECREASE]:
        xticks, xlabels, aug_errors, non_aug_errors = compute_lines(
            res_dir_aug, res_dir_non_aug, parameters, _CSV_PATTERN_BRIGHTNESS, quantity)
    else:
        raise ValueError

    # plot errors
    fig = plt.figure(figsize=_FIG_SIZE)
    plt.plot(xticks, aug_errors, marker='o', label='w/ Data Augmentation', color=colors[0])
    plt.plot(xticks, non_aug_errors, marker='x', label='w/o Data Augmentation', color=colors[1])
    plt.ylabel('Mean Absolute Error')
    plt.xlabel(_XLABELS[transformation])
    plt.xticks(xticks, xlabels)
    plt.grid(True)
    plt.legend(fancybox=False, fontsize=12)
    plt.tight_layout()

    if save_as is None:
        plt.show()
        plt.close()
        return

    plt.savefig(save_as, bbox_inches='tight', pad_inches=0.1, dpi=100)
    print(f'saved figure as {save_as}')
    plt.close(fig)


def compute_lines_rotation(res_dir_aug, res_dir_non_aug, rot_angles, quantity):
    aug_errors = []
    non_aug_errors = []
    xticks = []
    xlabels = []

    for i, (angle1, angle2) in enumerate(zip(rot_angles[:-1], rot_angles[1:])):
        csv_file_pred = _CSV_PATTERN_ROT.format(angle1, angle2, "test_predictions")
        csv_file_gt = _CSV_PATTERN_ROT.format(angle1, angle2, "test_ground_truths")

        # with augmentation
        aug_errors.append(calc_err(res_dir_aug, csv_file_pred, csv_file_gt, quantity))

        # without augmentation
        non_aug_errors.append(calc_err(res_dir_non_aug, csv_file_pred, csv_file_gt, quantity))

        xticks.append(i + 1)
        xlabels.append(f"{int(angle1)}-{int(angle2)}")

    return xticks, xlabels, aug_errors, non_aug_errors


def compute_lines(res_dir_aug, res_dir_non_aug, persp_params, csv_pattern, quantity):
    aug_errors = []
    non_aug_errors = []
    xticks = []
    xlabels = []

    for i, p in enumerate(persp_params):
        csv_file_pred = csv_pattern.format(p, "test_predictions")
        csv_file_gt = csv_pattern.format(p, "test_ground_truths")

        # with augmentation
        aug_errors.append(calc_err(res_dir_aug, csv_file_pred, csv_file_gt, quantity))

        # without augmentation
        non_aug_errors.append(calc_err(res_dir_non_aug, csv_file_pred, csv_file_gt, quantity))

        xticks.append(i + 1)
        xlabels.append(f"{p}")

    return xticks, xlabels, aug_errors, non_aug_errors


def calc_err(res_dir, csv_file_pred, csv_file_gt, quantity):
    # load data
    preds = os.path.join(res_dir, csv_file_pred)
    preds = pd.read_csv(preds)

    ground_truths = os.path.join(res_dir, csv_file_gt)
    ground_truths = pd.read_csv(ground_truths)

    # calculate errors
    errors = compute_errors(preds, ground_truths)

    return errors[quantity].mean()


if __name__ == '__main__':
    res_aug = '../results/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier'
    res_non_aug = '../results/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize/rey-multilabel-classifier'

    rotation_angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    perspective_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    contrast_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    brightness_decr_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    brightness_incr_params = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    save_dir = '../../../results/figures/semantic-transformations/'

    main(res_aug, res_non_aug, rotation_angles, transformation=TF_ROTATION, save_as=save_dir + 'rotations.pdf')
    main(res_aug, res_non_aug, perspective_params, transformation=TF_PERSPECTIVE, save_as=save_dir + 'perspective.pdf')
    main(res_aug, res_non_aug, contrast_params, transformation=TF_CONTRAST, save_as=save_dir + 'contrast.pdf')
    main(res_aug, res_non_aug, brightness_decr_params, transformation=_TF_BRIGHTNESS_DECREASE,
         save_as=save_dir + 'brightness_decrease.pdf')
    main(res_aug, res_non_aug, brightness_incr_params, transformation=_TF_BRIGHTNESS_INCREASE,
         save_as=save_dir + 'brightness_increase.pdf')
