from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

from constants import REGRESSION_LABELS, DATA_DIR, TF_ROTATION, TF_PERSPECTIVE, TF_CONTRAST, TF_BRIGHTNESS
from src.dataloaders.semantic_transforms_dataloader import get_dataloader
from src.dataloaders.semantic_transforms_dataset import STDataset

TF = TF_ROTATION

transform_specs = {
    TF_ROTATION: {'rotation_angles': [[5, 10], [20, 25], [35, 40]]},
    TF_PERSPECTIVE: {'distortion_scale': [0.3, 0.7, 0.9]},
    TF_BRIGHTNESS: {'brightness_factor': [0.1, 0.3, 0.7, 1.3, 1.7, 2.0]},
    TF_CONTRAST: {'contrast_factor': [0.1, 0.3, 1.0, 1.5]}
}


def main():
    save_dir = './results/figures/main-figures/robustness-examples'
    data_dir = os.path.join(DATA_DIR, 'serialized-data/data_232x300-seed_1')
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

    for tf, tf_params in transform_specs.items():
        param_str, param_vals = list(tf_params.items())[0]
        for p in param_vals:
            params = {param_str: p}
            dataset = STDataset(data_root=data_dir, labels=test_labels, label_type=REGRESSION_LABELS,
                                image_size=(2 * 232, 2 * 300), transform=tf, num_scores=4, do_normalize=False, **params)

            num_figs = 0
            for i in range(len(dataset)):
                img, label = dataset[i]
                score = label[-1]
                if score < 30:
                    continue

                num_figs += 1
                img = np.squeeze(img.numpy())
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                save_dir_temp = os.path.join(save_dir, f'example{num_figs}')

                if not os.path.exists(save_dir_temp):
                    os.makedirs(save_dir_temp)

                plt.savefig(os.path.join(save_dir_temp, f'example{num_figs}-{tf}-{p}.png'), bbox_inches='tight')
                plt.close()

                if num_figs > 5:
                    break


if __name__ == '__main__':
    main()
