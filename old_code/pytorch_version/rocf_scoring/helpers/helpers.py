import math
import os
import shutil
import numpy as np
import pandas as pd
from rocf_scoring.data_preprocessing.preprocess import preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class File:
    """
    A image file, used to match predicted labels with original images

    defined here instead of in dataloader.py, otherwise there is an error in the following situation:
    writing preprocessed serialized preprocessing while executing dataloader.py, later trying to read them while executing model.py
    consider for details: https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error
    """

    def __init__(self, filename, directory, dataset_name=""):
        self.filename = filename
        self.directory = directory
        self.path = directory + filename
        self.dataset = dataset_name



def is_number(s):
    try:
        number = float(s)
        return not math.isnan(number)
    except ValueError:
        return False

def create_directory(path, empty_dir = False):
    """ Creates directory if it doesnt exist yet, optionally deleting all files in there """
    if not os.path.exists(path):
        os.makedirs(path)

    if empty_dir:
        shutil.rmtree(path)
        os.makedirs(path)

    try:
        os.chmod(path, 0o777)
    except PermissionError:
        pass

def imgs_to_file(imgs, path="combined_image.jpg", undo_normalization = False):
    if not isinstance(imgs,list):
        raise ValueError("must provide a list of images")

    # find max dimension so all images can be displayed side by side
    max_y = 0
    for img in imgs:
        if img.shape[0] > max_y: max_y = img.shape[0]

    combined_img = np.zeros([max_y, 0])
    border = np.zeros([max_y, 3])
    for i, img in enumerate(imgs):
        if i != 0:
            combined_img = np.append(combined_img, border, axis=1)

        # expand to max dimension
        extra_y = max_y - img.shape[0]
        padding = np.ones((extra_y, img.shape[1]), dtype=img.dtype) / 2
        img = np.concatenate((img, padding), axis=0)

        if undo_normalization:
            # redo normalization (so its visually interpretable)
            img = preprocess.undo_normalization(img)
        combined_img = np.append(combined_img, img, axis=1)

    # add white border to bottom
    combined_img = np.append(combined_img, np.ones((10,combined_img.shape[1])), axis=0)

    combined_img = np.clip(combined_img,0,1)
    imsave(path, combined_img)


def check_if_in_range(value, upper_bound, lower_bound):
    if lower_bound <= value < upper_bound:
        return True
    return False


def visualize_training(results_dir):
    col_names = ["epoch", "n_iter", "loss", "acc"]

    train_res_path = os.path.join(results_dir, "train_results.csv")
    test_res_path = os.path.join(results_dir, "test_results.csv")

    train_res_df = pd.read_csv(train_res_path, names=col_names)
    test_res_df = pd.read_csv(test_res_path, names=col_names)


    # construct loss graph
    plt.plot(train_res_df["n_iter"], train_res_df["loss"], label='train loss')
    plt.plot(test_res_df["n_iter"], test_res_df["loss"], label='test loss')

    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()

    save_path=os.path.join(results_dir, "loss_plot.png")
    plt.savefig(save_path)
    plt.show()


    plt.plot(train_res_df["n_iter"], train_res_df["acc"], label='train acc')
    plt.plot(test_res_df["n_iter"], test_res_df["acc"], label='test acc')
    plt.ylabel('accuracy')
    plt.xlabel('iteration')
    plt.legend()

    save_path = os.path.join(results_dir, "acc_plot.png")
    plt.savefig(save_path)
    plt.show()



if __name__ == "__main__":
    results_dir = "/Users/felixasanger/Desktop/pytorch_rey-figure/rocf_scoring/training_results/20201109_12-06-01_test_felix"
    visualize_training(results_dir)