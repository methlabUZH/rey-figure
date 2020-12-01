import numpy as np
from config import DATA_DIR
import os
import csv
from skimage.io import imsave
import shutil

import preprocess


RANDOM = np.random.RandomState(42)


class BatchIds:
    def __init__(self, n, random = True, reset = True):
        self._n = n
        self._random = random
        self._reset = reset
        self._perm = self._get_permutation()
        self._curr_id = 0

    def empty(self):
        return  (not self._reset) and (self._n == self._curr_id)

    def _get_permutation(self):
        if(self._random):
            return RANDOM.permutation(self._n)
        else:
            return np.arange(self._n)

    def get_batch_ids(self, batch_size):
        if(self._curr_id + batch_size <self._n):
            end = self._curr_id + batch_size
            out = self._perm[self._curr_id:end]
            self._curr_id = end
            return out
        else:
            remaining = batch_size - (self._n - self._curr_id)
            out1 = self._perm[self._curr_id:self._n]
            if(self._reset):
                self._perm = self._get_permutation()
                out2 = self._perm[:remaining]
                self._curr_id = remaining
                return np.concatenate([out1, out2])
            else:
                if(self._curr_id >= self._n):
                    raise Exception("BatchIds: Reset is disabled. All ids read.")
                self._curr_id = self._n
                return out1


class File:
    """
    A image file, used to match predicted labels with original images

    defined here instead of in dataloader.py, otherwise there is an error in the following situation:
    writing preprocessed serialized data while executing dataloader.py, later trying to read them while executing model.py
    consider for details: https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error
    """
    def __init__(self, filename, directory, dataset_name = ""):
        self.filename = filename
        self.directory = directory
        self.path = directory + filename
        self.dataset = dataset_name


def log_validation_predictions(labels, predictions, files, log_filename, extra_information = np.array([])):
    """Writes the predictions of validation data to a csv file for inspection"""
    filenames = [f.filename for f in files]
    paths = [f.path for f in files]
    dataset_names = [f.dataset for f in files]
    labels = np.reshape(labels,labels.shape[0]) #reshape to 1d
    predictions = np.reshape(predictions,predictions.shape[0]) #reshape to 1d
    def extra_information_printable(row):
        # takes each row of extra information and returns a string (might be multi-dimensional)
        row = np.array2string(row, formatter={'float_kind':lambda x: "%.3f" % x})
        return row
    if extra_information is not None:
        extra_information = np.apply_along_axis(extra_information_printable, 1, extra_information)
    with open(log_filename, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        if extra_information is not None:
            writer.writerow(['label', 'prediction', 'data set', 'file name', 'path', 'extra_information'])
            writer.writerows(zip(labels, predictions, dataset_names, filenames, paths, extra_information))
        else:
            writer.writerow(['label', 'prediction', 'data set', 'file name', 'path'])
            writer.writerows(zip(labels, predictions, dataset_names, filenames, paths))
    csv_file.close()

def write_validation_errors(labels, predictions, files, runname):
   """
   For each training item writes the error to a file to be used to later classify into easy and hard figures
   """
   filenames = [f.filename for f in files]
   paths = [f.path for f in files]
   labels = np.reshape(labels, labels.shape[0])  # reshape to 1d
   predictions = np.reshape(predictions, predictions.shape[0])  # reshape to 1d
   errors = np.abs(labels-predictions)
   create_directory(DATA_DIR + "validation_errors")
   error_file = DATA_DIR + "validation_errors/errors_"+runname+".csv"
   mode = 'a' if os.path.isfile(error_file) else 'w'
   with open(error_file, mode=mode) as csv_file:
       writer = csv.writer(csv_file)
       writer.writerow(['label', 'prediction', 'abs_error', 'file name', 'path'])
       writer.writerows(zip(labels, predictions, errors, filenames, paths))
   csv_file.close()


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

# helper fuctions to write images to disk for inspection
# useful e.g. for debugging
def img_to_file(img, path="saved_image.jpg", undo_normalization = False):
    if undo_normalization:
        img = preprocess.undo_normalization(img)
    imsave(path, img)

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
