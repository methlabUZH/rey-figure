import math
from datetime import datetime as dt


def is_number(s):
    try:
        number = float(s)
        return not math.isnan(number)
    except ValueError:
        return False


def check_if_in_range(value, upper_bound, lower_bound):
    if lower_bound <= value < upper_bound:
        return True
    return False


def timestamp_human():
    return dt.now().strftime('%H:%M:%S')



class File:
    """
    An image file, used to match predicted labels with original images

    defined here instead of in dataloader.py, otherwise there is an error in the following situation:
    writing preprocessed serialized data while executing dataloader.py, later trying to read them while executing model.py
    consider for details: https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error
    """

    def __init__(self, filepath, dataset_name=""):
        self.filepath = filepath
        self.dataset = dataset_name
