import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from config import DATA_DIR, DEBUG

RAW = [
    {'name': "Mexico", 'labels': DATA_DIR + "raw/mexico_trainval.csv", 'dest': DATA_DIR + "raw/mexico_trainval_binned.csv"},
    {'name': "Colombia", 'labels': DATA_DIR + "raw/colombia_trainval.csv", 'dest': DATA_DIR + "raw/colombia_trainval_binned.csv"},
    {'name': "Brugger", 'labels': DATA_DIR + "raw/brugger_trainval.csv", 'dest': DATA_DIR + "raw/brugger_trainval_binned.csv"},
    {'name': "Mexico_test", 'labels': DATA_DIR + "raw/mexico.csv", 'dest': DATA_DIR + "raw/mexico_binned.csv"},
    {'name': "Colombia_test", 'labels': DATA_DIR + "raw/colombia.csv", 'dest': DATA_DIR + "raw/colombia_binned.csv"},
    {'name': "Brugger_test", 'labels': DATA_DIR + "raw/brugger.csv", 'dest': DATA_DIR + "raw/brugger_binned.csv"},
]

# Edges of bins
BIN_LOCATIONS = [(0, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30), (30, 32),
                 (32, 34), (34, 36), (36, 37)] # last bin for flawless images


# TODO implement using pandas
def bin_numbers(labels):
    bins = pd.IntervalIndex.from_tuples(BIN_LOCATIONS, closed='left')
    categorical_object = pd.cut(labels, bins)
    print(categorical_object.categories)
    binned_data = categorical_object.codes
    return binned_data


def one_hot_encoding(binned_data):
    onehot_encoder = OneHotEncoder(sparse=False)
    maxi = np.max(binned_data)
    onehot_encoder.fit(np.asarray(range(maxi+1)).reshape(-1, 1))
    onehot_encoded = onehot_encoder.transform(binned_data)
    print(onehot_encoded)
    return onehot_encoded


def read_labels(label_path):
    if DEBUG:
        print("Reading labels from: " + label_path)

    # read labels
    with open(label_path) as csv_file:
        labels_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in labels_reader]
        filenames = [row[0] for row in rows]
        labels = [row[1] for row in rows]

    return filenames, labels


def write_bins(filenames, labels, dest_path):
    if DEBUG:
        print("Writing bin file to: " + dest_path)

    new_labels = bin_numbers(labels).reshape(-1, 1)

    print(new_labels.shape)

    one_hot_labels = one_hot_encoding(new_labels)

    print(one_hot_labels)

    # write to csv
    with open(dest_path, 'w') as csv_file:
        labels_writer = csv.writer(csv_file)
        for name, label, old in zip(filenames, one_hot_labels.tolist(), labels):
            labels_writer.writerow([str(name)] + label)


entry = RAW[2]
path = entry['labels']
dest_path = entry['dest']
files, labels = read_labels(path)
labels = np.asarray(labels, dtype=float)
write_bins(files, labels, dest_path)
