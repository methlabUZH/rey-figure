import numpy as np
import csv
from cv2 import imread
from preprocess import preprocess, preprocess_basic
import os
from skimage.io import imsave
import shutil
from helpers import File, create_directory
from sklearn.utils import shuffle
import time



from config import DATA_DIR, DEBUG, REDO_PREPROCESSING, LOAD_ONLY_FEW

RAW = [
    {'name': "Mexico", 'images': DATA_DIR + "raw/Mexico/", 'labels': DATA_DIR + "raw/mexico_trainval.csv"},
    {'name': "Colombia", 'images': DATA_DIR + "raw/Colombia/", 'labels': DATA_DIR + "raw/colombia_trainval.csv"},
    {'name': "Brugger", 'images': DATA_DIR + "raw/Brugger_Daten/", 'labels': DATA_DIR + "raw/brugger_trainval.csv"}
]

# load only few images per preprocessing set, not all (to speed up testing)
# LOAD_ONLY_FEW = False
# Read in via config file

# preprocessed images (incl. labels) are written to disk for faster recovery
# if raw preprocessing or preprocessing changes, you have to redo it by setting this to True
# REDO_PREPROCESSING = False
# Read in via config file

# make splits deterministic
RANDOM_STATE = 123



class Dataset:
    """A preprocessing set consisting of images, labels, and files"""
    def __init__(self, name, images, files, labels = None, intermediates = None):
        self.images = images
        self.labels = labels
        self.intermediates = intermediates
        self.name = name
        self.files = files

    def split(self, train_indices, val_indices):
        train_images = self.images[train_indices]
        train_labels = self.labels[train_indices]
        train_intermediates = self.intermediates[train_indices]
        train_files = self.files[train_indices]
        val_images = self.images[val_indices]
        val_labels = self.labels[val_indices]
        val_intermediates = self.intermediates[val_indices]
        val_files = self.files[val_indices]

        train = Dataset(self.name + " TRAIN", train_images, train_files, train_labels, train_intermediates)
        val = Dataset(self.name + " VAL", val_images, val_files, val_labels, val_intermediates)
        return train, val

    def add_dataset(self, new_images):
        self.images = np.concatenate([self.images,new_images.images],axis=0)
        self.labels = np.concatenate([self.labels, new_images.labels], axis=0)
        self.files = np.concatenate([self.files, new_images.files], axis=0)

    def set_name(self,name):
        self.name = name

    def shuffle(self):
        self.images, self.labels, self.files = shuffle(self.images, self.labels, self.files, random_state=RANDOM_STATE)

    def visualize_dataset(self, n = 50, directory = DATA_DIR + "visualized/"):
        """
        Visualizes preprocessing set by writing original image (from path) vs image in preprocessing set to disk
        n is the number of items visualized, set to -1 if you want to visualize all of them
        """

        # create directory, delete all files in it
        create_directory(directory)
        dir = directory + self.name + "/"
        create_directory(dir,empty_dir=True)

        # go through n items in dataset and write comparison to disk
        for ind, image in enumerate(self.images):
            original_image = imread(self.files[ind].path)
            original_image = preprocess_basic(original_image)
            border = np.zeros([image.shape[0], 3])
            combined_img = np.append(original_image, border, axis=1)
            combined_img = np.append(combined_img, image, axis=1)
            try:
                imsave(dir + str(self.labels[ind]) + "---" + self.files[ind].filename, combined_img)
            except ValueError:
                print("Failed to write to file " + dir + str(self.labels[ind]) + "---" + self.files[ind].filename)
            if n > 0 and ind > n:
                break

    def __str__(self):
        return "DATA SET: {}\nData shape:{}\nLabels shape: {}\nFiles shape: {}".format(self.name, self.images.shape, self.labels.shape, len(self.files))


def join_datasets(name, sets):
    joined_set = Dataset(name, sets[0].images, sets[0].files, sets[0].labels)
    for s in sets[1:]:
        joined_set.add_dataset(s)
    return joined_set



def load_raw_data(image_path, label_path):
    if(DEBUG):
        print("Loading dataset "+image_path)
    #read labels
    with open(label_path) as csv_file:
        labels_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in labels_reader]
        filenames = [row[0] for row in rows]
        labels = [row[1] for row in rows]


    if(LOAD_ONLY_FEW):
        filenames = filenames[0:20]
        labels = labels[0:20]

    # read images
    valid_paths = [image_path + f for f in filenames]
    images = [imread(file) for file in valid_paths]

    return images, labels, filenames


def preprocess_images(images):
    if(DEBUG):
        print("Preprocessing " + str(len(images)) + " images...")
    for i in range(len(images)):
        images[i] = preprocess(images[i])
    return images


def preprocess_labels(labels):
    return np.array([np.float(x) for x in labels])


def collect_file_info(filenames, directory, dataset_name):
    """
    Creates a list of File objects of all the files in filenames
    Can later be used to match trained labels to their original image files
    """
    files = []
    for f in filenames:
        file = File(f, directory, dataset_name)
        files.append(file)
    return files


datasets = []

# check if preprocessed preprocessing available -> don't redo anything
if REDO_PREPROCESSING or not os.path.exists(DATA_DIR + "serialized/images.npy"):

    for set in RAW:
        images, labels, filenames = load_raw_data(set['images'], set['labels'])
        start_preprocessing = time.perf_counter()
        images = preprocess_images(images)
        end_preprocessing = time.perf_counter()
        print("preprocessing of dataset {} took {}s".format(set['name'], end_preprocessing-start_preprocessing))
        labels = preprocess_labels(labels)
        files = collect_file_info(filenames, set['images'], set['name'])
        new_dataset = Dataset(set['name'], np.asarray(images), files, np.asarray(labels))
        datasets.append(new_dataset)

    DATA = join_datasets("DATA", datasets)
    DATA.shuffle()

    # save to disk for later use
    if(DEBUG):
        print("Writing preprocessed preprocessing to disk...")
    create_directory(DATA_DIR + "serialized")
    np.save(DATA_DIR + 'serialized/images.npy', DATA.images)
    np.save(DATA_DIR + 'serialized/labels.npy', DATA.labels)
    np.save(DATA_DIR + 'serialized/files.npy', DATA.files)
    try:
        os.chmod(DATA_DIR + 'serialized/images.npy', 0o777)
        os.chmod(DATA_DIR + 'serialized/labels.npy', 0o777)
        os.chmod(DATA_DIR + 'serialized/files.npy', 0o777)
    except PermissionError:
        pass


else:
    # reload preprocessed preprocessing from disk
    if(DEBUG):
        print("Reading preprocessed preprocessing from disk...")
    images = np.load(DATA_DIR + 'serialized/images.npy')
    labels = np.load(DATA_DIR + 'serialized/labels.npy')
    files = np.load(DATA_DIR + 'serialized/files.npy')

    DATA = Dataset("DATA", images, files, labels)



print("preprocessing")
print(DATA)

if __name__ == "__main__":
    pass
    #DATA.visualize_dataset(n=-1)










