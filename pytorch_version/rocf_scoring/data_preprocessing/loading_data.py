import numpy as np
import csv
from cv2 import imread
import os
from rocf_scoring.helpers.helpers import is_number, File
from rocf_scoring.features.augmenter import simulate_augment, augment
from rocf_scoring.data_preprocessing.preprocess import preprocess
from rocf_scoring.helpers.helpers import create_directory


from sklearn.utils import shuffle
import time
import math
from collections import defaultdict
from config import DATA_DIR, DEBUG, DATA_AUGMENTATION, LOAD_ONLY_FEW, REDO_PREPROCESSING_LABELS, \
    REDO_PREPROCESSING_IMAGES, LABEL_FORMAT, MODEL_PATH, INTERMEDIATES, CONVERGENCE, TEST

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from helpers import File, create_directory, imgs_to_file


# make splits deterministic
RANDOM_STATE = 123

# FILENAME OF LABELS
#FILENAME_LABELS = "Data2018-11-29.csv"
FILENAME_LABELS = "Data07112018.csv"

class Item:
    def __init__(self, filename, assessment_id, id, score, visible, right_place, drawn_correctly):
        self.filename = filename
        self.assessment_id = assessment_id
        self.id = id
        self.score = float(score)
        self.visible = float(visible)
        self.right_place = float(right_place)
        self.drawn_correctly = float(drawn_correctly)

    def getScore(self):
        return self.score


class Assessment:
    def __init__(self, id, filename):
        self.items = {}
        self.id = id
        self.filename = filename
        self.invalid = False

    def addItem(self, item_id, score, visible, right_place, drawn_correctly):
        if item_id in self.items:
            # check if it's the same (double line) or different (error!)
            curr = self.items[item_id]
            if curr.score == float(score) and curr.visible == float(visible) and curr.right_place == float(right_place) \
                    and curr.drawn_correctly == float(drawn_correctly):
                # nothing to do, just double line
                pass
            else:
                #print("error, item {} already in list of items for figure {}, assessment {}, and is different from current -> remove assessment".format(item_id, self.filename, self.id))
                # mark as invalid -> cannot be used
                self.invalid = True
        else:
            self.items[item_id] = Item(self.filename, self.id, item_id, score, visible, right_place, drawn_correctly)

    def itemCount(self):
        return len(self.items)

    def getScore(self):
        return sum([self.items[i].getScore() for i in self.items])

    def isValid(self):
        # check if all scores and in correct range and not marked as invalid
        if self.itemCount() == 18 and not self.invalid:
            return True
        return False


class Figure:
    def __init__(self, filename):
        self.filename = filename
        self.assessments = {}

    def getAssessment(self, assessment_id):
        if assessment_id not in self.assessments:
            self.assessments[assessment_id] = Assessment(assessment_id, self.filename)
        return self.assessments[assessment_id]

    def getScores(self):
        # only scores of valid assessments
        return [a.getScore() for a in self.getValidAssessments()]

    def getMedianScore(self):
        return np.median(self.getScores())

    def getMedianScorePerItem(self):
        # get median score of each item
        scores = defaultdict(list)
        list_of_items = [i for i in self.getValidAssessments()[0].items]
        for ass in self.getValidAssessments():
            for i in list_of_items:
                if i in ass.items:
                    scores[i].append(ass.items[i].getScore())
        median_scores = [np.median(scores[s]) for s in sorted(scores.keys())]
        return median_scores

    def getMedianPartScorePerItem(self):
        # get median score of each part of each item (3 scores per item), as one list of dimension 54
        scores = defaultdict(list)
        list_of_items = [i for i in self.getValidAssessments()[0].items]
        for ass in self.getValidAssessments():
            for i in list_of_items:
                if i in ass.items:
                    scores[str(i)+"visible"].append(ass.items[i].visible)
                    scores[str(i)+"right_place"].append(ass.items[i].right_place)
                    scores[str(i)+"drawn_correctly"].append(ass.items[i].drawn_correctly)
        median_scores = [np.median(scores[s]) for s in sorted(scores.keys())]
        return median_scores

    def getImage(self, set = "train"):
        path = DATA_DIR + "Data_{}/".format(set) + self.filename
        image = imread(path)
        if image is None:
            print("Error: could not load file {}".format(path))
        return image

    def getValidAssessments(self):
        return [self.assessments[a] for a in self.assessments if self.assessments[a].isValid()]

    def hasValidAssessment(self):
        if len(self.getValidAssessments()) > 0:
            return True
        return False


class Dataset:
    """A data set consisting of images, labels, and files
    ONLY GONNA BE THERE TEMPORARILY
    """
    def __init__(self, name, images, files, labels = None, intermediates = None):
        self.images = images
        self.labels = labels
        self.name = name
        self.files = files
        self.intermediates = intermediates

    def split(self, train_indices, val_indices):
        train_images = self.images[train_indices]
        train_labels = self.labels[train_indices]
        train_files = self.files[train_indices]
        val_images = self.images[val_indices]
        val_labels = self.labels[val_indices]
        val_files = self.files[val_indices]

        if self.intermediates is None or self.intermediates.shape[0] != self.images.shape[0]:
            train_intermediates = None
            val_intermediates = None
        else:
            train_intermediates = self.intermediates[train_indices]
            val_intermediates = self.intermediates[val_indices]

        train = Dataset(self.name + " TRAIN", train_images, train_files, train_labels, train_intermediates)
        val = Dataset(self.name + " VAL", val_images, val_files, val_labels, val_intermediates)
        return train, val

    def shuffleAll(self):
        self.images, self.labels, self.files = shuffle(self.images, self.labels, self.files, random_state=RANDOM_STATE)

    def shuffleImages(self):
        # if labels and files have already been shuffled (deterministically), do the same here with images
        self.images = shuffle(self.images, random_state=RANDOM_STATE)

    def __str__(self):
        return "DATA SET: {}\nData shape:{}\nLabels shape: {}\nFiles shape: {}\n"\
            .format(self.name, self.images.shape, self.labels.shape, len(self.files))


def load_raw_data(set = "train"):

    if DEBUG:
        print("Loading new data {} ...".format(set))
        print(DATA_DIR + FILENAME_LABELS)

    raw_figures = {}
    with open(DATA_DIR + FILENAME_LABELS) as csv_file:
        labels_reader = csv.reader(csv_file, delimiter=',')


        rows = [row for row in labels_reader]
        rows = rows[1:] #first is label

        for ind, row in enumerate(rows):

            if not is_number(row[11]) or float(row[11]) >= 0.8:
                filename = row[5]
                assessment_id = row[0]
                # add this assessment / figure if in correct (train or test) directory
                if os.path.exists(DATA_DIR + "Data_{}/".format(set) + filename):
                    if not filename in raw_figures:
                        raw_figures[filename] = Figure(filename)

                    assessment = raw_figures[filename].getAssessment(assessment_id)
                    assessment.addItem(row[6], row[7], row[8], row[9], row[10])

    # convert dictionary of figures to list
    figures = [raw_figures[fig] for fig in raw_figures if raw_figures[fig].hasValidAssessment()]

    if DEBUG:
        print("{} figures in {} data have no valid assessment, left with {} good figures".format(len(raw_figures)-len(figures), set, len(figures)))

    if LOAD_ONLY_FEW:
        figures = figures[0:50]

    # get list of labels
    if LABEL_FORMAT == 'one-per-item':
        labels = [fig.getMedianScorePerItem() + [fig.getMedianScore()] for fig in figures]
    elif LABEL_FORMAT == 'three-per-item':
        labels = [fig.getMedianPartScorePerItem() + [fig.getMedianScore()] for fig in figures]
    else:
        labels = [fig.getMedianScore() for fig in figures]

    # get list of files
    directory = DATA_DIR + "Data_{}/".format(set)
    print("directory: ", directory)
    files = [File(fig.filename,directory) for fig in figures]

    return figures, labels, files

def prepare_dataset():
    """
    is only gonna be there temporarily
    :return: prepared data set
    """
    if REDO_PREPROCESSING_LABELS or REDO_PREPROCESSING_IMAGES:
        figures, labels, files = load_raw_data()

    if REDO_PREPROCESSING_LABELS:
        labels = preprocess_labels(labels)
        labels = np.asarray(labels)
        files = np.asarray(files)

        # shuffle labels and files, important that random_state is same as when the images where shuffled
        labels, files = shuffle(labels, files, random_state=RANDOM_STATE)

    else:
        if LABEL_FORMAT == 'one-per-item':
            labels = np.load(DATA_DIR + 'serialized/labels_oneperitem.npy')
        elif LABEL_FORMAT == 'three-per-item':
            labels = np.load(DATA_DIR + 'serialized/labels_threeperitem.npy')
        else:
            labels = np.load(DATA_DIR + 'serialized/labels.npy')
        files = np.load(DATA_DIR + 'serialized/files.npy')

    if REDO_PREPROCESSING_IMAGES:

        if not DATA_AUGMENTATION:
            # no data augmentation -> load everything

            if DEBUG:
                print("Loading images from disk ...")
            images = [fig.getImage() for fig in figures]

            start_preprocessing = time.perf_counter()
            images = preprocess_images(images)
            end_preprocessing = time.perf_counter()
            print("preprocessing of all images took {}s".format(end_preprocessing - start_preprocessing))

            if CONVERGENCE == 0:
                DATA = Dataset("DATA", np.asarray(images), files, labels)
            else:
                DATA = Dataset("DATA", np.asarray(images)[0:CONVERGENCE, :], files[0:CONVERGENCE],
                               labels[0:CONVERGENCE])

        else:
            # data augmentation -> load in batches, then combine

            batch_size = 50  # do data augmentation in batches
            l = 0
            create_directory(DATA_DIR + 'serialized/augmentation_batches')

            while batch_size * l < len(figures):
                if DEBUG:
                    print("\nData augmentation batch {} of {}...".format(l, int(np.ceil(len(figures) / batch_size))))
                    print("Loading images from disk ...")

                images = [fig.getImage() for fig in
                          figures[l * batch_size:np.min([(l + 1) * batch_size, len(figures)])]]

                start_batch = time.perf_counter()
                start_preprocessing = time.perf_counter()
                images = preprocess_images(images)
                end_preprocessing = time.perf_counter()
                if DEBUG:
                    print("preprocessing of {} images took {}s".format(len(images),
                                                                       end_preprocessing - start_preprocessing))

                # do actual data augmentation
                new_images = np.empty([np.shape(images)[0], 1, 116, 150])
                if DEBUG:
                    print("Doing simulated augmentation")
                start_aug = time.perf_counter()
                for i in range(np.shape(images)[0]):
                    new_images[i, 0, :, :] = simulate_augment(np.array(images)[i, :, :])
                end_aug = time.perf_counter()
                if DEBUG:
                    print("simulated augmentation of {} images took {}s".format(len(images), end_aug - start_aug))

                for k in range(9):
                    start_aug = time.perf_counter()
                    add_images = np.empty([np.shape(images)[0], 1, 116, 150])
                    print("Doing augmentation {}".format(k + 1))
                    for i in range(np.shape(images)[0]):
                        add_images[i, 0, :, :] = augment(np.array(images)[i, :, :], 10, 5, 1.5, 10)
                    new_images = np.concatenate((new_images, add_images), axis=1)
                    end_aug = time.perf_counter()
                    print("augmentation {} of {} images took {}s".format(k + 1, len(images), end_aug - start_aug))

                end_batch = time.perf_counter()
                np.save(DATA_DIR + 'serialized/augmentation_batches/batch' + str(l) + '.npy', new_images)
                print("Batch {} took {}s".format(l, end_batch - start_batch))
                l = l + 1

            # combine batches
            all_images = []
            for i in range(l):
                add_images = np.load(DATA_DIR + 'serialized/augmentation_batches/batch' + str(i) + '.npy')
                if i == 0:
                    all_images = add_images
                else:
                    all_images = np.concatenate((all_images, add_images), 0)
                print(np.shape(all_images))

            if CONVERGENCE == 0:
                DATA = Dataset("DATA", all_images, files, labels)
            else:
                DATA = Dataset("DATA", all_images[0:CONVERGENCE, :], files[0:CONVERGENCE], labels[0:CONVERGENCE])

        # shuffle images; labels and files have already been shuffled accordingly
        DATA.shuffleImages()

    else:
        # don't redo preprocessing for images -> load from disk
        if (DEBUG):
            print("Reading preprocessed data from disk...")

        if DATA_AUGMENTATION:
            images = np.load(DATA_DIR + 'serialized/images_aug.npy')
        else:
            images = np.load(DATA_DIR + 'serialized/images.npy')

        # no need to store in .npy since runs extremely fast (+ would be not needing GPU)
        if INTERMEDIATES:
            model = restore(MODEL_PATH)
            intermediates = np.empty((images.shape[0], 1024))
            for start, end in zip(range(0, images.shape[0], 256), range(256, images.shape[0] + 256, 256)):
                if end > images.shape[0]:
                    end = images.shape[0]
                intermediates[start:end, :] = model.get_intermediate(images[start:end, :])

        else:
            intermediates = None

        if CONVERGENCE == 0:
            DATA = Dataset("DATA", images, files, labels)
        else:
            DATA = Dataset("DATA", images[0:CONVERGENCE, :], files[0:CONVERGENCE], labels[0:CONVERGENCE])

    if REDO_PREPROCESSING_LABELS or REDO_PREPROCESSING_IMAGES:
        # save to disk for later use
        if (DEBUG):
            print("Writing preprocessed data to disk...")
        create_directory(DATA_DIR + "serialized")

    if REDO_PREPROCESSING_IMAGES:
        # save images depending on data augmentation or not
        if DATA_AUGMENTATION:
            np.save(DATA_DIR + 'serialized/images_aug.npy', DATA.images)
        else:
            np.save(DATA_DIR + 'serialized/images.npy', DATA.images)

    if REDO_PREPROCESSING_LABELS:
        # save labels depending on LABEL_FORMAT
        if LABEL_FORMAT == 'one-per-item':
            np.save(DATA_DIR + 'serialized/labels_oneperitem.npy', DATA.labels)
        elif LABEL_FORMAT == 'three-per-item':
            np.save(DATA_DIR + 'serialized/labels_threeperitem.npy', DATA.labels)
        else:
            np.save(DATA_DIR + 'serialized/labels.npy', DATA.labels)

        # save files
        np.save(DATA_DIR + 'serialized/files.npy', DATA.files)

    # change permission of written files
    try:
        if REDO_PREPROCESSING_IMAGES:
            if DATA_AUGMENTATION:
                os.chmod(DATA_DIR + 'serialized/images_aug.npy', 0o777)
            else:
                os.chmod(DATA_DIR + 'serialized/images.npy', 0o777)
        if REDO_PREPROCESSING_LABELS:
            if LABEL_FORMAT == 'one-per-item':
                os.chmod(DATA_DIR + 'serialized/labels_oneperitem.npy', 0o777)
            elif LABEL_FORMAT == 'three-per-item':
                os.chmod(DATA_DIR + 'serialized/labels_threeperitem.npy', 0o777)
            else:
                os.chmod(DATA_DIR + 'serialized/labels.npy', 0o777)
            os.chmod(DATA_DIR + 'serialized/files.npy', 0o777)
    except:
        pass

    # if test time -> prepare test data
    if TEST:
        figures_test, labels_test, files_test = load_raw_data("test")
        labels_test = preprocess_labels(labels_test)
        labels_test = np.asarray(labels_test)
        files_test = np.asarray(files_test)

        if DEBUG:
            print("Loading test images from disk ...")
        images_test = [fig.getImage('test') for fig in figures_test]

        start_preprocessing = time.perf_counter()
        images_test = preprocess_images(images_test, set="test")
        end_preprocessing = time.perf_counter()
        print("preprocessing of all test images took {}s".format(end_preprocessing - start_preprocessing))

        DATA_TEST = Dataset("DATA_TEST", np.asarray(images_test), files_test, labels_test)
    else:
        DATA_TEST = None


    if TEST:
        return DATA, DATA_TEST
    return DATA

def preprocess_dataset(figures, labels, files):
    """Loads raw data and preprocesses it.
    :return: preprocessed images, files, labels
    """

    preprocessed_labels = preprocess_labels(labels)
    preprocessed_labels = np.asarray(preprocessed_labels)
    files = np.asarray(files)

    # preprocessed_labels, files = shuffle(preprocessed_labels, files, random_state=RANDOM_STATE)

    if DEBUG:
        print("Loading images from disk ...")

    if len(figures) == 1:
        figures = [figures]

    images = [fig.getImage() for fig in figures]

    # start_preprocessing = time.perf_counter()
    preprocessed_images = preprocess_images(images)

    # end_preprocessing = time.perf_counter()
    # print("preprocessing of all images took {}s".format(end_preprocessing - start_preprocessing))

    # TODO: can be deleted later on
    # if CONVERGENCE == 0:
    #     DATA = Dataset("DATA", np.asarray(preprocessed_images), files, preprocessed_labels)
    # else:
    #     DATA = Dataset("DATA", np.asarray(preprocessed_images)[0:CONVERGENCE, :], files[0:CONVERGENCE],
    #                    preprocessed_labels[0:CONVERGENCE])

    #################################################
    #TODO: create function to save serialized data  #
    #################################################
    if REDO_PREPROCESSING_LABELS or REDO_PREPROCESSING_IMAGES:
        # save to disk for later use
        if (DEBUG):
            print("Writing preprocessed data to disk...")
        create_directory(DATA_DIR + "serialized")

    if REDO_PREPROCESSING_IMAGES:
        # save images depending on data augmentation or not
        if DATA_AUGMENTATION:
            np.save(DATA_DIR + 'serialized/images_aug.npy', preprocessed_images)
        else:
            np.save(DATA_DIR + 'serialized/images.npy', preprocessed_images)

    if REDO_PREPROCESSING_LABELS:
        # save labels depending on LABEL_FORMAT
        if LABEL_FORMAT == 'one-per-item':
            np.save(DATA_DIR + 'serialized/labels_oneperitem.npy', preprocessed_labels)
        elif LABEL_FORMAT == 'three-per-item':
            np.save(DATA_DIR + 'serialized/labels_threeperitem.npy', preprocessed_labels)
        else:
            np.save(DATA_DIR + 'serialized/labels.npy', preprocessed_labels)

        # save files
        np.save(DATA_DIR + 'serialized/files.npy', files)

    # change permission of written files
    try:
        if REDO_PREPROCESSING_IMAGES:
            if DATA_AUGMENTATION:
                os.chmod(DATA_DIR + 'serialized/images_aug.npy', 0o777)
            else:
                os.chmod(DATA_DIR + 'serialized/images.npy', 0o777)
        if REDO_PREPROCESSING_LABELS:
            if LABEL_FORMAT == 'one-per-item':
                os.chmod(DATA_DIR + 'serialized/labels_oneperitem.npy', 0o777)
            elif LABEL_FORMAT == 'three-per-item':
                os.chmod(DATA_DIR + 'serialized/labels_threeperitem.npy', 0o777)
            else:
                os.chmod(DATA_DIR + 'serialized/labels.npy', 0o777)
            os.chmod(DATA_DIR + 'serialized/files.npy', 0o777)
    except:
        pass
    #################################################
    # TODO: create function to save serialized data #
    #################################################

    #TODO: Add TEST part of prepare_dataset() function

    return preprocessed_images, preprocessed_labels, files



def preprocess_images(images, set="train") -> object:
    if(DEBUG):
        print("Preprocessing " + str(len(images)) + " images...")
    for i in range(len(images)):
        images[i] = preprocess(images[i])
        if set == "test" and DATA_AUGMENTATION:
            images[i] = simulate_augment(images[i])
        if i != 0 and i % 100 == 0:
            print("{}% ".format(i*100//len(images)), end='', flush=True)
    print(" ")
    return images

def preprocess_labels(labels):
    return labels


if __name__=="__main__":

    figures, labels, files = load_raw_data(set = "train")
    print("\nFound {} figures! ".format(len(figures)))

    print("Filename of the first figure: ", figures[0].filename)
    print("Assessments made for the first figure: \n{}".format(figures[0].assessments))
    print("Labels for the first figure", labels[0])
    print("File: ", files[0].path)

    print("\nDisplaying one of {} assessments of figure 0 in detail: ".format(len(figures[0].assessments)))

    for assessment in figures[0].assessments.items():
        print(assessment)
        print("Id: ", assessment[1].id)
        print("filename: ", assessment[1].filename)
        print("Items:")

        for item in assessment[1].items.items():
            print("\n\tfilename item: ", item[1].filename)
            print("\tassessment_id item: ", item[1].assessment_id)
            print("\tid item: ", item[1].id)
            print("\tscore item: ", item[1].score)
        break

    print("\nSores for figure[0]: ", figures[0].getScores())

    print("\nAssessments per figure: ")
    for figure in figures:
        print("{} assessments for figure: {}".format(len(figure.assessments), figure.filename))


    preprocessed_images, preprocessed_labels, files = preprocess_dataset(figures, labels, files)
