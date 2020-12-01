# currently only runs locally due to reading in from command line (interferes with config)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import os
import sys
from helpers import create_directory

from datetime import datetime

from dataloader import DATA
from config import RUN_NAME, LABEL_FORMAT


class Models:

    def __init__(self, data_train, data_val):
        self.train_images = np.reshape(data_train.images, (data_train.images.shape[0], -1))
        self.val_images = np.reshape(data_val.images, (data_val.images.shape[0], -1))
        self.train_labels = data_train.labels
        self.train_intermediates = data_train.intermediates
        self.val_intermediates = data_val.intermediates

    def knn(self, n_neighbors=20):
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(self.train_images, self.train_labels)
        return neigh.predict(self.val_images)

    def linear_regression(self):
        regr = LinearRegression()
        regr.fit(self.train_images, self.train_labels)
        return regr.predict(self.val_images)

    def random_forest(self):
        rforest = RandomForestRegressor(n_estimators=50)
        rforest.fit(self.train_images, self.train_labels)
        return rforest.predict(self.val_images)

    def adaboost_regressor(self):
        regr = AdaBoostRegressor(n_estimators=150)
        regr.fit(self.train_images, self.train_labels)
        return regr.predict(self.val_images)

    def gaussian_process(self):
        regr = GaussianProcessRegressor()
        if LABEL_FORMAT == 'one':
            regr.fit(self.train_intermediates, self.train_labels)
        elif LABEL_FORMAT == 'one-per-item':
            regr.fit(self.train_intermediates, self.train_labels[:, 18])
        return regr.predict(self.val_intermediates)


def cross_validate(k=10):
    val_results = []

    # name of this run, for tensorboard summaries etc
    run_name = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if len(RUN_NAME) > 0:
        run_name = run_name + "_" + RUN_NAME

    # create directories for tensorboard visualizations
    create_directory(os.path.join('../summaries', run_name))

    kf = KFold(n_splits=k)

    for i, (train_idx, val_idx) in enumerate(kf.split(DATA.images)):
        start_time = time.perf_counter()
        print("Currently in fold {}...".format(i+1))
        data_train, data_val = DATA.split(train_idx, val_idx)
        inst = Models(data_train, data_val)
        predictions = inst.gaussian_process()
        if LABEL_FORMAT == 'one':
            predictions = np.clip(predictions, 0, 36)

        if LABEL_FORMAT == 'one-per-item':
            res = mean_squared_error(predictions, data_val.labels[:, 18])
        elif LABEL_FORMAT == 'one':
            res = mean_squared_error(predictions, data_val.labels)

        val_results.append(res)
        end_time = time.perf_counter()
        delta = end_time - start_time
        print("Fold {} took {}s.".format(i+1, delta))
        print("MSE for fold was: {}...".format(res))
        print("Estimated remaining time = {}min".format(delta * (k - i - 1) / 60))


    print("\n\n\nCROSS VALIDATION RESULT:")
    print(val_results)
    print("mean: {}, standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    # write result to file, including current git commit information
    result_file_path = os.path.join('../summaries', run_name, "result.txt")
    with open(result_file_path, "a") as result_file:
        result_file.write("Model: adaboost regressor with 150 estimators")
        result_file.write("\nRun: {}, Number of folds: {}".format(run_name, k))
        result_file.write("\nValidation Results\n")
        result_file.write(str(val_results))
        result_file.write("\nMean: {}, Standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    return val_results


print("Running gaussian process regressor...")
result = cross_validate()
