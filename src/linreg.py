import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from model import CNN
from dataloader import DATA
from config import DEBUG, RUN_NAME
from datetime import datetime
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from helpers import create_directory

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
        data_train, data_val = DATA.split(train_idx,val_idx)
        regr = linear_model.LinearRegression()
        train_images = np.reshape(data_train.images, (data_train.images.shape[0], -1))
        val_images = np.reshape(data_val.images, (data_val.images.shape[0], -1))
        regr.fit(train_images, data_train.labels)
        labels_val_pred = regr.predict(val_images)
        res=mean_squared_error(labels_val_pred, data_val.labels)
        val_results.append(res)


    print("\n\n\nCROSS VALIDATION RESULT:")
    print(val_results)
    print("mean: {}, standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    # write result to file, including current git commit information
    result_file_path = os.path.join('../summaries', run_name, "result.txt")
    os.system('git log -1 --pretty=format:"Latest commit on machine: %h - %B" > {}'.format(result_file_path))
    os.system('git log -1 origin/master --pretty=format:"Latest commit on remote: %h - %B" >> {}'.format(result_file_path))
    with open(result_file_path, "a") as result_file:
        result_file.write("\nRun: {}, Number of folds: {}".format(run_name,k))
        result_file.write("\nValidation Results\n")
        result_file.write(str(val_results))
        result_file.write("\nMean: {}, Standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    return val_results


result = cross_validate()