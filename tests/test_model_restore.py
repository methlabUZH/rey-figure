import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
import os

# path append stuff because this file is in "wrong" folder to find imports
import sys
sys.path.append('../src')

from model import CNN
from dataloader import DATA
from config import DEBUG, RUN_NAME
from helpers import create_directory

EARLY_STOP_THRESHOLD = 5


def train_cnn(run_name, fold, data_train, data_val, restore_path = None):
    model = CNN(run_name, fold)

    if restore_path:
        model_path = os.path.join('../models', run_name, restore_path)
        model.restore_model(model_path)
        print("restoring from {}".format(model_path))
    else:
        print("no restoring, start training new model")

    n_epochs = 1
    evaluate_every = 30 # number of batches before validation

    if(DEBUG):
        print("\n\nStart training model. Run: {}, Fold: {}".format(run_name,fold))
        print("train")
        print(data_train)
        print("val")
        print(data_val)

    number_of_batches = np.ceil(data_train.images.shape[0] * n_epochs / model.batch_size)

    min_mse = 36**2  # value for zero prediction (max value)
    early_stop_cnt = 0

    for i in range(int(np.ceil(number_of_batches/evaluate_every))):
        model.fit(data_train.images, data_train.labels, trainingsteps = evaluate_every, globalstep=i)

        # detect early stopping
        mse_val = model.validate(data_val.images, data_val.labels, summary_writer = model.validation_writer,
                                 summary_step = ((i+1)*evaluate_every-1), files = data_val.files)
        if mse_val > min_mse:
            early_stop_cnt += 1
        else:
            early_stop_cnt = 0
            min_mse = mse_val
        if early_stop_cnt == EARLY_STOP_THRESHOLD:
            print("Stopping training early after {} steps".format((i + 1) * evaluate_every))
            break

        if(DEBUG):
            print("validation mse after {} steps (stop at {}): ".format((i+1)*evaluate_every, int(number_of_batches)))
            print(mse_val)

    print("Final validation mse:")
    log_results_filename = run_name + "__fold" + str(fold)
    mse_val = model.validate(data_val.images, data_val.labels, summary_writer=model.validation_writer,
                             summary_step=((i + 1) * evaluate_every - 1), files=data_val.files,
                             log_results = True, log_results_filename=log_results_filename)

    model.save_model()

    print(mse_val)
    return mse_val


def cross_validate(k=10):
    val_results = []

    # fixed run name to later restore
    run_name = "restore_test"

    if not os.path.exists(os.path.join('../models', run_name)):
        # nothing to restore
        os.makedirs(os.path.join('../summaries', run_name))
        os.makedirs(os.path.join('../models', run_name))
        restore = None

    else:
        # restore from last training
        restore = "model_fold0.ckpt"

    kf = KFold(n_splits=k)

    for i, (train_idx, val_idx) in enumerate(kf.split(DATA.images)):
        data_train, data_val = DATA.split(train_idx,val_idx)
        res = train_cnn(run_name, i, data_train, data_val, restore_path = restore)
        val_results.append(res)
        # only run once, this is a test, no CV here^^
        break






result = cross_validate()
