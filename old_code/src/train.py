import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
import os
import time
import csv

from model import CNN
from dataloader import DATA, DATA_TEST
from config import DEBUG, RUN_NAME, SKIP_FOLDS, RUN_PARAMETERS, TEST, DATA_AUGMENTATION, LABEL_FORMAT, BINNING, CONV_LAYERS
from helpers import create_directory

EARLY_STOP_THRESHOLD = 10



def train_cnn(run_name, fold, data_train, data_val):
    model = CNN(run_name, fold)

    if TEST:
        n_epochs = 60
    else:
        n_epochs = 150

    evaluate_every = 30 # number of batches before validation

    if(DEBUG):
        if TEST:
            print("\n\nStart training model on all training data_preprocessing. Run: {}".format(run_name, fold))
        else:
            print("\n\nStart training model. Run: {}, Fold: {}".format(run_name, fold))
            print("train")
            print(data_train)
            print("val")
            print(data_val)

    number_of_batches = np.ceil(data_train.images.shape[0] * n_epochs / model.batch_size)

    min_mse = 36**2  # value for zero prediction (max value)
    early_stop_cnt = 0

    for i in range(int(np.ceil(number_of_batches/evaluate_every))):
        start_time = time.perf_counter()
        model.fit(data_train.images, data_train.labels, trainingsteps = evaluate_every, globalstep=i)
        end_time = time.perf_counter()
        print("fitting {} batches of {} images took {}s".format(evaluate_every, model.batch_size, end_time - start_time))

        mse_val = model.validate(data_val.images, data_val.labels, summary_writer = model.validation_writer,
                                     summary_step = ((i+1)*evaluate_every-1), files = data_val.files)
        if not TEST:
            # detect early stopping
            if mse_val > min_mse:
                early_stop_cnt += 1
            else:
                early_stop_cnt = 0
                min_mse = mse_val
            if early_stop_cnt == EARLY_STOP_THRESHOLD:
                print("Stopping training early after {} steps".format((i + 1) * evaluate_every))
                break

        if(DEBUG):
            if TEST:
                print("test mse after {} of {} steps: ".format((i + 1) * evaluate_every, int(number_of_batches)))
                print(mse_val)
            else:
                print("validation mse after {} steps (stop at {}): ".format((i+1)*evaluate_every, int(number_of_batches)))
                print(mse_val)

    if TEST:
        print("Final test mse:")
        log_results_dir = os.path.join('../summaries', run_name, "test_results")
        create_directory(log_results_dir)
        log_results_filename = os.path.join(log_results_dir, "test_results.csv")
        mse_val = model.validate(data_val.images, data_val.labels, summary_writer=model.validation_writer,
                                 summary_step=((i + 1) * evaluate_every - 1), files=data_val.files,
                                 log_results=True, log_results_filename=log_results_filename)
    else:
        print("Final validation mse:")
        log_results_dir = os.path.join('../summaries', run_name, "validation_results")
        create_directory(log_results_dir)
        log_results_filename = os.path.join(log_results_dir, "validation_results__fold" + str(fold) + ".csv")
        mse_val = model.validate(data_val.images, data_val.labels, summary_writer=model.validation_writer,
                                 summary_step=((i + 1) * evaluate_every - 1), files=data_val.files,
                                 log_results = True, log_results_filename=log_results_filename)

    model.save_model()

    print(mse_val)

    return mse_val


def cross_validate(data, k=10, skip=[]):
    val_results = []

    # name of this run, for tensorboard summaries etc
    run_name = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if len(RUN_NAME) > 0:
        run_name = run_name + "_" + RUN_NAME

    # create directories for tensorboard visualizations and saved model
    create_directory(os.path.join('../summaries', run_name))
    create_directory(os.path.join('../models', run_name))

    kf = KFold(n_splits=k)
    for i, (train_idx, val_idx) in enumerate(kf.split(data.images)):
        if i in skip:
            print("skipping fold {}".format(i))
            continue
        data_train, data_val = data.split(train_idx,val_idx)
        res = train_cnn(run_name, i, data_train, data_val)
        val_results.append(res)

    print("\n\n\nCROSS VALIDATION RESULT:")
    print(val_results)
    print("mean: {}, standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    # write to csv results overview file
    if not os.path.exists('results.csv'):
        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            fields = ["Time", "Run name", "data_preprocessing augmentation", "label format", "binning", "conv layers", 'mean', 'val results']
            writer.writerow(fields)
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        fields = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), RUN_NAME, DATA_AUGMENTATION, LABEL_FORMAT, BINNING, CONV_LAYERS, str(np.mean(val_results)), str(val_results)]
        writer.writerow(fields)

    # write result to file, including current git commit information and config file
    result_file_path = os.path.join('../summaries', run_name, "result.txt")
    os.system('git log -1 --pretty=format:"Latest commit on machine: %h - %B" > {}'.format(result_file_path))
    os.system('git log -1 origin/master --pretty=format:"Latest commit on remote: %h - %B" >> {}'.format(result_file_path))
    with open(result_file_path, "a") as result_file:
        result_file.write("\nRun parameters:\n {}\n".format(RUN_PARAMETERS))
        result_file.write("\nRun: {}, Number of folds: {}".format(run_name,k))
        result_file.write("\nValidation Results\n")
        result_file.write(str(val_results))
        result_file.write("\nMean: {}, Standard dev: {}".format(np.mean(val_results), np.std(val_results)))

    return val_results, run_name


def test():
    # name of this run, for tensorboard summaries etc
    run_name = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if len(RUN_NAME) > 0:
        run_name = run_name + "_ALL_" + RUN_NAME

    # create directories for tensorboard visualizations and saved model
    create_directory(os.path.join('../summaries', run_name))
    create_directory(os.path.join('../models', run_name))

    results = []
    for i in range(10):
        res = train_cnn(run_name, i, DATA, DATA_TEST)
        results.append(res)


    print("\n\n\nTEST RESULTS:")
    print(results)

    # write result to file, including current git commit information and config file
    result_file_path = os.path.join('../summaries', run_name, "result.txt")
    os.system('git log -1 --pretty=format:"Latest commit on machine: %h - %B" > {}'.format(result_file_path))
    os.system(
        'git log -1 origin/master --pretty=format:"Latest commit on remote: %h - %B" >> {}'.format(result_file_path))
    with open(result_file_path, "a") as result_file:
        result_file.write("\nRun parameters:\n {}\n".format(RUN_PARAMETERS))
        result_file.write("\nRun: {}, train on all training data_preprocessing, test on test".format(run_name))
        result_file.write("\nTest Results\n")
        result_file.write(str(results))
        result_file.write("\nMean: {}, Standard dev: {}".format(np.mean(results), np.std(results)))


if __name__ == "__main__":
    if TEST:
        result = test()
    else:
        result, _ = cross_validate(DATA, skip=SKIP_FOLDS)
