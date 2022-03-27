import torch
from torch.utils.data import DataLoader
from rocf_scoring.config import (RESUME_CHECKPOINT, PATH_CHECKPOINT, SAVE_MODEL_CHECKPOINTS, RUN_NAME, TRAINING_RESULTS_DIR,
                                 TRAINED_MODELS)
from rocf_scoring.data_preprocessing.loading_data import load_raw_data
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
from rocf_scoring.features.dataset import ROCFDataset
from datetime import datetime
import csv
from sklearn.metrics import accuracy_score

def monitor_training_results(results_dir, monitor_dict, mode=None):

    if mode == "train":
        path = os.path.join(results_dir, "train_results.csv")

    if mode == "test":
        path = os.path.join(results_dir, "test_results.csv")

    elif mode == None:
        assert "Did not specify mode"

    list_of_elem = (item for key,item in monitor_dict.items())

    with open(path, 'a', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(list_of_elem)



def get_dataloaders():

    figures, labels, files = load_raw_data()
    figures_train, figures_test, labels_train, labels_test, files_train, files_test = train_test_split(figures,
                                                                                                       labels,
                                                                                                       files,
                                                                                                       test_size=0.2)


    train_dataset = ROCFDataset(figures_train, labels_train, files_train)
    test_dataset = ROCFDataset(figures_test, labels_test, files_test)

    train_loader = DataLoader(train_dataset,
                                   batch_size=16, shuffle=True,
                                   num_workers=1)
    test_loader = DataLoader(test_dataset,
                                  batch_size=16, shuffle=False,
                                  num_workers=1)

    return train_loader, test_loader


def directory_setup():
    """
    Sets up directories used to save models, checkpoints and training results
    :return:
    """

    # run name for model saving
    run_name = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if len(RUN_NAME) > 0:
        run_name = run_name + "_" + RUN_NAME

    # create directory for training results
    train_results_dir = os.path.join(TRAINING_RESULTS_DIR, run_name)
    if not os.path.exists(train_results_dir):
        os.makedirs(train_results_dir)

    # create directory to save trained model
    save_model_dir_final = os.path.join(TRAINED_MODELS, run_name, "final")
    if not os.path.exists(save_model_dir_final):
        os.makedirs(save_model_dir_final)

    # create directory to save model checkpoints
    if SAVE_MODEL_CHECKPOINTS:
        save_model_dir_checkpoints = os.path.join(TRAINED_MODELS, run_name, "checkpoints")
        if not os.path.exists(save_model_dir_checkpoints):
            os.makedirs(save_model_dir_checkpoints)
    else:
        save_model_dir_checkpoints = None

    return run_name, train_results_dir, save_model_dir_final, save_model_dir_checkpoints


def initialize_training_csv(results_dir):
    """
    NOT used at the moment!!!
    :param results_dir:
    :return:
    """
    columns = "epoch,n_iter,loss,pred,groundtruth"
    #columns = "epoch,n_iter,loss"

    train_results_path = os.path.join(results_dir, "train_results.csv")
    test_results_path = os.path.join(results_dir, "test_results.csv")

    with open(train_results_path, 'w', newline='\n') as fd:
        fd.write(columns)

    with open(test_results_path, 'w', newline='\n') as fd:
        fd.write(columns)


def train(model, criterion, optimizer, train_dataloader, test_dataloader, opt=None):

    # if running on GPU and we want to use cuda move model there
    # use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     net = model.cuda()

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0

    if RESUME_CHECKPOINT:
        ckpt = torch.load(PATH_CHECKPOINT)  # custom method for loading last checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("last checkpoint restored")
        print("start epoch: ", start_epoch)


    # TODO: read about summary writer
    # typically we use tensorboardX to keep track of experiments
    # writer = SummaryWriter(...)

    n_iter = start_n_iter
    for epoch in range(start_epoch, opt["EPOCHS"]):
        # set models to train mode
        model.train_epoch()

        # for loop going through dataset
        for data in train_dataloader:
            # preprocessing preparation
            imgs, labels, one_hot_label = data

            # if use_cuda:
            #     img = img.cuda()
            #     label = label.cuda()

            # forward and backward pass
            optimizer.zero_grad()
            outputs = model(imgs.float())

            loss = criterion(outputs, np.argmax(one_hot_label, axis=1))
            loss.backward()
            optimizer.step()

            # keeping track of losses outputs and groundtruths per epoch
            loss_np = loss.detach().numpy()
            outputs_np = np.argmax(outputs.detach().numpy(), axis=1)
            ground_truths_np = np.argmax(one_hot_label, axis=1)

            acc = accuracy_score(ground_truths_np, outputs_np)

            monitor = {
                "epoch" : epoch,
                "n_iter" : n_iter,
                "loss" : loss_np,
                "acc" : acc,
            }
            monitor_training_results(opt["TRAINING_RESULTS_DIR"], monitor, mode="train")

            n_iter += 1

        # save model checkpoint every 5 epochs
        if SAVE_MODEL_CHECKPOINTS:
            if epoch % 5 == 0:
                path = os.path.join(opt["CHECKPOINTS_DIR"], f"epoch_{epoch}")
                if not os.path.exists(path):
                    os.makedirs(path)
                path = os.path.join(path, "checkpoint.tar")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, path)


        # do a test pass every x epochs
        n_iter_test = max ( (epoch - 1) * len(train_dataloader), 0)

        if epoch % 3 == 0:
            print("epoch val: ", epoch)
            with torch.no_grad():
                # bring models to evaluation mode
                model.eval_model()

                for data in test_dataloader:
                    # preprocessing preparation
                    imgs, labels, one_hot_label = data
                    outputs = model(imgs.float())
                    loss = criterion(outputs, np.argmax(one_hot_label, axis=1))

                    loss_np = loss.detach().numpy()
                    outputs_np = np.argmax(outputs.detach().numpy(), axis=1)
                    ground_truths_np = np.argmax(one_hot_label, axis=1)

                    acc = accuracy_score(ground_truths_np, outputs_np)

                    monitor = {
                        "epoch": epoch,
                        "n_iter": n_iter_test,
                        "loss": loss_np,
                        "acc": acc,
                    }

                    monitor_training_results(opt["TRAINING_RESULTS_DIR"], monitor, mode="test")

                    n_iter_test += 1

    print("Finished Training!!!")