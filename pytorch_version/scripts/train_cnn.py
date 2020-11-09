# TODO: Define model, dataloaders,criterion, hyperparams
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from rocf_scoring.models.model import CNN, weights_init
from sklearn.model_selection import train_test_split
from rocf_scoring.features.dataset import ROCFDataset
from rocf_scoring.data_preprocessing.loading_data import load_raw_data
from scripts.training import train
from config import DEBUG, RUN_NAME, TRAINING_RESULTS_DIR, TRAINED_MODELS
from datetime import datetime
import os

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
    save_model_dir_checkpoints = os.path.join(TRAINED_MODELS, run_name, "checkpoints")
    if not os.path.exists(save_model_dir_checkpoints):
        os.makedirs(save_model_dir_checkpoints)

    return run_name, train_results_dir, save_model_dir_final, save_model_dir_checkpoints

def train_CNN():
    # setup dirctories to safe trained model and training results
    run_name, training_results_dir, save_model_dir_final, save_model_checkpoints = directory_setup()

    train_opt = {
        "LR": 0.001,
        "EPOCHS": 12,
        "CHECKPOINTS_DIR" : save_model_checkpoints,
    }
    torch.set_num_threads(1)


    # model initialization
    # initialize model weights of conv layers with xavier method
    model = CNN()
    model.apply(weights_init)

    # get train and test data loader
    train_data_loader, test_data_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_opt["LR"])

    train(model, criterion, optimizer, train_data_loader, test_data_loader, train_opt)

    # save final model
    final_model_path = os.path.join(save_model_dir_final, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    train_CNN()

    
