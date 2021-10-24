import torch
import torch.nn as nn
from torch.optim import Adam
from rocf_scoring.models.model import CNN, weights_init
from scripts.training import train, get_dataloaders, directory_setup, initialize_training_csv
import os
from rocf_scoring.helpers.helpers import visualize_training


def train_CNN():
    # setup dirctories to safe trained model and training results
    run_name, training_results_dir, save_model_dir_final, save_model_checkpoints = directory_setup()

    # set up csv files to display training results
    # initialize_training_csv(training_results_dir)

    train_opt = {
        "LR": 0.001,
        "EPOCHS": 10,
        "CHECKPOINTS_DIR" : save_model_checkpoints,
        "TRAINING_RESULTS_DIR" : training_results_dir,
    }
    torch.set_num_threads(1)


    # model initialization
    # initialize model weights of conv layers with xavier method
    model = CNN()
    model.apply(weights_init)

    # get train and test data_preprocessing loader
    train_data_loader, test_data_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_opt["LR"])

    train(model, criterion, optimizer, train_data_loader, test_data_loader, train_opt)

    # save final model
    final_model_path = os.path.join(save_model_dir_final, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # visualizing training results
    visualize_training(training_results_dir)

if __name__ == "__main__":
    train_CNN()


