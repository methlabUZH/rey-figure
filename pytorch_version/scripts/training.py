import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from rocf_scoring.models.model import CNN
from config import RESUME_CHECKPOINT, PATH_CHECKPOINT
import tqdm
import time
import numpy as np

def train(model, criterion, optimizer, train_dataloader, test_dataloader, opt=None):

    # if running on GPU and we want to use cuda move model there
    # use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     net = model.cuda()

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    # if RESUME_CHECKPOINT:
    #     ckpt = load_checkpoint(PATH_CHECKPOINT)  # custom method for loading last checkpoint
    #     net.load_state_dict(ckpt['net'])
    #     start_epoch = ckpt['epoch']
    #     start_n_iter = ckpt['n_iter']
    #     optim.load_state_dict(ckpt['optim'])
    #     print("last checkpoint restored")


    # TODO: read about summary writer
    # typically we use tensorboardX to keep track of experiments
    # writer = SummaryWriter(...)

    # now we start the main loop
    n_iter = start_n_iter

    predictions_train = {}
    losses_train = {}
    groundtruths_train = {}
    predictions_test = {}
    losses_test = {}
    groundtruths_test = {}


    for epoch in range(start_epoch, opt["EPOCHS"]):
        # set models to train mode
        model.train()

        # use prefetch_generator and tqdm for iterating through data
        # pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
        #             total=len(train_data_loader))
        # start_time = time.time()

        losses_epoch = np.asarray([])
        outputs_epoch = np.asarray([])
        groundtruths_epoche = np.asarray([])

        # for loop going through dataset
        for data in train_dataloader:
            # data preparation
            imgs, labels, one_hot_label = data

            # if use_cuda:
            #     img = img.cuda()
            #     label = label.cuda()


            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader

            # forward and backward pass
            optimizer.zero_grad()
            outputs = model(imgs.float())
            loss = criterion(outputs, np.argmax(one_hot_label, axis=1))
            loss.backward()
            optimizer.step()


            # keeping track of losses outputs and groundtruths per epoch
            loss_np = loss.detach().numpy()
            outputs_np = outputs.detach().numpy()
            ground_truths_np = np.argmax(one_hot_label, axis=1)
            losses_epoch = np.append(losses_epoch, loss_np)
            outputs_epoch = np.append(outputs_epoch, outputs_np)
            groundtruths_epoche = np.append(groundtruths_epoche, ground_truths_np)

            # udpate tensorboardX
            # writer.add_scalar(..., n_iter)

            # compute computation time and *compute_efficiency*

            # pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
            #     process_time / (process_time + prepare_time), epoch, opt.epochs))

        print("losses per epoche: ", losses_epoch)

        # keeping track of losses outputs and groundtruths per epoch
        predictions_train[epoch] = outputs_epoch
        losses_train[epoch] = losses_epoch
        groundtruths_train[epoch] = groundtruths_epoche


        # maybe do a test pass every x epochs
        if epoch % 3 == 0:
            print("epoch val: ", epoch)
            with torch.no_grad():
                # bring models to evaluation mode
                model.eval()
                ...
                # do some tests
                # pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader, ...)),
                #             total=len(test_data_loader))

                losses_epoch = np.asarray([])
                outputs_epoch = np.asarray([])
                groundtruths_epoche = np.asarray([])

                for data in test_dataloader:
                    # data preparation
                    imgs, labels, one_hot_label = data
                    outputs = model(imgs.float())
                    loss = criterion(outputs, np.argmax(one_hot_label, axis=1))
                    print("val loss: ", loss)

                    loss_np = loss.detach().numpy()
                    outputs_np = outputs.detach().numpy()
                    ground_truths_np = np.argmax(one_hot_label, axis=1)
                    losses_epoch = np.append(losses_epoch, loss_np)
                    outputs_epoch = np.append(outputs_epoch, outputs_np)
                    groundtruths_epoche = np.append(groundtruths_epoche, ground_truths_np)


            predictions_test[epoch] = outputs_epoch
            losses_test[epoch] = losses_epoch
            groundtruths_test[epoch] = groundtruths_epoche

            # save checkpoint if needed


    print("Finished Training!!!")