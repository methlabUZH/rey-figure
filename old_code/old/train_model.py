# import time
#
# import torch
# import torch.nn.functional as F  # noqa
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim import Optimizer
# from torch.utils.tensorboard import SummaryWriter
#
#
# def train_epoch(model: torch.nn.Module, criterion: torch.nn.MSELoss, optimizer: Optimizer, scheduler: ExponentialLR,
#                 use_cuda: bool, dataloader: DataLoader, writer: SummaryWriter):
#     total_items_loss, total_scores_loss = .0, .0
#     n_batches, n_images = 0, 0
#
#     # switch to train mode
#     model.train()
#
#     start_time = time.time()
#
#     for i, data in enumerate(dataloader):
#         # load data
#         imgs, labels = data
#         imgs = imgs.cuda() if use_cuda else imgs
#         labels = labels.cuda() if use_cuda else labels
#
#         # compute output
#         outputs = model(imgs.float())
#         loss_items = criterion(outputs, labels)
#         loss_scores = criterion(outputs[:, -1], labels[:, -1])
#
#         # set grads to zero; this is fater than optimizer.zero_grad()
#         for param in model.parameters():
#             param.grad = None
#
#         # train step
#         loss_items.backward()
#         optimizer.step()
#         scheduler.step()
#
#         n_images_in_batch = len(imgs)
#         n_images += n_images_in_batch
#         n_batches += 1
#
#         if i % 15 == 0:
#             height, width = imgs.size()[-2:]
#             imgs = imgs.view(imgs.size(0), -1)
#             imgs -= imgs.min(1, keepdim=True)[0]
#             imgs /= imgs.max(1, keepdim=True)[0]
#             imgs = imgs.view(imgs.size(0), 1, height, width)
#             writer.add_images('training-images', imgs)
#
#         # collect total loss
#         total_items_loss += loss_items.cpu().item() * n_images_in_batch
#         total_scores_loss += loss_scores.cpu().item() * n_images_in_batch
#
#     total_items_loss = total_items_loss / n_images
#     total_scores_loss = total_scores_loss / n_images
#
#     epoch_time = time.time() - start_time
#
#     return total_items_loss, total_scores_loss, epoch_time
#
#
# def eval_model(model: torch.nn.Module, criterion: torch.nn.MSELoss, use_cuda: bool, dataloader: DataLoader):
#     total_items_loss, total_scores_loss = .0, .0
#     n_batches = 0
#     n_images = 0
#
#     # switch to eval mode
#     model.eval()
#
#     for i, data in enumerate(dataloader):
#         # load data
#         imgs, labels = data
#         imgs = imgs.cuda() if use_cuda else imgs
#         labels = labels.cuda() if use_cuda else labels
#
#         # compute output, mse on all items + final score and only on the final score
#         outputs = model(imgs.float())
#
#         loss_items = criterion(outputs, labels)
#         loss_scores = criterion(outputs[:, -1], labels[:, -1])
#
#         n_batches += 1
#         n_images_in_batch = len(imgs)
#         n_images += n_images_in_batch
#
#         # collect total loss
#         total_items_loss += loss_items.cpu().item() * n_images_in_batch
#         total_scores_loss += loss_scores.cpu().item() * n_images_in_batch
#
#     total_items_loss /= n_images
#     total_scores_loss /= n_images
#
#     return total_items_loss, total_scores_loss
#
#
# def train_model(model, optimizer: Optimizer, train_data_loader: DataLoader, val_dataloader: DataLoader,
#                 config: ParsedConfig, results_dir, val_every=1):
#     use_cuda = torch.cuda.is_available()
#     if use_cuda:
#         print('cuda enabled')
#
#     if use_cuda:
#         torch.cuda.set_device(0)
#         model.cuda()
#         criterion = model.criterion.cuda()
#     else:
#         criterion = model.criterion
#
#     start_epoch = 1
#
#     # writer for summaries
#     writer = SummaryWriter(results_dir)
#     # writer.add_graph(model)
#
#     # exponentially decay learning rate
#     total_steps = (config.num_epochs + 1 - start_epoch) * len(train_data_loader)
#     gamma = (config.final_learning_rate / config.initial_learning_rate) ** (1.0 / total_steps)
#
#     scheduler = ExponentialLR(optimizer, gamma=gamma)
#     # stats = TrainStats(results_dir=results_dir)
#
#     print(f'[{timestamp_human()}] start training')
#
#     for epoch in range(start_epoch, config.num_epochs + 1):
#         # train for one epoch
#         train_items_loss, train_scores_loss, epoch_time = train_epoch(model, criterion,
#                                                                       optimizer=optimizer,
#                                                                       scheduler=scheduler,
#                                                                       use_cuda=use_cuda,
#                                                                       dataloader=train_data_loader,
#                                                                       writer=writer)
#
#         # compute validation loss every val_every epochs
#         if epoch % val_every == 0:
#             val_items_loss, val_scores_loss = eval_model(model, criterion, use_cuda, val_dataloader)
#
#             # add tensorboard summaries
#             writer.add_scalar('validation loss items + score', val_items_loss, global_step=epoch)
#             writer.add_scalar('validation loss total score', val_scores_loss, global_step=epoch)
#             writer.add_scalar('training loss items + score', train_items_loss, global_step=epoch)
#             writer.add_scalar('training loss total score', train_scores_loss, global_step=epoch)
#
#             images, labels = next(iter(val_dataloader))
#             writer.add_figure('predictions', plot_scores_preds(model, images, labels, use_cuda), global_step=epoch)
#             writer.flush()
#
#             print_stats(epoch, epoch_time, train_items_loss, train_scores_loss, val_items_loss, val_scores_loss,
#                         scheduler.get_last_lr()[0])
#
#     writer.flush()
#     writer.close()
#
#     print(f'[{timestamp_human()}] end training')
#
#
# def print_stats(epoch, epoch_time, train_loss_items, train_loss_score, val_loss_items, val_loss_score, learning_rate):
#     print_str = f'[{timestamp_human()}]\tepoch {epoch} | epoch time: {epoch_time:.2f}'
#     print_str += f'| train loss items: {train_loss_items:.4f} | train loss score: {train_loss_score:.4f}'
#     print_str += f'| val loss items: {val_loss_items:.4f} | val loss score: {val_loss_score:.4f}'
#     print_str += f'| learning rate: {learning_rate:.6f}'
#     print(print_str)
