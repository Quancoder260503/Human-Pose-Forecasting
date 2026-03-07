import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ConvLSTM import ConvLSTMPM
from ConvLSTMTrainingUtil import *
from data.PennAction import PennAction

"""
TODO: Fixing the joints, calculate the MSE for baseline, also PCK, 
      If available training the pose by CNN to classified the actions, which can be used as a metric to evaluate, 
      Read about LR_scheduler. 
"""

def get_model(time_step, device):
    """
    :param time_step:
    :param device:
    :return:
    """
    model = ConvLSTMPM(time_step = time_step).to(device)
    model = model.to(device)
    return model

def loss_function(pred_maps, gt_maps, time_step):
    """
    :param pred_maps:
    :param gt_maps:
    :param time_step:
    :return:
    """
    criterion = nn.MSELoss(reduction = 'mean')
    initial_pred = pred_maps[:, 0, :, :, :]
    initial_gt   = gt_maps[:, 0, :, :, :]
    initial_loss = criterion(initial_pred, initial_gt)
    total_loss = initial_loss
    for t in range(time_step):
        pred = pred_maps[:, t + 1]
        gt   = gt_maps[:, t, ]
        curr_loss = criterion(pred, gt)
        total_loss = total_loss + curr_loss
    return total_loss

def train_per_epoch(model, train_loader, optimizer, criterion, epoch, time_step, device):
    """
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param epoch:
    :param time_step:
    :param device:
    :return:
    """

    model.train()
    losses = []
    total_predictions = get_predictions_dict()
    print_step = len(train_loader) // 5
    for batch_idx, (images, _, gt_maps, center_map) in enumerate(train_loader):
        images = images.to(device)
        gt_maps = gt_maps.to(device)
        center_map = center_map.to(device)

        pred_heatmaps = model(images, center_map)

        loss = criterion(pred_heatmaps, gt_maps, time_step)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions = compute_metric(
            pred_maps = pred_heatmaps,
            gt_maps   = gt_maps.cpu().numpy(),
            threshold = gt_maps.shape[-1],
            time_step = time_step
        )

        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

        if iteration % print_step == 0:
            acc = total_predictions['total']['correct'] / total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.2f} | PCK accuracy = {acc:.2f}')

    return np.mean(losses), total_predictions

def evaluate_per_epoch(model, val_loader, criterion, time_step, device):
    """
    :param model:
    :param val_loader:
    :param criterion:
    :param time_step:
    :param device:
    :return:
    """
    model.eval()
    losses = []
    total_predictions = get_predictions_dict()
    with torch.no_grad():
     for batch_idx, (images, _, gt_maps, center_map) in enumerate(val_loader):
        images = images.to(device)
        gt_maps = gt_maps.to(device)
        center_map = center_map.to(device)

        pred_heatmaps = model(images, center_map)

        loss = criterion(pred_heatmaps, gt_maps, time_step)
        losses.append(loss.item())

        predictions = compute_metric(
            pred_maps = pred_heatmaps,
            gt_maps   = gt_maps.cpu().numpy(),
            threshold = gt_maps.shape[-1],
            time_step = time_step
        )
        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

    return np.mean(losses), total_predictions

def train(time_step,
          batch_size,
          learning_rate,
          betas,
          scheduler_step,
          scheduler_gamma,
          num_epochs,
          title,
    ):
    train_dataset = PennAction(sequence_train = time_step, transform = PennAction.get_default_transform(), train = True)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset  = PennAction(sequence_train = time_step, transform = PennAction.get_default_transform(), train = False)
    val_loader   = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    print('-' * 40)
    print('Number of train batches =', len(train_loader))
    print('Number of validation batches =', len(val_loader))

    print('-' * 40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(device, 'is available')
    model = get_model(time_step, device)
    if device == 'cuda':
        model = model.cuda(0)
        model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step, gamma = scheduler_gamma)
    criterion = loss_function

    best_accuracy = 0
    loss_list, metric_list = [], []
    print("Start Training ...\n")

    for epoch in range(num_epochs):
        train_loss, train_predictions = train_per_epoch(model, train_loader, optimizer, criterion, epoch, time_step, device)
        val_loss, val_predictions     = evaluate_per_epoch(model, val_loader, criterion, time_step, device)
        scheduler.step()

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc, curr_val_total_acc = train_acc['total'], val_acc['total']

        print(f'Training | Loss : {train_loss:.4f} | PCK accuracy : {curr_train_total_acc:.4f}')
        print(f'Validation | Loss : {val_loss:.4f} | PCK accuracy : {curr_val_total_acc:.4f}')

    plot(np.array(loss_list), np.array(metric_list), title)
    return model, loss_list, metric_list

if __name__ == '__main__':
    model, loss_list, metric_list = train(
        time_step = 7,
        batch_size = 4,
        learning_rate = 1-4,
        betas = (0.9, 0.999),
        scheduler_step = 40000,
        scheduler_gamma = 0.3333,
        num_epochs = 50,
        title = 'ConvLSTM Training',
    )

