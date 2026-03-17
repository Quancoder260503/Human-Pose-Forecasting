import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.LSTM import *
from PennActionTrainingUtil import *
from data.PennAction import PennAction
from NTUTrainingUtil import *
from data.NTU import NTU
import os 

def train_per_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """
    :param model: model to train 
    :param train_loader: train data loader 
    :param optimizer: set default to ADAM 
    :param criterion: loss function
    :param epoch: current epoch
    :param device: 'cuda' 
    """
    model.train()
    losses = []
    total_predictions = get_ntu_predictions_dict()
    print_step = 50
    for batch_idx, (observed_poses, target_poses) in enumerate(train_loader):
        observed_poses = observed_poses.to(device, dtype = torch.float32)
        target_poses   = target_poses.to(device, dtype = torch.float32)

        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]

        predicted_poses = model(observed_poses, prediction_len)
        loss = criterion(predicted_poses, target_poses, prediction_len) 
        losses.append(loss.item())

        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted_poses = predicted_poses.contiguous().view(batch, prediction_len, -1, 3) 
        target_poses    = target_poses.contiguous().view(batch, prediction_len, -1, 3) 

        predictions = compute_joints_metric(
            pred = predicted_poses[:, :, :, :2],  
            gt   = target_poses[:, :, :, :2],  
            time_step = prediction_len, 
        )

        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

        if batch_idx % print_step == 0:
            acc = total_predictions['total']['correct'] / total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.6f} | PCK accuracy = {acc:.4f}')

    return np.mean(losses), total_predictions


def evaluate_per_epoch(model, val_loader, criterion, device):
    """
    :param model: model 
    :param val_loader: validation dataloader
    :param criterion: loss function
    :param device: cuda 
    :return:
    """
    model.eval()
    losses = []
    total_predictions = get_ntu_predictions_dict()
    with torch.no_grad():
        for _, (observed_poses, target_poses) in enumerate(val_loader):
           observed_poses = observed_poses.to(device, dtype = torch.float32)
           target_poses   = target_poses.to(device,   dtype = torch.float32)
           batch          = target_poses.shape[0]
           prediction_len = target_poses.shape[1]
           predicted_poses = model(observed_poses, prediction_len)
           loss = criterion(predicted_poses, target_poses, prediction_len) 
           losses.append(loss.item())

           predicted_poses = predicted_poses.contiguous().view(batch, prediction_len, -1, 3) 
           target_poses    = target_poses.contiguous().view(batch, prediction_len, -1, 3)  

           predictions = compute_joints_metric(
              pred = predicted_poses[:, :, :, :2],  
              gt   = target_poses[:, :, :, :2],  
              time_step = prediction_len, 
           )
           for dict in predictions:
               for key in predictions[dict]:
                   total_predictions[dict][key] += predictions[dict][key]

    return np.mean(losses), total_predictions

def train(
          num_training_frame, 
          num_testing_frame, 
          batch_size,
          learning_rate,
          betas,
          scheduler_step, 
          scheduler_gamma, 
          num_epochs,
          title,
          num_joints, 
    ):
    """
    :param num_training_frame   : number of training frames 
    :param num_testing_frame    : number of testing frames
    :param batch_size           : sample per batch 
    :param learning_rate        : 5e-4 
    :param betas                : (0.9, 0.99) (AdamW betas) 
    :param scheduler_step       : lr scheduler step 
    :param scheduler_gamma      : lr scheduler gamma 
    :param num_epochs           : training epoch
    """
    # train_dataset = PennAction(
    #     num_training_frame = num_training_frame, 
    #     num_testing_frame  = num_testing_frame,
    #     train = True, 
    #     coordinates = True, 
    #     on_debug_mode = False,  
    # )
    train_dataset = NTU(
        num_training_frame = num_training_frame,
        num_testing_frame  = num_testing_frame,  
        train = True, 
        coordinates = True, 
    )
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    # val_dataset = PennAction(
    #     num_training_frame = num_training_frame, 
    #     num_testing_frame  = num_testing_frame,
    #     train = False,
    #     coordinates = True, 
    #     on_debug_mode = False,  
    # )

    val_dataset = NTU(
        num_training_frame = num_training_frame,
        num_testing_frame  = num_testing_frame,  
        train = False, 
        coordinates = True, 
    )
    val_loader   = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    print('-' * 40)
    print('Number of train batches =', len(train_loader))
    print('Number of validation batches =', len(val_loader))

    print('-' * 40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(device, 'is available')

    checkpoint = 'checkpoint/baseline/joint_lstm_ntu.pth.tar'
    
    start_epoch = 0 
    if os.path.exists(checkpoint): 
        model_dict = torch.load(checkpoint, weights_only = False) 
        start_epoch = model_dict['epoch'] + 1 
        model       = get_lstm_model(device)
        model.load_state_dict(model_dict['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)
   
    else : 
        model = get_lstm_model(device, pose_dim = num_joints) 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)
    criterion = LSTM_loss_function

    best_accuracy = 0
    loss_list, metric_list = [], []
    print("Start Training ...\n")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, train_predictions = train_per_epoch(model, train_loader, optimizer, criterion, epoch, device)
        val_loss, val_predictions     = evaluate_per_epoch(model, val_loader, criterion, device)
        
        scheduler.step(train_loss)

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc, curr_val_total_acc = train_acc['total'], val_acc['total']

        print(f'Training | Loss : {train_loss:.6f} | PCK accuracy : {curr_train_total_acc:.4f}')
        print(f'Validation | Loss : {val_loss:.6f} | PCK accuracy : {curr_val_total_acc:.4f}')

        if val_acc['total'] > best_accuracy : 
            best_accuracy = val_acc['total']
            save_checkpoint(epoch, model, optimizer, model_name = 'joint_lstm_ntu' , prefix_dir_name = 'checkpoint/baseline/')

    #plot(np.array(loss_list), np.array(metric_list), title)
    return model, loss_list, metric_list

if __name__ == '__main__':
    model, loss_list, metric_list = train(
        num_training_frame = 7, 
        num_testing_frame  = 7,  
        batch_size = 16,
        learning_rate = 8e-5,
        betas = (0.9, 0.999),
        scheduler_step = 40000,
        scheduler_gamma = 0.3333,
        num_epochs = 100,
        num_joints = 25, 
        title = 'Key Point LSTM Training on NTU',
    )

