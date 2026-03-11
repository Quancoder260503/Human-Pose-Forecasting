import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from JointLSTM import JointLSTM, LSTM_loss_function
#from PennActionTrainingUtil import *
#from data.PennAction import PennAction
from NTUTrainingUtil import *
from data.NTU import NTU
import os 


def get_model(device):
    model = JointLSTM(pose_dim = 25 * 2) # Joints for NTU
    model = model.to(device)
    return model

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
    print_step = 20
    for batch_idx, (observed_poses, target_poses) in enumerate(train_loader):
        observed_poses = observed_poses.to(device, dtype = torch.float32)
        target_poses   = target_poses.to(device, dtype = torch.float32)

        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]

        predicted_poses= model(observed_poses, prediction_len)
        loss = criterion(predicted_poses, target_poses, prediction_len) 

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions = compute_joints_metric(
            pred = predicted_poses.contiguous().view(batch, prediction_len, -1, 2),  
            gt   = target_poses.contiguous().view(batch, prediction_len, -1, 2),  
            time_step = prediction_len, 
        )

        for dict in predictions:
            for key in predictions[dict]:
                total_predictions[dict][key] += predictions[dict][key]

        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = total_predictions['total']['correct'] / total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.6f} | PCK accuracy = {acc:.4f}')

    return np.mean(losses), total_predictions

def evaluate_per_epoch(model, val_loader, criterion, device):
    """
    :param model:
    :param val_loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    losses = []
    total_predictions = get_ntu_predictions_dict()
    with torch.no_grad():
     for _, (observed_poses, target_poses) in enumerate(val_loader):
        observed_poses = observed_poses.to(device, dtype = torch.float32)
        target_poses   = target_poses.to(device, dtype = torch.float32)
        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]
        predicted_poses = model(observed_poses, prediction_len)
        loss = criterion(predicted_poses, target_poses, prediction_len) 
        losses.append(loss.item())

        predictions = compute_joints_metric(
            pred = predicted_poses.contiguous().view(batch, prediction_len, -1, 2),  
            gt   = target_poses.contiguous().view(batch, prediction_len, -1, 2),  
            time_step = prediction_len, 
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
  #  train_dataset = PennAction(total_frame = time_step, transform = PennAction.get_default_transform(), train = True, on_keypoints = True)
    train_dataset = NTU(total_frame = time_step, train = True)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

  #  val_dataset  = PennAction(total_frame = time_step, transform = PennAction.get_default_transform(), train = False, on_keypoints = True)
    val_dataset  = NTU(total_frame = time_step, train = False)
    val_loader   = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    print('-' * 40)
    print('Number of train batches =', len(train_loader))
    print('Number of validation batches =', len(val_loader))

    print('-' * 40)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(device, 'is available')

    checkpoint = 'checkpoint/joint_lstm_ntu.pth.tar'
    
    start_epoch = 0 
    if os.path.exists(checkpoint): 
        model_dict = torch.load(checkpoint, weights_only = False) 
        start_epoch = model_dict['epoch'] + 1 
        model       = get_model(device)
        model.load_state_dict(model_dict['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold = 1e-5)
   
    else : 
        model = get_model(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold = 1e-5)
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
            save_checkpoint(epoch, model, optimizer, model_name = 'joint_lstm_ntu' , prefix_dir_name = 'checkpoint/baseline_lstm/')

   # plot(np.array(loss_list), np.array(metric_list), title)
    return model, loss_list, metric_list

if __name__ == '__main__':
    model, loss_list, metric_list = train(
        time_step = 29, 
        batch_size = 16,
        learning_rate = 1e-4,
        betas = (0.9, 0.999),
        scheduler_step = 40000,
        scheduler_gamma = 0.3333,
        num_epochs = 100,
        title = 'Key Point LSTM Training',
    )

