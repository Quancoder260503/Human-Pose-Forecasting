import torch
import torch.nn as nn 
import torch.nn.functional as F
import os 
import numpy as np

class PosePredictionMLP(nn.Module):
    """
    A custom baseline network used for joint predictions 
    """ 
    def __init__(self, pose_dim, train_len, predict_len, hidden_layers, dropout = 0.1): 
        super(PosePredictionMLP, self).__init__()
        self.pose_dim      = pose_dim 
        self.train_len     = train_len 
        self.predict_len   = predict_len 
        self.hidden_layers = hidden_layers
        self.dropout     = 0.1
        self.mlp = nn.ModuleList()
        self.mlp.append(
            nn.Sequential(
               nn.Linear(train_len * pose_dim, hidden_layers[0]), 
               nn.ReLU(inplace = True), 
               nn.Dropout(dropout)
            )
        ) 
        for i in range(1, len(hidden_layers)): 
            self.mlp.append(
               nn.Sequential(
                  nn.Linear(hidden_layers[i - 1], hidden_layers[i]), 
                  nn.ReLU(inplace = True), 
                  nn.Dropout(dropout), 
               )
            )
        self.mlp.append(
            nn.Linear(hidden_layers[-1], predict_len * pose_dim)
        )

    def forward(self, inp): 
        """
        :param inp : tensor of shape (B, train_len, pose_dim)
        :return : 
           predicted jointed sequences : a tensor of shape(B, predicted_len, pose_dim) 
        """
        batch = inp.shape[0]
        inp = inp.contiguous().view(batch, -1)
        out = inp 
        for layer in self.mlp:   
            out = layer(out)
        out = out.contiguous().view(batch, self.predict_len, self.pose_dim)
        return out 
    

def mlp_loss_function(pred_joints, gt_joints, time_step):
    """
    :param pred_joints : (B, T, 39)
    :param gt_joints   : (B, T, 39)   
    :param time_step   : prediction time step 
    :return:
    """
    # Huber Loss
    criterion  = nn.SmoothL1Loss(reduction = 'mean')
    total_loss = 0
    for t in range(time_step):
        pred = pred_joints[:, t, :]
        gt   = gt_joints[:,   t, :]
        curr_loss = criterion(pred, gt)
        total_loss = total_loss + curr_loss
    return total_loss / time_step
