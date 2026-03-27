import torch
import torch.nn as nn 
import torch.nn.functional as F
import os 
import numpy as np

class PosePredictionMLP(nn.Module):
    def __init__(self, pose_dim, coordinates, train_len, predict_len, hidden_layers, dropout = 0.1): 
        """
        :param pose_dim      : per joint information
        :param coordinates   : training on coordinates
        :param train_len     : the length of the observed pose sequences 
        :param predict_len   : the length of the target   pose sequences
        :param hidden_layers : the size of the hidden layers of the network  
        :param dropout       : dropout probability ? 
        """

        super(PosePredictionMLP, self).__init__()
        self.pose_dim      = pose_dim * (3 if coordinates else 2)
        self.train_len     = train_len 
        self.predict_len   = predict_len 
        self.hidden_layers = hidden_layers
        self.dropout       = 0.1
        self.mlp = nn.ModuleList()
        self.mlp.append(
            nn.Sequential(
               nn.Linear(self.train_len * self.pose_dim, self.hidden_layers[0]), 
               nn.ReLU(inplace = True), 
               nn.Dropout(self.dropout)
            )
        ) 
        for i in range(1, len(hidden_layers)): 
            self.mlp.append(
               nn.Sequential(
                  nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]), 
                  nn.ReLU(inplace = True), 
                  nn.Dropout(self.dropout), 
               )
            )
        self.mlp.append(
            nn.Linear(self.hidden_layers[-1], self.predict_len * self.pose_dim)
        )

    def forward(self, observed_poses, prediction_len = 14): 
        """
        :param observed_poses : tensor of shape (batch, train_len, pose_dim)
        :return : 
           predicted joint sequences : a tensor of shape(batch, predicted_len, pose_dim) 
        """
        batch          = observed_poses.shape[0]
        observed_poses = observed_poses.contiguous().view(batch, -1)
        out            = observed_poses 
        for layer in self.mlp:   
            out = layer(out)
        out = out.contiguous().view(batch, self.predict_len, self.pose_dim)
        return out 
    

def mlp_loss_function(pred_joints, gt_joints, time_step):
    """
    :param pred_joints : (B, T, posedim)
    :param gt_joints   : (B, T, posedim)   
    :param time_step   : prediction time step 
    :return:
    """
    criterion  = nn.SmoothL1Loss(reduction = 'mean')
    total_loss = 0
    for t in range(time_step):
        pred = pred_joints[:, t, :]
        gt   = gt_joints[:,   t, :]
        curr_loss = criterion(pred, gt)
        total_loss = total_loss + curr_loss
    return total_loss / time_step


def get_mlp_model(pose_dim = 13, coordinates = True, train_len = 15, predict_len = 15, hidden_layers = [1024, 2048, 1024], dropout = 0.2, device = 'cuda'):
    model = PosePredictionMLP( 
        pose_dim      = pose_dim,
        coordinates   = coordinates, 
        train_len     = train_len, 
        predict_len   = predict_len, 
        hidden_layers = hidden_layers, 
        dropout       = dropout
    )
    return model
