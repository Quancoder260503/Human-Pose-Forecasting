import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePredictionLSTM(nn.Module):
    def __init__(self, pose_dim = 13, coordinates = 3, embedding_dim = 2048, hidden_dim = 256, num_layers = 2, dropout = 0.2, device = 'cuda'): 
        """
        :param pose_dim : per joint information
        :param embedding_dim : embedding dimension 
        :param hidden_dim    : Recurrent Neural Network hidden dimension 
        :param num_layers    : the number of layers for Recurrent Neural Network
        :param dropout       : dropout probability ? 
        :param device        : device of tensor 
        """
        super(PosePredictionLSTM, self).__init__()
        self.pose_dim       = pose_dim * (3 if coordinates else 2)
        self.embedding_dim  = embedding_dim 
        self.hidden_dim     = hidden_dim 
        self.num_layers     = num_layers
        self.dropout        = dropout 
        self.device         = device

        self.embedding_fc  = nn.Linear(self.pose_dim, self.embedding_dim) 

        self.encoder = nn.LSTM(
            input_size  = self.embedding_dim, 
            hidden_size = self.hidden_dim, 
            dropout     = self.dropout, 
            num_layers  = self.num_layers, 
            batch_first = True 
        )

        self.decoder = nn.LSTM(
            input_size  = self.embedding_dim, 
            hidden_size = self.hidden_dim, 
            dropout     = self.dropout, 
            num_layers  = self.num_layers, 
            batch_first = True 
        )

        self.output_fc = nn.Linear(self.hidden_dim, self.pose_dim)
    
    def forward(self, observed_poses, prediction_len = 14):
        """
        :param observed_poses : tensor of shape (Batch, Train_Len, pose_dim) 
        :return : tensor of shape 
        """
        batch, _, _ = observed_poses.shape

        lstm_state = (
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32), 
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        )

        embedded_poses = self.embedding_fc(observed_poses) # (batch, train_len, embedding_dim)
        output, lstm_state = self.encoder(embedded_poses, lstm_state) # output : (batch, train_len, hidden_dim)

        last_observed = observed_poses[:, -1, :].unsqueeze(1) # (Batch, 1, pose_dim)
        predicted_poses = [] 
        for t in range(prediction_len): 
             embedded_last = self.embedding_fc(last_observed) # (batch, 1, embedding_dim)
             output, lstm_state = self.decoder(embedded_last, lstm_state) # output : (batch, 1, hidden_dim) 
             curr_poses = output.squeeze(1)  # (batch, hidden_dim)
             curr_poses = self.output_fc(curr_poses) # (batch, pose_dim)
             predicted_poses.append(curr_poses)
             last_observed = curr_poses.unsqueeze(1) # (batch, 1, pose_dim)
        
        predicted_poses = torch.stack(predicted_poses, dim = 1) # (batch, Test_len, pose_dim) 

        return predicted_poses


def lstm_loss_function(pred_joints, gt_joints, time_step):
    """
    :param pred_joints : (B, T, pose_dim)
    :param gt_joints   : (B, T, pose_dim)   
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

def get_lstm_model(pose_dim = 13, coordinates = True, embedding_dim = 2048, hidden_dim = 256, num_layers = 2, dropout = 0.2, device = 'cuda'):
    model = PosePredictionLSTM(
        pose_dim = pose_dim, 
        coordinates = coordinates, 
        embedding_dim = embedding_dim, 
        hidden_dim = hidden_dim, 
        num_layers = num_layers, 
        dropout = dropout, 
        device = device
    ) 
    return model