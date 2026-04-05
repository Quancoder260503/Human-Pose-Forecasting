import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import math 

"""
Model human motion using VAE, note that the input for this model would have to be in 
the form of vectors
"""

class Encoder(nn.Module): 
    def __init__(self, pose_dim = 26, hidden_dim = 32, latent_dim = 16, num_layers = 2, dropout = 0.2, device = 'cuda'):
        """
        :param pose_dim   : per joint information
        :param hidden_dim : recurrent neural network hidden dimension 
        :param num_layers : recurrent neural network layers 
        :param dropout    : dropout probability ? 
        :param device     : tensor device 
        """
        super(Encoder, self).__init__() 
        self.pose_dim      = pose_dim
        self.hidden_dim    = hidden_dim 
        self.latent_dim    = latent_dim 
        self.num_layers    = num_layers 
        self.dropout       = dropout
        self.device        = device 
        
        self.encoder    = nn.LSTM(
            input_size  = pose_dim, 
            hidden_size = hidden_dim, 
            num_layers  = num_layers, 
            batch_first = True,
            dropout     = dropout, 
        )
        
        self.fc_mean    = nn.Linear(hidden_dim, latent_dim) 
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim) 
    
    def forward(self, observed_poses): 
        ''' 
        :param observed_poses : tensor of shape (B, train_len, posedim) 
        :return : 
          mean     : latent mean of shape (B,    latent_dim) 
          log_var  : latent_log_var of shape (B, latent_dim) 
        '''

        batch, _, _ = observed_poses.shape
        lstm_state = (
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32), 
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        )
        output, lstm_state = self.encoder(observed_poses, lstm_state) # output : (batch, train_len, hidden_dim)
        # get the hidden stete 
        encoder_out = output[:, -1, :] # (batch_size, hidden_dim) 
        mean    =  self.fc_mean(encoder_out)
        log_var =  self.fc_log_var(encoder_out) 
        return mean, log_var

class Decoder(nn.Module): 
    def __init__(self, pose_dim = 26, hidden_dim = 32, latent_dim = 16, num_layers = 1, dropout = 0.2, device = 'cuda'):
        """
        :param pose_dim   : per joint information
        :param hidden_dim : recurrent neural network hidden dimension 
        :param num_layers : recurrent neural network layers 
        :param dropout    : dropout probability ? 
        :param device     : tensor device 
        """
        super(Decoder, self).__init__()
        self.pose_dim      =  pose_dim
        self.hidden_dim    = hidden_dim 
        self.latent_dim    = latent_dim 
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.device        = device 
        
        self.decoder    = nn.LSTM(
            input_size  = pose_dim,  
            hidden_size = hidden_dim, 
            num_layers  = num_layers, 
            dropout     = dropout if num_layers > 1 else 0, 
            batch_first = True, 
        )
        self.proj_hidden_dim  = nn.Linear(latent_dim, hidden_dim)
        self.proj_pose_dim    = nn.Linear(hidden_dim, pose_dim)

    def forward(self, observed_poses, target_poses, latent_vector, predicted_len, teacher_forcing_rate): 
        '''
        :param : observed_poses : tensor of shape (B, T, pose_dim)
        :param : latent_vectors : tensor of shape (B, latent_dim) 
        :param : predicted_len  : prediction time frame 
        '''

        batch = observed_poses.shape[0]
        
        cell_state   = torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        hidden_state = self.proj_hidden_dim(latent_vector).unsqueeze(0) # (1, B, hidden_dim) 
        hidden_state = hidden_state.repeat(self.num_layers, 1, 1)

        curr_state   = (hidden_state, cell_state)
        last_observed = observed_poses[:, -1, :].unsqueeze(1) # (B, 1, pose_dim)
        predicted_poses = [] 
    
        for t in range(predicted_len):
           if target_poses is not None and random.random() < teacher_forcing_rate : 
               last_observed = last_observed if t == 0 else target_poses[:, t - 1, :].unsqueeze(1)
           output, curr_state = self.decoder(last_observed, curr_state)          # output : (B, 1, hidden_dim) 
           curr_pose = self.proj_pose_dim(output.squeeze(1)) # (B, pose_dim) 
           predicted_poses.append(curr_pose) 
           last_observed = curr_pose.unsqueeze(1) # (B, 1, pose_dim)
        predicted_poses = torch.stack(predicted_poses, dim = 1) # (B, T, pose_dim)
        return predicted_poses 
    

class RNNMotionVAE(nn.Module): 
    def __init__(self, Encoder, Decoder): 
        super(RNNMotionVAE, self).__init__()
        self.encoder = Encoder 
        self.decoder = Decoder 
    
    def reparameterization(self, mean, log_var): 
        epsilon = torch.randn_like(mean)   # (B, latent_dim) 
        var     = torch.exp(0.5 * log_var) # (B, latent_dim) 
        return epsilon * var + mean   
    
    def forward(self, observed_sequences, target_sequences, prediction_len, teacher_forcing_rate = 0.5): 
        '''
        :param : observed sequences (B, T, pose_dim)
        :predicted_len : prediction length 
        :return :
           return tensor of shape (B, predicted_len, pose_dim) 
        '''
        mean, log_var = self.encoder(observed_sequences)
        z = self.reparameterization(mean, log_var) 
        out = self.decoder(observed_sequences, target_sequences, z, prediction_len, teacher_forcing_rate)
        return out, mean, log_var

def vae_loss_function(pred_velos, gt_velos, mean, log_var, beta): 
    """
    :param pred_velos : tensor of shape (B, T, pose_dim)
    :param gt_velos   : tensor of shape (B, T, pose_dim) 
    :mean             : tensor of shape (B, latent_dim) 
    :log_var          : tensor of shape (B, latent_dim)
    :beta             : beta coefficient
    """
    assert pred_velos.shape == gt_velos.shape
    reconstruction_loss = torch.mean(torch.norm(pred_velos - gt_velos, dim = -1))
    kullback_leiber_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kullback_leiber_loss

def get_rnn_vae_model(
        pose_dim = 13,
        hidden_dim = 32, 
        latent_dim = 16, 
        encoding_layers = 2, 
        decoding_layers = 1, 
        dropout         = 0.2, 
        device          = 'cuda',  
    ):
    vae_encoder = Encoder(pose_dim = pose_dim * 2, hidden_dim = hidden_dim, latent_dim = latent_dim, num_layers = encoding_layers, dropout = dropout, device = device) 
    vae_decoder = Decoder(pose_dim = pose_dim * 2, hidden_dim = hidden_dim, latent_dim = latent_dim, num_layers = decoding_layers, dropout = dropout, device = device) 
    rnn_vae     = RNNMotionVAE(vae_encoder, vae_decoder) 
    return rnn_vae


