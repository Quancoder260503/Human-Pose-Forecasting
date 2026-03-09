import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Baseline VAE,
more technique can be applied rather than LSTM, i.e. a Transformer like architecture, 
find out another ways to model the input other than joints 

"""

class Encoder(nn.Module): 
    def __init__(self, input_dim = 26, hidden_dim = 64, latent_dim = 16, num_layers = 2, dropout = 0.1, device = 'cuda'):
        super(Encoder, self).__init__() 
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim 
        self.num_layers = num_layers 
        self.dropout    = dropout
        self.device     = device 

        self.encoder    = nn.LSTM(
            input_size  = input_dim, 
            hidden_size = hidden_dim, 
            num_layers  = num_layers, 
            batch_first = True,
            dropout     = dropout 
        )
        
        self.fc_mean    = nn.Linear(hidden_dim, latent_dim) 
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim) 
    
    def forward(self, observed_dim): 
        ''' 
        :param observed_dim : tensor of shape (B, Train_len, input_dim) 
        :return : 
          mean     : latent mean of shape (B, latent_dim) 
          log_var  : latent_log_var of shape (B, latent_dim) 
        '''

        batch, _, _ = observed_dim.shape
        lstm_state = (
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32), 
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        )
        output, lstm_state = self.encoder(observed_dim, lstm_state) # output : (batch, train_len, hidden_dim)
        # get the hidden stete 
        hidden_state = lstm_state[0] # (batch_size, hidden_dim) 
        mean    =  self.fc_mean(hidden_state)
        log_var =  self.fc_log_var(hidden_state) 
        return mean, log_var

class Decoder(nn.Module): 
    def __init__(self, input_dim = 26, hidden_dim = 64, latent_dim = 16, num_layers = 2, dropout = 0.1, device = 'cuda'):
        super(Decoder, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim 
        self.num_layers = num_layers
        self.dropout    = dropout
        self.device     = device 
        
        self.decoder    = nn.LSTM(
            input_size = input_dim,  
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            dropout = dropout, 
            batch_first = True, 
        )

        self.reconstruct_hidden = nn.Linear(latent_dim, hidden_dim)
        self.reconstruct_inp    = nn.Linear(hidden_dim, input_dim)

    def forward(self, observed_poses, latent_vector, predicted_len): 
        '''
        :param : observed_poses : tensor of shape (B, T, pose_dim)
        :param : latent_vectors : tensor of shape (B, latent_dim) 
        :param : predicted_len  : prediction time frame 
        '''

        batch = observed_poses.shape[0]
        
        cell_state   = torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        hidden_state = self.reconstruct_hidden(latent_vector) 

        curr_state   = (hidden_state, cell_state)
        last_observed = observed_poses[:, -1, :].unsqueeze(1) # (B, 1, pose_dim)
        predicted_poses = [] 
    
        for t in range(predicted_len):
           output, curr_state = self.decoder(last_observed, curr_state) # output : (B, 1, hidden_dim) 
           curr_pose = self.reconstruct_inp(output.contiguous().view(batch, -1)) # (B, pose_dim) 
           predicted_poses.append(curr_pose) 
           last_observed = curr_pose.unsqueeze(1) # (B, 1, pose_dim)
        predicted_poses = torch.stack(predicted_poses, dim = 1)
        return predicted_poses 
    

class JointVAE(nn.Module): 
    def __init__(self, Encoder, Decoder): 
        super(JointVAE, self).__init__()
        self.encoder = Encoder 
        self.decoder = Decoder 
    
    def reparameterization(self, mean, log_var): 
        epsilon = torch.randn_like(mean) # (B, latent_dim) 
        var     = torch.exp(log_var)
        return epsilon * var + mean # (B, latent_dim) 
    
    def forward(self, observed_sequences, predicted_len = 14): 
        '''
        :param : observed sequences (B, T, pose_dim)
        :predicted_len : prediction length 
        :return :
           return tensor of shape (B, predicted_len, pose_dim) 
        '''
        mean, log_var = self.encoder(observed_sequences)
        z = self.reparameterization(mean, log_var) 
        out = self.decoder(observed_sequences, z, predicted_len)
        return out 

if __name__ == "__main__":
    dummy = torch.randn(5, 10, 26).to('cuda') 
    model = JointVAE(Encoder(), Decoder()).to('cuda')
    out = model(dummy)
    print(out.shape) 