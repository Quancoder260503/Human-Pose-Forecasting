import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Encoder-Decoder LSTM for generating sequence of poses, the training data will try to minimise 
the result of the coordinates of the joints (Baseline Method)
'''

class JointLSTM(nn.Module):
    def __init__(self, pose_dim = 26, embedding_dim = 64, hidden_dim = 128, num_layers = 2, dropout = 0.1, device = 'cuda'): 
        super(JointLSTM, self).__init__()
        self.pose_dim      = pose_dim
        self.embedding_dim  = embedding_dim 
        self.hidden_dim     = hidden_dim 
        self.num_layers     = num_layers
        self.dropout        = dropout 
        self.device         = device

        self.embedding_fc  = nn.Linear(pose_dim, embedding_dim) 

        self.encoder = nn.LSTM(
            input_size  = embedding_dim, 
            hidden_size = hidden_dim, 
            dropout     = dropout, 
            num_layers  = num_layers, 
            batch_first = True 
        )

        self.decoder = nn.LSTM(
            input_size  = embedding_dim, 
            hidden_size = hidden_dim, 
            dropout     = dropout, 
            num_layers  = num_layers, 
            batch_first = True 
        )

        self.output_fc = nn.Linear(hidden_dim, pose_dim)
    
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

# if __name__ == "__main__":
#     dummy = torch.randn(4, 10, 26).to('cuda') 
#     model = JointLSTM().to('cuda')
#     out = model(dummy, 15)
#     print(out.shape)