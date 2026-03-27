import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import * 
from model.PositionalEncoding import * 

"""
Model human motion using VAE, note that the input for this model would have to be in 
the form of vectors
"""

class AttentionEncoder(nn.Module): 
    def __init__(self, pose_dim, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout, positional_encoding, latent_dim, device):

        """
        :param pose_dim            : pose dimension 
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (associated with the object through out the structures)
        :param query_dim           : query dimension (dimension of query space)
        :param value_dim           : value dimension (dimension of value space)
        :param num_layers          : number of (multi_head_attention + ffn) layers
        :param positional_encoding : positional_encoding 
        :param latent_dim          : latent dimension 
        :param dropout             : dropout probability ? 
        :param device              : tensor device 
        """
        super(AttentionEncoder, self).__init__() 

        self.pose_dim            = pose_dim
        self.num_heads           = num_heads
        self.model_dim           = model_dim 
        self.query_dim           = query_dim 
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.device              = device 
        self.latent_dim          = latent_dim
        self.positional_encoding = positional_encoding
        
        self.avg_pooling   = nn.AdaptiveAvgPool1d(1)
        self.apply_dropout = nn.Dropout(self.dropout) 
        self.embedding_fc  = nn.Linear(self.pose_dim, self.model_dim)
        self.layernorm     = nn.LayerNorm(self.model_dim)

        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layers() for _ in range(self.num_layers)
        ])
        
        self.fc_mean    = nn.Linear(self.model_dim, self.latent_dim) 
        self.fc_log_var = nn.Linear(self.model_dim, self.latent_dim)

        self.init_weights()

    def init_weights(self): 
        """
        Initialize the weights of the model 
        """
        for p in self.parameters():
            if p.dim() > 1 : 
                nn.init.xavier_uniform_(p, gain = 1.0) 
    
    def make_encoder_layers(self):
        """
        Create a Multihead Attn + FFN layer 
        """ 
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
               dim_model   = self.model_dim, 
               num_heads   = self.num_heads, 
               dim_queries = self.query_dim, 
               dim_values  = self.value_dim, 
               dropout     = self.dropout, 
               in_decoder  = False, 
               device      = self.device, 
            ), 
            PointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout, 
            )
        ])
        return encoder_layer

    def forward(self, observed_poses): 
        ''' 
        :param observed_poses : tensor of shape (B, train_len, pose_dim) 
        :return : 
          mean     : latent mean of shape (B,    latent_dim) 
          log_var  : latent_log_var of shape (B, latent_dim) 
        '''
        observed_poses = self.embedding_fc(observed_poses) # (B, train_len, model_dim)

        observed_poses = observed_poses * math.sqrt(self.model_dim)  + self.positional_encoding 
        observed_poses = self.layernorm(observed_poses)

        for encoder_layer in self.encoder_layers:
            observed_poses = encoder_layer[0](observed_poses, observed_poses) # multi_head_attention 
            observed_poses = encoder_layer[1](observed_poses) # ffn
        
        observed_poses =  self.layernorm(observed_poses)
        encoder_out    =  observed_poses.permute(0, 2, 1)  # (batch_size, model_dim, train_len)
        encoder_out    =  self.avg_pooling(encoder_out)    # (batch_size, model_dim, 1) 
        encoder_out    =  encoder_out.squeeze(2)
        mean           =  self.fc_mean(encoder_out)
        log_var        =  self.fc_log_var(encoder_out) 
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
        self.pose_dim      = pose_dim
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
        self.proj_hidden_dim = nn.Linear(latent_dim, hidden_dim)
        self.proj_pose_dim   = nn.Linear(hidden_dim, pose_dim)

    def forward(self, observed_poses, latent_vector, predicted_len): 
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

        for _ in range(predicted_len):
           output, curr_state = self.decoder(last_observed, curr_state)          # output : (B, 1, hidden_dim) 
           curr_pose          = self.proj_pose_dim(output.squeeze(1))            # (B, pose_dim) 
           predicted_poses.append(curr_pose) 
           last_observed = curr_pose.unsqueeze(1)                                # (B, 1, pose_dim)
        predicted_poses = torch.stack(predicted_poses, dim = 1)                  # (B, T, pose_dim)
        return predicted_poses 


class AttentionMotionVAE(nn.Module): 
    def __init__(self, 
        pose_dim            = 13,
        encoding_length     = 14, 
        num_heads           = 8,  
        model_dim           = 512, 
        query_dim           = 64, 
        value_dim           = 64, 
        fc_hidden_dim       = 2048, 
        num_encoding_layers = 6, 
        num_decoding_layers = 1, 
        dropout             = 0.2, 
        latent_dim          = 128, 
        temperature         = 10000, 
        alpha               = 1, 
        device          = 'cuda',  
    ): 
        super(AttentionMotionVAE, self).__init__()
        self.pose_dim            = pose_dim * 2
        self.encoding_length     = encoding_length
        self.num_heads           = num_heads
        self.model_dim           = model_dim 
        self.query_dim           = query_dim 
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_encoding_layers = num_encoding_layers 
        self.num_decoding_layers = num_decoding_layers
        self.dropout             = dropout 
        self.latent_dim          = latent_dim
        self.temperature         = temperature 
        self.alpha               = alpha 
        self.device              = device

        self.init_positional_encoding()

        self.encoder = AttentionEncoder(
            pose_dim            = self.pose_dim, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim, 
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_encoding_layers,
            positional_encoding = self.encoder_positional_encoding,   
            dropout             = self.dropout, 
            device              = self.device,
            latent_dim          = self.latent_dim, 
        )

        self.decoder = Decoder(
            pose_dim   = self.pose_dim, 
            hidden_dim = self.model_dim,
            latent_dim = self.latent_dim, 
            dropout    = self.dropout, 
            device     = self.device,  
        )
    
    def init_positional_encoding(self):
        encoder_positional = PositionalEncoding1D(
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha, 
        )
        self.encoder_positional_encoding = encoder_positional(sequence_length = self.encoding_length).contiguous().view(
            1, self.encoding_length, self.model_dim
        )
        # Do not train the distinguish noise 
        self.encoder_positional_encoding = nn.Parameter(self.encoder_positional_encoding, requires_grad = False) 

    
    def reparameterization(self, mean, log_var): 
        epsilon = torch.randn_like(mean) # (B, latent_dim) 
        var     = torch.exp(0.5 * log_var) # (B, latent_dim) 
        return epsilon * var + mean   
    
    def forward(self, observed_sequences, prediction_len): 
        '''
        :param : observed sequences (B, T, pose_dim)
        :predicted_len : prediction length 
        :return :
           return tensor of shape (B, predicted_len, pose_dim) 
        '''
        mean, log_var = self.encoder(observed_sequences)
        z = self.reparameterization(mean, log_var) 
        out = self.decoder(observed_sequences, z, prediction_len)
        return out, mean, log_var

def attention_vae_loss_function(pred_velos, gt_velos, mean, log_var): 
    """
    :param pred_velos : tensor of shape (B, T, pose_dim)
    :param gt_velos   : tensor of shape (B, T, pose_dim) 
    :mean             : tensor of shape (B, latent_dim) 
    :log_var          : tensor of shape (B, latent_dim)
    """
    assert pred_velos.shape == gt_velos.shape
    reconstruction_loss = torch.mean(torch.norm(pred_velos - gt_velos, dim = -1))
    kullback_leiber_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + 0.1 * kullback_leiber_loss


'''
method only used for velocity 
'''
def get_attention_vae_model(
        pose_dim            = 13,
        encoding_length     = 14, 
        num_heads           = 8,  
        model_dim           = 512, 
        query_dim           = 64, 
        value_dim           = 64, 
        fc_hidden_dim       = 2048, 
        num_encoding_layers = 6, 
        num_decoding_layers = 1, 
        dropout             = 0.2, 
        latent_dim          = 128, 
        temperature         = 10000, 
        alpha               = 1, 
        device          = 'cuda', 
    ): 

    model = AttentionMotionVAE(
        pose_dim            = pose_dim, 
        encoding_length     = encoding_length, 
        num_heads           = num_heads,
        model_dim           = model_dim, 
        query_dim           = query_dim, 
        value_dim           = value_dim, 
        fc_hidden_dim       = fc_hidden_dim, 
        num_encoding_layers = num_encoding_layers, 
        num_decoding_layers = num_decoding_layers, 
        dropout             = dropout,  
        latent_dim          = latent_dim, 
        temperature         = temperature, 
        alpha               = alpha, 
        device              = device, 
    )

    return model 

    
