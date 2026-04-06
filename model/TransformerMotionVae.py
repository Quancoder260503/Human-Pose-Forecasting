import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention import * 
from model.PositionalEncoding import * 

"""
Model human motion using VAE, note that the input for this model would have to be in 
the form of vectors
:TODO add a bit of distinguish noise for each joints 
"""

def KL_annealing(epoch, total_epochs, schedule_type = 'linear', start_beta = 0.0, end_beta = 1.0):
    progress = epoch / total_epochs
    if schedule_type == "linear":
        return start_beta + (end_beta - start_beta) * progress
    elif schedule_type == "cyclical":
        return end_beta * (1 + math.cos(math.pi * (1 - progress))) / 2
    elif schedule_type == "sigmoid":
        return start_beta + (end_beta - start_beta) / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

class MotionTransformerEncoder(nn.Module): 
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
        super(MotionTransformerEncoder, self).__init__() 

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
        encoder_out = self.embedding_fc(observed_poses) # (B, train_len, model_dim)
        encoder_out = encoder_out * math.sqrt(self.model_dim)  + self.positional_encoding 
        encoder_out = self.layernorm(encoder_out)

        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer[0](encoder_out, encoder_out) # multi_head_attention 
            encoder_out = encoder_layer[1](encoder_out) # ffn
        
        encoder_out     =  self.layernorm(encoder_out)
        avg_encoder_out =  encoder_out.permute(0, 2, 1)            # (batch_size, model_dim, train_len)
        
        avg_encoder_out =  self.avg_pooling(avg_encoder_out)           # (batch_size, model_dim, 1) 
        avg_encoder_out =  avg_encoder_out.squeeze(2)
        
        mean            =  self.fc_mean(avg_encoder_out)
        log_var         =  self.fc_log_var(avg_encoder_out)
        return encoder_out, mean, log_var


class MotionTransformerDecoder(nn.Module): 
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
        super(MotionTransformerDecoder, self).__init__()
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

        self.apply_dropout  = nn.Dropout(self.dropout) 
        self.proj_model_dim = nn.Linear(self.latent_dim, self.model_dim)
        self.proj_pose_dim  = nn.Linear(self.model_dim, self.pose_dim)
        self.layernorm      = nn.LayerNorm(self.model_dim)

        self.decoder_layers = nn.ModuleList([
            self.make_decoder_layers() for _ in range(self.num_layers)
        ])
    
     def make_decoder_layers(self):
        """
        Create a (Cross Multihead Attn + Self Multihead Attn + FFN layer)  
        """ 
        decoder_layer = nn.ModuleList([
            # Cross-Attention 
            MultiHeadAttention( 
              dim_model      = self.model_dim,
              num_heads      = self.num_heads, 
              dim_queries    = self.query_dim,
              dim_values     = self.value_dim, 
              dropout        = self.dropout,   
              self_attention = False, 
              in_decoder     = True, 
              device         = self.device, 
            ), 
            # Self attention 
            MultiHeadAttention(
               dim_model      = self.model_dim, 
               num_heads      = self.num_heads, 
               dim_queries    = self.query_dim, 
               dim_values     = self.value_dim, 
               dropout        = self.dropout, 
               self_attention = True, 
               in_decoder     = True, 
               device         = self.device, 
            ), 
            PointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout, 
            )
        ])
        return decoder_layer
     
     def forward(self, encoder_out, latent_vars): 
        """
        :param : encoder_out    : tensor of shape (B, train_len,     pose_dim)
        :param : latent_vars    : tensor of shape (B, prediction_len, latent_dim)
        :return : 
           a tensor of shape (B, prediction_len, model_dim)
        """
        decoder_out = self.proj_model_dim(latent_vars) # (B, prediction_len, model_dim) 
        decoder_out = decoder_out * math.sqrt(self.model_dim) + self.positional_encoding
        decoder_out = self.layernorm(decoder_out) 

        for decoder_layer in self.decoder_layers: 
            decoder_out = decoder_layer[0](decoder_out, encoder_out) 
            decoder_out = decoder_layer[1](decoder_out, decoder_out) 
            decoder_out = decoder_layer[2](decoder_out)
        
        decoder_out = self.layernorm(decoder_out)     # (B, prediction_len, model_dim) 
        decoder_out = self.proj_pose_dim(decoder_out) # (B, prediction_len, pose_dim)
        return decoder_out 

        

class TransformerMotionVAE(nn.Module): 
    def __init__(self, 
        pose_dim            = 13,
        encoding_length     = 14,
        decoding_length     = 14,  
        num_heads           = 8,  
        model_dim           = 512, 
        query_dim           = 64, 
        value_dim           = 64, 
        fc_hidden_dim       = 2048, 
        num_encoding_layers = 6, 
        num_decoding_layers = 6, 
        dropout             = 0.2, 
        latent_dim          = 128, 
        temperature         = 10000, 
        alpha               = 1, 
        device          = 'cuda',  
    ): 
        super(TransformerMotionVAE, self).__init__()
        self.pose_dim            = pose_dim * 2
        self.encoding_length     = encoding_length
        self.decoding_length     = decoding_length
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

        self.encoder = MotionTransformerEncoder(
            pose_dim            = self.pose_dim, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim, 
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_encoding_layers,
            positional_encoding = self.encoder_positional_encoding,   
            dropout             = self.dropout, 
            latent_dim          = self.latent_dim,
            device              = self.device, 
        )

        self.decoder = MotionTransformerDecoder(
            pose_dim            = self.pose_dim, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim, 
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_decoding_layers, 
            positional_encoding = self.decoder_positional_encoding, 
            dropout             = self.dropout, 
            latent_dim          = self.latent_dim, 
            device              = self.device, 
        )

        self.init_weights()
    
    def init_positional_encoding(self):
        encoder_positional = PositionalEncoding1D(
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha, 
        )
        decoder_positional = PositionalEncoding1D(
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha,
        )

        self.encoder_positional_encoding = encoder_positional(sequence_length = self.encoding_length).contiguous().view(
            1, self.encoding_length, self.model_dim
        )

        self.decoder_positional_encoding = decoder_positional(sequence_length = self.decoding_length).contiguous().view(
            1, self.decoding_length, self.model_dim 
        )
        
        # Do not train the distinguish noise (temporal noise) 
        self.encoder_positional_encoding = nn.Parameter(self.encoder_positional_encoding, requires_grad = False) 
        self.decoder_positional_encoding = nn.Parameter(self.decoder_positional_encoding, requires_grad = False)

    def init_weights(self): 
        """
        Initialize the weights of the model 
        """
        for p in self.parameters():
            if p.dim() > 1 : 
                nn.init.xavier_uniform_(p, gain = 1.0) 
    
    def reparameterization(self, mean, log_var): 
        batch, latent_dim = mean.shape 
        assert mean.shape == log_var.shape 
        epsilon = torch.randn((batch, self.decoding_length, latent_dim)).to(self.device)   # (B, decoding_length, latent_dim) 
        var     = torch.exp(0.5 * log_var) # (B, latent_dim) 
        var     = var.unsqueeze(1) 
        mean    = mean.unsqueeze(1)
        return epsilon * var + mean
    
    def forward(self, observed_sequences): 
        '''
        :param : observed sequences (B, T, pose_dim)
        :predicted_len : prediction length 
        :return :
           return tensor of shape (B, predicted_len, pose_dim) 
        '''
        encoder_out, mean, log_var = self.encoder(observed_sequences) # (B, encoding_length, model_dim) (B, latent_dim), (B, latent_dim)
        z = self.reparameterization(mean, log_var)  # (B, decoding_length, latent_dim)
        decoder_out = self.decoder(encoder_out, z)  # (B, decoding_length, pose_dim)
        return decoder_out, mean, log_var

def transformer_vae_loss_function(pred_velos, gt_velos, mean, log_var, beta): 
    """
    :param pred_velos : tensor of shape (B, T, pose_dim)
    :param gt_velos   : tensor of shape (B, T, pose_dim) 
    :mean             : tensor of shape (B, latent_dim) 
    :log_var          : tensor of shape (B, latent_dim)
    :beta             : beta coefficient
    """
    assert pred_velos.shape == gt_velos.shape
    reconstruction_loss = torch.mean(torch.norm(pred_velos - gt_velos, dim = -1))
    kullback_leiber_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    #print(reconstruction_loss, kullback_leiber_loss)
    return reconstruction_loss + beta * kullback_leiber_loss

'''
method only used for velocity 
'''
def get_transformer_motion_vae_model(
        pose_dim            = 13,
        encoding_length     = 14,
        decoding_length     = 14,  
        num_heads           = 8,  
        model_dim           = 512, 
        query_dim           = 64, 
        value_dim           = 64, 
        fc_hidden_dim       = 2048, 
        num_encoding_layers = 6, 
        num_decoding_layers = 6, 
        dropout             = 0.2, 
        latent_dim          = 128, 
        temperature         = 10000, 
        alpha               = 1, 
        device          = 'cuda', 
    ): 

    model = TransformerMotionVAE(
        pose_dim            = pose_dim, 
        encoding_length     = encoding_length,
        decoding_length     = decoding_length,  
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

    
