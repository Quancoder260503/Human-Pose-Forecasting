import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.Attention import *
from model.PositionalEncoding import *
import matplotlib.pyplot as plt

'''
For now, just experiment mainly on joints 
:TODO : visualize attention 
Model human motion using VAE, note that the input for this model would have to be in 
the form of vectors
:TODO add a bit of distinguish noise for each joints 
"""
'''     

class TransformerEncoder(nn.Module):
    """
    Encoder Architecture, encoded the observed sequences information,
    First step, is to try to plug it as a GRU hidden state for decoder 
    Second step, is to come up with Decoder part of Transformer 
    Third  step, extend this to latent space, get the noise sampled from the Encoder and use it to guide Diffusion Model
    """
    def __init__(self, pose_dim, positional_encoding, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout, device = 'cuda'): 
        """
        :param pose_dim            : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim           : query dimension 
        :param value_dim           : value dimension 
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(TransformerEncoder, self).__init__()
        self.pose_dim            = pose_dim 
        self.positional_encoding = positional_encoding
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.query_dim           = query_dim
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.apply_dropout       = nn.Dropout(dropout)
        self.embedding_fc        = nn.Linear(pose_dim, model_dim)
        self.layernorm           = nn.LayerNorm(self.model_dim) 
        self.device              = device 
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])

    
    def make_encoder_layer(self): 
        """
        Create a MultiheadAttn + FFC layer 
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
                dropout = self.dropout
            )
        ])
        return encoder_layer
    
    def forward(self, observed_poses): 
        """
        For now, only support input tensor of shape (B, train_len, pose_dim = J * 3 or J * 2) (x, y, visibility)
        :param observed_poses : tensor of shape (B, train_len, model_dim)
        : return:
            tensor of shape (B, train_len, model_dim) 
        """
        observed_poses = self.embedding_fc(observed_poses) # (B, train_len, model_dim) 
        observed_poses = self.layernorm(observed_poses)
        # Scale to reduce the effect of positional encoding 
        observed_poses = observed_poses * math.sqrt(self.model_dim) + self.positional_encoding 
        observed_poses = self.apply_dropout(observed_poses)
        for encoder_layer in self.encoder_layers: 
            observed_poses = encoder_layer[0](observed_poses, observed_poses)
            observed_poses = encoder_layer[1](observed_poses)
        observed_poses = self.layernorm(observed_poses) # (B, train_len, dim_model)
        return observed_poses

class TransformerDecoder(nn.Module): 
    def __init__(self, pose_dim, positional_encoding, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout, device = 'cuda'): 
        """
        :param pose_dim      : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param num_heads     : number of attention heads 
        :param model_dim     : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim     : query dimension 
        :param value_dim     : value dimension 
        :param fc_hidden_dim : fully connected layer hidden dim 
        :param num_layers    : number of (MultiheadAttn + FC) layers 
        :param dropout       : dropout probability 
        :param positional_encoding : distinguish noise 
        """

        super(TransformerDecoder, self).__init__() 
        self.pose_dim            = pose_dim 
        self.positional_encoding = positional_encoding 
        self.num_heads           = num_heads 
        self.model_dim           = model_dim 
        self.query_dim           = query_dim 
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers 
        self.device              = device 
        self.dropout             = dropout
        self.apply_dropout       = nn.Dropout(dropout)
        self.embedding_fc        = nn.Linear(pose_dim, model_dim)
        self.layernorm           = nn.LayerNorm(model_dim)
        self.decoder_layers      = nn.ModuleList([ 
            self.make_decoder_layer() for _ in range(num_layers)
        ])

    def make_decoder_layer(self):
        """
        Create a MultiheadAttn + FC layer for decoder layer  
        """
        decoder_layer = nn.ModuleList([
            # Cross_attention 
            MultiHeadAttention(
                num_heads   = self.num_heads, 
                dim_model   = self.model_dim, 
                dim_queries = self.query_dim, 
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = True, 
                device      = self.device, 
            ), 
            # Self_attention 
            MultiHeadAttention(
                num_heads   = self.num_heads, 
                dim_model   = self.model_dim, 
                dim_queries = self.query_dim,
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = True, 
                device      = self.device,  
            ), 
            # FC 
            PointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout, 
            )
        ])
        return decoder_layer 
    
    def forward(self, encoder_out, decoding_poses): 
        """
        For now, only support input tensor of shape (B, train_len, pose_dim = J * 3) (x, y, visibility)
        :param encoder_out    : what we get from the encoder, a tensor of shape ()
        :param decoding_poses : tensor of shape (B, train_len, pose_dim)
        : return:
            tensor of shape (B, train_len, model_dim) 
        """
        decoding_poses = self.embedding_fc(decoding_poses) # (B, decoding_len, model_dim) 
        decoding_poses = self.layernorm(decoding_poses)
        decoding_poses = decoding_poses * math.sqrt(self.model_dim) + self.positional_encoding # scale this up to reduce the effect of dim embedding 
        decoding_poses = self.apply_dropout(decoding_poses)
    
        
        for decoder_layer in self.decoder_layers: 
            # cross-attention 
            decoding_poses = decoder_layer[0](
                decoding_poses, 
                encoder_out, 
            )
            # self-attention 
            decoding_poses = decoder_layer[1]( 
                decoding_poses, 
                decoding_poses, 
            )
            decoding_poses = decoder_layer[2](
                decoding_poses, 
            )
            # (B, decoding_len, model_dim) 
        decoding_poses = self.layernorm(decoding_poses)
        return decoding_poses 

class PoseTransformer(nn.Module):
    def __init__(self, 
        pose_dim                        = 39,  
        encoding_length                 = 14,
        decoding_length                 = 14, 
        num_heads                       = 8,  
        query_dim                       = 64, 
        value_dim                       = 64, 
        model_dim                       = 512, 
        fc_hidden_dim                   = 2048,
        num_layers                      = 6,
        dropout                         = 0.2, 
        positional_encoding_temperature = 10000, 
        positional_encoding_alpha       = 1, 
        device                          = 'cuda', 
    ): 
        """
        :param pose_dim        : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param encoding_length : the length of the encoding pose sequences 
        :param decoding_length : the length of the decoding pose sequences 
        :param num_heads       : number of attention heads 
        :param model_dim       : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim       : query dimension 
        :param value_dim       : value dimension 
        :param fc_hidden_dim   : fully connected layer hidden dim 
        :param num_layers      : number of (MultiheadAttn + FC) layers 
        :param dropout         : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(PoseTransformer, self).__init__()
        self.pose_dim            = pose_dim
        self.encoding_length     = encoding_length 
        self.decoding_length     = decoding_length
        self.num_heads           = num_heads
        self.query_dim           = query_dim 
        self.value_dim           = value_dim
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout 
        self.device              = device

        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

        self.init_positional_encoding()

        self.encoder            = TransformerEncoder(
            pose_dim            = self.pose_dim, 
            positional_encoding = self.encoder_positional_encoding, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim,
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
        ) # Use Transformer encoder as the main decoder

        self.decoder            = TransformerDecoder(
            pose_dim            = self.pose_dim, 
            positional_encoding = self.decoder_positional_encoding, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim, 
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
        )

        self.output_fc = nn.Linear(self.model_dim, self.pose_dim) # map to the output space 
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

        # Do not train the distinguish noise 
        self.encoder_positional_encoding = nn.Parameter(self.encoder_positional_encoding, requires_grad = False) 
        self.decoder_positional_encoding = nn.Parameter(self.decoder_positional_encoding, requires_grad = False)

    def init_weights(self): 
        """
        Initialize the weights of the model 
        """
        for p in self.parameters():
            if p.dim() > 1 : 
                nn.init.xavier_uniform_(p, gain = 1.0) 

    def forward(self, encoding_poses, decoding_poses):
        """
        :param encoding_poses : tensor of shape (B, encoding_len, pose_dim)
        :param decoding_poses : tensor of shape (B, decoding_len, pose_dim)
        :return : tensor of shape (B, decoding_len, pose_dim) 
        """
        encoder_out     = self.encoder(encoding_poses)
        predicted_poses = self.decoder(encoder_out, decoding_poses)
        out             = self.output_fc(predicted_poses)
        return out

def transformer_loss_function(pred_joints, gt_joints, time_step):
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

def reconstruct_poses(transformer_model, observed_poses, prediction_len):
    """
    :param observed_poses : a tensor of shape (B, train_len, pose_dim)
    :param prediction_len  : the number of prediction frames
    function to inference transformers
    """
    generated_sequence = observed_poses[:, -1:, :].clone()  # (B, 1, pose_dim) 
    encoder_input = observed_poses  
    
    for _ in range(prediction_len):
        output = transformer_model(encoder_input, generated_sequence)
        next_pose = output[:, -1:, :]  # (B, 1, pose_dim)
        generated_sequence = torch.cat([generated_sequence, next_pose], dim = 1)
    
    predicted_poses = generated_sequence[:, 1:, :]
    return predicted_poses
    

if __name__ == "__main__":
    dummy = torch.randn(1, 14, 39).to('cuda') 
    model = PoseTransformer().to('cuda')
    out = model(dummy, dummy)
    print(out.shape)