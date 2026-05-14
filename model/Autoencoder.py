import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

from model.PositionalEncoding import *

'''
Encoding the observed sequences to guide the diffusion model.
'''

class ParallelEncoderLayer(nn.Module):
    """
    Dual attention layer, combining spatial and temporal attention. 
    The spatial attention is applied to each frame independently, while the temporal attention is 
    applied to each joint independently. 
    """
    def __init__(self, 
        num_joints, 
        num_heads, 
        model_dim, 
        fc_hidden_dim, 
        dropout, 
        device = 'cuda', 
    ): 
        """
        :param num_joints          : number of joints in the dataset 
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param pre_normalization   : forward mode, requires normalization afterwards ? 
        """
        super(ParallelEncoderLayer, self).__init__()
        self.num_joints          = num_joints 
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.dropout             = dropout
        
        self.dropout1            = nn.Dropout(self.dropout)
        self.dropout2            = nn.Dropout(self.dropout)
        self.dropout3            = nn.Dropout(self.dropout)
        self.dropout4            = nn.Dropout(self.dropout)

        self.linear1             = nn.Linear(model_dim, fc_hidden_dim) 
        self.linear2             = nn.Linear(fc_hidden_dim, model_dim)

        self.layernorm1          = nn.LayerNorm(self.model_dim)
        self.layernorm2          = nn.LayerNorm(self.model_dim)

        self.relu                = nn.ReLU(inplace = True)
        self.softmax             = nn.Softmax(dim = -1)

        
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.proj_weight = nn.Linear(self.model_dim * 2, 2)
        self.device = device 


    def add_embedding(self, inp, embedding):
        return inp + embedding if embedding is not None else inp 

    
    def forward(self, input, positional_embedding, spatial_embedding):
        """
        :param input                : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
        :param positional_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_embedding    : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        norm_input = self.layernorm1(input)

        batch_size, sequence_length, num_joints, model_dim = input.shape
       
        position_inp = norm_input.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim).clone()
        spatial_inp  = norm_input.contiguous().view(batch_size * sequence_length, num_joints, model_dim).clone()
        
        # Temporal attention
        pos_attn, position_weight = self.temporal_attention(
            query = self.add_positional_embedding(position_inp, positional_embedding), 
            key   = self.add_positional_embedding(position_inp, positional_embedding),  
            value = position_inp, 
            need_weights = True,
        ) 
        pos_attn = pos_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)

        # Spatial attention
        spatial_attn, spatial_weight = self.spatial_attention(
            query = self.add_spatial_embedding(spatial_inp, spatial_embedding), 
            key   = self.add_spatial_embedding(spatial_inp, spatial_embedding), 
            value = spatial_inp,
            need_weights = True
        )  # (B * T, J, D)
        spatial_attn = spatial_attn.contiguous().view(batch_size, sequence_length, num_joints, model_dim)

        
        stacked_self_attn  = torch.cat([pos_attn, spatial_attn], dim = -1) # (B, T, J, 2 * D) 
        stacked_self_attn  = self.proj_weight(stacked_self_attn) # (B, T, J, 2)
        combined_weight    = self.softmax(stacked_self_attn)     # (B, T, J, 2) 
        w_temporal, w_spatial = combined_weight[:, :, :, 0].unsqueeze(-1), combined_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1) 

        attn_out = w_temporal * pos_attn + w_spatial * spatial_attn # (B, T, J, D)

        attn_out = attn_out + input

        norm_attn_out = self.layernorm2(attn_out)
        ffn_out  = self.dropout3(self.relu(self.linear1(norm_attn_out)))
        ffn_out  = self.dropout4(self.linear2(ffn_out))

        out     = attn_out + ffn_out

        return out, [position_weight, spatial_weight]
    
    
class AutoencoderTransformer(nn.Module):
    def __init__(self, 
        num_joints, 
        num_heads, 
        model_dim, 
        fc_hidden_dim, 
        num_layers, 
        dropout, 
        device = 'cuda', 
    ): 
        """
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(AutoencoderTransformer, self).__init__()
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.device              = device 
        self.num_joints          = num_joints
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_encoder_layer(self): 
        layer = ParallelEncoderLayer(
            num_joints        = self.num_joints, 
            num_heads         = self.num_heads, 
            model_dim         = self.model_dim, 
            fc_hidden_dim     = self.fc_hidden_dim,
            dropout           = self.dropout, 
            device            = self.device, 
        )
        return layer
    
    
    def forward(self, src_sequence, pos_embedding): 
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param positional_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        out = src_sequence 
        attn_weight_lst = [] 
        prelist = []
        for index, module in enumerate(self.encoder_layers):
            if index < self.num_layers // 2 : 
                out, attn_weights = module(out, pos_embedding)
                prelist.append(out)
                attn_weight_lst.append(attn_weights)
            else : 
                out, attn_weights = module(out, pos_embedding)
                out = out + prelist[-1]
                prelist.pop()
                attn_weight_lst.append(attn_weights)

        out = self.layernorm(out)

        return out, attn_weight_lst

        
