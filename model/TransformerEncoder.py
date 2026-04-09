"""
Spatial-Temporal Transformer 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_heads, 
                model_dim, 
                fc_hidden_dim, 
                dropout, 
                device            = 'cuda', 
                pre_normalization  = False, 
        ): 
        """
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param pre_normalization   : forward mode, requires normalization afterwards ? 
        """
        super(TransformerEncoderLayer, self).__init__() 
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.dropout             = dropout
        
        self.dropout1            = nn.Dropout(self.dropout)
        self.dropout2            = nn.Dropout(self.dropout)
        self.dropout3            = nn.Dropout(self.dropout)

        self.linear1             = nn.Linear(model_dim, fc_hidden_dim) 
        self.linear2             = nn.Linear(fc_hidden_dim, model_dim)

        self.layernorm1          = nn.LayerNorm(self.model_dim)
        self.layernorm2          = nn.LayerNorm(self.model_dim)

        self.relu                = nn.ReLU(inplace = True)

        
        self._self_attention      = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )
        
        self.device              = device 
        self.pre_normalization   = pre_normalization


    def add_positional_embedding(self, inp, pos_embedding):
        if pos_embedding is not None :
            return inp + pos_embedding
        return inp  
    
    def forward_post_norm(self, src_sequence, pos_embedding):
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param pos_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        query = key = self.add_positional_embedding(src_sequence, pos_embedding)
        value = src_sequence

        self_attn_out, self_attn_weight = self._self_attention(query, key, value, need_weights = True)

        src_sequence = src_sequence + self.dropout1(self_attn_out)

        src_sequence = self.layernorm1(src_sequence)

        curr_sequence = self.linear1(src_sequence) 
        curr_sequence = self.relu(curr_sequence)
        curr_sequence = self.dropout2(curr_sequence) 
        
        curr_sequence = self.linear2(curr_sequence) 
        src_sequence  = src_sequence + self.dropout3(curr_sequence)
        output        = self.layernorm2(src_sequence)

        return output, self_attn_weight
    
    def forward_pre_norm(self, src_sequence, pos_embedding):
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param pos_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        src_sequence = self.layernorm1(src_sequence)
        
        query = key = self.add_positional_embedding(src_sequence, pos_embedding)
        value = src_sequence

        self_attn_out, self_attn_weight = self._self_attention(query, key, value, need_weights = True)

        src_sequence = src_sequence + self.dropout1(self_attn_out)
        src_sequence = self.layernorm2(src_sequence)

        curr_sequence  = self.linear1(src_sequence)
        curr_sequence  = self.relu(curr_sequence) 
        curr_sequence  = self.dropout2(curr_sequence) 

        curr_sequence  = self.linear2(curr_sequence)
        output         =  src_sequence + self.dropout3(curr_sequence)
        return output, self_attn_weight

    
    def forward(self, src_sequence, pos_embedding): 
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param pos_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization : 
            return self.forward_pre_norm(src_sequence, pos_embedding) 
        return self.forward_post_norm(src_sequence, pos_embedding)
    
   
class TransformerParallelDualEncoderLayer(nn.Module):
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
            device            = 'cuda', 
            pre_normalization = False, 
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
        super(TransformerParallelDualEncoderLayer, self).__init__()
        self.num_joints          = num_joints 
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.dropout             = dropout
        
        self.dropout1            = nn.Dropout(self.dropout)
        self.dropout2            = nn.Dropout(self.dropout)
        self.dropout3            = nn.Dropout(self.dropout)
        self.dropout4            = nn.Dropout(self.dropout)
        self.dropout5            = nn.Dropout(self.dropout)

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
        
        self.device              = device 
        self.pre_normalization   = pre_normalization


    def add_temporal_embedding(self, inp, temporal_embedding):
        if temporal_embedding is not None :
            return inp + temporal_embedding
        return inp  
    
    def add_spatial_embedding(self, inp, spatial_embedding):
        if spatial_embedding is not None : 
            return inp + spatial_embedding 
        return inp 
    
    
    def forward_post_norm(self, src_sequence, temporal_embedding, spatial_embedding):
        """
        :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
        :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        batch_size, sequence_length, num_joints, model_dim = src_sequence.shape
        temporal_inp = src_sequence.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
        spatial_inp  = src_sequence.contiguous().view(batch_size * sequence_length, num_joints, model_dim) 
        
        # Temporal attention
        query = key = self.add_temporal_embedding(temporal_inp, temporal_embedding)
        value = temporal_inp
        temporal_self_attn, temporal_weight = self.temporal_attention(query, key, value, need_weights = True) # (B * J, T, D)
        temporal_self_attn = self.dropout1(temporal_self_attn)

        # Spatial attention
        query = key = self.add_spatial_embedding(spatial_inp, spatial_embedding)
        value = spatial_inp
        spatial_self_attn, spatial_weight  = self.spatial_attention(query, key, value, need_weights = True)  # (B * T, J, D)
        spatial_self_attn = self.dropout2(spatial_self_attn)
      
        temporal_self_attn = temporal_self_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)
        spatial_self_attn  = spatial_self_attn.contiguous().view(batch_size, sequence_length, num_joints, model_dim)

        
        stacked_self_attn  = torch.cat([temporal_self_attn, spatial_self_attn], dim = -1) # (B, T, J, 2 * D) 
        stacked_self_attn  = self.proj_weight(stacked_self_attn) # (B, T, J, 2)
        combined_weight    = self.softmax(stacked_self_attn)     # (B, T, J, 2) 

        w_temporal, w_spatial = combined_weight[:, :, :, 0].unsqueeze(-1), combined_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1) 

        combined_self_attn = w_temporal * temporal_self_attn + w_spatial * spatial_self_attn # (B, T, J, D)

        src_sequence = src_sequence + self.dropout3(combined_self_attn)

        src_sequence = self.layernorm1(src_sequence)

        curr_sequence = self.linear1(src_sequence) 
        curr_sequence = self.relu(curr_sequence)
        curr_sequence = self.dropout4(curr_sequence) 
        
        curr_sequence = self.linear2(curr_sequence) 
        src_sequence  = src_sequence + self.dropout5(curr_sequence)
        output        = self.layernorm2(src_sequence)

        return output, [temporal_weight, spatial_weight]
    
    def forward_pre_norm(self, src_sequence, temporal_embedding, spatial_embedding):
        """
        :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
        :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        src_sequence = self.layernorm1(src_sequence)

        batch_size, sequence_length, num_joints, model_dim = src_sequence.shape

        temporal_inp = src_sequence.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
        spatial_inp  = src_sequence.contiguous().view(batch_size * sequence_length, num_joints, model_dim) 
        
        # Temporal attention
        query = key = self.add_temporal_embedding(temporal_inp, temporal_embedding)
        value = temporal_inp
        temporal_self_attn, temporal_weight = self.temporal_attention(query, key, value, need_weights = True) # (B * J, T, D)
        
        # Spatial attention
        query = key = self.add_spatial_embedding(spatial_inp, spatial_embedding)
        value = spatial_inp
        spatial_self_attn, spatial_weight  = self.spatial_attention(query, key, value, need_weights = True)  # (B * T, J, D)

        temporal_self_attn = temporal_self_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)
        spatial_self_attn  = spatial_self_attn.contiguous().view(batch_size, sequence_length, num_joints, model_dim)

        stacked_self_attn  = torch.cat([temporal_self_attn, spatial_self_attn], dim = -1) # (B, T, J, 2 * D) 
        stacked_self_attn  = self.proj_weight(stacked_self_attn) # (B, T, J, 2)
        combined_weight    = self.softmax(stacked_self_attn)     # (B, T, J, 2) 

        w_temporal, w_spatial = combined_weight[:, :, :, 0].unsqueeze(-1), combined_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1) 

        combined_self_attn = w_temporal * temporal_self_attn + w_spatial * spatial_self_attn # (B, T, J, D)
        
        src_sequence = src_sequence + self.dropout1(combined_self_attn)
        

        src_sequence = self.layernorm2(src_sequence)

        curr_sequence  = self.linear1(src_sequence)
        curr_sequence  = self.relu(curr_sequence) 
        curr_sequence  = self.dropout2(curr_sequence) 

        curr_sequence  = self.linear2(curr_sequence)
        output         =  src_sequence + self.dropout3(curr_sequence)
        
        return output, [temporal_weight, spatial_weight]
    
    def forward(self, src_sequence, temporal_embedding, spatial_embedding): 
        """
        :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
        :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization : 
            return self.forward_pre_norm(src_sequence, temporal_embedding, spatial_embedding) 
        return self.forward_post_norm(src_sequence, temporal_embedding, spatial_embedding)


# Frozen this idea for now 
# class TransformerStackedDualEncoder(nn.Module):
#     """
#     Stacked dual attention layer, combining spatial and temporal attention. 
#     The spatial attention is applied to each frame independently, while the temporal attention is 
#     applied to each joint independently. 
#     """
#     def __init__(self, 
#             num_joints, 
#             num_heads, 
#             model_dim, 
#             fc_hidden_dim, 
#             dropout, 
#             device            = 'cuda', 
#             pre_normalization = False, 
#         ): 
#         """
#         :param num_joints          : number of joints in the dataset 
#         :param num_heads           : number of attention heads 
#         :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
#         :param fc_hidden_dim       : fully connected layer hidden dim 
#         :param num_layers          : number of (MultiheadAttn + FC) layers 
#         :param dropout             : dropout probability 
#         :param pre_normalization   : forward mode, requires normalization afterwards ? 
#         """
#         super(TransformerStackedDualEncoder, self).__init__()
#         self.num_joints          = num_joints 
#         self.num_heads           = num_heads 
#         self.model_dim           = model_dim
#         self.fc_hidden_dim       = fc_hidden_dim 
#         self.dropout             = dropout
        
#         self.dropout1            = nn.Dropout(self.dropout)
#         self.dropout2            = nn.Dropout(self.dropout)
#         self.dropout3            = nn.Dropout(self.dropout)
#         self.dropout4            = nn.Dropout(self.dropout)
#         self.dropout5            = nn.Dropout(self.dropout)
#         self.dropout6            = nn.Dropout(self.dropout)


#         self.linear1             = nn.Linear(model_dim, fc_hidden_dim) 
#         self.linear2             = nn.Linear(fc_hidden_dim, model_dim)

#         self.layernorm1          = nn.LayerNorm(self.model_dim)
#         self.layernorm2          = nn.LayerNorm(self.model_dim)
#         self.layernorm3          = nn.LayerNorm(self.model_dim)
#         self.layernorm4          = nn.LayerNorm(self.model_dim)
#         self.layernorm5          = nn.LayerNorm(self.model_dim)
#         self.layernorm6          = nn.LayerNorm(self.model_dim)

#         self.relu                = nn.ReLU(inplace = True)
#         self.softmax             = nn.Softmax(dim = -1)

        
#         self.spatial_attention = nn.MultiheadAttention(
#             embed_dim   = self.model_dim, 
#             num_heads   = self.num_heads, 
#             dropout     = self.dropout, 
#             batch_first = True, 
#         )

#         self.temporal_attention = nn.MultiheadAttention(
#             embed_dim   = self.model_dim, 
#             num_heads   = self.num_heads, 
#             dropout     = self.dropout, 
#             batch_first = True, 

#         )

#         self.proj_weight = nn.Linear(self.model_dim * 2, 2)
        
#         self.device              = device 
#         self.pre_normalization   = pre_normalization


#     def add_temporal_embedding(self, inp, temporal_embedding):
#         if temporal_embedding is not None :
#             return inp + temporal_embedding
#         return inp  
    
#     def add_spatial_embedding(self, inp, spatial_embedding):
#         if spatial_embedding is not None : 
#             return inp + spatial_embedding 
#         return inp 
    

#     def temporal_spatial_attention(self, src_sequence, temporal_embedding, spatial_embedding):
#         batch_size, sequence_length, num_joints, model_dim = src_sequence.shape
        
#         # Temporal attention
#         temporal_inp = src_sequence.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
#         query = key = self.add_temporal_embedding(temporal_inp, temporal_embedding)
#         value = temporal_inp
#         temporal_self_attn, temporal_weight = self.temporal_attention(query, key, value, need_weights = True) # (B * J, T, D)
#         out_temporal_attn = temporal_inp + self.dropout1(temporal_self_attn)
#         out_temporal_attn = self.layernorm1(out_temporal_attn)
#         out_temporal_attn = out_temporal_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3) # (B, T, J, D) 


#         # Spatial attention
#         spatial_inp  = out_temporal_attn.contiguous().view(batch_size * sequence_length, num_joints, model_dim) 
#         query = key = self.add_spatial_embedding(spatial_inp, spatial_embedding)
#         value = spatial_inp
#         spatial_self_attn, spatial_weight  = self.spatial_attention(query, key, value, need_weights = True)  # (B * T, J, D)
#         out_spatial_attn = spatial_inp + self.dropout2(spatial_self_attn)
#         out = self.layernorm2(out_spatial_attn)
#         out = out.contiguous().view(batch_size, sequence_length, num_joints, model_dim)
        
#         return out, [temporal_weight, spatial_weight]
    
#     def spatial_temporal_attention(self, src_sequence, temporal_embedding, spatial_embedding):
#         batch_size, sequence_length, num_joints, model_dim = src_sequence.shape
        
#         # Spatial attention
#         spatial_inp  = src_sequence.contiguous().view(batch_size * sequence_length, num_joints, model_dim) 
#         query = key  = self.add_spatial_embedding(spatial_inp, spatial_embedding)
#         value = spatial_inp
#         spatial_self_attn, spatial_weight  = self.spatial_attention(query, key, value, need_weights = True)  # (B * T, J, D)
#         out_spatial_attn = spatial_inp + self.dropout3(spatial_self_attn)
#         out_spatial_attn = self.layernorm3(out_spatial_attn)
#         out_spatial_attn = out_spatial_attn.contiguous().view(batch_size, sequence_length, num_joints, model_dim)
        
#         # Temporal attention
#         temporal_inp = out_spatial_attn.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
#         query = key = self.add_temporal_embedding(temporal_inp, temporal_embedding)
#         value = temporal_inp
#         temporal_self_attn, temporal_weight = self.temporal_attention(query, key, value, need_weights = True) # (B * J, T, D)
#         out_temporal_attn = temporal_inp + self.dropout4(temporal_self_attn)
#         out_temporal_attn = self.layernorm4(out_temporal_attn)
#         out = out_temporal_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3) # (B, T, J, D)
        
#         return out, [temporal_weight, spatial_weight]

#     def forward_post_norm(self, src_sequence, temporal_embedding, spatial_embedding):
#         """
#         :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
#         :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
#         :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
#         :return 
#             output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
#             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
#         """

#         out_ts, [temporal_weight_ts, spatial_weight_ts] = self.temporal_spatial_attention(src_sequence, temporal_embedding, spatial_embedding) 
#         out_st, [temporal_weight_st, spatial_weight_st] = self.spatial_temporal_attention(src_sequence, temporal_embedding, spatial_embedding)

#         stack_out = torch.cat([out_ts, out_st], dim = -1) # (B, T, J, 2 * D) 
#         stack_weight = self.proj_weight(stack_out) # (B, T, J, 2)
#         stack_weight = self.softmax(stack_weight)   # (B, T, J, 2)
#         w_ts, w_st = stack_weight[:, :, :, 0].unsqueeze(-1), stack_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1)

#         src_sequence = w_ts * out_ts + w_st * out_st 
#         src_sequence = self.layernorm5(src_sequence)

#         curr_sequence = self.linear1(src_sequence) 
#         curr_sequence = self.relu(curr_sequence)
#         curr_sequence = self.dropout5(curr_sequence) 
        
#         curr_sequence = self.linear2(curr_sequence) 
#         src_sequence  = src_sequence + self.dropout6(curr_sequence)
#         output        = self.layernorm6(src_sequence)

#         return output, [temporal_weight_ts, spatial_weight_ts, temporal_weight_st, spatial_weight_st] 
    
#     def forward_pre_norm(self, src_sequence, temporal_embedding, spatial_embedding):
#         """
#         :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
#         :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
#         :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
#         :return 
#             output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
#             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
#         """
#         src_sequence = self.layernorm5(src_sequence)

#         batch_size, sequence_length, num_joints, model_dim = src_sequence.shape

#         out_ts, [temporal_weight_ts, spatial_weight_ts] = self.temporal_spatial_attention(src_sequence, temporal_embedding, spatial_embedding) 
#         out_st, [temporal_weight_st, spatial_weight_st] = self.spatial_temporal_attention(src_sequence, temporal_embedding, spatial_embedding)

#         stack_out = torch.cat([out_ts, out_st], dim = -1) # (B, T, J, 2 * D) 
#         stack_weight = self.proj_weight(stack_out)  # (B, T, J, 2)
#         stack_weight = self.softmax(stack_weight)   # (B, T, J, 2)
#         w_ts, w_st = stack_weight[:, :, :, 0].unsqueeze(-1), stack_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1)

#         src_sequence = w_ts * out_ts + w_st * out_st 

#         src_sequence = self.layernorm6(src_sequence)

#         curr_sequence  = self.linear1(src_sequence)
#         curr_sequence  = self.relu(curr_sequence) 
#         curr_sequence  = self.dropout5(curr_sequence) 

#         curr_sequence  = self.linear2(curr_sequence)
#         output         =  src_sequence + self.dropout6(curr_sequence)

#         return output, [temporal_weight_ts, spatial_weight_ts, temporal_weight_st, spatial_weight_st]
    
#     def forward(self, src_sequence, temporal_embedding, spatial_embedding): 
#         """
#         :param src_sequence       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim)
#         :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
#         :param spatial_embedding  : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
#         :return 
#              output       : a tensor of shape (batch_size, sequence_length, model_dim) 
#              attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
#         """
#         if self.pre_normalization : 
#             return self.forward_pre_norm(src_sequence, temporal_embedding, spatial_embedding) 
#         return self.forward_post_norm(src_sequence, temporal_embedding, spatial_embedding)
  

class TransformerEncoder(nn.Module):
    def __init__(self, 
            num_joints, 
            num_heads, 
            model_dim, 
            fc_hidden_dim, 
            num_layers, 
            dropout, 
            device = 'cuda', 
            pre_normalization = False, 
            attention_mode    = 'dual'
        ): 
        """
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(TransformerEncoder, self).__init__()
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.device              = device 
        self.pre_normalization   = pre_normalization
        self.num_joints          = num_joints
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.attention_mode      = attention_mode
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_encoder_layer(self): 
        if self.attention_mode == 'parallel_dual' : 
            layer = TransformerParallelDualEncoderLayer(
                num_joints        = self.num_joints, 
                num_heads         = self.num_heads, 
                model_dim         = self.model_dim, 
                fc_hidden_dim     = self.fc_hidden_dim,
                dropout           = self.dropout, 
                device            = self.device, 
                pre_normalization = self.pre_normalization, 
            )
            return layer
    
        elif self.attention_mode == 'vanilla' : 
            layer = TransformerEncoderLayer(
                num_heads         = self.num_heads, 
                model_dim         = self.model_dim, 
                fc_hidden_dim     = self.fc_hidden_dim,
                dropout           = self.dropout, 
                device            = self.device, 
                pre_normalization = self.pre_normalization
            )
            return layer
        else : 
            raise NotImplementedError(f"Attention mode {self.attention_mode} not implemented for encoder") 
    

    def forward_dual(self, src_sequence, temporal_embedding, spatial_embedding): 
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param temporal_embedding : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_embedding : a tensor of shape (1, num_joints, model_dim),       getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        out = src_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.encoder_layers[i](out, temporal_embedding, spatial_embedding) 
            attn_weight_lst.append(attn_weights)

        if self.pre_normalization : 
            out = self.layernorm(out)

        return out, attn_weight_lst
    
    def forward_single(self, src_sequence, temporal_embedding): 
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param temporal_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        out = src_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.encoder_layers[i](out, temporal_embedding) 
            attn_weight_lst.append(attn_weights)

        if self.pre_normalization : 
            out = self.layernorm(out)

        return out, attn_weight_lst
    
    def forward(self, src_sequence, temporal_embedding = None, spatial_embedding = None):
        if self.attention_mode in 'parallel_duel':
            return self.forward_dual(src_sequence, temporal_embedding, spatial_embedding)
        else : 
            return self.forward_single(src_sequence, temporal_embedding)



