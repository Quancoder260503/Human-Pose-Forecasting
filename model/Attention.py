import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

"""
Attention module (MultiHeadAttention + Feed Forward Network)
"""

class MultiHeadAttention(nn.Module): 
    def __init__(self, num_heads, dim_model, dim_queries, dim_values, dropout, self_attention = False, in_decoder = False, device = 'cuda'): 
        """
        :param num_heads   : number of attention heads 
        :param dim_model   : model dimension
        :param dim_queries : query dimension 
        :param dim_values  : value dimension 
        :param dropout     : dropout probability 
        :param in_decoder  : in_decoder phase 
        """

        super(MultiHeadAttention, self).__init__() 
        self.num_heads      = num_heads 
        self.dim_model      = dim_model 
        self.dim_queries    = dim_queries 
        self.dim_values     = dim_values 
        self.dropout        = nn.Dropout(dropout)
        self.in_decoder     = in_decoder
        self.proj_q         = nn.Linear(dim_model, num_heads * dim_queries) 
        self.proj_k         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_v         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_o         = nn.Linear(num_heads * dim_values, dim_model)
        self.layernorm      = nn.LayerNorm(dim_model)
        self.softmax        = nn.Softmax(dim = -1) 
        self.self_attention = self_attention
        self.device         = device 

    def forward(self, q_poses, kv_poses): 
        """
        :param q_poses : input query sequences, a tensor of shape (B, query_len, dim_model) 
        :param kv_poses : input_kv   sequences, a tensor of shape (B, keyval_len, dim_model)  
        :return 
            a tensor of shape (B, query_len, dim_model)
        """
        batch_size = q_poses.shape[0] 
        query_len  = q_poses.shape[1]
        keyval_len = kv_poses.shape[1]
        
        self_attention = self.self_attention
        
        curr_q_poses = q_poses.clone() # (B, J, dim_model) 
        q_poses = self.layernorm(q_poses)   
        
        # if not self attention, then no need to normalize since the encoder has already done it 
        if self_attention:
            kv_poses = self.layernorm(kv_poses) 

        query = self.proj_q(q_poses)  # (B, query_len, query_dim * num_heads) 
        key   = self.proj_k(kv_poses) # (B, query_len, query_dim * num_heads)
        val   = self.proj_v(kv_poses) # (B, query_len, value_dim * num_heads) 

        query = query.contiguous().view(batch_size, query_len, self.num_heads, -1) # (B, query_len, num_heads, query_dim) 
        key   = key.contiguous().view(batch_size, keyval_len,  self.num_heads, -1) # (B, keyval_len, num_heads, query_dim)
        val   = val.contiguous().view(batch_size, keyval_len,  self.num_heads, -1) # (B, keyval_len, num_heads, value_dim)


        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, query_len,  -1) # (B * n_heads, query_len,  query_dim) 
        key   = key.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   keyval_len, -1) # (B * n_heads, keyval_len, query_dim) 
        val   = val.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   keyval_len, -1) # (B * n_heads, keyval_len, value_dim)

        attention_weights = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.dim_queries) # (B * n_heads,  query_len, keyval_len) 
        
        if self.in_decoder and self_attention : 
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(device = self.device) # Lower triangle matrix (actually use this to filter out the future) 
            attention_weights = attention_weights.masked_fill(~not_future_mask, -1e9) 

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights) 

        attn_values = torch.bmm(attention_weights, val)  # (B * n_heads, query_len, value_dim)
        attn_values = attn_values.contiguous().view(batch_size, self.num_heads, query_len, self.dim_values)
        attn_values = attn_values.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, -1) # (B, J, n_heads * value_dim)
        out_poses = self.proj_o(attn_values) # (B, query_len, dim_model) 
        out_poses = self.dropout(out_poses) + curr_q_poses # (B, query_len, dim_model) 
        return out_poses


class PointWiseFFN(nn.Module): 
    """
    Position-wise feed forward network layer 
    """
    def __init__(self, d_model, d_inner, dropout): 
        """
        :param d_model : size of the vectors throughout the transformer model
        :param d_inner : hidden size 
        :param dropout : dropout prob ? 
        """
        super(PointWiseFFN, self).__init__()
        self.d_model = d_model 
        self.d_inner = d_inner 
        self.layer_norm = nn.LayerNorm(d_model) 
        self.fc_1       = nn.Linear(d_model, d_inner) 
        self.relu       = nn.ReLU(inplace = True) 
        self.fc_2       = nn.Linear(d_inner, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, poses): 
        ''' 
        :param poses : a tensor of shape (batch_size, query_len, dim_model)
        '''
        prev_poses = poses.clone() 
        poses      = self.layer_norm(poses) 
        
        # First layer 
        poses = self.fc_1(poses) # (batch_size, query_len, d_inner) 
        poses = self.relu(poses)
        poses = self.dropout(poses) 

        # Second Layer 
        poses = self.fc_2(poses) # (batch_size, query_len, dim_model) 
        poses = self.dropout(poses) + prev_poses 
        return poses 