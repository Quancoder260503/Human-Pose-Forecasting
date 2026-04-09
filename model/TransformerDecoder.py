import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt

class TransformerDecoderLayer(nn.Module):
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
        super(TransformerDecoderLayer, self).__init__() 
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
        self.layernorm3          = nn.LayerNorm(self.model_dim)

        self.relu                = nn.ReLU(inplace = True)

        
        self._self_attention      = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self._cross_attention    = nn.MultiheadAttention(
            embed_dim   =  self.model_dim, 
            num_heads   =  self.num_heads, 
            dropout     =  self.dropout, 
            batch_first = True, 
        )
        
        self.device              = device 
        self.pre_normalization   = pre_normalization


    def add_positional_embedding(self, inp, pos_embedding):
        if pos_embedding is not None :
            inp = inp + pos_embedding 
        return inp  
    
    def forward_post_norm(self, memory, tgt_sequence, pos_embedding, query_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim) 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param pos_embedding   : a tensor of shape (batch_size, pos_sequence_length, 1), getting broadcasted 
        :param query_embedding : a tensor of shape (batch_size, tgt_sequence_length, 1), getting broadcasted
        :return 
            output       : a tensor of shape (batch_size, sequence_length, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """

        # Self-attention 
        key = query = self.add_positional_embedding(tgt_sequence, query_embedding)
        value = tgt_sequence 

        self_attn_out, self_attn_weight = self._self_attention(query, key, value, need_weights = True)

        tgt_sequence = tgt_sequence + self.dropout1(self_attn_out)

        tgt_sequence = self.layernorm1(tgt_sequence)

        key   = self.add_positional_embedding(memory, pos_embedding)
        query = self.add_positional_embedding(tgt_sequence, query_embedding)
        value = memory 

        cross_attn_out, cross_attn_weight = self._cross_attention(query, key, value, need_weights = True)

        tgt_sequence = tgt_sequence + self.dropout2(cross_attn_out)

        tgt_sequence = self.layernorm2(tgt_sequence)


        curr_sequence = self.linear1(tgt_sequence) 
        curr_sequence = self.relu(curr_sequence)
        curr_sequence = self.dropout3(curr_sequence) 
        
        curr_sequence = self.linear2(curr_sequence) 
        src_sequence  = tgt_sequence + self.dropout4(curr_sequence)
        
        output        = self.layernorm3(src_sequence)
        return output, [self_attn_weight, cross_attn_weight]
    
    
    def forward_pre_norm(self, memory, tgt_sequence, pos_embedding, query_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim), encoder output 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :param pos_embedding   : a tensor of shape (batch_size, pos_sequence_length, 1), getting broadcasted 
        :param query_embedding : a tensor of shape (batch_size, tgt_sequence_length, 1), getting broadcasted
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             self_attn_weights  : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
             cross_attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length)
        """
        
        # Self-attention 
        tgt_sequence = self.layernorm1(tgt_sequence)
        
        query = key = self.add_positional_embedding(tgt_sequence, query_embedding)
        value = tgt_sequence

        self_attn_out, self_attn_weight = self._self_attention(query, key, value, need_weights = True)

        tgt_sequence = tgt_sequence + self.dropout1(self_attn_out)
        tgt_sequence = self.layernorm2(tgt_sequence)

        query = self.add_positional_embedding(tgt_sequence, query_embedding) 
        key   = self.add_positional_embedding(memory, pos_embedding)
        value = memory 
        
        cross_attn_out, cross_attn_weight = self._cross_attention(query, key, value, need_weights = True)

        tgt_sequence = tgt_sequence + self.dropout2(cross_attn_out)
        tgt_sequence = self.layernorm3(tgt_sequence)


        curr_sequence  = self.linear1(tgt_sequence)
        curr_sequence  = self.relu(curr_sequence) 
        curr_sequence  = self.dropout3(curr_sequence) 

        curr_sequence  = self.linear2(curr_sequence)
        output         = tgt_sequence + self.dropout4(curr_sequence)
        return output, [self_attn_weight, cross_attn_weight]


    
    def forward(self, memory, tgt_sequence, pos_embedding, query_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim) 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :param pos_embedding   : a tensor of shape (1, pos_sequence_length, model_dim), getting broadcasted 
        :param query_embedding : a tensor of shape (1, tgt_sequence_length, model_dim), getting broadcasted
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization : 
            return self.forward_pre_norm(memory, tgt_sequence, pos_embedding, query_embedding) 
        return self.forward_post_norm(memory, tgt_sequence, pos_embedding, query_embedding)
       
class TransformerParallelDualDecoderLayer(nn.Module):
    """
    Dual attention layer, combining spatial and temporal attention. 
    The spatial attention is applied to each frame independently, while the 
    temporal attention is applied to each joint independently. 
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
        super(TransformerParallelDualDecoderLayer, self).__init__()
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
        self.dropout6            = nn.Dropout(self.dropout)
        self.dropout7            = nn.Dropout(self.dropout)
    

        self.linear1             = nn.Linear(model_dim, fc_hidden_dim) 
        self.linear2             = nn.Linear(fc_hidden_dim, model_dim)

        self.layernorm1          = nn.LayerNorm(self.model_dim)
        self.layernorm2          = nn.LayerNorm(self.model_dim)
        self.layernorm3          = nn.LayerNorm(self.model_dim)

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

        self.temporal_cross_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.proj_self_weight = nn.Linear(self.model_dim * 2, 2)
        self.proj_cross_weight = nn.Linear(self.model_dim * 2, 2)

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
    
    def self_attention_modules(
        self, 
        tgt_sequences, 
        temporal_query_embedding, 
        spatial_query_embedding
    ):

        batch_size, sequence_length, num_joints, model_dim = tgt_sequences.shape

        temporal_inp = tgt_sequences.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
        spatial_inp  = tgt_sequences.contiguous().view(batch_size * sequence_length, num_joints, model_dim) 
        
        # Temporal attention
        query = key = self.add_temporal_embedding(temporal_inp, temporal_query_embedding)
        value = temporal_inp
        temporal_self_attn, temporal_self_weight = self.temporal_attention(query, key, value, need_weights = True) # (B * J, T, D)
        temporal_self_attn = self.dropout1(temporal_self_attn)

        # Spatial attention
        query = key = self.add_spatial_embedding(spatial_inp, spatial_query_embedding)
        value = spatial_inp
        spatial_self_attn, spatial_self_weight  = self.spatial_attention(query, key, value, need_weights = True)  # (B * T, J, D)
        spatial_self_attn = self.dropout2(spatial_self_attn)
      
        temporal_self_attn = temporal_self_attn.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)
        spatial_self_attn  = spatial_self_attn.contiguous().view(batch_size, sequence_length, num_joints, model_dim)
        
        stacked_self_attn  = torch.cat([temporal_self_attn, spatial_self_attn], dim = -1) # (B, T, J, 2 * D) 
        stacked_self_attn  = self.proj_self_weight(stacked_self_attn) # (B, T, J, 2)
        combined_weight    = self.softmax(stacked_self_attn)     # (B, T, J, 2) 

        w_temporal, w_spatial = combined_weight[:, :, :, 0].unsqueeze(-1), combined_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1) 

        combined_self_attn = w_temporal * temporal_self_attn + w_spatial * spatial_self_attn # (B, T, J, D)
        return combined_self_attn, [temporal_self_weight, spatial_self_weight]
    
    def cross_attention_modules( 
        self, 
        memory, 
        tgt_sequences, 
        encoder_spatial_embedding, 
        encoder_temporal_embedding, 
        spatial_query_embedding, 
        temporal_query_embedding
    ):
        
        batch_size, tgt_sequence_length, num_joints, model_dim = tgt_sequences.shape
        batch_size, src_sequence_length, _, _                  = memory.shape 
        

        temporal_inp = tgt_sequences.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, tgt_sequence_length, model_dim)
        spatial_inp  = tgt_sequences.contiguous().view(batch_size * tgt_sequence_length, num_joints, model_dim) 

        temporal_memory = memory.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, src_sequence_length, model_dim)
        spatial_memory  = memory.contiguous().view(batch_size * src_sequence_length, num_joints, model_dim)
        
        # Temporal Cross Attention
        query = self.add_temporal_embedding(temporal_inp,      temporal_query_embedding) 
        key   = self.add_temporal_embedding(temporal_memory, encoder_temporal_embedding)
        value = temporal_memory
        temporal_cross_attn, temporal_cross_weight = self.temporal_cross_attention(query, key, value, need_weights = True)
        temporal_cross_attn = self.dropout3(temporal_cross_attn)
        
        # Spatial Cross Attention 
        query = self.add_spatial_embedding(spatial_inp, spatial_query_embedding)
        key   = self.add_spatial_embedding(spatial_memory, encoder_spatial_embedding)
        value = spatial_memory
        spatial_cross_attn, spatial_cross_weight = self.spatial_cross_attention(query, key, value, need_weights = True)
        spatial_cross_attn = self.dropout4(spatial_cross_attn)

        stacked_cross_attn  = torch.cat([temporal_cross_attn, spatial_cross_attn], dim = -1) # (B, T, J, 2 * D) 
        stacked_cross_attn  = self.proj_cross_weight(stacked_cross_attn) # (B, T, J, 2)
        combined_weight    = self.softmax(stacked_cross_attn)            # (B, T, J, 2) 

        w_temporal, w_spatial = combined_weight[:, :, :, 0].unsqueeze(-1), combined_weight[:, :, :, 1].unsqueeze(-1) # (B, T, J, 1) 

        combined_cross_attn = w_temporal * temporal_cross_attn + w_spatial * spatial_cross_attn # (B, T, J, D)
        return combined_cross_attn, [temporal_cross_weight, spatial_cross_weight]

    def forward_post_norm(self, 
            memory, 
            tgt_sequences, 
            encoder_spatial_embedding, 
            encoder_temporal_embedding, 
            spatial_query_embedding, 
            temporal_query_embedding
        ):
        """
        :param memory             : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences       : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_spatial_embedding  : a tensor of shape (1, num_joints, model_dim) 
        :param encoder_temporal_embedding : a tensor of shape (1, sequence_length, model_dim) 
        :param temporal_query_embedding   : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_query_embedding    : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        # self-attention 
        self_attn_out, self_attn_weight = self.self_attention_modules(
           tgt_sequences            = tgt_sequences, 
           temporal_query_embedding = temporal_query_embedding, 
           spatial_query_embedding  = spatial_query_embedding, 
        )

        tgt_sequences = tgt_sequences + self.dropout5(self_attn_out)
        tgt_sequences = self.layernorm1(tgt_sequences)

        # cross-attention
        cross_attn_out, cross_attn_weight = self.cross_attention_modules(
            memory                      = memory, 
            tgt_sequences               = tgt_sequences, 
            encoder_spatial_embedding   = encoder_spatial_embedding, 
            encoder_temporal_embedding  = encoder_temporal_embedding, 
            temporal_query_embedding    = temporal_query_embedding, 
            spatial_query_embedding     = spatial_query_embedding, 
        )

        tgt_sequences = tgt_sequences + self.dropout6(cross_attn_out)
        tgt_sequences = self.layernorm2(tgt_sequences)


        # FFN   
        curr_sequence = self.linear1(tgt_sequences) 
        curr_sequence = self.relu(curr_sequence)
        curr_sequence = self.dropout6(curr_sequence) 
        
        curr_sequence = self.linear2(curr_sequence) 
        tgt_sequences = tgt_sequences + self.dropout7(curr_sequence)
        output        = self.layernorm3(tgt_sequences)

        return output, [self_attn_weight, cross_attn_weight]

    def forward_pre_norm(self, 
            memory, 
            tgt_sequences, 
            encoder_spatial_embedding, 
            encoder_temporal_embedding, 
            spatial_query_embedding, 
            temporal_query_embedding
        ):
        """
        :param memory             : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences       : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_spatial_embedding  : a tensor of shape (1, num_joints, model_dim) 
        :param encoder_temporal_embedding : a tensor of shape (1, sequence_length, model_dim) 
        :param temporal_query_embedding   : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_query_embedding    : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """

        # Self-attention
        tgt_sequences = self.layernorm1(tgt_sequences)
        self_attn_out, self_attn_weight = self.self_attention_modules(
           tgt_sequences            = tgt_sequences, 
           temporal_query_embedding = temporal_query_embedding, 
           spatial_query_embedding  = spatial_query_embedding, 
        )
        tgt_sequences = tgt_sequences + self.dropout5(self_attn_out)

        # Cross-attention
        tgt_sequences = self.layernorm2(tgt_sequences)
        cross_attn_out, cross_attn_weight = self.cross_attention_modules(
            memory                      = memory, 
            tgt_sequences               = tgt_sequences, 
            encoder_spatial_embedding   = encoder_spatial_embedding, 
            encoder_temporal_embedding  = encoder_temporal_embedding, 
            temporal_query_embedding    = temporal_query_embedding, 
            spatial_query_embedding     = spatial_query_embedding, 
        )
        tgt_sequences = tgt_sequences + self.dropout6(cross_attn_out)
        
        # FFN 
        tgt_sequences = self.layernorm3(tgt_sequences)
          
        curr_sequence = self.linear1(tgt_sequences) 
        curr_sequence = self.relu(curr_sequence)
        curr_sequence = self.dropout6(curr_sequence) 
        
        curr_sequence = self.linear2(curr_sequence) 
        output        = tgt_sequences + self.dropout7(curr_sequence)

        return output, [self_attn_weight, cross_attn_weight]


    def forward(self,             
            memory, 
            tgt_sequences, 
            encoder_spatial_embedding, 
            encoder_temporal_embedding, 
            spatial_query_embedding, 
            temporal_query_embedding
        ):
        """
        :param memory             : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences       : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_spatial_embedding  : a tensor of shape (1, num_joints, model_dim) 
        :param encoder_temporal_embedding : a tensor of shape (1, sequence_length, model_dim) 
        :param temporal_query_embedding   : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_query_embedding    : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization:
            return self.forward_pre_norm(
                memory                      = memory, 
                tgt_sequences               = tgt_sequences, 
                encoder_spatial_embedding   = encoder_spatial_embedding, 
                encoder_temporal_embedding  = encoder_temporal_embedding, 
                temporal_query_embedding    = temporal_query_embedding, 
                spatial_query_embedding     = spatial_query_embedding, 
            )
        else : 
            return self.forward_post_norm(
                memory                      = memory, 
                tgt_sequences               = tgt_sequences, 
                encoder_spatial_embedding   = encoder_spatial_embedding, 
                encoder_temporal_embedding  = encoder_temporal_embedding, 
                temporal_query_embedding    = temporal_query_embedding, 
                spatial_query_embedding     = spatial_query_embedding, 
            )

class TransformerDecoder(nn.Module):
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
        super(TransformerDecoder, self).__init__()
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
        self.decoder_layers = nn.ModuleList([
            self.make_decoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_decoder_layer(self): 
        if self.attention_mode == 'parallel_dual' : 
            layer = TransformerParallelDualDecoderLayer(
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
            layer = TransformerDecoderLayer(
                num_heads         = self.num_heads, 
                model_dim         = self.model_dim, 
                fc_hidden_dim     = self.fc_hidden_dim,
                dropout           = self.dropout, 
                device            = self.device, 
                pre_normalization = self.pre_normalization
            )
            return layer
        else : 
            raise NotImplementedError(f"Attention mode {self.attention_mode} not implemented for decoder") 
    

    def forward_dual(self, 
            memory, 
            tgt_sequence, 
            encoder_temporal_embedding, 
            encoder_spatial_embedding, 
            temporal_embedding, 
            spatial_embedding
        ): 
        """
        """
        
        out = tgt_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.decoder_layers[i](
                memory                     = memory, 
                tgt_sequence               = out, 
                encoder_temporal_embedding = encoder_temporal_embedding, 
                encoder_spatial_embedding  = encoder_spatial_embedding, 
                temporal_query_embedding   = temporal_embedding, 
                spatial_query_embedding    = spatial_embedding, 
            )
            attn_weight_lst.append(attn_weights)

        if self.pre_normalization : 
            out = self.layernorm(out)

        return out, attn_weight_lst
    
    def forward_single(self, memory, tgt_sequence, pos_embedding, query_embedding): 
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim), encoder output 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :param pos_embedding   : a tensor of shape (batch_size, pos_sequence_length, 1), getting broadcasted 
        :param query_embedding : a tensor of shape (batch_size, tgt_sequence_length, 1), getting broadcasted
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             self_attn_weights  : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
             cross_attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length)
        """
        out = tgt_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.decoder_layers[i](memory, out, pos_embedding, query_embedding) 
            attn_weight_lst.append(attn_weights)

        if self.pre_normalization : 
            out = self.layernorm(out)

        return out, attn_weight_lst
    
    def foward(self, 
        memory, 
        tgt_sequence, 
        query_embedding            = None, 
        encoder_temporal_embedding = None, 
        encoder_spatial_embedding  = None, 
        decoder_temporal_embedding = None, 
        decoder_spatial_embedding  = None
    ):
        if self.attention_mode == 'parallel_dual':
            return self.forward_dual(
                memory = memory, 
                tgt_sequence = tgt_sequence, 
                encoder_temporal_embedding = encoder_temporal_embedding, 
                encoder_spatial_embedding  = encoder_spatial_embedding, 
                temporal_embedding         = decoder_temporal_embedding, 
                spatial_embedding          = decoder_spatial_embedding, 
            )
        
        elif self.attention_mode == 'vanilla':
            return self.forward_single(
                memory          = memory, 
                tgt_sequence    = tgt_sequence, 
                pos_embedding   = encoder_temporal_embedding, 
                query_embedding = query_embedding, 
            )
