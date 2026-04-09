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
    
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_heads, 
                 model_dim, 
                 fc_hidden_dim, 
                 num_layers, 
                 dropout, 
                 device = 'cuda', 
                 pre_normalization = False, 
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
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_encoder_layer(self): 
        layer = TransformerEncoderLayer(
           num_heads         = self.num_heads, 
           model_dim         = self.model_dim, 
           fc_hidden_dim     = self.fc_hidden_dim,
           dropout           = self.dropout, 
           device            = self.device, 
           pre_normalization = self.pre_normalization
        )
        return layer 
    
    def forward(self, src_sequence, pos_embedding): 
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param pos_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        out = src_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.encoder_layers[i](out, pos_embedding) 
            attn_weight_lst.append(attn_weights)

        if self.pre_normalization : 
            out = self.layernorm(out)

        return out, attn_weight_lst
    
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
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim), encoder output 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
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
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim), encoder output 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :param pos_embedding   : a tensor of shape (batch_size, pos_sequence_length, 1), getting broadcasted 
        :param query_embedding : a tensor of shape (batch_size, tgt_sequence_length, 1), getting broadcasted
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization : 
            return self.forward_pre_norm(memory, tgt_sequence, pos_embedding, query_embedding) 
        return self.forward_post_norm(memory, tgt_sequence, pos_embedding, query_embedding)
    
class TransformerDecoder(nn.Module):
    def __init__(self, 
                num_heads, 
                model_dim, 
                fc_hidden_dim, 
                num_layers, 
                dropout, 
                device = 'cuda', 
                pre_normalization = False, 
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
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.decoder_layers      = nn.ModuleList([
            self.make_decoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_decoder_layer(self): 
        layer = TransformerDecoderLayer(
           num_heads         = self.num_heads, 
           model_dim         = self.model_dim, 
           fc_hidden_dim     = self.fc_hidden_dim,
           dropout           = self.dropout, 
           device            = self.device, 
           pre_normalization = self.pre_normalization
        )
        return layer 
    
    def forward(self, memory, tgt_sequence, pos_embedding, query_embedding): 
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


class Transformer(nn.Module):
    def __init__(self, 
        pose_dim                        = 39,  
        encoding_length                 = 14,
        decoding_length                 = 14, 
        num_heads                       = 8,  
        model_dim                       = 512, 
        fc_hidden_dim                   = 2048,
        num_layers                      = 6,
        dropout                         = 0.2, 
        positional_encoding_temperature = 10000, 
        positional_encoding_alpha       = 1, 
        device                          = 'cuda', 
        pre_normalization               = False, 
    ): 
        """
        :param pose_dim        : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param encoding_length : the length of the encoding pose sequences 
        :param decoding_length : the length of the decoding pose sequences 
        :param num_heads       : number of attention heads 
        :param model_dim       : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim   : fully connected layer hidden dim 
        :param num_layers      : number of (MultiheadAttn + FC) layers 
        :param dropout         : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(Transformer, self).__init__()
        self.pose_dim            = pose_dim
        self.encoding_length     = encoding_length 
        self.decoding_length     = decoding_length
        self.num_heads           = num_heads
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout 
        self.device              = device
        self.pre_normalization   = pre_normalization

        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

        self.init_positional_encoding()

        self.encoder            = TransformerEncoder(
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
            pre_normalization   = self.pre_normalization, 
        )

        self.decoder            = TransformerDecoder(
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
            pre_normalization   = self.pre_normalization, 
        )
        self.proj_model              = nn.Linear(self.pose_dim,  self.model_dim)
        self.proj_out                = nn.Linear(self.model_dim, self.pose_dim) # map to the output space
        self.decoder_query_embedding = nn.Embedding(self.decoding_length, self.model_dim)
        self.init_weights() 

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

    def init_weights(self): 
        """
        Initialize the weights of the model 
        """
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad : 
                nn.init.xavier_uniform_(p, gain = 1.0) 


    def forward(self, src_sequences):
        """
        :param encoding_poses : tensor of shape (B, encoding_len, pose_dim)
        :param decoding_poses : tensor of shape (B, decoding_len, pose_dim)
        :return : 
             tensor of shape (B, decoding_len, pose_dim) 
        """

        batch_size, _, _ = src_sequences.shape

        tgt_sequences    = src_sequences[:, -1, :].clone().unsqueeze(1) # (B, 1, pose_dim)
        tgt_sequences    = tgt_sequences.repeat(1, self.decoding_length, 1) # (B, decoding_len, pose_dim)

        src_sequences = self.proj_model(src_sequences)  
        tgt_sequences = self.proj_model(tgt_sequences)

        memory, _       = self.encoder(src_sequences, self.encoder_positional_encoding)
        query_embedding = self.decoder_query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        out, _           = self.decoder(memory, tgt_sequences, self.encoder_positional_encoding, query_embedding)
        out              = self.proj_out(out) 
        return out  


def transformer_loss_function(pred_joints, gt_joints):
    criterion = nn.SmoothL1Loss(reduction = 'mean')
    return criterion(pred_joints, gt_joints)

def get_transformer_model(
    pose_dim        = 13, 
    coordinates     = True, 
    encoding_length = 14, 
    decoding_length = 14, 
    num_heads       = 8, 
    model_dim       = 512, 
    fc_hidden_dim   = 2048, 
    num_layers      = 6, 
    dropout         = 0.2, 
    positional_encoding_temperature = 10000, 
    positional_encoding_alpha       = 1, 
    device                          = 'cuda', 
    pre_normalization               = False, 
):
    model = Transformer(
        pose_dim                        = pose_dim * (3 if coordinates else 2),  
        encoding_length                 = encoding_length,
        decoding_length                 = decoding_length, 
        num_heads                       = num_heads,  
        model_dim                       = model_dim, 
        fc_hidden_dim                   = fc_hidden_dim,
        num_layers                      = num_layers,
        dropout                         = dropout, 
        positional_encoding_temperature = positional_encoding_temperature, 
        positional_encoding_alpha       = positional_encoding_alpha, 
        device                          = device, 
        pre_normalization               = pre_normalization, 
    )
    return model 