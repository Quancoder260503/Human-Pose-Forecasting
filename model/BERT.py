"""
Use the encoder only for prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

from model.Encoder            import *    
from model.BERTPoseEncoderDecoder        import *  
from model.PositionalEncoding import *

class BERT(nn.Module):
    def __init__(self, 
        num_joints                      = 13, 
        joint_dim                       = 3, 
        window_size                     = 14,  
        num_heads                       = 8,  
        model_dim                       = 512, 
        fc_hidden_dim                   = 2048,
        num_layers                      = 6,
        dropout                         = 0.2, 
        positional_encoding_temperature = 10000, 
        positional_encoding_alpha       = 1, 
        device                          = 'cuda', 
        pre_normalization               = False, 
        query_selection                 = False, 
        attention_mode                  = 'parallel_dual', 
        pose_encoder_type               = 'mlp', 
        pose_decoder_type               = 'mlp', 
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
        :param query_selection : perform query selection strategy or not.  
        """
        super(BERT, self).__init__()
        self.num_joints          = num_joints
        self.joint_dim           = joint_dim
        self.window_size         = window_size
        self.num_heads           = num_heads
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout 
        self.device              = device
        self.pre_normalization   = pre_normalization
        self.query_selection     = query_selection
        self.attention_mode      = attention_mode
        self.pose_encoder_type   = pose_encoder_type
        self.pose_decoder_type   = pose_decoder_type

        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

        self.init_attention_module()
        self.init_positional_encoding()
        self.init_pose_encoder()
        self.init_pose_decoder()
        self.spatial_embedding = nn.Embedding(self.num_joints, self.model_dim)
        self.init_weights()
        torch.nn.init.trunc_normal_(self.spatial_embedding.weight, std = 0.02)



    def init_attention_module(self):
        self.attention_module = Encoder(
            num_joints        = self.num_joints, 
            num_heads         = self.num_heads, 
            model_dim         = self.model_dim, 
            num_layers        = self.num_layers, 
            fc_hidden_dim     = self.fc_hidden_dim, 
            dropout           = self.dropout, 
            device            = self.device, 
            pre_normalization = self.pre_normalization, 
            attention_mode    = self.attention_mode, 
        )

    def init_pose_encoder(self): 
        if self.pose_encoder_type == 'mlp':
            self.encoder = nn.Linear(self.joint_dim, self.model_dim)
        elif self.pose_encoder_type == 'conv1D' : 
            self.encoder = BERTPose1DEncoder(in_channels = self.model_dim, out_channels = self.joint_dim, num_joints = self.num_joints)
        else : 
            raise NotImplementedError(f'encoder {self.pose_decoder_type} has not been implemented')    
        
    def init_pose_decoder(self): 
        if self.pose_decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(self.model_dim, 128), 
                nn.ReLU(inplace = True), 
                nn.Dropout(self.dropout), 
                nn.Linear(128, self.joint_dim), 
            )
        elif self.pose_decoder_type == 'conv1D' : 
            self.decoder = BERTPose1DDecoder(in_channels = self.model_dim, out_channels = self.joint_dim, num_joints = self.num_joints)
        else : 
            raise NotImplementedError(f'decoder {self.pose_decoder_type} has not been implemented')

    
    def init_positional_encoding(self):
        positional_encoding = PositionalEncoding1D(
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha, 
        )
        self.positional_encoding = positional_encoding(sequence_length = self.window_size).contiguous().view(
            1, self.window_size, self.model_dim
        )
        # Do not train the distinguish noise 
        self.positional_encoding = nn.Parameter(self.positional_encoding, requires_grad = False) 

    def init_weights(self):
        for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, src_sequences, prediction_len = 14, get_attn_weight = False):
        """
        Technically this is still in autoregressive fashion

        :param encoding_poses : tensor of shape (B, window_size, pose_dim)
        :param prediction_len : the length of the prediction, can be arbitrily long 
        :return : 
             tensor of shape (B, decoding_len, pose_dim) 
        """

        batch_size, encoding_length, _ = src_sequences.shape
        src_sequences = src_sequences.contiguous().view(batch_size, encoding_length, self.num_joints, self.joint_dim)
        attn_weights = [] 
        curr_inp = src_sequences
        out_lst = []
        for _ in range(0, prediction_len, self.window_size):
            encoder_inp = self.encoder(curr_inp)  # (batch_size, window_size, num_joints, model_dim)
            memory, self_attn_weight = self.attention_module(
                src_sequence = encoder_inp, 
                positional_embedding = self.positional_encoding, 
                spatial_embedding    = self.spatial_embedding.weight.unsqueeze(0)
            )  # (batch_size, window_size, num_joints, model_dim) 

            out = self.decoder(memory) # (batch_size, encoding_length, num_joints, joint_dim) 
            curr_inp = out 
            out_lst.append(out.contiguous().view(batch_size, self.window_size, -1))
            attn_weights.append(self_attn_weight)
        out_lst = torch.cat(out_lst, dim = 1) 
        out_lst = out_lst[:, : prediction_len, :]

        if get_attn_weight : 
            return out_lst, attn_weights
        else : 
            return out_lst
        
def get_bert_model(
    num_joints      = 13, 
    coordinates     = True, 
    window_size     = 14, 
    num_heads       = 8, 
    model_dim       = 512, 
    fc_hidden_dim   = 2048, 
    num_layers      = 6, 
    dropout         = 0.2, 
    positional_encoding_temperature = 10000, 
    positional_encoding_alpha       = 1, 
    device                          = 'cuda', 
    pre_normalization               = False,
    attention_mode                  = 'parallel',  
    pose_encoder_type               = 'mlp', 
    pose_decoder_type               = 'mlp', 
):
    model = BERT(
        num_joints                      = num_joints, 
        joint_dim                       = (3 if coordinates else 2), 
        window_size                     = window_size, 
        num_heads                       = num_heads, 
        model_dim                       = model_dim, 
        fc_hidden_dim                   = fc_hidden_dim, 
        num_layers                      = num_layers, 
        dropout                         = dropout, 
        positional_encoding_alpha       = positional_encoding_alpha, 
        positional_encoding_temperature = positional_encoding_temperature, 
        device                          = device, 
        pre_normalization               = pre_normalization, 
        attention_mode                  = attention_mode, 
        pose_encoder_type               = pose_encoder_type, 
        pose_decoder_type               = pose_decoder_type, 
    )
    return model 
