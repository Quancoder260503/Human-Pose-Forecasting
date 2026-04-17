import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt
from model.Encoder     import *
from model.Decoder     import *
from model.PoseEncoder import * 
from scipy.optimize import linear_sum_assignment

class VanillaTransformer(nn.Module):
    def __init__(self, 
        num_joints                      = 13, 
        joint_dim                       = 3, 
        encoding_length                 = 14,
        decoding_length                 = 14, 
        num_heads                       = 8,  
        model_dim                       = 512, 
        fc_hidden_dim                   = 2048,
        num_layers                      = 6,
        dropout                         = 0.2, 
        positional_encoding_temperature = 10000, 
        positional_encoding_alpha       = 1, 
        joint_attn_embed_dim            = 128, 
        joint_attn_num_layers           = 2, 
        joint_attn_num_heads            = 4, 
        joint_attn_fc_hidden_dim        = 512, 
        device                          = 'cuda', 
        pre_normalization               = False, 
        query_selection                 = False, 
        residual_connection             = False, 
        proj_model_type                 = 'mlp',  
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
        super(VanillaTransformer, self).__init__()
        self.num_joints          = num_joints
        self.joint_dim           = joint_dim
        self.pose_dim            = self.num_joints * self.joint_dim
        self.encoding_length     = encoding_length 
        self.decoding_length     = decoding_length
        self.num_heads           = num_heads
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout 
        self.device              = device
        self.pre_normalization   = pre_normalization
        self.query_selection     = query_selection
        self.residual_connection = residual_connection
        self.proj_model_type     = proj_model_type

        # joint attention module 
        self.joint_attn_embed_dim     = joint_attn_embed_dim
        self.joint_attn_num_heads     = joint_attn_num_heads
        self.joint_attn_num_layers    = joint_attn_num_layers
        self.joint_attn_fc_hidden_dim = joint_attn_fc_hidden_dim


        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

        self.init_positional_encoding()
        self.init_model_proj()

        self.encoder            = Encoder(
            num_joints          = self.num_joints, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
            pre_normalization   = self.pre_normalization, 
            attention_mode      = 'vanilla', 
        )

        self.decoder            = Decoder(
            num_joints          = self.num_joints, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
            pre_normalization   = self.pre_normalization,
            attention_mode      = 'vanilla',  
        )


        self.proj_out                = nn.Linear(self.model_dim, self.pose_dim) # map to the output space        
        self.decoder_query_embedding = nn.Embedding(self.decoding_length, self.model_dim)
        self.encoder_joint_embedding = nn.Embedding(self.num_joints, self.joint_attn_embed_dim)
        self.dropout1                = nn.Dropout(self.dropout)
        
        """
        Taking inspiration from : 
          https://github.com/idiap/potr/blob/1e194130cd012d3c9a6137052f402e6f7fb5b71b/models/Transformer.py
        """
        if self.query_selection : 
            self.position_predictor = nn.Linear(self.model_dim, self.decoding_length)



        self.init_weights()
        torch.nn.init.trunc_normal_(self.decoder_query_embedding.weight, std = 0.02)  
        torch.nn.init.trunc_normal_(self.encoder_joint_embedding.weight, std = 0.02) 

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
        for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def query_selecton_fn(self, indices, src_sequences):
        """ 
        Taking inspiration from : 
        https://github.com/idiap/potr/blob/1e194130cd012d3c9a6137052f402e6f7fb5b71b/models/Transformer.py

        Handles the way queries ar egenerated copying items from the input 
        """
        batch_size = src_sequences.shape[0]
        decoder_inputs = torch.FloatTensor(batch_size, self.decoding_length, self.pose_dim).to(self.device)

        for i in range(batch_size): 
            for j in range(self.decoding_length):
                src_index, target_index = indices[i][0][j], indices[i][1][j] 
                decoder_inputs[i, target_index] = src_sequences[i, src_index]
        return decoder_inputs

    def process_index_selection(self, memory, one_to_one_selection = False):
        """
        :param memory : encoder with memory shape [batch_size, src_len, model_dim]
        :return 
          a tuple with two list of i and j matrix entries of m 
        """

        batch_size = memory.shape[0] 

        prob_matrix = self.position_predictor(memory) # (batch_size, src_len, tgt_length)

        if one_to_one_selection : 
            # Hungarian algorithm
            soft_matrix = torch.softmax(prob_matrix, dim = -1) 
            indices = [linear_sum_assignment(soft_matrix[i].cpu().detach(), maximize = True) for i in range(batch_size)]
        else : 
            soft_matrix = torch.softmax(prob_matrix, dim = -1)
            indices_row = torch.argmax(soft_matrix, dim = 1)
            indices = [(indices_row[i], list(range(prob_matrix.size()[2]))) for i in range(batch_size)]
        
        return indices, soft_matrix

    def init_model_proj(self): 
        if self.proj_model_type == 'mlp': 
            self.proj_model = nn.Linear(self.pose_dim, self.model_dim)
        elif self.proj_model_type == 'conv1d':
            self.proj_model = Pose1DEncoder(
                in_channels = self.joint_dim, 
                out_channels = self.model_dim, 
                num_joints = self.num_joints
            )
        elif self.proj_model_type == 'attn': 
            self.proj_model = JointAttention(
                num_joints    = self.num_joints, 
                joint_dim     = self.joint_dim, 
                embed_dim     = self.joint_attn_embed_dim,  
                model_dim     = self.model_dim, 
                num_heads     = self.joint_attn_num_heads,
                num_layers    = self.joint_attn_num_layers,
                dropout       = self.dropout,
                fc_hidden_dim = self.joint_attn_fc_hidden_dim,  
            )
        else : 
            raise NotImplementedError(f'proj type {self.proj_model_type} is not implemented')
    
    def forward(self, src_sequences, get_attn_weight = False):
        """
        :param encoding_poses : tensor of shape (B, encoding_len, pose_dim)
        :param decoding_poses : tensor of shape (B, decoding_len, pose_dim)
        :return : 
             tensor of shape (B, decoding_len, pose_dim) 
        """

        batch_size, _, _ = src_sequences.shape
        encoder_inp      = self.proj_model(src_sequences, self.encoder_joint_embedding.weight.unsqueeze(0))  

        memory, encoder_attn_weight = self.encoder(
            src_sequence         = encoder_inp, 
            positional_embedding = self.encoder_positional_encoding, 
        )
        
        if self.query_selection :
           indices, prob_matrix = self.process_index_selection(memory)
           tgt_sequences        = self.query_selection_fn(indices, src_sequences)

        else :  
           tgt_sequences    = src_sequences[:, -1, :].clone().unsqueeze(1) # (B, 1, pose_dim)
           tgt_sequences    = tgt_sequences.repeat(1, self.decoding_length, 1) # (B, decoding_len, pose_dim)
        
        decoder_inp      = self.proj_model(tgt_sequences, self.encoder_joint_embedding.weight.unsqueeze(0))
        
        query_embedding  = self.decoder_query_embedding.weight.unsqueeze(0)
        
        dec_out_lst, decoder_attn_weight = self.decoder(
            memory                      = memory, 
            tgt_sequences               = decoder_inp, 
            enc_positional_embedding    = self.encoder_positional_encoding,  
            dec_positional_embedding    = query_embedding, 
        ) 
        out_lst = []
        for i in range(len(dec_out_lst)):
            out  = self.proj_out(dec_out_lst[i])
            if self.residual_connection : 
                out = self.dropout1(out) + tgt_sequences
            out_lst.append(out) 
        
        if get_attn_weight : 
            return out_lst, encoder_attn_weight, decoder_attn_weight 
        else : 
            return out_lst


def get_transformer_model(
    num_joints       = 13, 
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
    transformer_type                = 'vanilla', 
    residual_connection             = False, 
    joint_attn_embed_dim            = 128, 
    joint_attn_num_layers           = 2, 
    joint_attn_num_heads            = 4, 
    joint_attn_fc_hidden_dim        = 512, 
    proj_model_type                 = 'attn',  
):
    if transformer_type == 'vanilla':
        model = VanillaTransformer(
            num_joints                      = num_joints, 
            joint_dim                       = (3 if coordinates else 2), 
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
            residual_connection             = residual_connection,  
            joint_attn_embed_dim            = joint_attn_embed_dim, 
            joint_attn_num_layers           = joint_attn_num_layers, 
            joint_attn_num_heads            = joint_attn_num_heads, 
            joint_attn_fc_hidden_dim        = joint_attn_fc_hidden_dim, 
            proj_model_type                 = proj_model_type, 
        )
        return model 
    else : 
        raise NotImplementedError('Transformer type not supported for now')
    
