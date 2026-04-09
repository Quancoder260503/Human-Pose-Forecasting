import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt
from model.TransformerEncoder import *
from model.TransformerDecoder import *
from scipy.optimize import linear_sum_assignment

class VanillaTransformer(nn.Module):
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
        query_selection                 = False, 
        residual_connection             = False, 
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
        self.query_selection     = query_selection
        self.residual_connection = residual_connection

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
        
        """
        Taking inspiration from : 
          https://github.com/idiap/potr/blob/1e194130cd012d3c9a6137052f402e6f7fb5b71b/models/Transformer.py
        """
        if self.query_selection : 
            self.position_predictor = nn.Linear(self.model_dim, self.decoding_length)



        self.init_weights()
        torch.nn.init.trunc_normal_(self.decoder_query_embedding.weight, std = 0.02)  

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


    def forward(self, src_sequences):
        """
        :param encoding_poses : tensor of shape (B, encoding_len, pose_dim)
        :param decoding_poses : tensor of shape (B, decoding_len, pose_dim)
        :return : 
             tensor of shape (B, decoding_len, pose_dim) 
        """

        batch_size, _, _ = src_sequences.shape
        encoder_inp      = self.proj_model(src_sequences)  

        memory, _       = self.encoder(encoder_inp, self.encoder_positional_encoding)
        
        if self.query_selection :
           indices, prob_matrix = self.process_index_selection(memory)
           tgt_sequences        = self.query_selection_fn(indices, src_sequences)

        else :  
           tgt_sequences    = src_sequences[:, -1, :].clone().unsqueeze(1) # (B, 1, pose_dim)
           tgt_sequences    = tgt_sequences.repeat(1, self.decoding_length, 1) # (B, decoding_len, pose_dim)
            
        decoder_inp = self.proj_model(tgt_sequences)
        query_embedding  = self.decoder_query_embedding.weight.unsqueeze(0)
        out, _           = self.decoder(memory, decoder_inp, self.encoder_positional_encoding, query_embedding)
        out              = self.proj_out(out) 
        return out  


class DualTransformer(nn.Module):
    def __init__(self, 
        num_joints        = 13, 
        joint_dim         = 3, 
        encoding_length   = 14, 
        decoding_length   = 14, 
        num_heads         = 8, 
        model_dim         = 512, 
        fc_hidden_dim     = 2048, 
        num_layers        = 6, 
        dropout           = 0.1, 
        positional_encoding_temperature = 10000,
        positional_encoding_alpha       = 1, 
        device = 'cuda', 
        pre_normalization   = False, 
        query_selection     = False, 
        residual_connection = False, 
    ):
        super(DualTransformer, self).__init__()
        self.num_joints                      = num_joints 
        self.joint_dim                       = joint_dim 
        self.encoding_length                 = encoding_length 
        self.decoding_length                 = decoding_length 
        self.num_heads                       = num_heads 
        self.model_dim                       = model_dim 
        self.fc_hidden_dim                   = fc_hidden_dim 
        self.num_layers                      = num_layers 
        self.dropout                         = dropout 
        self.positional_encoding_temperature = positional_encoding_temperature 
        self.positional_encoding_alpha       = positional_encoding_alpha 
        self.device                          = device 
        self.pre_normalization               = pre_normalization 
        self.query_selection                 = query_selection
        self.residual_connection             = residual_connection

        self.init_positional_encoding() 

        self.encoder = TransformerParallelDualEncoderLayer(
            num_joints        = self.num_joints, 
            num_heads         = self.num_heads, 
            model_dim         = self.model_dim, 
            fc_hidden_dim     = self.fc_hidden_dim, 
            dropout           = self.dropout, 
            device            = self.device, 
            pre_normalization = self.pre_normalization,
        )

        self.decoder = TransformerParallelDualDecoderLayer(
            num_joints        = self.num_joints,
            num_heads         = self.num_heads, 
            model_dim         = self.model_dim, 
            fc_hidden_dim     = self.fc_hidden_dim,
            dropout           = self.dropout, 
            device            = self.device, 
            pre_normalization = self.pre_normalization,  
        )

        self.encoder_spatial_embedding  = nn.Embedding(self.num_joints,      self.model_dim)
        self.decoder_spatial_embedding  = nn.Embedding(self.num_joints,      self.model_dim)
        self.encoder_temporal_embedding = nn.Embedding(self.encoding_length, self.model_dim)    
        self.decoder_temporal_embedding = nn.Embedding(self.decoding_length, self.model_dim) 

        self.proj_model = nn.Linear(self.joint_dim, self.model_dim)
        self.proj_joint = nn.Linear(self.model_dim, self.joint_dim)

        if self.query_selection : 
            self.position_predictor = nn.Linear(self.num_joints * self.model_dim, self.decoding_length)


        self.init_weights()

        torch.nn.init.trunc_normal_(self.encoder_spatial_embedding.weight,  std = 0.02) 
        torch.nn.init.trunc_normal_(self.decoder_spatial_embedding.weight,  std = 0.02)
        torch.nn.init.trunc_normal_(self.encoder_temporal_embedding.weight, std = 0.02)
        torch.nn.init.trunc_normal_(self.decoder_temporal_embedding.weight, std = 0.02)
        
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
        decoder_inputs = torch.FloatTensor(batch_size, self.decoding_length, self.num_joints, self.joint_dim).to(self.device)

        for i in range(batch_size): 
            for j in range(self.decoding_length):
                src_index, target_index = indices[i][0][j], indices[i][1][j] 
                decoder_inputs[i, target_index] = src_sequences[i, src_index]
        return decoder_inputs

    def process_index_selection(self, memory, one_to_one_selection = False):
        """
        :param memory : encoder with memory shape [batch_size, src_len, num_joints, joint_dim]
        :return 
          a tuple with two list of i and j matrix entries of m 
        """

        batch_size, src_len, _, _ = memory.shape 
        inp  = memory.contiguous().view(batch_size, src_len, -1)

        prob_matrix = self.position_predictor(inp) # (batch_size, src_len, tgt_length)

        if one_to_one_selection : 
            # Hungarian algorithm
            soft_matrix = torch.softmax(prob_matrix, dim = -1) 
            indices = [linear_sum_assignment(soft_matrix[i].cpu().detach(), maximize = True) for i in range(batch_size)]
        else : 
            soft_matrix = torch.softmax(prob_matrix, dim = -1)
            indices_row = torch.argmax(soft_matrix, dim = 1)
            indices = [(indices_row[i], list(range(prob_matrix.size()[2]))) for i in range(batch_size)]
        
        return indices, soft_matrix


    def forward(self, src_sequences):
        """
        :param src_sequences : tensor of shape (B, encoding_len, pose_dim)
        :return : 
             tensor of shape (B, decoding_len, pose_dim) 
        """
        batch_size, encoding_length, _ = src_sequences.shape
        src_sequences = src_sequences.contiguous().view(batch_size, encoding_length, self.num_joints, self.joint_dim)

        encoder_inp = self.proj_model(src_sequences) 

        memory, _ = self.encoder(
            src_sequence       = encoder_inp, 
            temporal_embedding = self.encoder_temporal_embedding.weight.unsqueeze(0),
            spatial_embedding  = self.encoder_spatial_embedding.weight.unsqueeze(0),  
        )

        if self.query_selection :
           indices, prob_matrix = self.process_index_selection(memory)
           tgt_sequences        = self.query_selection_fn(indices, src_sequences)

        else :  
           tgt_sequences    = src_sequences[:, -1, :, :].clone().unsqueeze(1) # (batch_size, 1, num_joints, joint_dim)
           tgt_sequences    = tgt_sequences.repeat(1, self.decoding_length, 1, 1) # (batch_size, decoding_len, num_joints, joint_dim)
        
        decoder_inp = self.proj_model(tgt_sequences)
        
        out, _   = self.decoder(
            memory                    = memory, 
            tgt_sequences             = decoder_inp, 
            encoder_spatial_embedding = self.encoder_spatial_embedding.weight.unsqueeze(0), 
            encoder_temporal_embedding= self.encoder_temporal_embedding.weight.unsqueeze(0), 
            spatial_query_embedding   = self.decoder_spatial_embedding.weight.unsqueeze(0), 
            temporal_query_embedding  = self.decoder_temporal_embedding.weight.unsqueeze(0), 
        )

        out = self.proj_joint(out) 
        out = out.contiguous().view(batch_size, self.decoding_length, -1)
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
    transformer_type                = 'vanilla',  
):
    if transformer_type == 'vanilla':
        model = VanillaTransformer(
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
    elif transformer_type == 'parallel_duel':
        model = DualTransformer(
            num_joints         = pose_dim, 
            joint_dim          = (3 if coordinates else 2), 
            encoding_length    = encoding_length, 
            decoding_length    = decoding_length, 
            num_heads          = num_heads, 
            model_dim          = model_dim, 
            fc_hidden_dim      = fc_hidden_dim, 
            num_layers         = num_layers, 
            dropout                         = dropout, 
            positional_encoding_alpha       = positional_encoding_alpha, 
            positional_encoding_temperature = positional_encoding_temperature, 
            device                          = device, 
            pre_normalization               = pre_normalization 
        )
        return model 
    else : 
        raise NotImplementedError('Transformer type not supported for now')
    
