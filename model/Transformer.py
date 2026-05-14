import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt
from model.PoseAutoencoder import * 
from scipy.optimize import linear_sum_assignment

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
    
    def add_positional_embedding(self, seq, pos_embedding):
        return seq + pos_embedding if pos_embedding is not None else seq
    
    def forward_post_norm(self, src_sequence, pos_embedding):
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :param pos_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        self_attn, self_attn_weight = self._self_attention(
            query        = self.add_positional_embedding(src_sequence, pos_embedding), 
            key          = self.add_positional_embedding(src_sequence, pos_embedding), 
            value        = src_sequence, 
            need_weights = True,
        )
        self_attn = self.dropout1(self_attn)
        self_attn_out = self.layernorm1(src_sequence + self_attn)


        ffn_out = self.linear1(self_attn_out) 
        ffn_out = self.relu(ffn_out) 
        ffn_out = self.dropout2(ffn_out) 
        
        ffn_out = self.linear2(ffn_out) 
        ffn_out = self.dropout3(ffn_out)
        output  = self.layernorm2(self_attn_out + ffn_out)

        return output, self_attn_weight
    
    
    
    def forward_pre_norm(self, src_sequence, pos_embedding):
        """
        :param src_sequence  : a tensor of shape (batch_size, sequence_length, model_dim)
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        norm_src_seq = self.layernorm1(src_sequence)


        self_attn, self_attn_weight = self._self_attention(
            query = self.add_positional_embedding(norm_src_seq, pos_embedding),
            key   = self.add_positional_embedding(norm_src_seq, pos_embedding),
            value = norm_src_seq, 
            need_weights = True, 
        )

        self_attn = self.dropout1(self_attn)
        self_attn = self_attn + src_sequence
        
        norm_self_attn = self.layernorm2(self_attn)

        ffn_out  = self.linear1(norm_self_attn) 
        ffn_out  = self.relu(ffn_out) 
        ffn_out  = self.dropout2(ffn_out) 

        ffn_out  = self.linear2(ffn_out)
        ffn_out  = self.dropout3(ffn_out)
        output   = self_attn + ffn_out
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


class Encoder(nn.Module):
    def __init__(self, 
        num_joints, 
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
        super(Encoder, self).__init__()
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.device              = device 
        self.pre_normalization   = pre_normalization
        self.num_joints          = num_joints
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])
        print("Use pre normalization" if self.pre_normalization else "Use post norm")
    
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
        :param positional_embedding : a tensor of shape (batch_size, sequence_length, 1), getting broadcasted 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        
        out = src_sequence 
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            if self.pre_normalization or i > 0:
               out, attn_weights = self.encoder_layers[i](out, pos_embedding)
            else : 
               out, attn_weights = self.encoder_layers[i](out, None) 
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
        device              = 'cuda', 
        pre_normalization   = False, 
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

        
        self._self_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self._cross_attention = nn.MultiheadAttention(
            embed_dim   =  self.model_dim, 
            num_heads   =  self.num_heads, 
            dropout     =  self.dropout, 
            batch_first = True, 
        )
        
        self.device              = device 
        self.pre_normalization   = pre_normalization

    def add_positional_embedding(self, seq, pos_embedding):
        return seq + pos_embedding if pos_embedding is not None else seq

    def forward_post_norm(self, memory, tgt_sequence, enc_pos_embedding, dec_pos_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim) 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :return 
            output       : a tensor of shape (batch_size, sequence_length, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """

        # Self-attention 

        self_attn, self_attn_weight = self._self_attention(
            query = self.add_positional_embedding(tgt_sequence, dec_pos_embedding), 
            key   = self.add_positional_embedding(tgt_sequence, dec_pos_embedding), 
            value = self.add_positional_embedding(tgt_sequence, dec_pos_embedding), 
            need_weights = True, 
        )

        self_attn = self.dropout1(self_attn)

        self_attn_out = self.layernorm1(tgt_sequence + self_attn)


        cross_attn, cross_attn_weight = self._cross_attention(
            query = self.add_positional_embedding(self_attn_out, dec_pos_embedding), 
            key   = self.add_positional_embedding(memory, enc_pos_embedding), 
            value = memory, 
            need_weights = True, 
        )

        cross_attn = self.dropout2(cross_attn)
        attn_out   = self.layernorm2(cross_attn + self_attn_out)

        ffn_out = self.linear1(attn_out) 
        ffn_out = self.relu(ffn_out)
        ffn_out = self.dropout3(ffn_out) 
        
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout4(ffn_out)
        
        output        = self.layernorm3(ffn_out + attn_out)
        return output, [self_attn_weight, cross_attn_weight]
    
    
    def forward_pre_norm(self, memory, tgt_sequence, enc_pos_embedding, dec_pos_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length,   model_dim), encoder output 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length,   model_dim), 
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             self_attn_weights  : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
             cross_attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length)
        """
        
        # Self-attention 
        tgt_norm = self.layernorm1(tgt_sequence)

        self_attn, self_attn_weight = self._self_attention(
            query = self.add_positional_embedding(tgt_norm, dec_pos_embedding),  
            key   = self.add_positional_embedding(tgt_norm, dec_pos_embedding), 
            value = self.add_positional_embedding(tgt_norm, dec_pos_embedding),  
            need_weights = True, 
        )

        self_attn = self.dropout1(self_attn)
        self_attn_out = tgt_sequence + self_attn

        norm_self_attn = self.layernorm2(self_attn_out)

        cross_attn, cross_attn_weight = self._cross_attention(
            query = self.add_positional_embedding(norm_self_attn, dec_pos_embedding), 
            key   = self.add_positional_embedding(memory, enc_pos_embedding), 
            value = memory, # could adjust
            need_weights = True, 
        )

        cross_attn   = self.dropout2(cross_attn)
        attn_out     = self_attn_out + cross_attn

        norm_attn_out = self.layernorm3(attn_out)
        ffn_out  = self.linear1(norm_attn_out)
        ffn_out  = self.relu(ffn_out) 
        ffn_out  = self.dropout3(ffn_out) 

        ffn_out  = self.linear2(ffn_out)
        ffn_out  = self.dropout4(ffn_out)
        output   = attn_out + ffn_out
        return output, [self_attn_weight, cross_attn_weight]

    
    def forward(self, memory, tgt_sequence, enc_pos_embedding, dec_pos_embedding):
        """
        :param memory          : a tensor of shape (batch_size, src_sequence_length, model_dim) 
        :param tgt_sequence    : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :return 
             output       : a tensor of shape (batch_size, sequence_length, model_dim) 
             attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        if self.pre_normalization : 
            return self.forward_pre_norm(memory, tgt_sequence, enc_pos_embedding, dec_pos_embedding) 
        return self.forward_post_norm(memory, tgt_sequence, enc_pos_embedding, dec_pos_embedding)
    
    
class Decoder(nn.Module):
    def __init__(self, 
        num_joints, 
        num_heads, 
        model_dim, 
        fc_hidden_dim, 
        num_layers, 
        dropout, 
        device              = 'cuda', 
        pre_normalization   = False, 
        use_query_embedding = False, 
        intermediate        = False, 
    ): 
        """
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(Decoder, self).__init__()
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.device              = device 
        self.pre_normalization   = pre_normalization
        self.num_joints          = num_joints
        self.layernorm           = nn.LayerNorm(self.model_dim)
        self.use_query_embedding = use_query_embedding
        self.decoder_layers = nn.ModuleList([
            self.make_decoder_layer() for _ in range(self.num_layers)
        ])
        self.intermediate = intermediate 
        print("Use pre normalization" if self.pre_normalization else "Use post norm")

    
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

    
    def forward(self, memory, tgt_sequences, enc_pos_embedding, dec_pos_embedding): 
        """
        :param memory            : a tensor of shape (batch_size, src_sequence_length, model_dim), encoder output 
        :param tgt_sequence      : a tensor of shape (batch_size, tgt_sequence_length, model_dim)
        :return 
            output             : a tensor of shape (batch_size, sequence_length, model_dim) 
            self_attn_weights  : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
            cross_attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length)
        """
        if self.use_query_embedding :
           out = torch.zeros_like(tgt_sequences)
        else : 
           out = tgt_sequences

        out_lst = []
        attn_weight_lst = [] 
        for i in range(self.num_layers): 
            if self.pre_normalization or i > 0: 
               out, attn_weights = self.decoder_layers[i](memory, out, enc_pos_embedding, dec_pos_embedding)
            else : 
               out, attn_weights = self.decoder_layers[i](memory, out, enc_pos_embedding, None) 
               
            attn_weight_lst.append(attn_weights)
            if self.pre_normalization : 
                out = self.layernorm(out)
            
            if self.intermediate or (i + 1 == self.num_layers): 
               out_lst.append(out)

        return out_lst, attn_weight_lst
    
    


class PoseTransformer(nn.Module):
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
        use_query_embedding             = False, 
        intermediate                    = False, 
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
        super(PoseTransformer, self).__init__()
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
        self.use_query_embedding = use_query_embedding
        self.intermediate        = intermediate

        # joint attention module 
        self.joint_attn_embed_dim     = joint_attn_embed_dim
        self.joint_attn_num_heads     = joint_attn_num_heads
        self.joint_attn_num_layers    = joint_attn_num_layers
        self.joint_attn_fc_hidden_dim = joint_attn_fc_hidden_dim


        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

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
            use_query_embedding = self.use_query_embedding, 
            intermediate        = self.intermediate, 
        )


        self.proj_out                = nn.Linear(self.model_dim, self.pose_dim) # map to the output space 
        
        if self.use_query_embedding:        
           self.decoder_query_embedding = nn.Embedding(self.decoding_length, self.model_dim)
        
        self.dropout1                = nn.Dropout(self.dropout)
        
        """
        Taking inspiration from : 
          https://github.com/idiap/potr/blob/1e194130cd012d3c9a6137052f402e6f7fb5b71b/models/Transformer.py
        """
        if self.query_selection : 
            self.position_predictor = nn.Linear(self.model_dim, self.decoding_length)



        self.init_weights()
        self.init_positional_encoding()

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

        self.encoder_positional_encoding = encoder_positional(sequence_length = self.encoding_length)
        self.decoder_positional_encoding = decoder_positional(sequence_length = self.decoding_length)
        # Do not train the distinguish noise 
        self.encoder_positional_encoding = nn.Parameter(self.encoder_positional_encoding, requires_grad = False)
        self.decoder_positional_encoding = nn.Parameter(self.decoder_positional_encoding, requires_grad = False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean = 0.0, std = 0.05)

            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_normal_(m.weight)

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
            self.proj_model = PoseConv1DCEncoder(
                in_channels = self.joint_dim, 
                out_channels = self.model_dim, 
                num_joints = self.num_joints
            )
        elif self.proj_model_type == 'attn': 
            self.proj_model = JointAttentionEncoder(
                num_joints    = self.num_joints, 
                joint_dim     = self.joint_dim, 
                embed_dim     = self.joint_attn_embed_dim,  
                model_dim     = self.model_dim, 
                num_heads     = self.joint_attn_num_heads,
                num_layers    = self.joint_attn_num_layers,
                dropout       = self.dropout,
                fc_hidden_dim = self.joint_attn_fc_hidden_dim,  
            )
            self.encoder_joint_embedding = nn.Embedding(self.num_joints, self.joint_attn_embed_dim)
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
        if isinstance(self.proj_model, JointAttentionEncoder): 
            encoder_inp      = self.proj_model(src_sequences, self.encoder_joint_embedding.weight)  
        else : 
            encoder_inp      = self.proj_model(src_sequences)

        encoder_inp = encoder_inp + self.encoder_positional_encoding

        memory, encoder_attn_weight = self.encoder(encoder_inp, self.encoder_positional_encoding)
        
        if self.query_selection :
           indices, prob_matrix = self.process_index_selection(memory)
           tgt_sequences        = self.query_selection_fn(indices, src_sequences)
        else : 
           tgt_sequences    = src_sequences[:, -1, :].clone().unsqueeze(1) 
           tgt_sequences    = tgt_sequences.repeat(1, self.decoding_length, 1)
        
        if isinstance(self.proj_model, JointAttentionEncoder): 
            decoder_inp      = self.proj_model(tgt_sequences, self.encoder_joint_embedding.weight)
        else : 
            decoder_inp      = self.proj_model(tgt_sequences)
        
        decoder_inp = decoder_inp + self.decoder_positional_encoding

        dec_out_lst, decoder_attn_weight = self.decoder(memory, decoder_inp, self.encoder_positional_encoding, self.decoder_positional_encoding)
        out_lst = []
        for i in range(len(dec_out_lst)):
            out  = self.proj_out(dec_out_lst[i])
            if self.residual_connection : 
                out = self.dropout1(out) + tgt_sequences
            out_lst.append(out) 

        print(src_sequences.mean(), src_sequences.std())    

        print("src_sequences", src_sequences.abs().mean().item())
        print("encoder_inp",   encoder_inp.abs().mean().item())
        print("memory",        memory.abs().mean().item())
        print("decoder_inp",   decoder_inp.abs().mean().item())
        for i, dec_out in enumerate(dec_out_lst):
            print(f"dec_out[{i}]", dec_out.abs().mean().item())
        print("row std across timesteps:", dec_out_lst[-1].std(dim=1).mean().item())
    
    
        print("ENCODER ATTENTION WEIGHT")
        print(encoder_attn_weight)
        print("DECODER ATTENTION WEIGHT")
        print(decoder_attn_weight)
    
        
        if get_attn_weight : 
            return out_lst, encoder_attn_weight, decoder_attn_weight 
        else : 
            return out_lst


def get_transformer_model(
    num_joints      = 17, 
    joint_dim       = 3, 
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
    residual_connection             = False, 
    joint_attn_embed_dim            = 128, 
    joint_attn_num_layers           = 2, 
    joint_attn_num_heads            = 4, 
    joint_attn_fc_hidden_dim        = 512, 
    proj_model_type                 = 'attn',  
    use_query_embedding             = False, 
    intermediate                    = False, 
):
    model = PoseTransformer(
        num_joints                      = num_joints, 
        joint_dim                       = joint_dim, 
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
        use_query_embedding             = use_query_embedding, 
        intermediate                    = intermediate, 
    )
    return model 
    