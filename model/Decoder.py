import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from model.PositionalEncoding import *
import matplotlib.pyplot as plt

class VanillaDecoderLayer(nn.Module):
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
        super(VanillaDecoderLayer, self).__init__() 
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

        self_attn, self_attn_weight = self._self_attention(
            query = self.add_positional_embedding(tgt_sequence, query_embedding), 
            key   = self.add_positional_embedding(tgt_sequence, query_embedding), 
            value = self.add_positional_embedding(tgt_sequence, query_embedding), 
            need_weights = True
        )

        self_attn = self.dropout1(self_attn)

        self_attn_out = self.layernorm1(tgt_sequence + self_attn)
       

        cross_attn, cross_attn_weight = self._cross_attention(
            query = self.add_positional_embedding(self_attn_out, query_embedding), 
            key   = self.add_positional_embedding(memory,          pos_embedding), 
            value = memory, 
            need_weights = True
        )

        cross_attn = self.dropout2(cross_attn)
        attn_out = self.layernorm2(cross_attn + self_attn_out)

        ffn_out = self.linear1(attn_out) 
        ffn_out = self.relu(ffn_out)
        ffn_out = self.dropout3(ffn_out) 
        
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout4(ffn_out)
        
        output        = self.layernorm3(ffn_out + attn_out)
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

        self_attn, self_attn_weight = self._self_attention(
            query = self.add_positional_embedding(tgt_sequence, query_embedding),  
            key   = self.add_positional_embedding(tgt_sequence, query_embedding), 
            value = self.add_positional_embedding(tgt_sequence, query_embedding), 
            need_weights = True
        )

        self_attn = self.dropout1(self_attn)
        self_attn_out  = self.layernorm2(tgt_sequence + self_attn)

        cross_attn, cross_attn_weight = self._cross_attention(
            query = self.add_positional_embedding(self_attn_out, query_embedding), 
            key   = self.add_positional_embedding(memory, pos_embedding), 
            value = memory, 
            need_weights = True, 
        )


        cross_attn   = self.dropout2(cross_attn)
        attn_out     = self.layernorm3(self_attn_out + cross_attn)


        ffn_out  = self.linear1(attn_out)
        ffn_out  = self.relu(ffn_out) 
        ffn_out  = self.dropout3(ffn_out) 

        ffn_out  = self.linear2(ffn_out)
        ffn_out  = self.dropout4(ffn_out)
        output   = attn_out + ffn_out
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
    

class JointAttentionDecoderLayer(nn.Module):
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
        super(JointAttentionDecoderLayer, self).__init__()
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
    

        self.linear1             = nn.Linear(model_dim, fc_hidden_dim) 
        self.linear2             = nn.Linear(fc_hidden_dim, model_dim)

        self.layernorm1          = nn.LayerNorm(self.model_dim)
        self.layernorm2          = nn.LayerNorm(self.model_dim)
        self.layernorm3          = nn.LayerNorm(self.model_dim)
        self.layernorm4          = nn.LayerNorm(self.model_dim)
        self.layernorm5          = nn.LayerNorm(self.model_dim)

        self.relu                = nn.ReLU(inplace = True)
        self.softmax             = nn.Softmax(dim = -1)

        
        self.joint_attention_1 = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.joint_attention_2 = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.proj_self_attn  = nn.Linear(self.model_dim * self.num_joints, self.model_dim)
        self.proj_cross_attn = nn.Linear(self.model_dim * self.num_joints, self.model_dim) 
        self.enc_proj_attn   = nn.Linear(self.model_dim * self.num_joints, self.model_dim)
        self.proj_pose_dim   = nn.Linear(self.model_dim, self.model_dim * self.num_joints)


        self.self_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim   = self.model_dim, 
            num_heads   = self.num_heads, 
            dropout     = self.dropout, 
            batch_first = True, 
        )


        self.device              = device 
        self.pre_normalization   = pre_normalization

    def add_positional_embedding(self, inp, positional_embedding):
        if positional_embedding is not None :
            return inp + positional_embedding
        return inp  
    
    def add_spatial_embedding(self, inp, spatial_embedding):
        if spatial_embedding is not None : 
            return inp + spatial_embedding 
        return inp 
    
    def forward_post_norm(self, 
            memory, 
            tgt_sequences, 
            enc_positional_embedding, 
            dec_spatial_embedding, 
            dec_positional_embedding, 
        ):
        """
        :param memory             : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences       : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_temporal_embedding : a tensor of shape (1, sequence_length, model_dim) 
        :param temporal_query_embedding   : a tensor of shape (1, sequence_length, model_dim), getting broadcasted 
        :param spatial_query_embedding    : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """ 

        batch_size, sequence_length, num_joints, model_dim = tgt_sequences.shape
        batch_size, src_length, _, _                       = memory.shape

        inp = tgt_sequences.clone()
        inp = inp.contiguous().view(batch_size * sequence_length, num_joints, model_dim)

        joint_attn_1, joint_attn_1_weight = self.joint_attention_1(
            query        = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            key          = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            value        = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            need_weights = True, 
        )

        joint_attn_1     = self.dropout1(joint_attn_1)
        joint_attn_1_out = self.layernorm1(joint_attn_1 + inp)
        joint_attn_1_out = joint_attn_1_out.contiguous().view(batch_size, sequence_length, num_joints, model_dim) # attention between joints within frame 
        
        joint_attn_1_out = joint_attn_1_out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
        
        joint_attn_2, joint_attn_2_weight = self.joint_attention_2(
            query  = self.add_positional_embedding(joint_attn_1_out, dec_positional_embedding), 
            key    = self.add_positional_embedding(joint_attn_1_out, dec_positional_embedding), 
            value  = joint_attn_1_out, 
            need_weights = True, 
        )

        joint_attn_2 = self.dropout2(joint_attn_2)
        joint_attn_2_out = self.layernorm2(joint_attn_2 + joint_attn_1_out)
        joint_attn_2_out = joint_attn_2_out.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)


        self_attn_in = joint_attn_2_out.contiguous().view(batch_size, sequence_length, num_joints * model_dim)
        self_attn_in = self.proj_self_attn(self_attn_in) 


        self_attn, self_attn_weight = self.self_attention(
            query = self.add_positional_embedding(self_attn_in, dec_positional_embedding), 
            key   = self.add_positional_embedding(self_attn_in, dec_positional_embedding), 
            value = self_attn_in, 
            need_weights = True, 
        )

        self_attn  = self.dropout3(self_attn)
        self_attn_out = self.layernorm3(self_attn + self_attn_in)

        memory = memory.contiguous().view(batch_size, src_length, -1)
        enc_out = self.proj_cross_attn(memory)

        cross_attn, cross_attn_weight = self.cross_attention(
            query = self.add_positional_embedding(self_attn_out, dec_positional_embedding), 
            key   = self.add_positional_embedding(enc_out      , enc_positional_embedding), 
            value = enc_out, 
            need_weights = True, 
        )

        cross_attn = self.dropout4(cross_attn)
        attn_out = self.layernorm4(self_attn_out + cross_attn)

        attn_out = self.proj_pose_dim(attn_out)
        attn_out = attn_out.contiguous().view(batch_size, sequence_length, num_joints, model_dim)

        ffn_out = self.linear1(attn_out) 
        ffn_out = self.relu(ffn_out)
        ffn_out = self.dropout5(ffn_out) 
        
        ffn_out = self.linear2(ffn_out) 
        ffn_out = self.dropout6(ffn_out)
        
        out = self.layernorm5(ffn_out + attn_out)

        return out, [joint_attn_1_weight, joint_attn_2_weight, self_attn_weight, cross_attn_weight]

    def forward_pre_norm(self, 
            memory, 
            tgt_sequences, 
            enc_positional_embedding, 
            dec_spatial_embedding, 
            dec_positional_embedding, 
        ):
        """
        :param memory                     : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences              : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_temporal_embedding : a tensor of shape (1, enc_sequence_length, model_dim) 
        :param dec_positional_embedding   : a tensor of shape (1, dec_sequence_length, model_dim), getting broadcasted 
        :param dec_spatial_embedding      : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """ 

        batch_size, sequence_length, num_joints, model_dim = tgt_sequences.shape
        batch_size, src_length, _, _                       = memory.shape

        tgt_sequences   = self.layernorm1(tgt_sequences)

        inp = tgt_sequences.clone()
        inp = inp.contiguous().view(batch_size * sequence_length, num_joints, model_dim)

        joint_attn_1, joint_attn_1_weight = self.joint_attention_1(
            query        = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            key          = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            value        = self.add_spatial_embedding(inp, dec_spatial_embedding), 
            need_weights = True, 
        )

        joint_attn_1     = self.dropout1(joint_attn_1)
        joint_attn_1_out = self.layernorm2(joint_attn_1 + inp)
        joint_attn_1_out = joint_attn_1_out.contiguous().view(batch_size, sequence_length, num_joints, model_dim) # attention between joints within frame 
        
        joint_attn_1_out = joint_attn_1_out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_joints, sequence_length, model_dim)
        
        joint_attn_2, joint_attn_2_weight = self.joint_attention_2(
            query  = self.add_positional_embedding(joint_attn_1_out, dec_positional_embedding), 
            key    = self.add_positional_embedding(joint_attn_1_out, dec_positional_embedding), 
            value  = joint_attn_1_out, 
            need_weights = True, 
        )

        joint_attn_2 = self.dropout2(joint_attn_2)
        joint_attn_2_out = self.layernorm3(joint_attn_2 + joint_attn_1_out)
        joint_attn_2_out = joint_attn_2_out.contiguous().view(batch_size, num_joints, sequence_length, model_dim).permute(0, 2, 1, 3)


        self_attn_in = joint_attn_2_out.contiguous().view(batch_size, sequence_length, num_joints * model_dim)
        self_attn_in = self.proj_self_attn(self_attn_in)

        self_attn, self_attn_weight = self.self_attention(
            query = self.add_positional_embedding(self_attn_in, dec_positional_embedding), 
            key   = self.add_positional_embedding(self_attn_in, dec_positional_embedding), 
            value = self_attn_in, 
            need_weights = True, 
        )
        self_attn  = self.dropout3(self_attn)
        self_attn_out = self.layernorm4(self_attn + self_attn_in) # (batch_size, sequence_length, model_dim) 

        memory = memory.contiguous().view(batch_size, src_length, -1)
        enc_out = self.enc_proj_attn(memory) # (batch_size, sequence_length, model_dim)

        cross_attn, cross_attn_weight = self.cross_attention(
            query = self.add_positional_embedding(self_attn_out, dec_positional_embedding), 
            key   = self.add_positional_embedding(enc_out      , enc_positional_embedding), 
            value = enc_out, 
            need_weights = True, 
        )

        cross_attn = self.dropout4(cross_attn)
        attn_out = self.layernorm5(self_attn_out + cross_attn)

        attn_out = self.proj_pose_dim(attn_out)

        attn_out = attn_out.contiguous().view(batch_size, sequence_length, num_joints, model_dim)

        ffn_out = self.linear1(attn_out) 
        ffn_out = self.relu(ffn_out)
        ffn_out = self.dropout5(ffn_out) 
        
        ffn_out = self.linear2(ffn_out) 
        ffn_out = self.dropout6(ffn_out)
        out     = attn_out + ffn_out

        return out, [joint_attn_1_weight, joint_attn_2_weight, self_attn_weight, cross_attn_weight]


    def forward(self,             
            memory, 
            tgt_sequences, 
            enc_positional_embedding, 
            dec_spatial_embedding, 
            dec_positional_embedding, 
        ):
        """
        :param memory                     : a tensor of shape (batch_size, src_sequence_length, num_joints, model_dim)
        :param tgt_sequences              : a tensor of shape (batch_size, tgt_sequence_length, num_joints, model_dim)
        :param encoder_temporal_embedding : a tensor of shape (1, enc_sequence_length, model_dim) 
        :param dec_positional_embedding   : a tensor of shape (1, dec_sequence_length, model_dim), getting broadcasted 
        :param dec_spatial_embedding      : a tensor of shape (1, num_joints,      model_dim), getting broadcasted 
        :return 
            output       : a tensor of shape (batch_size, sequence_length, num_joints, model_dim) 
            attn_weights : a tensor of shape (batch_size, query_sequence_length, kv_sequence_length) 
        """
        fn = self.forward_pre_norm if self.pre_normalization else self.forward_post_norm
        return fn(
            memory                      = memory, 
            tgt_sequences               = tgt_sequences, 
            enc_positional_embedding    = enc_positional_embedding, 
            dec_spatial_embedding       = dec_spatial_embedding, 
            dec_positional_embedding    = dec_positional_embedding, 
        )


class Decoder(nn.Module):
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
        self.attention_mode      = attention_mode
        self.decoder_layers = nn.ModuleList([
            self.make_decoder_layer() for _ in range(self.num_layers)
        ])

    
    def make_decoder_layer(self): 
        if self.attention_mode == 'vanilla' : 
            layer = VanillaDecoderLayer(
                num_heads         = self.num_heads, 
                model_dim         = self.model_dim, 
                fc_hidden_dim     = self.fc_hidden_dim,
                dropout           = self.dropout, 
                device            = self.device, 
                pre_normalization = self.pre_normalization
            )
            return layer
        elif self.attention_mode == 'joint':
            layer = JointAttentionDecoderLayer(
                num_joints        = self.num_joints, 
                num_heads         = self.num_heads, 
                model_dim         = self.model_dim, 
                fc_hidden_dim     = self.fc_hidden_dim,
                dropout           = self.dropout, 
                device            = self.device, 
                pre_normalization = self.pre_normalization, 
            )
            return layer
        else : 
            raise NotImplementedError(f"Attention mode {self.attention_mode} not implemented for decoder") 
    

    def forward_joint(self, 
            memory, 
            tgt_sequences, 
            enc_positional_embedding, 
            dec_spatial_embedding, 
            dec_positional_embedding, 
        ): 
        
        out = tgt_sequences 
        attn_weight_lst = [] 
        out_lst = [] 
        for i in range(self.num_layers): 
            out, attn_weights = self.decoder_layers[i](
                memory                     = memory, 
                tgt_sequences              = out, 
                enc_positional_embedding   = enc_positional_embedding, 
                dec_spatial_embedding      = dec_spatial_embedding, 
                dec_positional_embedding   = dec_positional_embedding, 
            )
            attn_weight_lst.append(attn_weights)
            if self.pre_normalization : 
               out = self.layernorm(out)
            out_lst.append(out)

        return out_lst, attn_weight_lst
    
    def forward_single(self, memory, tgt_sequences, pos_embedding, query_embedding): 
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
        out = tgt_sequences
        out_lst = []
        attn_weight_lst = [] 

        for i in range(self.num_layers): 
            out, attn_weights = self.decoder_layers[i](memory, out, pos_embedding, query_embedding) 
            attn_weight_lst.append(attn_weights)
            if self.pre_normalization : 
                out = self.layernorm(out)
            out_lst.append(out)

        return out_lst, attn_weight_lst
    
    def forward(self, 
        memory, 
        tgt_sequences, 
        enc_positional_embedding = None, 
        dec_spatial_embedding      = None, 
        dec_positional_embedding   = None, 
    ):
        if self.attention_mode == 'vanilla':
            return self.forward_single(
                memory          = memory, 
                tgt_sequences   = tgt_sequences, 
                pos_embedding   = enc_positional_embedding, 
                query_embedding = dec_positional_embedding, 
            )
        elif self.attention_mode == 'joint':
            return self.forward_joint(
                memory                     = memory, 
                tgt_sequences              = tgt_sequences, 
                enc_positional_embedding   = enc_positional_embedding, 
                dec_spatial_embedding      = dec_spatial_embedding, 
                dec_positional_embedding   = dec_positional_embedding, 
            )
        else : 
            raise NotImplementedError(f"Attention mode {self.attention_mode} not implemented for decoder") 
    
        
        