import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt


class Pose1DFeedForwardEncoder(nn.Module):
    def __init__(self, pose_dim, model_dim):
        super(Pose1DFeedForwardEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.model_dim = model_dim
        self.model = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, model_dim)
        )
    
    def forward(self, inp):
      return self.model(inp)
    
class Pose1DFeedForwardDecoder(nn.Module):
    def __init__(self, model_dim, pose_dim):
        super(Pose1DFeedForwardDecoder, self).__init__()
        self.model_dim = model_dim
        self.pose_dim  = pose_dim 
        self.model = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, pose_dim)
        )
    def forward(self, inp):
        return self.model(inp)
    

class PoseConv1DCEncoder(nn.Module): 
    def __init__(self, in_channels, out_channels, num_joints): 
        super(PoseConv1DCEncoder, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels 
        self.num_joints   = num_joints

        self.model = nn.Sequential(
            nn.Conv1d(in_channels = self.in_channels, out_channels = 32, kernel_size = 9), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True), 
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3), 
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace = True), 
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3), 
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace = True), 
            nn.Conv1d(in_channels = 128, out_channels = self.out_channels,  kernel_size = 3), 
            nn.BatchNorm1d(self.out_channels), 
            nn.ReLU(inplace = True),
            nn.Conv1d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = 3), 
        )

    def forward(self, inp): 
        """
        :param : 
          inp : a tensor of shape (batch_size, pose_dim)
        return : 
          out : a tensor of shape (batch_size, model_dim)
        """

        batch_size, pose_dim = inp.shape
        assert pose_dim % self.num_joints == 0, "pose dim is not divisble by joint dim" 
        joint_dim = pose_dim // self.num_joints 
        
        inp = inp.contiguous().view(batch_size, self.num_joints, joint_dim)
        inp = inp.permute(0, 2, 1) # (batch_size , joint_dim, self.num_joints)
        out = self.model(inp)
        out = out.contiguous().view(batch_size, -1)
        return out 
    

class JointAttentionEncoder(nn.Module):
    def __init__(self, num_joints, joint_dim, embed_dim, model_dim, num_layers, num_heads, dropout, fc_hidden_dim): 
        super(JointAttentionEncoder, self).__init__()
        self.num_joints     = num_joints
        self.joint_dim      = joint_dim 
        self.model_dim      = model_dim 
        self.embed_dim      = embed_dim 
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.fc_hidden_dim  = fc_hidden_dim
        self.dropout        = dropout

        for i in range(self.num_layers): 
          setattr(self, f'attention_{i + 1}', nn.MultiheadAttention(
              embed_dim   = embed_dim, 
              num_heads   = num_heads, 
              dropout     = dropout,
              batch_first = True, 
          ))
          setattr(self, f'layernorm_1_{i + 1}', nn.LayerNorm(embed_dim))
          setattr(self, f'layernorm_2_{i + 1}', nn.LayerNorm(embed_dim))
          setattr(self, f'layernorm_3_{i + 1}', nn.LayerNorm(embed_dim))
          setattr(self, f'dropout_1_{i + 1}'  , nn.Dropout(dropout))
          setattr(self, f'dropout_2_{i + 1}'  , nn.Dropout(dropout))
          setattr(self, f'dropout_3_{i + 1}'  , nn.Dropout(dropout))
          setattr(self, f'ffn_1_{i + 1}',       nn.Linear(embed_dim, fc_hidden_dim))
          setattr(self, f'ffn_2_{i + 1}',       nn.Linear(fc_hidden_dim, embed_dim))


        self.proj_embed = nn.Linear(self.joint_dim, self.embed_dim)
        self.proj_model = nn.Linear(self.num_joints * self.embed_dim, self.model_dim) 
        self.relu       = nn.ReLU(inplace = True)

        

    def add_pose_embedding(self, inp, embedding): 
        return inp if embedding is None else inp + embedding 
    
    def forward(self, inp, pose_embedding): 
        """
        :param : 
          inp : a tensor of shape (batch_size, pose_dim)
        return : 
          out : a tensor of shape (batch_size, model_dim)
        """

        batch_size, pose_dim = inp.shape
        assert pose_dim % self.num_joints == 0, "pose dim is not divisble by joint dim" 
        joint_dim = pose_dim // self.num_joints 
        
        inp = inp.contiguous().view(batch_size, self.num_joints, joint_dim)
        pose_embedding = pose_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.proj_embed(inp)
        attn_weights = []

        out = self.add_pose_embedding(out, pose_embedding)
 
        for i in range(self.num_layers): 
            out = getattr(self, f'layernorm_1_{i + 1}')(out)
            attn_out, attn_weight = getattr(self, f'attention_{i + 1}')(
              query = self.add_pose_embedding(out, pose_embedding), 
              key   = self.add_pose_embedding(out, pose_embedding), 
              value = out, 
            )
            attn_out = getattr(self, f'dropout_1_{i + 1}')(attn_out)
            ffn_in   = getattr(self, f'layernorm_2_{i + 1}')(attn_out + out)

            ffn_out  = getattr(self, f'ffn_1_{i + 1}')(ffn_in)
            ffn_out  = self.relu(ffn_out)
            ffn_out  = getattr(self, f'dropout_2_{i + 1}')(ffn_out)
            ffn_out  = getattr(self, f'ffn_2_{i + 1}')(ffn_out)
            ffn_out  = getattr(self, f'dropout_3_{i + 1}')(ffn_out)
            out      = getattr(self, f'layernorm_3_{i + 1}')(ffn_out + ffn_in)
            attn_weights.append(attn_weight)
        
        out = out.contiguous().view(batch_size, -1)
        out = self.proj_model(out)
        return out



class PoseAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(PoseAutoencoder, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, inp, pose_embedding = None): 
        latent = self.encoder(inp, pose_embedding)
        out    = self.decoder(latent)
        return out