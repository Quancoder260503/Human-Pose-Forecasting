import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

"""
BERT decoder, takes input from the Encoder, produce the prediction based on the extracted information 
from the hidden space. 
"""

class BERTPose1DEncoder(nn.Module): 
  def __init__(self, in_channels, out_channels, num_joints): 
      super(BERTPose1DEncoder, self).__init__()
      self.in_channels  = in_channels
      self.out_channels = out_channels 
      self.num_joints   = num_joints

      self.model = nn.Sequential(
          nn.Conv1d(in_channels = self.in_channels, out_channels = 32, kernel_size = 7, stride = 1, padding = 3), 
          nn.BatchNorm1d(32),
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 2), 
          nn.BatchNorm1d(64), 
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), 
          nn.BatchNorm1d(128), 
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 128, out_channels = self.out_channels, kernel_size = 3, stride = 1, padding = 1), 
          nn.BatchNorm1d(self.out_channels), 
          nn.ReLU(inplace = True),
          nn.Conv1d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = 3, stride = 1, padding = 1), 
      )

  def forward(self, inp): 
      """
      :param : 
        inp : a tensor of shape (batch_size, seq_len, num_joint, joint_dim)
      return : 
        out : a tensor of shape (batch_size, seq_len, num_joint, model_dim)
      """

      batch_size, seq_len, _, joint_dim = inp.shape

      inp = inp.contiguous().view(batch_size * seq_len, self.num_joints,  joint_dim)
      inp = inp.permute(0, 2, 1) # (batch_size * seq_len, joint_dim, self.num_joints)
      out = self.model(inp)
      out = out.contiguous().view(batch_size, seq_len, self.model, -1)
      return out 


class BERTPose1DDecoder(nn.Module): 
  def __init__(self, in_channels, out_channels, num_joints): 
      super(BERTPose1DDecoder, self).__init__()
      self.in_channels  = in_channels
      self.out_channels = out_channels 
      self.num_joints   = num_joints

      self.model = nn.Sequential(
          nn.Conv1d(in_channels = self.in_channels, out_channels = 128, kernel_size = 7, stride = 1, padding = 3), 
          nn.BatchNorm1d(128),
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 1, padding = 2), 
          nn.BatchNorm1d(64), 
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1), 
          nn.BatchNorm1d(32), 
          nn.ReLU(inplace = True), 
          nn.Conv1d(in_channels = 32, out_channels = self.out_channels, kernel_size = 3, stride = 1, padding = 1), 
          nn.BatchNorm1d(self.out_channels), 
          nn.ReLU(inplace = True),
          nn.Conv1d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = 3, stride = 1, padding = 1), 
      )

  def forward(self, inp): 
      """
      :param : 
        inp : a tensor of shape (batch_size, seq_len, num_joint, model_dim)
      return : 
        out : a tensor of shape (batch_size, seq_len, num_joint * joint_dim)
      """

      batch_size, seq_len, _, model_dim = inp.shape

      inp = inp.contiguous().view(batch_size, seq_len,  self.num_joints,  model_dim)
      inp = inp.contiguous().view(batch_size * seq_len, self.num_joints,  model_dim)
      inp = inp.permute(0, 2, 1) # (batch_size * seq_len, model_dim, self.num_joints)
      out = self.model(inp)
      out = out.contiguous().view(batch_size, seq_len, -1)
      return out 
    

    
# if __name__ == '__main__':
#     dof = 3
#     output_channels = 512
#     n_joints = 17
#     seq_len = 49

#     model  = Pose1DDecoder(in_channels = 512, out_channels = dof, num_joints = 17)
#     inputs = torch.FloatTensor(10, seq_len, n_joints, 512)
#     X = model(inputs)
#     print(X.size())