import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
from model.ModelUtils import *

"""
Graph Convolutional Network (GCN) for human pose forecasting. This code is modified based on 
https://github.com/wei-mao-2019/LearnTrajDep/blob/master/utils/model.py
"""

def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module

class GraphConvolution(nn.Module): 
  def __init__(self, in_features, out_features, bias = True, node_n = 48):
    super(GraphConvolution, self).__init__()
    self.in_features = in_features 
    self.out_features = out_features 
    self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)) 
    self.attn = torch.nn.Parameter(torch.FloatTensor(node_n, node_n))
    if bias : 
      self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
      torch.nn.init.zeros_(self.bias)
    else : 
      self.register_parameter('bias', None)
    self.init_weight() 

  def init_weight(self):
    stdev = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdev, stdev) 
    self.attn.data.uniform_(-stdev, stdev) 
    if self.bias is not None: 
      self.bias.data.uniform_(-stdev, stdev) 

  def forward(self, input): 
    support = torch.matmul(input, self.weight) 
    output = torch.matmul(self.attn, support) 
    if self.bias is not None: 
      output = output + self.bias 
    return output 

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
      + str(self.in_features) + ' -> ' \
      + str(self.out_features) + ')'

class GraphConvBlockv1(nn.Module): 
  def __init__(
    self, 
    in_features, 
    p_dropout, 
    normalizer, 
    embedding_dim,
    activation,  
    bias = True,  
    node_n = 48, 
  ):
    super(GraphConvBlockv1, self).__init__() 
    self.in_features = in_features 
    self.out_features = in_features 
    self.gc_1 = GraphConvolution(in_features, in_features, node_n = node_n, bias = bias)
    self.norm_1 = normalizer( 
      latent_dim = in_features, 
      embedding_dim = embedding_dim, 
      dropout = p_dropout, 
    )
    self.gc_2 = GraphConvolution(in_features, in_features, node_n = node_n, bias = bias) 
    self.norm_2 = normalizer(
      latent_dim = in_features, 
      embedding_dim = embedding_dim, 
      dropout = p_dropout, 
    )
    self.dropout = nn.Dropout(p = p_dropout) 
    self.act = get_activation(activation) 

  def forward(self, x, cond): 
    # gc layer 1 
    resi = self.norm_1(x, cond) 
    resi = self.gc_1(resi) 
    resi = self.act(resi)
    resi = self.dropout(resi) 

    # gc layer 2 
    resi = self.norm_2(resi, cond)
    resi = self.gc_2(resi)
    resi = self.act(resi) 
    resi = self.dropout(resi) 
    return x + resi 

class GraphConvNetv1(nn.Module): 
  def __init__(
    self, 
    input_feature, 
    hidden_feature, 
    embedding_dim, 
    p_dropout, 
    use_adaptive, 
    num_stage, 
    node_n, 
    activation = nn.GELU(approximate = 'tanh'), 
  ):
    super(GraphConvNetv1, self).__init__() 
    self.num_stage = num_stage
    self.gc_1 = GraphConvolution(
      in_features = input_feature, 
      out_features = hidden_feature, 
      bias = True, 
      node_n = node_n
    ) 
    normalizer = AdaptiveNormLayer if use_adaptive else ScaleShiftLayer

    self.norm_1 = normalizer(
      latent_dim = input_feature, 
      embedding_dim = embedding_dim,
      dropout = p_dropout, 
    )

    self.gcbs = [] 
    for _ in range(num_stage): 
      self.gcbs.append(GraphConvBlockv1(
        in_features = hidden_feature,
        node_n = node_n, 
        p_dropout = p_dropout, 
        normalizer = normalizer, 
        activation = activation, 
        embedding_dim = embedding_dim, 
      ))

    self.gcbs = nn.ModuleList(self.gcbs) 

    self.final_gc = zero_module(
      GraphConvolution(
       in_features = hidden_feature, 
       out_features = input_feature, 
       node_n = node_n, 
      )
    )

    self.dropout = nn.Dropout(p = p_dropout)
    self.act = get_activation(activation) 

  def forward(self, x, cond): 
    """
    :params x : a tensor of shape [batch_size, num_dct_channels, num_joint * 3] 
    :return : tensor of the same shape as x 
    """
    resi = self.norm_1(x, cond) 
    resi = self.gc_1(resi) 
    resi = self.act(resi) 
    resi = self.dropout(resi) 

    for i in range(self.num_stage): 
      resi = self.gcbs[i](resi, cond)

    resi = self.final_gc(resi)
    x = x + resi
    return x 



def graph_conv_net_v1(params): 
  return GraphConvNetv1(
    input_feature = params['num_dct_channels'], 
    hidden_feature = params['gcn_dim'], 
    p_dropout = params['gcn_dropout'], 
    num_stage = params['num_gcn_stages'], 
    node_n = params['pose_dim'], 
    embedding_dim = params['model_dim'], 
    use_adaptive = params['gcn_use_adaptive'], 
    activation = params['activation'], 
  )