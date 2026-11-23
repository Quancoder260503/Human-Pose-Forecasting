import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
from model.ModelUtils import *

"""
Implementation of the Motion Diffusion Transformer (MDT) and the Motion Refinement Network (MRN) proposed in my thesis.
The point is that we are going to use the informative outputs of the transformer, and learns the structural relativity 
of the human joint graph via a Graph Convolutional Network (GCN) to refine the prediction. I also tried the implementation 
of Diffusion Transformer (https://arxiv.org/pdf/2212.09748) and include the Squeeze-and-Excitation (SE, https://arxiv.org/pdf/1709.01507) 
block, which is quite popular for frequency-band data. The results show a notable improvement in the prediction accuracy compares 
to the other research works. 
"""

class MotionDiffusionTransformerLayer(nn.Module):
  def __init__(
    self, 
    model_dim, 
    ffn_dim, 
    embedding_dim, 
    num_head, 
    dropout, 
    activation, 
    num_channels, 
    se_ratio, 
    use_max_pooling, 
    use_adaptive, 
    use_concat, 
  ):
    super(MotionDiffusionTransformerLayer, self).__init__()

    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.embedding_dim = embedding_dim
    self.num_head = num_head 
    self.dropout = dropout
    self.num_channels = num_channels 
    self.use_adaptive = use_adaptive
    self.use_concat = use_concat

    use_stylization = (self.use_adaptive is False)

    self.attention_layer = (StyleMultiHeadAttention if use_stylization else AdaptiveMultiHeadAttention) (
      model_dim = model_dim, 
      num_head  = num_head, 
      embedding_dim = embedding_dim, 
      dropout = dropout,
    )
    
    self.ffn_layer = (StyleFeedForwardNetwork if use_stylization else AdaptiveFeedForwardNetwork)(
      model_dim = model_dim,
      ffn_dim = ffn_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
      activation = activation,   
    ) 
     
    self.excitation_layer = SqueezeAndExcitation(
      channels = num_channels, 
      ratio = se_ratio, 
      act_fn = activation, 
      use_max_pooling = use_max_pooling, 
    )
    
    if self.use_concat: 
      self.concat = nn.Linear(self.model_dim * 2, self.model_dim)

  def forward(self, input, embedding): 
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
      a tensor of shape (B, T, model_dim)
    """       
    if self.use_concat: 
      input = self.concat(input)
    input = self.excitation_layer(input)
    out = self.attention_layer(input, embedding)
    out = self.ffn_layer(out, embedding)
    return out 


class MotionDiffusionTransformer(nn.Module):
  def __init__(
    self, 
    input_features, 
    prediction_len, 
    model_dim, 
    ffn_dim, 
    num_layers,
    num_heads, 
    dropout = 0.2, 
    activation = 'GeLU', 
    embedding_max_period = 10000,
    se_ratio = 4, 
    use_max_pooling = False,
    use_adaptive = True,  
    num_classes = None, 
    dataset = 'h36m', 
  ):
    super(MotionDiffusionTransformer, self).__init__()
    self.input_features = input_features
    self.prediction_len = prediction_len 
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_layers = num_layers 
    self.num_heads = num_heads 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.embedding_max_period = embedding_max_period
    self.use_adaptive = use_adaptive
    self.se_ratio = se_ratio 
    self.use_max_pooling = use_max_pooling 
    self.num_classes = num_classes
    self.dataset = dataset 

 
    self.sequence_embedding = nn.Parameter(torch.randn(prediction_len, model_dim))

    self.conditional_embedding = nn.Linear(self.input_features * self.prediction_len, self.time_embedding_dim)

    self.time_embedding = nn.Sequential(
      nn.Linear(self.model_dim, self.time_embedding_dim), 
      nn.SiLU(), 
      nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
    )

    self.attn_blocks = torch.nn.ModuleList() 

    for index in range(self.num_layers):
      self.attn_blocks.append(
        MotionDiffusionTransformerLayer(
          model_dim = self.model_dim, 
          ffn_dim = self.ffn_dim, 
          embedding_dim = self.time_embedding_dim,
          num_head = self.num_heads, 
          dropout = self.dropout_p, 
          activation = self.activation,  
          use_max_pooling = self.use_max_pooling, 
          se_ratio = self.se_ratio,
          num_channels = self.prediction_len,  
          use_adaptive = self.use_adaptive, 
          use_concat = (index >= self.num_layers // 2), 
        )
      )

      self.mid_block = MotionDiffusionTransformerLayer(
        model_dim = self.model_dim, 
        ffn_dim = self.ffn_dim, 
        embedding_dim = self.time_embedding_dim,
        num_head = self.num_heads, 
        dropout = self.dropout_p, 
        activation = self.activation,  
        use_max_pooling = self.use_max_pooling, 
        se_ratio = self.se_ratio,
        num_channels = self.prediction_len,  
        use_adaptive = self.use_adaptive, 
        use_concat = False, 
      )


    self.pose_encoder = nn.Linear(input_features, model_dim)
    self.pose_decoder = (AdaptiveFinalTransformerLayer if self.use_adaptive else StyleFinalTransformerLayer)(
      model_dim = model_dim, 
      output_dim = input_features, 
      embedding_dim = self.time_embedding_dim,
    )

    
  def forward(self, input, timesteps, memory = None):
    """
    :param input : a tensor of shape (B, T_{prediction}, input_features) 
    :param timesteps : a tensor of shape (B, ) 
    :param memory : if not None, can be the observed seqs information 
    """

    batch_size, num_channels, _ = input.shape
  

    time_embedding = self.time_embedding(
      timestep_embedding(
        timesteps = timesteps, 
        dim = self.model_dim, 
        max_period = self.embedding_max_period, 
      )
    )

    total_embedding = time_embedding

    if memory is not None: 
      enc_memory = memory.contiguous().view(batch_size, -1)
      total_embedding = total_embedding + self.conditional_embedding(enc_memory)
    

    x = self.pose_encoder(input) 
    x = x + self.sequence_embedding.unsqueeze(0)[:, :num_channels, :]


    enc_list = [] 
    for module in self.attn_blocks[:self.num_layers // 2]: 
      x = module(x, total_embedding)
      enc_list.append(x)

    x = self.mid_block(x, total_embedding)

    for module in self.attn_blocks[self.num_layers // 2:]:
      x = module(torch.cat([x, enc_list[-1]], dim = -1), total_embedding)
      enc_list.pop() 
    
    output = self.pose_decoder(x, time_embedding)

    return output, total_embedding

  
def motion_diffusion_transformer(params):
  return MotionDiffusionTransformer(
    input_features = params['pose_dim'], 
    prediction_len = params['n_pred'], 
    model_dim = params['model_dim'], 
    ffn_dim = params['ffn_dim'], 
    num_layers = params['num_layers'],
    num_heads = params['num_heads'], 
    dropout = params['dropout'], 
    activation = params['activation'], 
    embedding_max_period = params['embedding_max_period'], 
    se_ratio = params['se_ratio'], 
    use_max_pooling = params['use_max_pooling'], 
    use_adaptive = params['use_adaptive'], 
    num_classes = params['num_classes'], 
    dataset = params['dataset'], 
  )


class MotionRefinementNet(nn.Module): 
  def __init__(self, predictor : nn.Module, corrector : nn.Module):
    super(MotionRefinementNet, self).__init__()
    self.predictor = predictor  
    self.corrector = corrector 

  def forward(self, x, t, memory = None):
    x, embed = self.predictor(x, t, memory) 
    x = x.permute(0, 2, 1) 
    out = self.corrector(x, embed)
    out = out.permute(0, 2, 1) 
    return out 
