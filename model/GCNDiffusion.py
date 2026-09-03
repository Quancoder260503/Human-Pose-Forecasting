# remove after finish testing 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import numpy as np 
from model.ModelUtils import *
from model.DiffusionTransformer import ( 
  ScaleShiftLayer, 
  SqueezeAndExcitation, 
  MultiHeadAttention, 
  AdaptiveMultiHeadAttention, 
  FeedForwardNetwork, 
  AdaptiveFeedForwardNetwork,  
) 

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

class GCBlock(nn.Module): 
  def __init__(
    self, 
    in_features, 
    p_dropout, 
    normalizer, 
    embedding_dim, 
    bias = True, 
    activation = nn.GELU(approximate = 'tanh'), 
    node_n = 48, 
  ):
    super(GCBlock, self).__init__() 
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
    self.act = activation 

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

class GCN(nn.Module): 
  def __init__(
    self, 
    input_feature, 
    hidden_feature, 
    embedding_dim, 
    p_dropout, 
    normalizer, 
    num_stage, 
    node_n, 
    activation = nn.GELU(approximate = 'tanh'), 
  ):
    super(GCN, self).__init__() 
    self.num_stage = num_stage
    self.gc_1 = GraphConvolution(
      in_features = input_feature, 
      out_features = hidden_feature, 
      bias = True, 
      node_n = node_n
    ) 
    self.norm_1 = normalizer(
      latent_dim = input_feature, 
      embedding_dim = embedding_dim,
      dropout = p_dropout, 
    )
    self.gcbs = [] 
    for _ in range(num_stage): 
      self.gcbs.append(GCBlock(
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
    self.act = activation 

  def forward(self, x, cond): 
    """
    :params x : a tensor of shape [batch_size, num_dct_channels, num_joint * 3] 
    :return : tensor of the same shape as x 
    """
    x = x.permute(0, 2, 1) # [b, n, c]
    resi = self.norm_1(x, cond) 
    resi = self.gc_1(resi) 
    resi = self.act(resi) 
    resi = self.dropout(resi) 

    for i in range(self.num_stage): 
      resi = self.gcbs[i](resi, cond)

    resi = self.final_gc(resi)
    x = x + resi
    x = x.permute(0, 2, 1)
    return x 

class GCNDiffusionTransformerLayer(nn.Module):
  def __init__(
    self, 
    model_dim, 
    ffn_dim, 
    embedding_dim, 
    num_head, 
    dropout, 
    activation,  
    use_excitation,
    num_channels, 
    use_adaptive, 
    se_ratio, 
    use_max_pooling, 
    use_concat, 
  ):
    super(GCNDiffusionTransformerLayer, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.embedding_dim = embedding_dim
    self.num_head = num_head 
    self.dropout = dropout
    self.use_excitation = use_excitation
    self.num_channels = num_channels 
    self.use_adaptive = use_adaptive
    self.use_excitation = use_excitation 
    self.use_max_pooling = use_max_pooling 
    self.se_ratio = se_ratio
    self.use_concat = use_concat

    self.attention_layer = (AdaptiveMultiHeadAttention if self.use_adaptive else MultiHeadAttention) (
      model_dim = model_dim, 
      num_head  = num_head, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    )
    
    self.ffn_layer = (AdaptiveFeedForwardNetwork if self.use_adaptive else FeedForwardNetwork)(
      model_dim = model_dim,
      ffn_dim = ffn_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
      activation = activation,  
    ) 
    if self.use_excitation: 
      self.excitation_layer = SqueezeAndExcitation(
        channels = num_channels, 
        ratio = se_ratio, 
        act_fn = activation, 
        use_max_pooling = use_max_pooling, 
      ) 
    if self.use_concat: 
      self.concat = nn.Linear(self.model_dim * 2, self.model_dim)


  def forward(self, x, embedding): 
    """
    :param input     : a tensor of shape (B, num_dct_channels, input_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
      a tensor of shape (B, T, input_dim)
    """       
    if self.use_concat: 
      x = self.concat(x) 
    if self.use_excitation: 
      x = self.excitation_layer(x)
    out = self.attention_layer(x, embedding)
    out = self.ffn_layer(out, embedding)
    return out 

class ConditionalTransformerLayer(nn.Module):
  def __init__(self, model_dim, output_dim, embedding_dim):
    super(ConditionalTransformerLayer, self).__init__()
    self.model_dim = model_dim 
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim 

    self.scale_shift_norm = nn.Sequential(
      nn.SiLU(), 
      nn.Linear(embedding_dim, 2 * model_dim), 
    )
    self.layernorm = nn.LayerNorm(self.model_dim)
    self.linear = zero_module(nn.Linear(model_dim, output_dim))

  def forward(self, input, embedding):
    """
    :param input     : a tensor of shape (B, num_dct_channels, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, num_dct_channels, model_dim)
    """  
    embedding_out = self.scale_shift_norm(embedding).unsqueeze(1)
    shift, scale =  torch.chunk(embedding_out, 2, dim = -1)
    norm_input = self.layernorm(input) * (scale + 1.) + shift
    out = self.linear(norm_input)
    return out 

class GCNDiffusionTransformer(nn.Module): 
  def __init__(
    self, 
    input_dim, 
    num_features, 
    model_dim, 
    ffn_dim, 
    num_layers,
    num_heads, 
    dropout = 0.2, 
    activation = 'GeLU', 
    embedding_max_period = 10000,
    learnable_embedding = True, 
    num_gcn_stages = 2, 
    pos_encoding_params = (10000, 1), 
    use_excitation = False, 
    gcn_dim = 256, 
    se_ratio = 4, 
    use_max_pooling = False,
    use_adaptive = True,  
    flatten_arc = False, 
    num_classes = False,
    use_action = False,     
    use_mid_block = False, 
    use_concat = False, 
  ):
    super(GCNDiffusionTransformer, self).__init__()
    self.input_dim = input_dim
    self.num_features = num_features 
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_layers = num_layers 
    self.num_heads = num_heads 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.embedding_max_period = embedding_max_period
    self.learnable_embedding = learnable_embedding 
    self._pos_encoding_params = pos_encoding_params
    self.use_excitation = use_excitation
    self.use_adaptive = use_adaptive
    self.se_ratio = se_ratio 
    self.use_max_pooling = use_max_pooling 
    self.num_gcn_states = num_gcn_stages
    self.gcn_dim = gcn_dim
    self.flatten_arc = flatten_arc
    self.num_classes = num_classes
    self.use_action = use_action
    self.use_mid_block = use_mid_block
    self.use_concat = use_concat

    if self.learnable_embedding : 
      self.sequence_embedding = nn.Parameter(torch.randn(num_features, model_dim))
    else : 
      positional_encoding = PositionalEncoding1D(
        model_dim = input_dim,
        temperature = self._pos_encoding_params[0],
        alpha = self._pos_encoding_params[1]
      )
      sequence_embedding = positional_encoding(self.num_features)
      self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)
    
    self.conditional_embedding = nn.Linear(
      self.num_features * self.input_dim, 
      self.time_embedding_dim
    )

    self.time_embedding = nn.Sequential(
      nn.Linear(self.model_dim, self.time_embedding_dim), 
      nn.SiLU(), 
      nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
    )

    self.posterior_decoder_block = nn.ModuleList()

    self.attn_blocks = []
    # post-gcn blocks
    self.post_gcn_blocks = GCN(
      input_feature = self.num_features,
      hidden_feature = self.gcn_dim,
      embedding_dim = self.time_embedding_dim, 
      p_dropout = self.dropout_p, 
      normalizer = ScaleShiftLayer, 
      num_stage = self.num_gcn_states,
      node_n = self.input_dim, 
      activation = get_activation(self.activation), 
    ) 

    for index in range(self.num_layers):
      self.posterior_decoder_block.append(
        GCNDiffusionTransformerLayer(
          model_dim = self.model_dim, 
          ffn_dim = self.ffn_dim, 
          embedding_dim = self.time_embedding_dim,
          num_head = self.num_heads, 
          dropout = self.dropout_p, 
          activation = self.activation, 
          num_channels = self.num_features,   
          use_adaptive = self.use_adaptive, 
          se_ratio = self.se_ratio, 
          use_max_pooling = self.use_max_pooling, 
          use_excitation = self.use_excitation, 
          use_concat = (self.use_concat and index >= self.num_layers // 2), 
        )
      )
    self.pose_encoder = nn.Linear(input_dim, model_dim)
    self.proj_p = ConditionalTransformerLayer(
      model_dim = model_dim, 
      output_dim = input_dim, 
      embedding_dim = self.time_embedding_dim, 
    )
    if self.use_mid_block: 
      self.mid_block = GCNDiffusionTransformerLayer(
        model_dim = self.model_dim, 
        ffn_dim = self.ffn_dim, 
        embedding_dim = self.time_embedding_dim,
        num_head = self.num_heads, 
        dropout = self.dropout_p, 
        activation = self.activation, 
        use_excitation = self.use_excitation, 
        use_max_pooling = self.use_max_pooling, 
        se_ratio = self.se_ratio,
        num_channels = self.num_features, 
        use_adaptive = self.use_adaptive, 
        use_concat = False, 
      )
    if self.use_action: 
      self.action_embed = torch.nn.Embedding(self.num_classes + 1, self.model_dim)
    
  def forward(self, input, timesteps, memory = None, actions = None):
    """
    :param input : a tensor of shape (B, num_dct_channels, num_joints * 3) 
    :param timesteps : a tensor of shape (B, ) 
    :param memory : if not None, can be the observed seqs information of shape (B, num_dct_channels, num_joints * 3)
    """
    if self.use_action == False: 
      assert actions is None 

    batch_size, num_dct_channels, _ = input.shape

    time_embedding = self.time_embedding(
      timestep_embedding(
        timesteps = timesteps, 
        dim = self.model_dim, 
        max_period = self.embedding_max_period, 
      )
    )
    total_embedding = time_embedding
    if memory is not None : 
      enc_memory = memory.contiguous().view(batch_size, -1)
      total_embedding  = total_embedding + self.conditional_embedding(enc_memory)

    if self.use_action and actions is not None:
      action_emb = self.action_embed(actions) * (actions != 0).float().unsqueeze(-1)
      total_embedding = total_embedding + action_emb

    x = self.pose_encoder(input)
    x = x + self.sequence_embedding.unsqueeze(0)[:, :num_dct_channels, :]
    

    if self.flatten_arc == False: 
      enc_list = [] 
      for _, module in enumerate(self.posterior_decoder_block[: self.num_layers // 2]): 
        enc_list.append(x) 
        x = module(x, total_embedding)

        
      if self.use_mid_block: 
        x = self.mid_block(x, total_embedding)

      for _, module, in enumerate(self.posterior_decoder_block[self.num_layers // 2 : ]): 
        if self.use_concat == False: 
          x = module(x, total_embedding)
          x += enc_list[-1]  
        else : 
          x = module(torch.cat([x, enc_list[-1]], dim = -1), total_embedding)
        enc_list.pop() 
    else : 
      for _, module in enumerate(self.posterior_decoder_block): 
        x = module(x, total_embedding)  

    out = self.proj_p(x, total_embedding) 
    out = self.post_gcn_blocks(out, total_embedding)
    return out

  
def gcn_transformer_diffusion(params):
  return GCNDiffusionTransformer(
    input_dim = params['pose_dim'], 
    num_features = params['num_dct_channels'], 
    model_dim = params['model_dim'], 
    ffn_dim = params['ffn_dim'], 
    num_layers = params['num_layers'],
    num_heads = params['num_heads'], 
    dropout = params['dropout'], 
    activation = params['activation'],
    embedding_max_period = params['embedding_max_period'], 
    learnable_embedding = params['learnable_embedding'], 
    pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
    se_ratio = params['se_ratio'], 
    use_excitation = params['use_excitation'], 
    use_max_pooling = params['use_max_pooling'], 
    use_adaptive = params['use_adaptive'],
    num_gcn_stages = params['num_gcn_stages'],  
    gcn_dim = params['gcn_dim'], 
    flatten_arc = params['flatten_arc'], 
    num_classes = params['num_classes'], 
    use_action = params['use_action'], 
    use_mid_block = params['use_mid_block'], 
    use_concat = params['use_concat'],  
  )


