import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import math 
import numpy as np 
from model.ModelUtils import *

class SqueezeAndExcitation(nn.Module): 
  def __init__(self, channels, ratio = 4, act_fn = 'GeLU', use_max_pooling = False):
    super(SqueezeAndExcitation, self).__init__()
    self.channels = channels 
    self.ratio = ratio 
    self.use_max_pooling = use_max_pooling 

    self.squeeze = nn.AdaptiveAvgPool1d(1) if not \
      self.use_max_pooling else nn.AdaptiveMaxPool1d(1)

    self.excitation = nn.Sequential(
      nn.Linear(channels, channels // ratio, bias = False), 
      get_activation(activation_fn = act_fn),
      zero_module(nn.Linear(channels // ratio, channels, bias = False)), 
      nn.Sigmoid(), 
    )
  
  def forward(self, x): 
    batch_size, channels, _ = x.shape 
    y = self.squeeze(x).view(batch_size, channels) 
    y = self.excitation(y).view(batch_size, channels, 1) 
    return x + x * y.expand_as(x) 

class MultiHeadAttention(nn.Module): 
  def __init__(self, model_dim, num_head, dropout):
    super(MultiHeadAttention, self).__init__()
    self.model_dim = model_dim  
    self.num_head = num_head
    assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads" 
    self.dropout = nn.Dropout(dropout)
    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.out_proj = nn.Linear(self.model_dim, self.model_dim)

  def forward(self, x): 
    """
    :param x : a tensor of shape (B, T, model_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """       

    # FIX: was reading from `input` (undefined / builtin) instead of `x`
    batch_size, seq_len, _ = x.shape
    latent_dim = self.model_dim // self.num_head
        
    query = self.query_proj(x).contiguous().\
        view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, seq_len, -1)

    key = self.key_proj(x).contiguous().\
        view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, seq_len, -1)
        
    value = self.value_proj(x).contiguous().\
        view(batch_size, seq_len,  self.num_head, -1).permute(0, 2, 1, 3)

    value = value.contiguous().view(batch_size * self.num_head, seq_len, -1)

    attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim) 
    attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
    attention_out = torch.bmm(attention_matrix, value) 

    attention_out = attention_out.contiguous().\
            view(batch_size, self.num_head, seq_len, -1).permute(0, 2, 1, 3)
    attention_out = attention_out.contiguous().\
            view(batch_size, seq_len, -1)
    out = self.out_proj(attention_out)
    return out 

class FeedForwardNetwork(nn.Module): 
  def __init__(self, model_dim, ffn_dim, dropout = 0.1, activation = 'GeLU'):
    super(FeedForwardNetwork, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim  

    self.linear_1 = nn.Linear(model_dim, ffn_dim)
    self.linear_2 = nn.Linear(ffn_dim, model_dim) 
    self.act = get_activation(activation_fn = activation)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, input):
    ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(input)))) 
    return ffn_out

class MultiHeadCrossAttention(nn.Module):
  def __init__(self, model_dim, num_head, dropout): 
    super(MultiHeadCrossAttention, self).__init__()
    self.model_dim = model_dim 
    self.num_head = num_head
    self.dropout = nn.Dropout(dropout)
    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.out_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)

  def forward(self, input, memory): 
    """
    :param input : a tensor of shape (B, num_channels, model_dim) 
    :param memory : a tensor of shape (B, num_channels, model_dim)
    :return : 
        a tensor of shape (B, T, model_dim)
    """     
    batch_size, tgt_seq_length, _ = input.shape
    _, src_seq_length, _ = memory.shape

    latent_dim = self.model_dim // self.num_head

    query = self.query_proj(input).contiguous().\
        view(batch_size, tgt_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, tgt_seq_length, -1)
        
    key = self.key_proj(memory).contiguous().\
        view(batch_size, src_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    value = self.value_proj(memory).contiguous().\
        view(batch_size, src_seq_length,  self.num_head, -1).permute(0, 2, 1, 3)

    value = value.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
    cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))
        
    cross_attention_out = torch.bmm(cross_attention_matrix, value) 

    cross_attention_out = cross_attention_out.contiguous().\
        view(batch_size, self.num_head, tgt_seq_length, -1).permute(0, 2, 1, 3)
    cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_seq_length, -1)
    out = self.out_proj(cross_attention_out)
    return out 

class RewardTransformerLayer(nn.Module):
  def __init__(
    self, 
    model_dim, 
    ffn_dim,  
    num_head, 
    dropout, 
    activation, 
    use_excitation,
    num_channels, 
    se_ratio, 
    use_max_pooling, 
    use_adaptive, 
  ):
    super(RewardTransformerLayer, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_head = num_head 
    self.dropout = dropout
    self.use_excitation = use_excitation
    self.num_channels = num_channels 
    self.use_adaptive = use_adaptive

    self.norm_a = nn.LayerNorm(model_dim) 
    self.norm_b = nn.LayerNorm(model_dim)
    self.norm_c = nn.LayerNorm(model_dim)
    self.norm_m = nn.LayerNorm(model_dim)

    self.self_attn = MultiHeadAttention(
      model_dim = model_dim, 
      num_head = num_head, 
      dropout = dropout, 
    )
    
    self.ffn_layer = FeedForwardNetwork(
      model_dim = model_dim,
      ffn_dim = ffn_dim, 
      dropout = dropout, 
      activation = activation,  
    ) 
    self.cross_attention = MultiHeadCrossAttention(
      model_dim = model_dim,  
      num_head = num_head, 
      dropout = dropout, 
    ) 
    if self.use_excitation : 
      self.excitation_layer = SqueezeAndExcitation(
        channels = num_channels, 
        ratio = se_ratio, 
        act_fn = activation, 
        use_max_pooling = use_max_pooling, 
      )

  def forward(self, input, memory): 
    x = input 
    if self.use_excitation: 
      x = self.excitation_layer(x)
    x = x + self.self_attn(self.norm_a(x))
    x = x + self.cross_attention(self.norm_b(x), self.norm_m(memory))
    out = x + self.ffn_layer(self.norm_c(x))
    return out 

class RewardEncoderLayer(nn.Module): 
  def __init__(
    self, 
    model_dim, 
    ffn_dim,  
    num_head, 
    dropout, 
    activation, 
    use_excitation,
    num_channels, 
    se_ratio, 
    use_max_pooling, 
    use_adaptive, 
  ):
    super(RewardEncoderLayer, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_head = num_head 
    self.dropout = dropout
    self.use_excitation = use_excitation
    self.num_channels = num_channels 
    self.use_adaptive = use_adaptive

    self.norm_a = nn.LayerNorm(model_dim) 
    self.norm_b = nn.LayerNorm(model_dim)

    self.self_attn = MultiHeadAttention(
    model_dim = model_dim, 
    num_head = num_head, 
    dropout = dropout, 
    )
    
    self.ffn_layer = FeedForwardNetwork(
    model_dim = model_dim,
    ffn_dim = ffn_dim, 
    dropout = dropout, 
    activation = activation,  
    ) 
    if self.use_excitation : 
      self.excitation_layer = SqueezeAndExcitation(
        channels = num_channels, 
        ratio = se_ratio, 
        act_fn = activation, 
        use_max_pooling = use_max_pooling, 
      )
  
  def forward(self, x): 
    if self.use_excitation: 
      x = self.excitation_layer(x)
    x = x + self.self_attn(self.norm_a(x))
    out = x + self.ffn_layer(self.norm_b(x))
    return out 
  
class RewardTransformer(nn.Module):
  def __init__(
    self, 
    input_features, 
    num_dct_channels, 
    model_dim, 
    ffn_dim, 
    num_layers,
    num_heads, 
    dropout = 0.2, 
    activation = 'GeLU', 
    learnable_embedding = True, 
    pos_encoding_params = (10000, 1), 
    use_excitation = False, 
    se_ratio = 4, 
    use_max_pooling = False,
    code_size = 128, 
  ):
    super(RewardTransformer, self).__init__()
    self.input_features = input_features
    self.num_dct_channels = num_dct_channels
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_layers = num_layers 
    self.num_heads = num_heads 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.learnable_embedding = learnable_embedding 
    self._pos_encoding_params = pos_encoding_params
    self.use_excitation = use_excitation
    self.se_ratio = se_ratio 
    self.use_max_pooling = use_max_pooling 
    self.code_size = code_size 
    self.hidden_dim = model_dim

    self.pose_encoder = nn.Linear(input_features, model_dim)

    if self.learnable_embedding : 
      self.sequence_embedding = nn.Parameter(torch.randn(num_dct_channels, model_dim))
    else : 
      positional_encoding = PositionalEncoding1D(
        model_dim = self.model_dim,
        temperature = self._pos_encoding_params[0],
        alpha = self._pos_encoding_params[1]
      )
      sequence_embedding = positional_encoding(self.num_dct_channels)
      self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)
    

    self.attn_blocks = [] 

    for _ in range(self.num_layers):
      self.attn_blocks.append(
        RewardTransformerLayer(
          model_dim = self.model_dim, 
          ffn_dim = self.ffn_dim, 
          num_head = self.num_heads, 
          dropout = self.dropout_p, 
          activation = self.activation, 
          use_excitation = self.use_excitation, 
          use_max_pooling = self.use_max_pooling, 
          se_ratio = self.se_ratio,
          num_channels = self.num_dct_channels,
          use_adaptive = False,
        )
      )
    self.attn_blocks = nn.ModuleList(self.attn_blocks)
    self.avg_module = torch.nn.AdaptiveAvgPool1d(1)
    self.code_embedder = nn.Sequential(
      nn.Linear(self.num_dct_channels, self.hidden_dim),
      get_activation(self.activation), 
      nn.Dropout(p = self.dropout_p), 
      nn.Linear(self.hidden_dim, self.code_size), 
    )

    
  def forward(self, input, memory):
    """
    :param input : a tensor of shape (B, num_dct_channels, input_features) 
    :param memory : if not None, can be the observed seqs information 
    """

    x = self.pose_encoder(input) # (B, num_dct_channels, model_dim)
    x = x + self.sequence_embedding.unsqueeze(0)[:, :self.num_dct_channels, :]

    for _, module in enumerate(self.attn_blocks): 
      x = module(x, memory)

    out = self.avg_module(x).squeeze(-1) # optionally add a learnable layer after this 
    out = self.code_embedder(out)
    return out

class RewardEncoder(nn.Module):
  def __init__(
    self, 
    input_features, 
    num_dct_channels, 
    model_dim, 
    ffn_dim, 
    num_layers,
    num_heads, 
    dropout = 0.2, 
    activation = 'GeLU', 
    learnable_embedding = True, 
    pos_encoding_params = (10000, 1), 
    use_excitation = False, 
    se_ratio = 4, 
    use_max_pooling = False,
    code_size = 128, 
  ):
    super(RewardEncoder, self).__init__()
    self.input_features = input_features
    self.num_dct_channels = num_dct_channels
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_layers = num_layers 
    self.num_heads = num_heads 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.learnable_embedding = learnable_embedding 
    self._pos_encoding_params = pos_encoding_params
    self.use_excitation = use_excitation
    self.se_ratio = se_ratio 
    self.use_max_pooling = use_max_pooling 
    self.code_size = code_size
    self.hidden_dim = model_dim

    self.pose_encoder = nn.Linear(input_features, model_dim)

    if self.learnable_embedding : 
      self.sequence_embedding = nn.Parameter(torch.randn(num_dct_channels, model_dim))
    else : 
      positional_encoding = PositionalEncoding1D(
        model_dim = self.model_dim,
        temperature = self._pos_encoding_params[0],
        alpha = self._pos_encoding_params[1]
      )
      sequence_embedding = positional_encoding(self.num_dct_channels)
      self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)
    

    self.attn_blocks = [] 

    for _ in range(self.num_layers):
      self.attn_blocks.append(
        RewardEncoderLayer(
          model_dim = self.model_dim, 
          ffn_dim = self.ffn_dim, 
          num_head = self.num_heads, 
          dropout = self.dropout_p, 
          activation = self.activation, 
          use_excitation = self.use_excitation, 
          use_max_pooling = self.use_max_pooling, 
          se_ratio = self.se_ratio,
          num_channels = self.num_dct_channels,
          use_adaptive = False,
        )
      )
    self.attn_blocks = nn.ModuleList(self.attn_blocks)
    self.avg_module = torch.nn.AdaptiveAvgPool1d(1)
    self.code_embedder = nn.Sequential(
      nn.Linear(self.num_dct_channels, self.hidden_dim),
      get_activation(self.activation), 
      nn.Dropout(p = self.dropout_p), 
      nn.Linear(self.hidden_dim, self.code_size), 
    )
    
  def forward(self, x):
    x = self.pose_encoder(x) # (B, num_dct_channels, model_dim)
    x = x + self.sequence_embedding.unsqueeze(0)[:, :self.num_dct_channels, :]
    for _, module in enumerate(self.attn_blocks): 
      x = module(x)
    out = self.avg_module(x).squeeze(-1) 
    out = self.code_embedder(out)
    return out

class GraphConvolution(nn.Module):
  def __init__(self, in_features, out_features, output_nodes = 48, bias = False):
    super(GraphConvolution, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self._output_nodes = output_nodes

    self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
    self.att = torch.nn.Parameter(torch.FloatTensor(output_nodes, output_nodes))
    
    if bias:
      self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    self.att.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)


  def forward(self, x):
    support = torch.matmul(x, self.weight)
    output = torch.matmul(self.att, support)

    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GC_Block(nn.Module):

  def __init__(self, in_features, p_dropout, output_nodes = 48, bias = False):
    """Constructor.

    Args:
      in_features: Number of input and output features.
      p_dropout: Dropout used in the layers.
      output_nodes: Number  of output nodes in the graph.
    """
    super(GC_Block, self).__init__()
    self.in_features = in_features
    self.out_features = in_features

    self.gc1 = GraphConvolution(
      in_features, 
      in_features,
      output_nodes = output_nodes, 
      bias = bias
    )
    self.bn1 = nn.BatchNorm1d(output_nodes * in_features)
    self.gc2 = GraphConvolution(
      in_features, 
      in_features, 
      output_nodes = output_nodes, 
      bias = bias
    )
    self.bn2 = nn.BatchNorm1d(output_nodes * in_features)
    self.do = nn.Dropout(p_dropout)
    self.act_f = nn.Tanh()

  def forward(self, x):
    """Forward pass of the residual module"""
    y = self.gc1(x)
    b, n, f = y.shape
    y = self.bn1(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    y = self.gc2(y)
    b, n, f = y.shape
    y = self.bn2(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)
    return y + x

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
  
class GCN(nn.Module):
  def __init__(self, input_feature, hidden_feature, p_dropout, num_stage = 1, node_n = 48, act_f = nn.GELU(approximate = 'tanh')):
    super(GCN, self).__init__()
    self.num_stage = num_stage
    self.gc1 = GraphConvolution(input_feature, hidden_feature, output_nodes = node_n)
    self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
    self.gcbs = []
    for _ in range(num_stage):
       self.gcbs.append(GC_Block(hidden_feature, p_dropout = p_dropout, output_nodes = node_n))

    self.gcbs = nn.ModuleList(self.gcbs)
    self.gc7 = GraphConvolution(hidden_feature, input_feature, output_nodes = node_n)

    self.do = nn.Dropout(p_dropout)
    self.act_f = act_f

  def forward(self, x):
    y = self.gc1(x)
    b, n, f = y.shape
    y = self.bn1(y.view(b, -1)).view(b, n, f)
    y = self.act_f(y)
    y = self.do(y)

    for i in range(self.num_stage):
        y = self.gcbs[i](y)

    y = self.gc7(y)
    y = y + x

    return y

class GCNRewardModel(nn.Module): 
  def __init__(
    self, 
    node_n, 
    num_dct_channels, 
    hidden_features, 
    num_gcn_stages, 
    dropout,
    activation, 
    code_size, 
  ):
    super(GCNRewardModel, self).__init__()
    self.node_n = node_n
    self.num_dct_channels = num_dct_channels
    self.hidden_features = hidden_features 
    self.num_gcn_stages = num_gcn_stages
    self.act_f = get_activation(activation)
    self.code_size = code_size
    self.hidden_dim = hidden_features
    self.gcn_mod = GCN(
      input_feature = self.num_dct_channels, 
      hidden_feature = self.hidden_features, 
      p_dropout = dropout, 
      num_stage = self.num_gcn_stages, 
      node_n = node_n,
      act_f = self.act_f,   
    )
    self.avg_module = torch.nn.AdaptiveAvgPool1d(1)
    self.code_embedder = nn.Sequential(
      nn.Linear(self.num_dct_channels, self.hidden_dim),
      self.act_f, 
      nn.Dropout(p = dropout), 
      nn.Linear(self.hidden_dim, self.code_size), 
    )
    
    
  def forward(self, x):
    x = self.gcn_mod(x.permute(0, 2, 1))
    out = self.avg_module(x.permute(0, 2, 1)).squeeze(-1)
    out = self.code_embedder(out)
    return out 

def reward_transformer(params):
  return RewardTransformer(
    input_features = params['pose_dim'], 
    num_dct_channels = params['num_dct_channels'], 
    model_dim = params['model_dim'], 
    ffn_dim = params['ffn_dim'], 
    num_layers = params['num_layers'],
    num_heads = params['num_heads'], 
    dropout = params['dropout'], 
    activation = params['activation'], 
    learnable_embedding = params['learnable_embedding'], 
    pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
    se_ratio = params['se_ratio'], 
    use_excitation = params['use_excitation'], 
    use_max_pooling = params['use_max_pooling'], 
    code_size = params['code_size'], 
  )

def reward_encoder(params):
  return RewardEncoder(
    input_features = params['pose_dim'], 
    num_dct_channels = params['num_dct_channels'], 
    model_dim = params['model_dim'], 
    ffn_dim = params['ffn_dim'], 
    num_layers = params['num_layers'],
    num_heads = params['num_heads'], 
    dropout = params['dropout'], 
    activation = params['activation'], 
    learnable_embedding = params['learnable_embedding'], 
    pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
    se_ratio = params['se_ratio'], 
    use_excitation = params['use_excitation'], 
    use_max_pooling = params['use_max_pooling'], 
    code_size = params['code_size'], 
  )

def reward_gcn(params):
  return GCNRewardModel(
    node_n = params['pose_dim'],
    num_dct_channels = params['num_dct_channels'], 
    hidden_features = params['model_dim'], 
    num_gcn_stages = params['num_gcn_stages'], 
    dropout = params['dropout'], 
    code_size = params['code_size'], 
    activation = params['activation'], 
  )

class ExpNLLLoss(_WeightedLoss): 
  def __init__(
   self, 
   weight = None, 
   size_average = None, 
   ignore_index = -100,
   reduce = None, 
   reduction = 'mean'
  ):
    super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
    self.ignore_index = ignore_index

  def forward(self, input, target): 
    x = torch.log(input + 1e-9)
    return F.nll_loss(
      x, target, weight = self.weight, ignore_index = self.ignore_index, reduction = self.reduction  
    )

class InfoNCE(nn.Module): 
  def __init__(self, encoder, prior_encoder, sample_model = None, sampler = None, num_samples = 0, temperature = 0.07):
    super(InfoNCE, self).__init__()
    self.temperature = temperature 
    self.contrast_loss = ExpNLLLoss()
    self.encoder = encoder 
    self.prior_encoder = prior_encoder 
    self.sample_model = sample_model
    self.sampler = sampler 
    self.num_samples = num_samples 

  def sampling(self, trajectory_dct, trajectory_dct_mod, mode_dict): 
    """
    :param trajectory_dct : compressed DCT of the entire true sequence [b, num_dct_channels, pose_dim]
    :param trajectory_dct_mod : compressed DCT of the prior/observed-padded sequence [b, num_dct_channels, pose_dim]
    """
    return self.sampler.sample_ddim(
      model = self.sampler_model, 
      traj_dct = trajectory_dct, 
      traj_dct_mod = trajectory_dct_mod, 
      mode_dict = mode_dict, 
    )


  def forward_sampler(self, trajectory_dct, trajectory_dct_mod, mode_dict = None):
    """
    :param trajectory_dct : compressed DCT of the entire true sequence [b, num_dct_channels, pose_dim]
    :param trajectory_dct_mod : compressed DCT of the prior/observed-padded sequence [b, num_dct_channels, pose_dim]
    """
    batch_size = trajectory_dct.shape[0] 
    samples_arr = []
    for _ in range(self.num_samples):
      sample = self.sampling(trajectory_dct_mod, trajectory_dct_mod, mode_dict) # mod out since we assume that the entire frame is unseen 
      samples_arr.append(sample)

    samples_arr = torch.stack(samples_arr, dim = 1)  # [b, K, num_dct_channels, pose_dim]
    samples_trajectory = samples_arr.contiguous().view(-1, samples_arr.shape[2], samples_arr.shape[3])

    enc_trajectory_dct_sample = self.encoder(samples_trajectory)  # [b*K, code_size]
    enc_trajectory_dct_sample = enc_trajectory_dct_sample.contiguous().view(batch_size, \
      self.num_samples, enc_trajectory_dct_sample.shape[-1])  # [b, K, code_size]

    enc_trajectory_dct_main = self.encoder(trajectory_dct).unsqueeze(1)  # [b, 1, code_size]
    enc_trajectory_dct = torch.cat([enc_trajectory_dct_main, enc_trajectory_dct_sample], dim=1)  # [b, 1+K, code_size]
    enc_trajectory_dct_mod = self.prior_encoder(trajectory_dct_mod).unsqueeze(1)  # [b, 1, code_size]

    enc_trajectory_dct = F.normalize(enc_trajectory_dct, dim = -1)
    enc_trajectory_dct_mod = F.normalize(enc_trajectory_dct_mod, dim = -1)

    all_to_prior = torch.matmul(enc_trajectory_dct, enc_trajectory_dct_mod.transpose(-2, -1)) / self.temperature  # [b, 1 + K, 1]
    prior_to_all = torch.matmul(enc_trajectory_dct_mod, enc_trajectory_dct.transpose(-2, -1)) / self.temperature  # [b, 1, 1 + K]

    all_to_prior = F.softmax(all_to_prior.squeeze(-1), dim = -1)   # [b, 1 + K]
    prior_to_all = F.softmax(prior_to_all.squeeze(1), dim = -1)    # [b, 1 + K]


    # the true trajectory is always placed at index 0, so target class is always 0
    targets = torch.zeros(batch_size, device = trajectory_dct.device, dtype = torch.long)

    all_to_prior_loss = self.contrast_loss(input = all_to_prior, target = targets)
    prior_to_all_loss = self.contrast_loss(input = prior_to_all, target = targets)
    loss = (all_to_prior_loss + prior_to_all_loss) / 2.0

    with torch.no_grad():
      pred_all_to_prior = all_to_prior.argmax(dim = -1)
      pred_prior_to_all = prior_to_all.argmax(dim = -1)
      acc_all_to_prior = (pred_all_to_prior == targets).float().mean()
      acc_prior_to_all = (pred_prior_to_all == targets).float().mean()
      acc = (acc_all_to_prior + acc_prior_to_all) / 2.0

    return loss, acc

  
  def forward_batch(self, trajectory_dct, trajectory_dct_mod): 
    """
    :param trajectory_dct : compressed
     of the entire sequences [b, num_dct_channels, pose_dim] 
    :param trajectory_dct_mod : compressed of the prior [b, num_dct_channels, pose_dim]
    """
    batch_size = trajectory_dct.shape[0]
    enc_trajectory_dct = self.encoder(trajectory_dct) # [b, num_dct_channels]
    enc_trajectory_dct_mod = self.prior_encoder(trajectory_dct_mod) # [b, num_dct_channels]

    enc_trajectory_dct = F.normalize(enc_trajectory_dct, dim = -1) 
    enc_trajectory_dct_mod = F.normalize(enc_trajectory_dct_mod, dim = -1)

    all_to_prior = torch.matmul(enc_trajectory_dct, enc_trajectory_dct_mod.t()) / self.temperature 
    prior_to_all = torch.matmul(enc_trajectory_dct_mod, enc_trajectory_dct.t()) / self.temperature  

    all_to_prior = F.softmax(all_to_prior, dim = -1)
    prior_to_all = F.softmax(prior_to_all, dim = -1)


    targets = torch.arange(batch_size, device = trajectory_dct.device, dtype = torch.long)
    all_to_prior_loss = self.contrast_loss(input = all_to_prior, target = targets)
    prior_to_all_loss = self.contrast_loss(input = prior_to_all, target = targets) 
    loss = (all_to_prior_loss + prior_to_all_loss) / 2.0
    with torch.no_grad(): 
      pred_all_to_prior = all_to_prior.argmax(dim = -1)
      pred_prior_to_all = prior_to_all.argmax(dim = -1)
      acc_all_to_prior = (pred_all_to_prior == targets).float().mean()
      acc_prior_to_all = (pred_prior_to_all == targets).float().mean()
      acc = (acc_all_to_prior + acc_prior_to_all) / 2.0
    return loss, acc

  def forward(self, trajectory_dct, trajectory_dct_mod, mode_dict = None): 
    if self.sampler is None: 
      return self.forward_batch(trajectory_dct, trajectory_dct_mod)
    return self.forward_sampler(trajectory_dct, trajectory_dct_mod, mode_dict)


# For this we need to ensure that output is a scalar 
class ConditioningInfoNCE(nn.Module): 
  def __init__(self, model, sample_model = None, sampler = None, num_samples = 0, temperature = 0.07):
    super(ConditioningInfoNCE, self).__init__()
    self.temperature = temperature 
    self.contrast_loss = ExpNLLLoss()
    self.cond_prob_estimator = model 
    self.sample_model = sample_model
    self.sampler = sampler 
    self.num_samples = num_samples 
    

  def sampling(self, trajectory_dct, trajectory_dct_mod, mode_dict): 
    """
    :param trajectory_dct : compressed DCT of the entire true sequence [b, num_dct_channels, pose_dim]
    :param trajectory_dct_mod : compressed DCT of the prior/observed-padded sequence [b, num_dct_channels, pose_dim]
    """
    return self.sampler.sample_ddim(
      model = self.sampler_model, 
      traj_dct = trajectory_dct, 
      traj_dct_mod = trajectory_dct_mod, 
      mode_dict = mode_dict, 
    )

  def forward(self, trajectory_dct, trajectory_dct_mod, mode_dict = None):
    """
    :param trajectory_dct : compressed DCT of the entire true sequence [b, num_dct_channels, pose_dim]
    :param trajectory_dct_mod : compressed DCT of the prior/observed-padded sequence [b, num_dct_channels, pose_dim]
    : training of this is going to be slow since we need to generated samples for learning 
    """
    batch_size = trajectory_dct.shape[0] 
    estimated_score = [self.cond_prob_estimator(trajectory_dct, trajectory_dct_mod)]
    for _ in range(self.num_samples):
      sample = self.sampling(trajectory_dct_mod, trajectory_dct_mod, mode_dict) # mod out since we assume that the entire frame is unseen 
      estimated_score.append(self.cond_prob_estimator(sample, trajectory_dct_mod)) # 

    estimated_score = torch.stack(estimated_score, dim = 1).squeeze(-1)  # [b, 1 + K]
    estimated_score = F.normalize(estimated_score, dim = -1) / self.temperature
    estimated_score = F.softmax(estimated_score, dim = -1)  
    targets = torch.arange(batch_size, device = trajectory_dct.device, dtype = torch.long)
    loss = self.contrast_loss(input = estimated_score, target = targets)
    with torch.no_grad():
      pred = estimated_score.argmax(dim = -1)
      acc = (pred == targets).float().mean()
    return loss, acc

  