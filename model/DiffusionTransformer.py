import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AdaptiveNormLayer(nn.Module):
  def __init__(self, latent_dim, embedding_dim, dropout = 0.1):
    super(AdaptiveNormLayer, self).__init__()
    
    self.latent_dim = latent_dim 
    self.embedding_dim = embedding_dim
    self.dropout = dropout 
    
    self.embedding_proj = zero_module(
      nn.Sequential(
       nn.SiLU(), 
       nn.Linear(embedding_dim, latent_dim * 3),
      )
    )
    self.layernorm = nn.LayerNorm(self.latent_dim)

  def forward(self, latent_input, embedding_input): 
    """
    :param latent_input    : a tensor of shape (B, T, latent_dim)
    :param embedding_input : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, latent_dim)
    """

    embedding_out = self.embedding_proj(embedding_input).unsqueeze(1) # (B, 1, 3 * latent_dim) 
    scale, shift, gate = torch.chunk(embedding_out, 3, dim = -1) 
    out = self.layernorm(latent_input) * (1. + scale) + shift
    return out, gate 
  
class ScaleShiftLayer(nn.Module):
  def __init__(self, latent_dim, embedding_dim, dropout = 0.1): 
    super(ScaleShiftLayer, self).__init__()
    self.latent_dim = latent_dim
    self.embedding_dim = embedding_dim 
    self.layernorm = nn.LayerNorm(self.latent_dim)

    self.embedding_proj = nn.Sequential(
      nn.SiLU(), 
      nn.Linear(embedding_dim, latent_dim * 2), 
    )

    self.proj_out = nn.Sequential(
      nn.SiLU(), 
      nn.Dropout(p = dropout),  
      zero_module(nn.Linear(latent_dim, latent_dim))
    )
  def forward(self, latent_input, embedding_input): 
    embedding_out = self.embedding_proj(embedding_input).unsqueeze(1) # (B, 1, 2* latent_dim) 
    scale, shift = torch.chunk(embedding_out, 2, dim = -1) 
    out = self.layernorm(latent_input) * (1. + scale) + shift
    out = self.proj_out(out) 
    return out 


class AdaptiveFeedForwardNetwork(nn.Module): 
  def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
    super(AdaptiveFeedForwardNetwork, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.embedding_dim = embedding_dim 

    self.adaptive_layernorm = AdaptiveNormLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    ) 

    self.linear_1 = nn.Linear(model_dim, ffn_dim)
    self.linear_2 = nn.Linear(ffn_dim, model_dim)

    def get_activation(activation_fn):
      if activation_fn   == 'GeLU':
        return nn.GELU() 
      elif activation_fn == 'ReLU':
        return nn.ReLU() 
      elif activation_fn == 'LeakyReLU':
        return nn.LeakyReLU() 
      else : 
        raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')
            
    self.act = get_activation(activation_fn = activation)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, input, embedding):
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """  

    norm_inp, gate = self.adaptive_layernorm(input, embedding)
    ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(norm_inp)))) 
    ffn_out = ffn_out * gate 
    out = input + ffn_out
    return out 
    
class AdaptiveMultiHeadAttention(nn.Module): 
  def __init__(self, model_dim, embedding_dim, num_head, dropout):
    super(AdaptiveMultiHeadAttention, self).__init__()
    self.model_dim = model_dim 
    self.embedding_dim = embedding_dim 
    self.num_head = num_head

    assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads" 

    self.dropout = nn.Dropout(dropout)
    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.adaptive_layernorm = AdaptiveNormLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    )

  def forward(self, input, embedding): 
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """       

    batch_size, seq_len, _ = input.shape
    latent_dim = self.model_dim // self.num_head
    
    norm_inp, gate = self.adaptive_layernorm(input, embedding)
        
    query = self.query_proj(norm_inp).contiguous().\
            view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, seq_len, -1)

    key = self.key_proj(norm_inp).contiguous().\
            view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, seq_len, -1)
        
    value = self.value_proj(norm_inp).contiguous().\
            view(batch_size, seq_len,  self.num_head, -1).permute(0, 2, 1, 3)
    value = value.contiguous().view(batch_size * self.num_head, seq_len, -1)

    attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim) 
    attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
    attention_out = torch.bmm(attention_matrix, value) 

    attention_out = attention_out.contiguous().\
            view(batch_size, self.num_head, seq_len, -1).permute(0, 2, 1, 3)
    attention_out = attention_out.contiguous().\
            view(batch_size, seq_len, -1)

    attention_out = attention_out * gate

    out = attention_out + input
    return out 
        
class AdaptiveMultiHeadCrossAttention(nn.Module):
  def __init__(self, model_dim, embedding_dim, num_head, dropout): 
    super(AdaptiveMultiHeadCrossAttention, self).__init__()
    self.model_dim = model_dim 
    self.embedding_dim = embedding_dim
    self.num_head = num_head 
        
    self.dropout = nn.Dropout(dropout)

    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    
    self.adaptive_layernorm = AdaptiveNormLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    )

    self.proj_out = zero_module(nn.Linear(self.model_dim, self.model_dim)) 

    self.memory_layernorm = nn.LayerNorm(self.model_dim)

  def forward(self, input, embedding, memory): 
    """
    :param input     : a tensor of shape (B, T_{predict}, model_dim) 
    :param memory    : a tensor of shape (B, T_{observed}, model_dim)
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """     
    batch_size, tgt_seq_length, _ = input.shape
    _, src_seq_length, _ = memory.shape

    latent_dim = self.model_dim // self.num_head

    norm_input, gate = self.adaptive_layernorm(input, embedding)
    norm_memory = self.memory_layernorm(memory)

    query = self.query_proj(norm_input).contiguous().\
        view(batch_size, tgt_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, tgt_seq_length, -1)
        
    key = self.key_proj(norm_memory).contiguous().\
        view(batch_size, src_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    value = self.value_proj(norm_memory).contiguous().\
        view(batch_size, src_seq_length,  self.num_head, -1).permute(0, 2, 1, 3)
    value = value.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
    cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))
        
    cross_attention_out = torch.bmm(cross_attention_matrix, value) 

    cross_attention_out = cross_attention_out.contiguous().\
        view(batch_size, self.num_head, tgt_seq_length, -1).permute(0, 2, 1, 3)
    cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_seq_length, -1)
    out = gate * self.proj_out(cross_attention_out) + input
    return out 


class FeedForwardNetwork(nn.Module): 
  def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
    super(FeedForwardNetwork, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.embedding_dim = embedding_dim 

    self.scale_shift_norm = ScaleShiftLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    ) 

    self.linear_1 = nn.Linear(model_dim, ffn_dim)
    self.linear_2 = nn.Linear(ffn_dim, model_dim) 

    self.act = get_activation(activation_fn = activation)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, input, embedding):
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """  
    ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(input)))) 
    out = input + self.scale_shift_norm(ffn_out, embedding) 
    return out 
  
class MultiHeadAttention(nn.Module): 
  def __init__(self, model_dim, embedding_dim, num_head, dropout):
    super(MultiHeadAttention, self).__init__()
    self.model_dim = model_dim 
    self.embedding_dim = embedding_dim 
    self.num_head = num_head

    assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads" 

    self.dropout = nn.Dropout(dropout)
    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.scale_shift_norm = ScaleShiftLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    )
    self.layernorm = nn.LayerNorm(self.model_dim)

  def forward(self, input, embedding): 
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """       

    batch_size, seq_len, _ = input.shape
    latent_dim = self.model_dim // self.num_head
    
    norm_inp = self.layernorm(input)
        
    query = self.query_proj(norm_inp).contiguous().\
            view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, seq_len, -1)

    key = self.key_proj(norm_inp).contiguous().\
            view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, seq_len, -1)
        
    value = self.value_proj(norm_inp).contiguous().\
            view(batch_size, seq_len,  self.num_head, -1).permute(0, 2, 1, 3)
    value = value.contiguous().view(batch_size * self.num_head, seq_len, -1)

    attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim) 
    attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
    attention_out = torch.bmm(attention_matrix, value) 

    attention_out = attention_out.contiguous().\
            view(batch_size, self.num_head, seq_len, -1).permute(0, 2, 1, 3)
    attention_out = attention_out.contiguous().\
            view(batch_size, seq_len, -1)

    out = self.scale_shift_norm(attention_out, embedding) + input
    return out 
      
class MultiHeadCrossAttention(nn.Module):
  def __init__(self, model_dim, embedding_dim, num_head, dropout): 
    super(MultiHeadCrossAttention, self).__init__()
    self.model_dim = model_dim 
    self.embedding_dim = embedding_dim
    self.num_head = num_head 
        
    self.dropout = nn.Dropout(dropout)

    self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
    self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
    
    self.scale_shift_norm = ScaleShiftLayer(
      latent_dim = model_dim, 
      embedding_dim = embedding_dim, 
      dropout = dropout, 
    )
    self.layernorm = nn.LayerNorm(self.model_dim)
    self.memory_layernorm = nn.LayerNorm(self.model_dim)

  def forward(self, input, embedding, memory): 
    """
    :param input     : a tensor of shape (B, T_{predict}, model_dim) 
    :param memory    : a tensor of shape (B, T_{observed}, model_dim)
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """     
    batch_size, tgt_seq_length, _ = input.shape
    _, src_seq_length, _ = memory.shape

    latent_dim = self.model_dim // self.num_head

    norm_input = self.layernorm(input)
    norm_memory = self.memory_layernorm(memory)
    query = self.query_proj(norm_input).contiguous().\
        view(batch_size, tgt_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    query = query.contiguous().view(batch_size * self.num_head, tgt_seq_length, -1)
        
    key = self.key_proj(norm_memory).contiguous().\
        view(batch_size, src_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
    key = key.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    value = self.value_proj(norm_memory).contiguous().\
        view(batch_size, src_seq_length,  self.num_head, -1).permute(0, 2, 1, 3)
    value = value.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

    cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
    cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))
        
    cross_attention_out = torch.bmm(cross_attention_matrix, value) 

    cross_attention_out = cross_attention_out.contiguous().\
        view(batch_size, self.num_head, tgt_seq_length, -1).permute(0, 2, 1, 3)
    cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_seq_length, -1)


    out = self.scale_shift_norm(cross_attention_out, embedding) + input
    return out 


class DiffusionTransformerLayer(nn.Module):
  def __init__(
    self, 
    model_dim, 
    ffn_dim, 
    embedding_dim, 
    num_head, 
    dropout, 
    activation, 
    use_cross_attention, 
    use_excitation,
    num_channels, 
    se_ratio, 
    use_max_pooling, 
    use_adaptive, 
    use_concat, 
  ):
    super(DiffusionTransformerLayer, self).__init__()
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.embedding_dim = embedding_dim
    self.num_head = num_head 
    self.dropout = dropout
    self.use_cross_attention = use_cross_attention
    self.use_excitation = use_excitation
    self.num_channels = num_channels 
    self.use_adaptive = use_adaptive
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
    
    if self.use_cross_attention : 
      self.cross_attention = (AdaptiveMultiHeadCrossAttention if self.use_adaptive else MultiHeadCrossAttention)(
        model_dim = model_dim, 
        embedding_dim = embedding_dim, 
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

    if self.use_concat: 
      self.concat = nn.Linear(self.model_dim * 2, self.model_dim)

  def forward(self, input, embedding, memory = None): 
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
      a tensor of shape (B, T, model_dim)
    """       
    if self.use_concat: 
      input = self.concat(input)

    if self.use_excitation: 
      input = self.excitation_layer(input)
    out = self.attention_layer(input, embedding)
    if self.use_cross_attention: 
      assert memory is not None, "Memory is none, cannot perform Cross Attention"
      out = self.cross_attention(out, embedding, memory)
    out = self.ffn_layer(out, embedding)
    return out 


class FinalTransformerLayer(nn.Module):
  def __init__(self, model_dim, output_dim, embedding_dim):
    super(FinalTransformerLayer, self).__init__()
    self.model_dim = model_dim 
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim 

    self.scale_shift_norm = zero_module(
      nn.Sequential(
        nn.SiLU(), 
        nn.Linear(embedding_dim, 2 * model_dim), 
      )
    )
    self.layernorm = nn.LayerNorm(self.model_dim)

    self.linear = zero_module(nn.Linear(model_dim, output_dim))

  def forward(self, input, embedding):
    """
    :param input     : a tensor of shape (B, T, model_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
        a tensor of shape (B, T, model_dim)
    """  
    embedding_out = self.scale_shift_norm(embedding).unsqueeze(1)
    shift, scale =  torch.chunk(embedding_out, 2, dim = -1) 
    norm_input = self.layernorm(input) * (scale + 1.) + shift
    out = self.linear(norm_input)
    return out 
  

class DiffusionTransformer(nn.Module):
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
    use_cross_attention = False, 
    embedding_max_period = 10000,
    learnable_embedding = True, 
    pos_encoding_params = (10000, 1), 
    flatten_arc = False, 
    use_excitation = False, 
    se_ratio = 4, 
    use_max_pooling = False,
    use_adaptive = True,  
    num_classes = None, 
    use_action = False, 
    use_mid_block = False, 
    use_concat = False, 
  ):
    super(DiffusionTransformer, self).__init__()
    self.input_features = input_features
    self.prediction_len = prediction_len 
    self.model_dim = model_dim 
    self.ffn_dim = ffn_dim 
    self.num_layers = num_layers 
    self.use_cross_attention = use_cross_attention 
    self.num_heads = num_heads 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.embedding_max_period = embedding_max_period
    self.learnable_embedding = learnable_embedding 
    self._pos_encoding_params = pos_encoding_params
    self.flatten_arc = flatten_arc 
    self.use_excitation = use_excitation
    self.use_adaptive = use_adaptive
    self.se_ratio = se_ratio 
    self.use_max_pooling = use_max_pooling 
    self.num_classes = num_classes
    self.use_action = use_action
    self.use_mid_block = use_mid_block
    self.use_concat = use_concat

    self.pose_encoder = nn.Linear(input_features, model_dim)

    if self.learnable_embedding : 
      self.sequence_embedding = nn.Parameter(torch.randn(prediction_len, model_dim))
    else : 
      positional_encoding = PositionalEncoding1D(
        model_dim = self.model_dim,
        temperature = self._pos_encoding_params[0],
        alpha = self._pos_encoding_params[1]
      )
      sequence_embedding = positional_encoding(self.prediction_len)
      self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)
    
    self.conditional_embedding = nn.Linear(
      self.input_features * self.prediction_len, 
      self.time_embedding_dim
    )

    self.time_embedding = nn.Sequential(
      nn.Linear(self.model_dim, self.time_embedding_dim), 
      nn.SiLU(), 
      nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
    )

    self.posterior_decoder_block = nn.ModuleList()

    for index in range(self.num_layers):
      self.posterior_decoder_block.append(
        DiffusionTransformerLayer(
          model_dim = self.model_dim, 
          ffn_dim = self.ffn_dim, 
          embedding_dim = self.time_embedding_dim,
          num_head = self.num_heads, 
          dropout = self.dropout_p, 
          activation = self.activation, 
          use_cross_attention = self.use_cross_attention, 
          use_excitation = self.use_excitation, 
          use_max_pooling = self.use_max_pooling, 
          se_ratio = self.se_ratio,
          num_channels = self.prediction_len,  
          use_adaptive = self.use_adaptive, 
          use_concat = (self.use_concat and (index >= self.num_layers // 2))
        )
      )

    if self.use_mid_block: 
      self.mid_block = DiffusionTransformerLayer(
        model_dim = self.model_dim, 
        ffn_dim = self.ffn_dim, 
        embedding_dim = self.time_embedding_dim,
        num_head = self.num_heads, 
        dropout = self.dropout_p, 
        activation = self.activation, 
        use_cross_attention = self.use_cross_attention, 
        use_excitation = self.use_excitation, 
        use_max_pooling = self.use_max_pooling, 
        se_ratio = self.se_ratio,
        num_channels = self.prediction_len,  
        use_adaptive = self.use_adaptive, 
        use_concat = False, 
      )
    
    self.final_layer = FinalTransformerLayer(
      model_dim = model_dim, 
      output_dim = input_features, 
      embedding_dim = self.time_embedding_dim, 
    )

    if self.use_action: 
      self.action_embed = torch.nn.Embedding(self.num_classes + 1, self.model_dim)
      nn.init.normal_(self.action_embed.weight, mean = 0.0, std = 0.02)
    
    
  def forward(self, input, timesteps, memory = None, actions = None):
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
    if memory is not None : 
      enc_memory = memory.contiguous().view(batch_size, -1)
      total_embedding  = total_embedding + self.conditional_embedding(enc_memory)

    if self.use_action and actions is not None:
      action_emb = self.action_embed(actions) * (actions != 0).float().unsqueeze(-1)
      total_embedding = total_embedding + action_emb

    x = self.pose_encoder(input) # (B, T_{prediction}, model_dim)
    x = x + self.sequence_embedding.unsqueeze(0)[:, :num_channels, :]
    
    if self.flatten_arc == False: 
      enc_list = [] 
      for index, module in enumerate(self.posterior_decoder_block): 
        if index < self.num_layers // 2: 
          enc_list.append(x) 
          x = module(x, total_embedding, memory)
      if self.use_mid_block: 
       x = self.mid_block(x, total_embedding, memory)

      for _, module, in enumerate(self.posterior_decoder_block[self.num_layers // 2 : ]): 
        if self.use_concat == False: 
          x = module(x, total_embedding, memory)
          x += enc_list[-1]  
        else : 
          x = module(torch.cat([x, enc_list[-1]], dim = -1), total_embedding, memory)
        enc_list.pop() 
    else : 
      for index, module in enumerate(self.posterior_decoder_block): 
        x = module(x, total_embedding, memory)
    
    output = self.final_layer(x, total_embedding)

    return output 

def diffusion_transformer(params):
  return DiffusionTransformer(
    input_features = params['pose_dim'], 
    prediction_len = params['num_dct_channels'], 
    model_dim = params['model_dim'], 
    ffn_dim = params['ffn_dim'], 
    num_layers = params['num_layers'],
    num_heads = params['num_heads'], 
    dropout = params['dropout'], 
    activation = params['activation'], 
    use_cross_attention = params['use_cross_attention'], 
    embedding_max_period = params['embedding_max_period'], 
    learnable_embedding = params['learnable_embedding'], 
    pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
    flatten_arc = params['flatten_arc'],
    se_ratio = params['se_ratio'], 
    use_excitation = params['use_excitation'], 
    use_max_pooling = params['use_max_pooling'], 
    use_adaptive = params['use_adaptive'], 
    num_classes = params['num_classes'], 
    use_action = params['use_action'],
    use_mid_block = params['use_mid_block'], 
    use_concat = params['use_concat'],  
  )


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math 
# import numpy as np 
# from model.ModelUtils import *


# class MotionDiffusionTransformerLayer(nn.Module):
#   def __init__(
#     self, 
#     model_dim, 
#     ffn_dim, 
#     embedding_dim, 
#     num_head, 
#     dropout, 
#     activation, 
#     use_cross_attention, 
#     use_excitation,
#     num_channels, 
#     se_ratio, 
#     use_max_pooling, 
#     use_pose_enc_dec, 
#     use_adaptive, 
#     use_concat, 
#     do_zero_module, 
#   ):
#     super(MotionDiffusionTransformerLayer, self).__init__()
#     self.model_dim = model_dim 
#     self.ffn_dim = ffn_dim 
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head 
#     self.dropout = dropout
#     self.use_cross_attention = use_cross_attention
#     self.use_excitation = use_excitation
#     self.num_channels = num_channels 
#     self.use_adaptive = use_adaptive
#     self.use_concat = use_concat
#     self.do_zero_module = do_zero_module
#     self.use_pose_enc_dec = use_pose_enc_dec

#     use_stylization = (self.use_pose_enc_dec) or (self.use_adaptive is False)

#     self.attention_layer = (MultiHeadAttention if use_stylization else AdaptiveMultiHeadAttention) (
#       model_dim = model_dim, 
#       num_head  = num_head, 
#       embedding_dim = embedding_dim, 
#       dropout = dropout,
#       do_zero_module = do_zero_module,  
#     )
    
#     self.ffn_layer = (FeedForwardNetworkv1 if self.use_pose_enc_dec else (FeedForwardNetwork\
#       if use_stylization else AdaptiveFeedForwardNetwork))(
#       model_dim = model_dim,
#       ffn_dim = ffn_dim, 
#       embedding_dim = embedding_dim, 
#       dropout = dropout, 
#       activation = activation,  
#       do_zero_module = do_zero_module, 
#     ) 
    
#     if self.use_cross_attention : 
#       self.cross_attention = (MultiHeadCrossAttention if use_stylization else AdaptiveMultiHeadCrossAttention) (
#         model_dim = model_dim, 
#         embedding_dim = embedding_dim, 
#         num_head = num_head, 
#         dropout = dropout, 
#         do_zero_module = do_zero_module, 
#       ) 
#     if self.use_excitation : 
#       self.excitation_layer = SqueezeAndExcitation(
#         channels = num_channels, 
#         ratio = se_ratio, 
#         act_fn = activation, 
#         use_max_pooling = use_max_pooling, 
#         do_zero_module = do_zero_module, 
#       )

#     if self.use_concat: 
#       self.concat = nn.Linear(self.model_dim * 2, self.model_dim)

#   def forward(self, input, embedding, memory = None): 
#     """
#     :param input     : a tensor of shape (B, T, model_dim) 
#     :param embedding : a tensor of shape (B, embedding_dim) 
#     :return : 
#       a tensor of shape (B, T, model_dim)
#     """       
#     if self.use_concat: 
#       input = self.concat(input)

#     if self.use_excitation: 
#       input = self.excitation_layer(input)
#     out = self.attention_layer(input, embedding)
#     if self.use_cross_attention and memory is not None:
#       out = self.cross_attention(out, embedding, memory)
#     out = self.ffn_layer(out, embedding)
#     return out 

# class MotionDiffusionTransformer(nn.Module):
#   def __init__(
#     self, 
#     input_features, 
#     prediction_len, 
#     model_dim, 
#     ffn_dim, 
#     num_layers,
#     num_heads, 
#     prior_len, 
#     dropout = 0.2, 
#     activation = 'GeLU', 
#     use_cross_attention = False, 
#     embedding_max_period = 10000,
#     learnable_embedding = True, 
#     pos_encoding_params = (10000, 1), 
#     flatten_arc = False, 
#     use_excitation = False, 
#     se_ratio = 4, 
#     use_max_pooling = False,
#     use_adaptive = True,  
#     num_classes = None, 
#     use_action = False, 
#     use_mid_block = False, 
#     use_concat = False, 
#     do_zero_module = True, 
#     attention_resolution = None,
#     use_pose_enc_dec = False,
#     dataset = 'h36m', 
#   ):
#     super(MotionDiffusionTransformer, self).__init__()
#     self.input_features = input_features
#     self.prediction_len = prediction_len 
#     self.model_dim = model_dim 
#     self.ffn_dim = ffn_dim 
#     self.num_layers = num_layers 
#     self.use_cross_attention = use_cross_attention 
#     self.num_heads = num_heads 
#     self.dropout_p = dropout 
#     self.activation = activation 
#     self.time_embedding_dim = model_dim
#     self.embedding_max_period = embedding_max_period
#     self.learnable_embedding = learnable_embedding 
#     self._pos_encoding_params = pos_encoding_params
#     self.flatten_arc = flatten_arc 
#     self.use_excitation = use_excitation
#     self.use_adaptive = use_adaptive
#     self.se_ratio = se_ratio 
#     self.use_max_pooling = use_max_pooling 
#     self.num_classes = num_classes
#     self.use_action = use_action
#     self.use_mid_block = use_mid_block
#     self.use_concat = use_concat
#     self.do_zero_module = do_zero_module
#     self.use_cross_attention = use_cross_attention
#     self.prior_len = prior_len
#     self.use_pose_enc_dec = use_pose_enc_dec
#     self.dataset = dataset 

#     if use_cross_attention : 
#       assert attention_resolution is not None, "Attention resolution cannot be none when use cross attention"

#     if self.learnable_embedding : 
#       self.sequence_embedding = nn.Parameter(torch.randn(prediction_len, model_dim))
#     else : 
#       positional_encoding = PositionalEncoding1D(
#         model_dim = self.model_dim,
#         temperature = self._pos_encoding_params[0],
#         alpha = self._pos_encoding_params[1]
#       )
#       sequence_embedding = positional_encoding(self.prediction_len)
#       self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)

#     if self.use_cross_attention is False: 
#       self.conditional_embedding = nn.Linear(
#         self.input_features * self.prior_len, 
#         self.time_embedding_dim
#       )

#     self.time_embedding = nn.Sequential(
#       nn.Linear(self.model_dim, self.time_embedding_dim), 
#       nn.SiLU(), 
#       nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
#     )

#     self.attn_blocks = torch.nn.ModuleList() 

#     for index in range(self.num_layers):
#       self.attn_blocks.append(
#         MotionDiffusionTransformerLayer(
#           model_dim = self.model_dim, 
#           ffn_dim = self.ffn_dim, 
#           embedding_dim = self.time_embedding_dim,
#           num_head = self.num_heads, 
#           dropout = self.dropout_p, 
#           activation = self.activation, 
#           use_cross_attention = True if (self.use_cross_attention and index in attention_resolution) else False, 
#           use_excitation = self.use_excitation, 
#           use_max_pooling = self.use_max_pooling, 
#           se_ratio = self.se_ratio,
#           num_channels = self.prediction_len,  
#           use_adaptive = self.use_adaptive, 
#           use_concat = (self.use_concat and index >= self.num_layers // 2),
#           do_zero_module = do_zero_module, 
#           use_pose_enc_dec = self.use_pose_enc_dec, 
#         )
#       )

#     if self.use_mid_block: 
#       self.mid_block = MotionDiffusionTransformerLayer(
#         model_dim = self.model_dim, 
#         ffn_dim = self.ffn_dim, 
#         embedding_dim = self.time_embedding_dim,
#         num_head = self.num_heads, 
#         dropout = self.dropout_p, 
#         activation = self.activation, 
#         use_cross_attention = False, 
#         use_excitation = self.use_excitation, 
#         use_max_pooling = self.use_max_pooling, 
#         se_ratio = self.se_ratio,
#         num_channels = self.prediction_len,  
#         use_adaptive = self.use_adaptive, 
#         do_zero_module = do_zero_module, 
#         use_concat = False, 
#         use_pose_enc_dec = self.use_pose_enc_dec, 
#       )

#     if self.use_pose_enc_dec:
#       self.pose_encoder = PoseEncoder(dataset_name = dataset, input_dim = input_features)
#       self.pose_decoder = PoseDecoder(dataset_name = dataset, input_dim = input_features, latent_dim = model_dim)
#     else :
#       self.pose_encoder = nn.Linear(input_features, model_dim)
#       self.pose_decoder = (AdaptiveFinalTransformerLayer if self.use_adaptive else FinalTransformerLayer)(
#         model_dim = model_dim, 
#         output_dim = input_features, 
#         embedding_dim = self.time_embedding_dim,
#         do_zero_module = do_zero_module,  
#       )

#     if self.use_action: 
#       self.action_embed = torch.nn.Embedding(self.num_classes + 1, self.model_dim)
#       nn.init.normal_(self.action_embed.weight, mean = 0.0, std = 0.02)
    
    
#   def forward(self, input, timesteps, memory = None, actions = None):
#     """
#     :param input : a tensor of shape (B, T_{prediction}, input_features) 
#     :param timesteps : a tensor of shape (B, ) 
#     :param memory : if not None, can be the observed seqs information 
#     """
#     if self.use_action == False: 
#       assert actions is None 

#     batch_size, num_channels, _ = input.shape
  

#     time_embedding = self.time_embedding(
#       timestep_embedding(
#         timesteps = timesteps, 
#         dim = self.model_dim, 
#         max_period = self.embedding_max_period, 
#       )
#     )

#     total_embedding = time_embedding

#     if memory is not None:  
#       if self.use_cross_attention is False: 
#        enc_memory = memory.contiguous().view(batch_size, -1)
#        total_embedding = total_embedding + self.conditional_embedding(enc_memory)
#       else : 
#         memory = self.pose_encoder(memory)

#     if self.use_action and actions is not None:
#       action_emb = self.action_embed(actions) * (actions != 0).float().unsqueeze(-1)
#       total_embedding = total_embedding + action_emb

#     x = self.pose_encoder(input) # (B, T_{prediction}, model_dim)
#     x = x + self.sequence_embedding.unsqueeze(0)[:, :num_channels, :]

#     if self.flatten_arc: 
#       # flat transformer 
#       for index, module in enumerate(self.attn_blocks): 
#         x = module(x, total_embedding, memory)
#     else :
#       if self.use_concat:  
#         enc_list = [] 
#         for module in self.attn_blocks[:self.num_layers // 2]: 
#           x = module(x, total_embedding, memory)
#           enc_list.append(x)
   
#         if self.use_mid_block: 
#           x = self.mid_block(x, total_embedding, memory)

#         for module in self.attn_blocks[self.num_layers // 2:]:
#           x = module(torch.cat([x, enc_list[-1]], dim = -1), total_embedding, memory)
#           enc_list.pop() 
      

#       else: 
#         enc_list = [] 
#         for module in self.attn_blocks[:self.num_layers // 2]: 
#           enc_list.append(x)
#           x = module(x, total_embedding, memory)
        
#         if self.use_mid_block: 
#           x = self.mid_block(x, total_embedding, memory)
        
#         for module in self.attn_blocks[self.num_layers // 2:]: 
#           x = module(x, total_embedding, memory) 
#           x += enc_list[-1]
#           enc_list.pop()
  

#     if isinstance(self.pose_decoder, PoseDecoder):
#       output = self.pose_decoder(x)
#     else : 
#       output = self.pose_decoder(x, total_embedding)

#     return output, total_embedding

  
# def motion_diffusion_transformer(params):
#   return MotionDiffusionTransformer(
#     input_features = params['pose_dim'], 
#     prediction_len = params['n_pred'], 
#     prior_len = params['n_prior'], 
#     model_dim = params['model_dim'], 
#     ffn_dim = params['ffn_dim'], 
#     num_layers = params['num_layers'],
#     num_heads = params['num_heads'], 
#     dropout = params['dropout'], 
#     activation = params['activation'], 
#     use_cross_attention = params['use_cross_attention'], 
#     embedding_max_period = params['embedding_max_period'], 
#     learnable_embedding = params['learnable_embedding'], 
#     pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
#     flatten_arc = params['flatten_arc'],
#     se_ratio = params['se_ratio'], 
#     use_excitation = params['use_excitation'], 
#     use_max_pooling = params['use_max_pooling'], 
#     use_adaptive = params['use_adaptive'], 
#     num_classes = params['num_classes'], 
#     use_action = params['use_action'],
#     use_mid_block = params['use_mid_block'], 
#     use_concat = params['use_concat'],  
#     do_zero_module = params['do_zero_module'], 
#     use_pose_enc_dec = params['use_pose_enc_dec'], 
#     attention_resolution = params['attention_resolution'], 
#   )


# class MotionRefinementNet(nn.Module): 
#   def __init__(self, predictor : nn.Module, corrector : nn.Module):
#     super(MotionRefinementNet, self).__init__()
#     self.predictor = predictor  
#     self.corrector = corrector 

#   def forward(self, x, t, memory = None, actions = None):
#     x, embed = self.predictor(x, t, memory, actions) 
#     x = x.permute(0, 2, 1) 
#     out = self.corrector(x, embed)
#     out = out.permute(0, 2, 1) 
#     return out 

