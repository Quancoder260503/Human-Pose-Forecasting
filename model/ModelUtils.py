import os
import sys
import logging
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np

"""
Utility functions for building models. This file contains various helper functions and classes that are used in the model implementations,
including Multi Head Attention, Feedforward Networks, which play the backbone role in the model architecture.
"""

logger = logging.getLogger(__name__)


def make_flatten_func(joint_dim):
  return lambda i : list(range(i * joint_dim, i * joint_dim + joint_dim))


def get_part_slice(joint_indices, func):
  flat = []
  for j in joint_indices:
    flat.extend(func(j))
  return flat


def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module


def timestep_embedding(timesteps, dim, max_period = 10000):
   """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
   """
   half = dim // 2
   freqs = torch.exp(-math.log(max_period) * torch.\
                     arange(start = 0, end = half, dtype=torch.float32) / half).to(device=timesteps.device)
   args = timesteps[:, None].float() * freqs[None]
   embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
   if dim % 2:
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
   return embedding


def get_activation(activation_fn):
  if activation_fn == 'GeLU':
    return nn.GELU(approximate = 'tanh')
  elif activation_fn == 'ReLU':
    return nn.ReLU()
  elif activation_fn == 'LeakyReLU':
    return nn.LeakyReLU()
  elif activation_fn == 'ELU':
    return nn.ELU()
  elif activation_fn == 'SoftPlus':
    return nn.Softplus()
  elif activation_fn == 'SiLU':
    return nn.SiLU()
  elif activation_fn == 'Sig':
    return nn.Sigmoid()
  elif activation_fn == 'Tanh':
    return nn.Tanh()
  else :
    raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')


class SqueezeAndExcitation(nn.Module):
  def __init__(self, channels, ratio = 4, act_fn = 'GeLU', use_max_pooling = False):
    super(SqueezeAndExcitation, self).__init__()
    print(f"SqueezeAndExcitation.__init__ called (channels = {channels}, ratio = {ratio}, "
      f"activation_fn = {act_fn}, use_max_pooling = {use_max_pooling})")
    self.channels = channels
    self.ratio = ratio
    self.use_max_pooling = use_max_pooling
    self.squeeze = nn.AdaptiveAvgPool1d(1) if not \
      self.use_max_pooling else nn.AdaptiveMaxPool1d(1)

    out_linear = zero_module(nn.Linear(channels // ratio, channels, bias = False))

    self.excitation = nn.Sequential(
      nn.Linear(channels, channels // ratio, bias = False),
      get_activation(activation_fn = act_fn),
      out_linear,
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
    print(f"AdaptiveNormLayer.__init__ called (latent_dim = {latent_dim}, "
      f"embedding_dim = {embedding_dim}, dropout = {dropout})")

    self.latent_dim = latent_dim
    self.embedding_dim = embedding_dim
    self.dropout = dropout

    proj_linear = zero_module(nn.Linear(embedding_dim, latent_dim * 3, bias = True))

    self.embedding_proj = nn.Sequential(
      nn.SiLU(),
      proj_linear,
    )
    self.layernorm = nn.LayerNorm(self.latent_dim)

  def forward(self, latent_input, embedding_input):
    """
    :param latent_input    : a tensor of shape (B, T, latent_dim)
    :param embedding_input : a tensor of shape (B, embedding_dim)
    :return :
        a tensor of shape (B, T, latent_dim)
    """
    embedding_out = self.embedding_proj(embedding_input).unsqueeze(1)  # (B, 1, 3 * latent_dim)
    scale, shift, gate = torch.chunk(embedding_out, 3, dim = -1)
    out = self.layernorm(latent_input) * (1. + scale) + shift
    return out, gate


class ScaleShiftLayer(nn.Module):
  def __init__(self, latent_dim, embedding_dim, dropout = 0.1):
    super(ScaleShiftLayer, self).__init__()
    print(f"ScaleShiftLayer.__init__ called (latent_dim = {latent_dim}, "
      f"embedding_dim = {embedding_dim}, dropout = {dropout})")
    self.latent_dim = latent_dim
    self.embedding_dim = embedding_dim
    self.layernorm = nn.LayerNorm(self.latent_dim)
    self.embedding_proj = nn.Sequential(
      nn.SiLU(),
      nn.Linear(embedding_dim, latent_dim * 2),
    )
    out_linear = zero_module(nn.Linear(latent_dim, latent_dim))

    self.proj_out = nn.Sequential(
      nn.SiLU(),
      nn.Dropout(p = dropout),
      out_linear,
    )

  def forward(self, latent_input, embedding_input):
    embedding_out = self.embedding_proj(embedding_input).unsqueeze(1)  # (B, 1, 2 * latent_dim)
    scale, shift = torch.chunk(embedding_out, 2, dim = -1)
    out = self.layernorm(latent_input) * (1. + scale) + shift
    out = self.proj_out(out)
    return out


class AdaptiveFeedForwardNetwork(nn.Module):
  def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
    super(AdaptiveFeedForwardNetwork, self).__init__()
    print(f"AdaptiveFeedForwardNetwork.__init__ called (model_dim = {model_dim}, "
      f"ffn_dim = {ffn_dim}, embedding_dim = {embedding_dim}, dropout = {dropout}, "
      f"activation_fn = {activation})"
    )
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

    self.act = get_activation(activation_fn = activation)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, input, embedding):
    """
    :param input : a tensor of shape (B, T, model_dim)
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
    print(f"AdaptiveMultiHeadAttention.__init__ called (model_dim = {model_dim}, "
      f"embedding_dim = {embedding_dim}, num_head = {num_head}, dropout = {dropout})"
    )
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
    :param input : a tensor of shape (B, T, model_dim)
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


class StyleFeedForwardNetwork(nn.Module):
  def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
    super(StyleFeedForwardNetwork, self).__init__()
    print(f"StyleFeedForwardNetwork.__init__ called (model_dim = {model_dim}, "
      f"ffn_dim = {ffn_dim}, embedding_dim = {embedding_dim}, dropout = {dropout}, "
      f"activation_fn = {activation})"
    )
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


class StyleFeedForwardNetworkv1(nn.Module):
  "Design to adapt with pose encoder, pose_decoder"
  def __init__(self, ffn_dim, branch, embedding_dim, dropout = 0.1, activation = 'GeLU'):
    super(StyleFeedForwardNetworkv1, self).__init__()
    print(f"StyleFeedForwardNetworkv1.__init__ called (ffn_dim = {ffn_dim}, branch = {branch}, "
      f"embedding_dim = {embedding_dim}, dropout = {dropout}, activation = {activation})"
    )
    self.linear1_list = nn.ModuleList()
    self.linear2_list = nn.ModuleList()
    self.latent_dim = [96, 64, 64, 64, 64, 160]
    self.branch = branch
    self.embedding_dim = embedding_dim
    for i in range(len(self.latent_dim)):
      self.linear1_list.append(nn.Linear(self.latent_dim[i], ffn_dim))
      self.linear2_list.append(nn.Linear(ffn_dim, self.latent_dim[i]))
    self.act = get_activation(activation)
    self.dropout = nn.Dropout(dropout)
    self.scale_shift_norm = ScaleShiftLayer(
      latent_dim = sum(self.latent_dim) * branch,
      embedding_dim = embedding_dim,
      dropout = dropout,
    )

  def get_ffn_out(self, i, feat):
    feat = self.dropout(self.act(self.linear1_list[i](feat)))
    feat = self.linear2_list[i](feat)
    return feat

  def forward(self, motion, embed):
    B, T, D = motion.shape
    head_feat = self.get_ffn_out(0, motion[:, :, 0 : 96].contiguous())
    larm_feat = self.get_ffn_out(1, motion[:, :, 96 : (96 + 64)].contiguous())
    rarm_feat = self.get_ffn_out(2, motion[:, :, (96 + 64) : (96 + 64 * 2)].contiguous())
    lleg_feat = self.get_ffn_out(3, motion[:, :, (96 + 64 * 2) : (96 + 64 * 3)].contiguous())
    rleg_feat = self.get_ffn_out(4, motion[:, :, (96 + 64 * 3) : (96 + 64 * 4)].contiguous())
    body_feat = self.get_ffn_out(5, motion[:, :, (96 + 64 * 4) : (96 + 64 * 4 + 160)].contiguous())
    output = [head_feat, larm_feat, rarm_feat, lleg_feat, rleg_feat, body_feat]
    y = torch.cat(output, dim = -1)
    y = motion.reshape(B, T, D) + self.scale_shift_norm(y, embed)
    return y


class StyleMultiHeadAttention(nn.Module):
  def __init__(self, model_dim, embedding_dim, num_head, dropout):
    super(StyleMultiHeadAttention, self).__init__()
    print(f"StyleMultiHeadAttention.__init__ called (model_dim = {model_dim}, "
      f"embedding_dim = {embedding_dim}, num_head = {num_head}, dropout = {dropout})"
    )
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


class StyleFinalTransformerLayer(nn.Module):
  def __init__(self, model_dim, output_dim, embedding_dim):
    super(StyleFinalTransformerLayer, self).__init__()
    print(f"StyleFinalTransformerLayer.__init__ called (model_dim = {model_dim}, "
      f"output_dim = {output_dim}, embedding_dim = {embedding_dim})"
    )
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
    shift, scale = torch.chunk(embedding_out, 2, dim = -1)
    norm_input = self.layernorm(input) * (scale + 1.) + shift
    out = self.linear(norm_input)
    return out


class AdaptiveFinalTransformerLayer(nn.Module):
  def __init__(self, model_dim, output_dim, embedding_dim):
    super(AdaptiveFinalTransformerLayer, self).__init__()
    print(f"AdaptiveFinalTransformerLayer.__init__ called (model_dim = {model_dim}, "
      f"output_dim = {output_dim}, embedding_dim = {embedding_dim})"
    )
    self.model_dim = model_dim
    self.output_dim = output_dim
    self.embedding_dim = embedding_dim
    self.adaptive_norm = zero_module(
      nn.Sequential(
        nn.SiLU(),
        nn.Linear(embedding_dim, 2 * model_dim + output_dim),
      )
    )
    self.layernorm = nn.LayerNorm(self.model_dim)
    self.linear = nn.Linear(model_dim, output_dim)

  def forward(self, input, embedding):
    """
    :param input     : a tensor of shape (B, T, model_dim)
    :param embedding : a tensor of shape (B, embedding_dim)
    :return :
      a tensor of shape (B, T, model_dim)
    """
    embedding_out = self.adaptive_norm(embedding).unsqueeze(1)
    shift, scale, gate = torch.split(embedding_out, [self.model_dim, self.model_dim, self.output_dim], dim = -1)
    norm_input = self.layernorm(input) * (scale + 1.) + shift
    out = self.linear(norm_input) * gate
    return out