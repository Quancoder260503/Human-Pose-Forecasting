import os
import sys
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np

def normal_init_(layer, mean_, sd_, bias, norm_bias = True):
  """Intialization of layers with normal distribution with mean and bias"""
  classname = layer.__class__.__name__
  # Only use the convolutional layers of the module
  #if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
  if classname.find('Linear') != -1:
    print('[INFO] (normal_init) Initializing layer {}'.format(classname))
    layer.weight.data.normal_(mean_, sd_)
    if norm_bias:
      layer.bias.data.normal_(bias, 0.05)
    else:
      layer.bias.data.fill_(bias)


def weight_init(module, mean_ = 0, sd_ = 0.004, bias = 0.0, norm_bias = False, init_fn_ = normal_init_):
  """Initialization of layers with normal distribution"""
  moduleclass = module.__class__.__name__
  try:
    for layer in module:
      if layer.__class__.__name__ == 'Sequential':
        for l in layer:
          init_fn_(l, mean_, sd_, bias, norm_bias)
      else:
        init_fn_(layer, mean_, sd_, bias, norm_bias)
  except TypeError:
    init_fn_(module, mean_, sd_, bias, norm_bias)


def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
  classname = layer.__class__.__name__
  if classname.find('Linear') != -1:
    print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
    nn.init.xavier_uniform_(layer.weight.data)
    # nninit.xavier_normal(layer.bias.data)
    if norm_bias:
      layer.bias.data.normal_(0, 0.05)
    else:
      layer.bias.data.zero_()

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


class PositionalEncoding1D(object):
  def __init__(self, model_dim = 512, temperature = 10000, alpha = 1):
    self._model_dim = model_dim
    self._temperature = temperature
    self._alpha = alpha

  def __call__(self, seq_length):
    angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(self._model_dim)[np.newaxis, :])

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = pos_encoding.astype(np.float32)

    return torch.from_numpy(pos_encoding)

  def get_angles(self, pos, i):
    angle_rates = 1 / np.power(self._temperature, (2 *  (i//2)) / np.float32(self._model_dim))
    return self._alpha*pos * angle_rates
