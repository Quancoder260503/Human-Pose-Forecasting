import os
import sys
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np

def make_flatten_func(joint_dim):
  return lambda i : list(range(i * joint_dim, i * joint_dim + joint_dim))

def get_part_slice(joint_indices, func):
  flat = []
  for j in joint_indices:
    flat.extend(func(j))
  return flat

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
    return nn.SoftPlus() 
  elif activation_fn == 'SiLU':
    return nn.SiLU() 
  elif activation_fn == 'Sig': 
    return nn.Sigmoid() 
  elif activation_fn == 'Tanh':
    return nn.Tanh() 
  else : 
    raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')


# class SqueezeAndExcitation(nn.Module): 
#   def __init__(self, channels, ratio = 4, act_fn = 'GeLU', use_max_pooling = False, zero_module = None):
#     super(SqueezeAndExcitation, self).__init__()
#     self.channels = channels 
#     self.ratio = ratio 
#     self.use_max_pooling = use_max_pooling 
#     self.squeeze = nn.AdaptiveAvgPool1d(1) if not \
#       self.use_max_pooling else nn.AdaptiveMaxPool1d(1)

#     out_linear = nn.Linear(channels // ratio, channels, bias = False)
#     if zero_module is not None:
#       out_linear = zero_module(out_linear)

#     self.excitation = nn.Sequential(
#       nn.Linear(channels, channels // ratio, bias = False), 
#       get_activation(activation_fn = act_fn),
#       out_linear, 
#       nn.Sigmoid(), 
#     )
  
#   def forward(self, x): 
#     batch_size, channels, _ = x.shape 
#     y = self.squeeze(x).view(batch_size, channels) 
#     y = self.excitation(y).view(batch_size, channels, 1) 
#     return x + x * y.expand_as(x) 


# class AdaptiveNormLayer(nn.Module):
#   def __init__(self, latent_dim, embedding_dim, dropout = 0.1, zero_module = None):
#     super(AdaptiveNormLayer, self).__init__()
    
#     self.latent_dim = latent_dim 
#     self.embedding_dim = embedding_dim
#     self.dropout = dropout 

#     proj_linear = nn.Linear(embedding_dim, latent_dim * 3)
#     if zero_module is not None:
#       proj_linear = zero_module(proj_linear)

#     self.embedding_proj = nn.Sequential(
#       nn.SiLU(), 
#       proj_linear,
#     )
#     self.layernorm = nn.LayerNorm(self.latent_dim)

#   def forward(self, latent_input, embedding_input): 
#     """
#     :param latent_input    : a tensor of shape (B, T, latent_dim)
#     :param embedding_input : a tensor of shape (B, embedding_dim) 
#     :return : 
#         a tensor of shape (B, T, latent_dim)
#     """

#     embedding_out = self.embedding_proj(embedding_input).unsqueeze(1) # (B, 1, 3 * latent_dim) 
#     scale, shift, gate = torch.chunk(embedding_out, 3, dim = -1) 
#     out = self.layernorm(latent_input) * (1. + scale) + shift
#     return out, gate 
  
# class ScaleShiftLayer(nn.Module):
#   def __init__(self, latent_dim, embedding_dim, dropout = 0.1, zero_module = None): 
#     super(ScaleShiftLayer, self).__init__()
#     self.latent_dim = latent_dim
#     self.embedding_dim = embedding_dim 
#     self.layernorm = nn.LayerNorm(self.latent_dim)
#     self.embedding_proj = nn.Sequential(
#       nn.SiLU(), 
#       nn.Linear(embedding_dim, latent_dim * 2), 
#     )

#     out_linear = nn.Linear(latent_dim, latent_dim)
#     if zero_module is not None:
#       out_linear = zero_module(out_linear)

#     self.proj_out = nn.Sequential(
#       nn.SiLU(), 
#       nn.Dropout(p = dropout),  
#       out_linear
#     )

#   def forward(self, latent_input, embedding_input): 
#     embedding_out = self.embedding_proj(embedding_input).unsqueeze(1) # (B, 1, 2* latent_dim) 
#     scale, shift = torch.chunk(embedding_out, 2, dim = -1) 
#     out = self.layernorm(latent_input) * (1. + scale) + shift
#     out = self.proj_out(out) 
#     return out

# class PoseEncoder(nn.Module):
#   def __init__(self, dataset_name='h36m', input_dim = 48, joint_dim = 3):
#     super(PoseEncoder, self).__init__()
#     self.dataset_name = dataset_name
#     func = make_flatten_func(joint_dim)
#     self.input_dim = input_dim

#     if dataset_name == "h36m":
#       self.head_slice = get_part_slice([6, 7, 8, 9], func)   
#       self.larm_slice = get_part_slice([10, 11, 12], func)  
#       self.rarm_slice = get_part_slice([13, 14, 15], func)  
#       self.lleg_slice = get_part_slice([3, 4, 5], func)  
#       self.rleg_slice = get_part_slice([0, 1, 2], func) 
#       self.body_slice = get_part_slice(range(16), func)     
#     elif dataset_name == "human_eva":
#       # unchanged / separate — flag if this also needs realigning to your HumanEva preprocessing
#       pass
#     else:
#       raise ValueError('dataset not implemented')

#     self.head_embed = nn.Linear(len(self.head_slice), 96)
#     self.larm_embed = nn.Linear(len(self.larm_slice), 64)
#     self.rarm_embed = nn.Linear(len(self.rarm_slice), 64)
#     self.lleg_embed = nn.Linear(len(self.lleg_slice), 64)
#     self.rleg_embed = nn.Linear(len(self.rleg_slice), 64)
#     self.body_embed = nn.Linear(len(self.body_slice), 160)

#     assert len(set(self.body_slice)) == input_dim

#   def forward(self, motion):
#     # motion : [B, num_dct_channels, n_joints * joint_dim]
#     head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
#     larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
#     rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
#     lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
#     rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
#     body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
#     feat = torch.cat((head_feat, larm_feat, rarm_feat, lleg_feat, rleg_feat, body_feat), dim = -1)
#     return feat

# class PoseDecoder(nn.Module):
#   def __init__(self, dataset_name = 'h36m', joint_dim = 3, input_dim = 48, latent_dim = 512):
#     super(PoseDecoder, self).__init__()
#     assert latent_dim == 512, "This design is fixed for the experiment that used this module"
#     self.dataset_name = dataset_name
#     self.output_dim = input_dim
#     func = make_flatten_func(joint_dim)
#     if dataset_name == "h36m":
#       self.head_slice = get_part_slice([6, 7, 8, 9], func)   
#       self.larm_slice = get_part_slice([10, 11, 12], func)  
#       self.rarm_slice = get_part_slice([13, 14, 15], func)  
#       self.lleg_slice = get_part_slice([3, 4, 5], func)  
#       self.rleg_slice = get_part_slice([0, 1, 2], func) 
#       self.body_slice = get_part_slice(range(16), func)     
#     elif dataset_name == "human_eva":
#       # unchanged / separate — flag if this also needs realigning
#       pass
#     else:
#       raise ValueError('dataset not implemented')

#     self.stem_out = nn.Linear(latent_dim, len(self.stem_slice))
#     self.head_out = nn.Linear(latent_dim, len(self.head_slice))
#     self.larm_out = nn.Linear(latent_dim, len(self.larm_slice))
#     self.rarm_out = nn.Linear(latent_dim, len(self.rarm_slice))
#     self.lleg_out = nn.Linear(latent_dim, len(self.lleg_slice))
#     self.rleg_out = nn.Linear(latent_dim, len(self.rleg_slice))
#     self.body_out = nn.Linear(latent_dim, len(self.body_slice))

#   def forward(self, motion):
#     B, T = motion.shape[:2]
#     D = self.latent_dim
#     head_feat = self.head_out(motion[:, :, 0 : 96].contiguous())
#     larm_feat = self.larm_out(motion[:, :, 96 : (96 + 64)].contiguous())
#     rarm_feat = self.rarm_out(motion[:, :, (96 + 64) : (96 + 64 * 2)].contiguous())
#     lleg_feat = self.lleg_out(motion[:, :, (96 + 64 * 2) : (96 + 64 * 3)].contiguous())
#     rleg_feat = self.rleg_out(motion[:, :, (96 + 64 * 3) : (96 + 64 * 4)].contiguous())
#     body_feat = self.body_out(motion[:, :, (96 + 64 * 4) : (96 + 64 * 4 + 160)].contiguous())

#     output = torch.zeros(B, T, self.output_dim).type_as(motion)
#     output[:, :, self.head_slice] = head_feat
#     output[:, :, self.larm_slice] = larm_feat
#     output[:, :, self.rarm_slice] = rarm_feat
#     output[:, :, self.lleg_slice] = lleg_feat
#     output[:, :, self.rleg_slice] = rleg_feat

#     output = (output + body_feat) / 2.0 
#     return output


# class AdaptiveNormLayer(nn.Module):
#   def __init__(self, latent_dim, embedding_dim, do_zero_module = False, dropout = 0.1):
#     super(AdaptiveNormLayer, self).__init__()

#     self.latent_dim = latent_dim
#     self.embedding_dim = embedding_dim
#     self.dropout = dropout

#     proj_linear = nn.Linear(embedding_dim, latent_dim * 3, bias = True)
#     if do_zero_module:
#       proj_linear = _zero_module(proj_linear)

#     self.embedding_proj = nn.Sequential(
#       nn.SiLU(),
#       proj_linear,
#     )
#     self.layernorm = nn.LayerNorm(self.latent_dim)

#   def forward(self, latent_input, embedding_input):
#     """
#     :param latent_input    : a tensor of shape (B, T, latent_dim)
#     :param embedding_input : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, latent_dim)
#     """
#     embedding_out = self.embedding_proj(embedding_input).unsqueeze(1)  # (B, 1, 3 * latent_dim)
#     scale, shift, gate = torch.chunk(embedding_out, 3, dim = -1)
#     out = self.layernorm(latent_input) * (1. + scale) + shift
#     return out, gate


# class ScaleShiftLayer(nn.Module):
#   def __init__(self, latent_dim, embedding_dim, do_zero_module = False, dropout = 0.1):
#     super(ScaleShiftLayer, self).__init__()
#     self.latent_dim = latent_dim
#     self.embedding_dim = embedding_dim
#     self.layernorm = nn.LayerNorm(self.latent_dim)
#     self.embedding_proj = nn.Sequential(
#       nn.SiLU(),
#       nn.Linear(embedding_dim, latent_dim * 2),
#     )

#     out_linear = nn.Linear(latent_dim, latent_dim)
#     if do_zero_module:
#       out_linear = _zero_module(out_linear)

#     self.proj_out = nn.Sequential(
#       nn.SiLU(),
#       nn.Dropout(p = dropout),
#       out_linear,
#     )

#   def forward(self, latent_input, embedding_input):
#     embedding_out = self.embedding_proj(embedding_input).unsqueeze(1)  # (B, 1, 2 * latent_dim)
#     scale, shift = torch.chunk(embedding_out, 2, dim = -1)
#     out = self.layernorm(latent_input) * (1. + scale) + shift
#     out = self.proj_out(out)
#     return out


# class AdaptiveFeedForwardNetwork(nn.Module):
#   def __init__(self, model_dim, do_zero_module, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
#     super(AdaptiveFeedForwardNetwork, self).__init__()
#     self.model_dim = model_dim
#     self.ffn_dim = ffn_dim
#     self.embedding_dim = embedding_dim

#     self.adaptive_layernorm = AdaptiveNormLayer(
#       latent_dim = model_dim,
#       do_zero_module = do_zero_module,
#       embedding_dim = embedding_dim,
#       dropout = dropout,
#     )

#     self.linear_1 = nn.Linear(model_dim, ffn_dim)
#     self.linear_2 = nn.Linear(ffn_dim, model_dim)

#     self.act = get_activation(activation_fn = activation)
#     self.dropout = nn.Dropout(p = dropout)

#   def forward(self, input, embedding):
#     """
#     :param input : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#       a tensor of shape (B, T, model_dim)
#     """
#     norm_inp, gate = self.adaptive_layernorm(input, embedding)
#     ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(norm_inp))))
#     ffn_out = ffn_out * gate
#     out = input + ffn_out
#     return out


# class AdaptiveMultiHeadAttention(nn.Module):
#   def __init__(self, model_dim, embedding_dim, num_head, dropout, do_zero_module = False):
#     super(AdaptiveMultiHeadAttention, self).__init__()
#     self.model_dim = model_dim
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head

#     assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads"

#     self.dropout = nn.Dropout(dropout)
#     self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.adaptive_layernorm = AdaptiveNormLayer(
#       latent_dim = model_dim,
#       embedding_dim = embedding_dim,
#       do_zero_module = do_zero_module,
#       dropout = dropout,
#     )

#   def forward(self, input, embedding):
#     """
#     :param input : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     batch_size, seq_len, _ = input.shape
#     latent_dim = self.model_dim // self.num_head

#     norm_inp, gate = self.adaptive_layernorm(input, embedding)

#     query = self.query_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
#     query = query.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     key = self.key_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
#     key = key.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     value = self.value_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len,  self.num_head, -1).permute(0, 2, 1, 3)
#     value = value.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
#     attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
#     attention_out = torch.bmm(attention_matrix, value)

#     attention_out = attention_out.contiguous().\
#             view(batch_size, self.num_head, seq_len, -1).permute(0, 2, 1, 3)
#     attention_out = attention_out.contiguous().\
#             view(batch_size, seq_len, -1)

#     attention_out = attention_out * gate

#     out = attention_out + input
#     return out


# class AdaptiveMultiHeadCrossAttention(nn.Module):
#   def __init__(self, model_dim, embedding_dim, num_head, dropout, do_zero_module = True):
#     super(AdaptiveMultiHeadCrossAttention, self).__init__()
#     self.model_dim = model_dim
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head

#     self.dropout = nn.Dropout(dropout)

#     self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)

#     self.adaptive_layernorm = AdaptiveNormLayer(
#       latent_dim = model_dim,
#       embedding_dim = embedding_dim,
#       dropout = dropout,
#     )

#     proj_out_linear = nn.Linear(self.model_dim, self.model_dim)
#     if do_zero_module:
#       proj_out_linear = _zero_module(proj_out_linear)
#     self.proj_out = proj_out_linear

#     self.memory_layernorm = nn.LayerNorm(self.model_dim)

#   def forward(self, input, embedding, memory):
#     """
#     :param input     : a tensor of shape (B, T_{predict}, model_dim)
#     :param memory    : a tensor of shape (B, T_{observed}, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     batch_size, tgt_seq_length, _ = input.shape
#     _, src_seq_length, _ = memory.shape

#     latent_dim = self.model_dim // self.num_head

#     norm_input, gate = self.adaptive_layernorm(input, embedding)
#     norm_memory = self.memory_layernorm(memory)

#     query = self.query_proj(norm_input).contiguous().\
#         view(batch_size, tgt_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
#     query = query.contiguous().view(batch_size * self.num_head, tgt_seq_length, -1)

#     key = self.key_proj(norm_memory).contiguous().\
#         view(batch_size, src_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
#     key = key.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

#     value = self.value_proj(norm_memory).contiguous().\
#         view(batch_size, src_seq_length,  self.num_head, -1).permute(0, 2, 1, 3)
#     value = value.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

#     cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
#     cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))

#     cross_attention_out = torch.bmm(cross_attention_matrix, value)

#     cross_attention_out = cross_attention_out.contiguous().\
#         view(batch_size, self.num_head, tgt_seq_length, -1).permute(0, 2, 1, 3)
#     cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_seq_length, -1)
#     out = gate * self.proj_out(cross_attention_out) + input
#     return out


# class FeedForwardNetwork(nn.Module):
#   def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU', do_zero_module = False):
#     super(FeedForwardNetwork, self).__init__()
#     self.model_dim = model_dim
#     self.ffn_dim = ffn_dim
#     self.embedding_dim = embedding_dim

#     self.scale_shift_norm = ScaleShiftLayer(
#       latent_dim = model_dim,
#       embedding_dim = embedding_dim,
#       do_zero_module = do_zero_module,
#       dropout = dropout,
#     )

#     self.linear_1 = nn.Linear(model_dim, ffn_dim)
#     self.linear_2 = nn.Linear(ffn_dim, model_dim)

#     self.act = get_activation(activation_fn = activation)
#     self.dropout = nn.Dropout(p = dropout)

#   def forward(self, input, embedding):
#     """
#     :param input     : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(input))))
#     out = input + self.scale_shift_norm(ffn_out, embedding)
#     return out


# class FeedForwardNetworkv1(nn.Module):
#   "Design to adapt with pose encoder, pose_decoder"
#   def __init__(self, ffn_dim, branch, embedding_dim, dropout = 0.1, activation = 'GeLU', do_zero_module = False):
#     super(FeedForwardNetworkv1, self).__init__()
#     self.linear1_list = nn.ModuleList()
#     self.linear2_list = nn.ModuleList()
#     self.latent_dim = [96, 64, 64, 64, 64, 160]
#     self.branch = branch
#     self.embedding_dim = embedding_dim
#     for i in range(len(self.latent_dim)):
#       self.linear1_list.append(nn.Linear(self.latent_dim[i], ffn_dim))
#       self.linear2_list.append(nn.Linear(ffn_dim, self.latent_dim[i]))
#     self.act = get_activation(activation)
#     self.dropout = nn.Dropout(dropout)
#     self.scale_shift_norm = ScaleShiftLayer(
#       latent_dim = sum(self.latent_dim) * branch,
#       embedding_dim = embedding_dim,
#       do_zero_module = do_zero_module,
#       dropout = dropout,
#     )

#   def get_ffn_out(self, i, feat):
#     feat = self.dropout(self.act(self.linear1_list[i](feat)))
#     feat = self.linear2_list[i](feat)
#     return feat

#   def forward(self, motion, embed):
#     B, T, D = motion.shape
#     head_feat = self.get_ffn_out(0, motion[:, :, 0 : 96].contiguous())
#     larm_feat = self.get_ffn_out(1, motion[:, :, 96 : (96 + 64)].contiguous())
#     rarm_feat = self.get_ffn_out(2, motion[:, :, (96 + 64) : (96 + 64 * 2)].contiguous())
#     lleg_feat = self.get_ffn_out(3, motion[:, :, (96 + 64 * 2) : (96 + 64 * 3)].contiguous())
#     rleg_feat = self.get_ffn_out(4, motion[:, :, (96 + 64 * 3) : (96 + 64 * 4)].contiguous())
#     body_feat = self.get_ffn_out(5, motion[:, :, (96 + 64 * 4) : (96 + 64 * 4 + 160)].contiguous())
#     output = [head_feat, larm_feat, rarm_feat, lleg_feat, rleg_feat, body_feat]
#     y = torch.cat(output, dim = -1)
#     y = motion.reshape(B, T, D) + self.scale_shift_norm(y, embed)
#     return y


# class MultiHeadAttention(nn.Module):
#   def __init__(self, model_dim, embedding_dim, num_head, dropout, do_zero_module = False):
#     super(MultiHeadAttention, self).__init__()
#     self.model_dim = model_dim
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head

#     assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads"

#     self.dropout = nn.Dropout(dropout)
#     self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.scale_shift_norm = ScaleShiftLayer(
#       latent_dim = model_dim,
#       embedding_dim = embedding_dim,
#       do_zero_module = do_zero_module,
#       dropout = dropout,
#     )
#     self.layernorm = nn.LayerNorm(self.model_dim)

#   def forward(self, input, embedding):
#     """
#     :param input     : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     batch_size, seq_len, _ = input.shape
#     latent_dim = self.model_dim // self.num_head

#     norm_inp = self.layernorm(input)

#     query = self.query_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
#     query = query.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     key = self.key_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
#     key = key.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     value = self.value_proj(norm_inp).contiguous().\
#             view(batch_size, seq_len,  self.num_head, -1).permute(0, 2, 1, 3)
#     value = value.contiguous().view(batch_size * self.num_head, seq_len, -1)

#     attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
#     attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
#     attention_out = torch.bmm(attention_matrix, value)

#     attention_out = attention_out.contiguous().\
#             view(batch_size, self.num_head, seq_len, -1).permute(0, 2, 1, 3)
#     attention_out = attention_out.contiguous().\
#             view(batch_size, seq_len, -1)

#     out = self.scale_shift_norm(attention_out, embedding) + input
#     return out


# class MultiHeadCrossAttention(nn.Module):
#   def __init__(self, model_dim, embedding_dim, num_head, dropout, do_zero_module = False):
#     super(MultiHeadCrossAttention, self).__init__()
#     self.model_dim = model_dim
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head

#     self.dropout = nn.Dropout(dropout)

#     self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False)
#     self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)

#     self.scale_shift_norm = ScaleShiftLayer(
#       latent_dim = model_dim,
#       embedding_dim = embedding_dim,
#       do_zero_module = do_zero_module,
#       dropout = dropout,
#     )
#     self.layernorm = nn.LayerNorm(self.model_dim)
#     self.memory_layernorm = nn.LayerNorm(self.model_dim)

#   def forward(self, input, embedding, memory):
#     """
#     :param input     : a tensor of shape (B, T_{predict}, model_dim)
#     :param memory    : a tensor of shape (B, T_{observed}, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     batch_size, tgt_seq_length, _ = input.shape
#     _, src_seq_length, _ = memory.shape

#     latent_dim = self.model_dim // self.num_head

#     norm_input = self.layernorm(input)
#     norm_memory = self.memory_layernorm(memory)
#     query = self.query_proj(norm_input).contiguous().\
#         view(batch_size, tgt_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
#     query = query.contiguous().view(batch_size * self.num_head, tgt_seq_length, -1)

#     key = self.key_proj(norm_memory).contiguous().\
#         view(batch_size, src_seq_length, self.num_head, -1).permute(0, 2, 1, 3)
#     key = key.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

#     value = self.value_proj(norm_memory).contiguous().\
#         view(batch_size, src_seq_length,  self.num_head, -1).permute(0, 2, 1, 3)
#     value = value.contiguous().view(batch_size * self.num_head, src_seq_length, -1)

#     cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
#     cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))

#     cross_attention_out = torch.bmm(cross_attention_matrix, value)

#     cross_attention_out = cross_attention_out.contiguous().\
#         view(batch_size, self.num_head, tgt_seq_length, -1).permute(0, 2, 1, 3)
#     cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_seq_length, -1)
#     out = self.scale_shift_norm(cross_attention_out, embedding) + input
#     return out


# class FinalTransformerLayer(nn.Module):
#   def __init__(self, model_dim, output_dim, embedding_dim, do_zero_module = True):
#     super(FinalTransformerLayer, self).__init__()
#     self.model_dim = model_dim
#     self.output_dim = output_dim
#     self.embedding_dim = embedding_dim

#     scale_shift_seq = nn.Sequential(
#       nn.SiLU(),
#       nn.Linear(embedding_dim, 2 * model_dim),
#     )
#     linear_out = nn.Linear(model_dim, output_dim)
#     if do_zero_module:
#       scale_shift_seq = _zero_module(scale_shift_seq)
#       linear_out = _zero_module(linear_out)

#     self.scale_shift_norm = scale_shift_seq
#     self.layernorm = nn.LayerNorm(self.model_dim)
#     self.linear = linear_out

#   def forward(self, input, embedding):
#     """
#     :param input     : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     embedding_out = self.scale_shift_norm(embedding).unsqueeze(1)
#     shift, scale = torch.chunk(embedding_out, 2, dim = -1)
#     norm_input = self.layernorm(input) * (scale + 1.) + shift
#     out = self.linear(norm_input)
#     return out


# class AdaptiveFinalTransformerLayer(nn.Module):
#   def __init__(self, model_dim, output_dim, embedding_dim, do_zero_module = True):
#     super(AdaptiveFinalTransformerLayer, self).__init__()
#     self.model_dim = model_dim
#     self.output_dim = output_dim
#     self.embedding_dim = embedding_dim

#     adaptive_seq = nn.Sequential(
#       nn.SiLU(),
#       nn.Linear(embedding_dim, 2 * model_dim + output_dim),
#     )
#     linear_out = nn.Linear(model_dim, output_dim)
#     if do_zero_module:
#       adaptive_seq = _zero_module(adaptive_seq)

#     self.adaptive_norm = adaptive_seq 
#     self.layernorm = nn.LayerNorm(self.model_dim)
#     self.linear = linear_out

#   def forward(self, input, embedding):
#     """
#     :param input     : a tensor of shape (B, T, model_dim)
#     :param embedding : a tensor of shape (B, embedding_dim)
#     :return :
#         a tensor of shape (B, T, model_dim)
#     """
#     embedding_out = self.adaptive_norm(embedding).unsqueeze(1)
#     shift, scale, gate = torch.split(embedding_out, [self.model_dim, self.model_dim, self.output_dim], dim = -1)
#     norm_input = self.layernorm(input) * (scale + 1.) + shift
#     out = self.linear(norm_input) * gate
#     return out