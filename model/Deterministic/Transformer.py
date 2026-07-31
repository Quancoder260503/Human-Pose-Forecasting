import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
from model.Deterministic.GraphConvNet import * 
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from model.ModelUtils import *


class PositionalEncoding1D(object):
  def __init__(self, model_dim = 512, temperature = 10000, alpha=1):
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

  def visualize(self, num_visualized):
    pos_encoding = self(num_visualized).numpy()[0]
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, self.d_model))
    plt.ylabel('Position in Sequence')
    plt.title('1D Positional Encoding')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def create_look_ahead_mask(seq_length, is_nonautoregressive = False):
  if is_nonautoregressive:
    return np.zeros((seq_length, seq_length), dtype = np.float32)
  x = np.ones((seq_length, seq_length), dtype=np.float32)
  mask = np.triu(x, 1).astype(np.float32)
  return mask  # (seq_len, seq_len)

class EncoderLayer(nn.Module):
  def __init__(self, model_dim = 256, num_heads = 8, dim_ffn = 2048, dropout = 0.1, init_fn = normal_init_):
    super(EncoderLayer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout

    self._self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout)
    self._relu = nn.ReLU()
    self._dropout_layer = nn.Dropout(self._dropout)

    self._linear1 = nn.Linear(model_dim, self._dim_ffn)
    self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
    self._norm1 = nn.LayerNorm(model_dim, eps = 1e-5)
    self._norm2 = nn.LayerNorm(model_dim, eps = 1e-5)

    weight_init(self._linear1, init_fn_ = init_fn)
    weight_init(self._linear2, init_fn_ = init_fn)

  def forward(self, source_seq_, pos_encodings):
    """Computes decoder layer forward pass with pre normalization.

    Args:
    source_seq: [sequence_length, batch_size, model_dim].
    pos_encodings: [sequence_length, model_dim].

    Returns:
    Tensor of shape [sequence_length, batch_size, model_dim].
    """
    # add positional encodings to the input sequence
    # for self attention query is the same as key
    source_seq = self._norm1(source_seq_)
    query = source_seq + pos_encodings
    key = query
    value = source_seq

    attn_output, attn_weights = self._self_attn(
        query, 
        key, 
        value, 
        need_weights = True
    )

    norm_attn_ = self._dropout_layer(attn_output) + source_seq_
    norm_attn = self._norm2(norm_attn_)

    output = self._linear1(norm_attn)
    output = self._relu(output)
    output = self._dropout_layer(output)
    output = self._linear2(output)
    output = self._dropout_layer(output) + norm_attn_

    return output, attn_weights


class TransformerEncoder(nn.Module):
  def __init__(self, num_layers = 6, model_dim = 256, num_heads = 8, dim_ffn = 2048, dropout = 0.1, init_fn = normal_init_):
    super(TransformerEncoder, self).__init__()
    """Transforme encoder initialization."""
    self._num_layers = num_layers
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._norm = nn.LayerNorm(model_dim, eps = 1e-5)

    self._encoder_stack = self.init_encoder_stack(init_fn)

  def init_encoder_stack(self, init_fn):
    stack = nn.ModuleList()
    for _ in range(self._num_layers):
      layer = EncoderLayer(
        model_dim = self._model_dim,
        num_heads = self._num_heads,
        dim_ffn   = self._dim_ffn,
        dropout   = self._dropout,
        init_fn   = init_fn,
      )
      stack.append(layer)
    return stack

  def forward(self, input_sequence, pos_encodings):
    """Computes decoder forward pass.

    Args:
    source_seq: [sequence_length, batch_size, model_dim].
    pos_encodings: [sequence_length, model_dim].

    Returns:
    Tensor of shape [sequence_length, batch_size, model_dim].
    """
    outputs = input_sequence
    for l in range(self._num_layers):
      outputs, attn_weights = self._encoder_stack[l](outputs, pos_encodings)

    if self._norm:
      outputs = self._norm(outputs)
    return outputs, attn_weights


class DecoderLayer(nn.Module):
  def __init__(self, model_dim = 256, num_heads = 8, dim_ffn = 2048, dropout = 0.1, init_fn = normal_init_):
    super(DecoderLayer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    
    self._self_attn = nn.MultiheadAttention(
        model_dim, num_heads, dropout = dropout
    )
    self._multihead_attn = nn.MultiheadAttention(
        model_dim, num_heads, dropout = dropout
    )

    self._linear1 = nn.Linear(model_dim, dim_ffn)
    self._linear2 = nn.Linear(dim_ffn, model_dim)
    self._relu = nn.ReLU()

    self._norm1 = nn.LayerNorm(model_dim)
    self._norm2 = nn.LayerNorm(model_dim)
    self._norm3 = nn.LayerNorm(model_dim)

    self._dropout1 = nn.Dropout(dropout)
    self._dropout2 = nn.Dropout(dropout)
    self._dropout3 = nn.Dropout(dropout)
    self._dropout4 = nn.Dropout(dropout)

    weight_init(self._linear1, init_fn_ = init_fn)
    weight_init(self._linear2, init_fn_ = init_fn)


  def forward(self, target_seq_, memory, dec_pos_encodings, mask_look_ahead):
    """Forward pass of the layer with pre normalization.

    Args:
    target_seq: [target_seq_length, batch_size, model_dim]
    memory: [source_seq_length, batch_size, model_dim]
    mask_look_ahead: []
    mask_target_padding:
    """
    target_seq = self._norm1(target_seq_)
    q = k = v =  target_seq + dec_pos_encodings
    
    self_attn, self_attn_weights = self._self_attn(
      query = q, 
      key = k, 
      value = v,
      attn_mask = mask_look_ahead, 
    )

    self_attn = self._dropout1(self_attn)
    out_self_attn = self._norm2(self_attn + target_seq_)

    attn, attn_weights = self._multihead_attn(
      query = out_self_attn + dec_pos_encodings, 
      key   = memory,
      value = memory, 
    )

    attn = self._dropout2(attn)
    attn = out_self_attn + attn
    out_attn = self._norm3(attn)

    ffn_output = self._linear1(out_attn)
    ffn_output = self._relu(ffn_output)
    ffn_output = self._dropout4(ffn_output)
    ffn_output = self._linear2(ffn_output)

    ffn_output = self._dropout3(ffn_output) + attn

    return ffn_output, self_attn_weights, attn_weights


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers = 6, model_dim = 256, num_heads = 8, dim_ffn = 2048, dropout = 0.1, init_fn = normal_init_):
    super(TransformerDecoder, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._num_layers = num_layers
    self._norm = nn.LayerNorm(model_dim, eps = 1e-5)
    self._decoder_stack = self.init_decoder_stack(init_fn)

  def init_decoder_stack(self, init_fn):
    stack = nn.ModuleList()
    for _ in range(self._num_layers):
      layer = DecoderLayer(
        model_dim = self._model_dim,
        num_heads = self._num_heads,
        dim_ffn   = self._dim_ffn,
        dropout   = self._dropout,
        init_fn   = init_fn,
      )
      stack.append(layer)
    return stack

  def forward(self, target_seq, memory, dec_pos_encodings, mask_look_ahead = None, get_attn_weights = False):
    seq_length = target_seq.size()[0]
    output_list = []
    attn_weights_list = [] if get_attn_weights else None
    outputs = target_seq

    for layer in range(self._num_layers):
      outputs, self_attn_weights, attn_weights = self._decoder_stack[layer](
        target_seq_       = outputs, 
        memory            = memory,
        dec_pos_encodings = dec_pos_encodings, 
        mask_look_ahead   = mask_look_ahead, 
      )
      if get_attn_weights:
        attn_weights_list.append(attn_weights)
      output_list.append(self._norm(outputs))
    return output_list, attn_weights_list


class Transformer(nn.Module):
  def __init__(self, num_encoder_layers = 6, num_decoder_layers = 6, model_dim = 256, num_heads = 8,\
                dim_ffn = 2048, dropout = 0.1, init_fn = normal_init_, target_seq_len = 25):
    """Implements the Transformer model for sequence-to-sequence modeling."""
    super(Transformer, self).__init__()
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._tgt_seq_len = target_seq_len

    self._encoder = TransformerEncoder(
      num_layers = num_encoder_layers,
      model_dim = model_dim,
      num_heads = num_heads,
      dim_ffn = dim_ffn,
      dropout = dropout,
      init_fn = init_fn,
    )
    self._decoder = TransformerDecoder(
      num_layers = num_decoder_layers,
      model_dim = model_dim,
      num_heads = num_heads,
      dim_ffn = dim_ffn,
      dropout = dropout,
      init_fn = init_fn,
    )


  def forward(self, source_seq, target_seq, encoder_position_encodings = None, decoder_position_encodings = None, \
            mask_look_ahead = None, get_attn_weights = False):  
    memory, enc_weights = self._encoder(source_seq, encoder_position_encodings)

    out_attn, out_weights = self._decoder(
      target_seq = target_seq,
      memory = memory,
      dec_pos_encodings = decoder_position_encodings,
      mask_look_ahead = mask_look_ahead, 
      get_attn_weights = get_attn_weights,
    )
    
    out_weights_ = None
    enc_weights_ = None

    if get_attn_weights:
      out_weights_, enc_weights_ = out_weights, enc_weights
    
    return out_attn, memory, out_weights_, enc_weights_


class PoseTransformer(nn.Module):
  def __init__(self,
    pose_dim = 51,
    model_dim = 256,
    num_encoder_layers = 6,
    num_decoder_layers = 6,
    num_heads = 8,
    dim_ffn = 2048,
    dropout = 0.1,
    target_seq_length = 25,
    source_seq_length = 50,
    init_fn = xavier_init_,
    pose_encoder   = None,
    pose_decoder   = None,
    pos_encoding_params=(10000, 1)
  ):
    """Initialization of pose transformers."""
    super(PoseTransformer, self).__init__()
    self._target_seq_length = target_seq_length
    self._source_seq_length = source_seq_length
    self._pose_dim  = pose_dim
    self._model_dim = model_dim
    self._num_decoder_layers = num_decoder_layers
    self._mlp_dim = model_dim 
    self._pose_encoder = pose_encoder
    self._pose_decoder = pose_decoder
    thisname = self.__class__.__name__
    self._pos_encoding_params = pos_encoding_params

    self._transformer = Transformer(
      num_encoder_layers = num_encoder_layers,
      num_decoder_layers = num_decoder_layers,
      model_dim = model_dim,
      num_heads = num_heads,
      dim_ffn = dim_ffn,
      dropout = dropout,
      init_fn = init_fn,
      target_seq_len = target_seq_length
    )

    t_params = filter(lambda p: p.requires_grad, self._transformer.parameters())
    nparams = sum([np.prod(p.size()) for p in t_params])
    print('[INFO] ({}) Transformer has {} parameters!'.format(thisname, nparams))

    self._pos_encoder = PositionalEncoding1D(
      model_dim = self._model_dim,
      temperature = self._pos_encoding_params[0],
      alpha = self._pos_encoding_params[1]
    )
    self._pos_decoder = PositionalEncoding1D(
      model_dim = self._model_dim,
      temperature = self._pos_encoding_params[0],
      alpha = self._pos_encoding_params[1]
    )


    self.init_position_encodings()

  def init_position_encodings(self):
    mask_look_ahead =  torch.from_numpy(create_look_ahead_mask(self._target_seq_length, is_nonautoregressive = True))
    encoder_pos_encodings = self._pos_encoder(self._source_seq_length).view(self._source_seq_length, 1, self._model_dim)
    decoder_pos_encodings = self._pos_decoder(self._target_seq_length).view(self._target_seq_length, 1, self._model_dim)

    self._encoder_pos_encodings = nn.Parameter(encoder_pos_encodings, requires_grad = False)
    self._decoder_pos_encodings = nn.Parameter(decoder_pos_encodings, requires_grad = False)
    self._mask_look_ahead       = nn.Parameter(mask_look_ahead,       requires_grad = False)



  def forward(self, input_pose_seq_, target_pose_seq_, get_attn_weights = False):
    """Compute forward pass for training and non recursive inference.
    Args:
        input_pose_seq_: Source sequence [batch_size, src_len, skeleton_dim].
        target_pose_seq_: Query target sequence [batch_size, tgt_len, skeleton_dim].
        mask_target_padding: Mask for target masking with ones where elements 
            belong to the padding elements of shape [batch_size, tgt_len, skeleton_dim].
        get_attn_weights: Boolean to indicate if attention weights should be returned.
    Returns:
    """
    # 1) Encode the sequence with given pose encoder
    # [batch_size, sequence_length, model_dim]
    input_pose_seq  = input_pose_seq_
    target_pose_seq = target_pose_seq_
    if self._pose_encoder is not None:
      input_pose_seq = self._pose_encoder(input_pose_seq)
      target_pose_seq = self._pose_encoder(target_pose_seq)

    # 2) compute the look-ahead mask and the positional encodings
    # [sequence_length, batch_size, model_dim]
    input_pose_seq = torch.transpose(input_pose_seq, 0, 1)
    target_pose_seq = torch.transpose(target_pose_seq, 0, 1)


    attn_output, memory, attn_weights, enc_weights = self._transformer(
      input_pose_seq,
      target_pose_seq,
      encoder_position_encodings = self._encoder_pos_encodings,
      decoder_position_encodings = self._decoder_pos_encodings,
      mask_look_ahead = self._mask_look_ahead, 
      get_attn_weights = get_attn_weights,
    )

    out_sequence = []
    target_pose_seq_ = torch.transpose(target_pose_seq_, 0, 1)

    for l in range(self._num_decoder_layers):
      out_sequence_ = self._pose_decoder(attn_output[l].view(-1, self._model_dim))
      out_sequence_ = out_sequence_.view(self._target_seq_length, -1, self._pose_dim)
      out_sequence_ = out_sequence_ + target_pose_seq_[:, :, :]
      out_sequence_ = torch.transpose(out_sequence_, 0, 1)
      out_sequence.append(out_sequence_)

    return out_sequence, attn_weights, enc_weights


def get_transformer_model(params, pose_encoder_fn, pose_decoder_fn):
  init_fn = normal_init_ if params['init_fn'] == 'normal_init' else xavier_init_
  return PoseTransformer(
    pose_dim  = params['pose_dim'],
    model_dim = params['model_dim'],
    num_encoder_layers = params['num_encoder_layers'],
    num_decoder_layers = params['num_decoder_layers'],
    num_heads = params['num_heads'],
    dim_ffn = params['dim_ffn'],
    dropout = params['dropout'],
    target_seq_length = params['target_seq_len'],
    source_seq_length = params['source_seq_len'] - (0 if params['include_last_obs'] else 1),
    init_fn = init_fn,
    pose_encoder = pose_encoder_fn(params),
    pose_decoder = pose_decoder_fn(params),
    pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha'])
  )