import torch
import torch.nn as nn

"""
A classifier for computing FID features.
"""

class ClassifierForFID(nn.Module):
  def __init__(
    self,
    input_size = 48,
    hidden_size = 128,
    hidden_layer = 2,
    output_size = 15,
    feature_dim = 30,
    use_noise = None,
    zero_init_hidden = False,
    device = None,
    dtype = torch.float32,
  ):
    super(ClassifierForFID, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.hidden_layer = hidden_layer
    self.output_size = output_size
    self.feature_dim = feature_dim
    self.use_noise = use_noise
    self.zero_init_hidden = zero_init_hidden
    self.device = device
    self.dtype = dtype

    self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
    self.linear1 = nn.Linear(hidden_size, feature_dim)
    self.linear2 = nn.Linear(feature_dim, output_size)

  def init_hidden(self, num_samples, device = None, dtype = None):
    """
    :param num_samples : the batch size
    :param device      : overrides self.device when given
    :param dtype       : overrides self.dtype when given
    :return :
      a tensor of shape (hidden_layer, num_samples, hidden_size)
    """
    init_fn = torch.zeros if self.zero_init_hidden else torch.randn
    return init_fn(
      self.hidden_layer,
      num_samples,
      self.hidden_size,
      device = device if device is not None else self.device,
      dtype = dtype if dtype is not None else self.dtype,
      requires_grad = False,
    )

  def get_fid_features(self, motion_sequence, hidden_unit = None):
    """
    :param motion_sequence : a tensor of shape (B, input_size, T)
    :param hidden_unit     : a tensor of shape (hidden_layer, B, hidden_size), or None
    :return :
      a tensor of shape (B, feature_dim)
    """
    motion_sequence = motion_sequence.permute(2, 0, 1).contiguous()
    motion_sequence = motion_sequence.to(self.dtype)

    if hidden_unit is None:
      hidden_unit = self.init_hidden(
        num_samples = motion_sequence.size(1),
        device = motion_sequence.device,
        dtype = motion_sequence.dtype,
      )

    gru_out, _ = self.recurrent(motion_sequence, hidden_unit)
    features = torch.tanh(self.linear1(gru_out[-1, :, :]))
    return features

  def forward(self, motion_sequence, hidden_unit = None):
    """
    :param motion_sequence : a tensor of shape (B, input_size, T)
    :param hidden_unit     : a tensor of shape (hidden_layer, B, hidden_size), or None
    :return :
      a tensor of shape (B, output_size)
    """
    features = self.get_fid_features(motion_sequence, hidden_unit)
    return self.linear2(features)

