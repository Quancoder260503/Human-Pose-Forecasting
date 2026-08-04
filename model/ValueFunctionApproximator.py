import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F 
from model.DiffusionTransformer import DiffusionTransformer
from model.ModelUtils import *

def network(in_size, out_size, h_size, num_layers, activation): 
  network = [] 
  for _ in range(num_layers): 
    network.append(nn.Linear(in_size, h_size))
    network.append(nn.GELU(approximate = 'tanh'))  
    in_size = h_size 
  network.append(nn.Linear(in_size, out_size))
  network.append(get_activation(activation))
  return network 

class ConditionalLinear(nn.Module): 
  def __init__(self, input_dim, output_dim, num_diffusion_steps):
    super(ConditionalLinear, self).__init__() 
    self.input_dim = input_dim 
    self.output_dim = output_dim 
    self.num_diffusion_steps = num_diffusion_steps 
    
    self.linear_1 = nn.Linear(input_dim, output_dim) 
    self.embed_1 = nn.Embedding(num_diffusion_steps, output_dim) 
    self.embed_1.weight.data.uniform_()
    torch.nn.init.xavier_normal_(self.linear_1.weight) 

  def forward(self, x, y): 
    out = self.linear_1(x) 
    gamma = self.embed_1(y) 
    out = out * gamma.view(-1, self.output_dim)
    return out 
  
class ValueApproximator(nn.Module): 
  def __init__(self, num_diffusion_steps, img_shape, layer_dims, activation):
    super(ValueApproximator, self).__init__() 
    self.num_diffusion_steps = num_diffusion_steps 
    self.img_shape = img_shape 
    self.num_layers = len(layer_dims)
    self.linear_0 = ConditionalLinear(int(np.prod(img_shape)), layer_dims[0], num_diffusion_steps) 
    for i in range(1, len(layer_dims)): 
      setattr(self, f"linear_{i}", ConditionalLinear(
        input_dim = layer_dims[i - 1], 
        output_dim = layer_dims[i], 
        num_diffusion_steps = num_diffusion_steps
      ))
    self.final_layer = nn.Linear(layer_dims[-1], 1) 
    torch.nn.init.xavier_normal_(self.final_layer.weight) 

    self.act = get_activation(activation)

  def forward(self, x, time): 
    x = x.view(x.shape[0], -1)   
    x = self.act(self.linear_0(x, time))
    for i in range(1, self.num_layers):  
      x = getattr(self, f"linear_{i}")(x, time) 
      x = self.act(x) 
    return self.final_layer(x)

class LinearValueApproximator(nn.Module): 
  def __init__(self, num_diffusion_steps, input_dim, layer_dims, activation):
    super(LinearValueApproximator, self).__init__() 
    self.num_diffusion_steps = num_diffusion_steps 
    self.input_dim = input_dim
    self.num_layers = len(layer_dims)
    self.linear_0 = ConditionalLinear(input_dim, layer_dims[0], num_diffusion_steps) 
    for i in range(1, len(layer_dims)): 
      setattr(self, f"linear_{i}", ConditionalLinear(
        input_dim = layer_dims[i - 1], 
        output_dim = layer_dims[i], 
        num_diffusion_steps = num_diffusion_steps
      ))
    self.final_layer = nn.Linear(layer_dims[-1], 1) 
    torch.nn.init.xavier_normal_(self.final_layer.weight) 
    self.act = get_activation(activation)

  def forward(self, x, timesteps, cond): 
    x = x.view(x.shape[0], -1) 
    cond = cond.view(cond.shape[0], -1)
    x = torch.cat([x, cond], dim = -1)
    x = self.act(self.linear_0(x, timesteps))
    for i in range(1, self.num_layers):  
      x = getattr(self, f"linear_{i}")(x, timesteps) 
      x = self.act(x) 
    return self.final_layer(x)

class TransformerValueApproximator(nn.Module): 
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
    use_cross_attention = False, 
    embedding_max_period = 10000,
    learnable_embedding = True, 
    pos_encoding_params = (10000, 1), 
    flatten_arc = False, 
    use_excitation = False, 
    se_ratio = 4, 
    use_max_pooling = False,
    use_adaptive = True,  
):
    super(TransformerValueApproximator, self).__init__() 
    self.transformer = DiffusionTransformer(
      input_features = input_features, 
      prediction_len = num_dct_channels, 
      model_dim = model_dim, 
      ffn_dim = ffn_dim, 
      num_layers = num_layers,
      num_heads = num_heads, 
      dropout = dropout, 
      activation = activation, 
      use_cross_attention = use_cross_attention, 
      embedding_max_period = embedding_max_period,
      learnable_embedding = learnable_embedding, 
      pos_encoding_params = pos_encoding_params, 
      flatten_arc = flatten_arc, 
      use_excitation = use_excitation, 
      se_ratio = se_ratio, 
      use_max_pooling = use_max_pooling,
      use_adaptive = use_adaptive,  
    )
    self.avg_module = torch.nn.AdaptiveAvgPool1d(1)
    self.value_approx = nn.Sequential(
      nn.Linear(num_dct_channels, model_dim),
      get_activation(activation), 
      nn.Dropout(p = dropout), 
      nn.Linear(model_dim, 1), 
    )

  def forward(self, x, timesteps, cond): 
    x = self.transformer(x, timesteps, cond) 
    x = self.avg_module(x) 
    out = self.value_approx(x) 
    return out 


def get_linear_value_model(params):
  return LinearValueApproximator(
    num_diffusion_steps = params['num_diffusion_steps'], 
    input_dim = params['num_dct_channels'] * params['pose_dim'] * 2, 
    layer_dims = params['value_network_layer_dims'], 
    activation = params['value_activation']  
  ) 

def get_transformer_value_model(params):
  return TransformerValueApproximator(
    input_features = params['pose_dim'], 
    num_dct_channels = params['num_dct_channels'], 
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
  )
