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
    input_dim, 
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
  ):
    super(GCNDiffusionTransformerLayer, self).__init__()
    self.input_dim = input_dim 
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

    #self.proj_m = nn.Linear(self.input_dim, self.model_dim)
    #self.proj_o = nn.Linear(self.model_dim, self.input_dim)
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

  def forward(self, input, embedding): 
    """
    :param input     : a tensor of shape (B, num_dct_channels, input_dim) 
    :param embedding : a tensor of shape (B, embedding_dim) 
    :return : 
      a tensor of shape (B, T, input_dim)
    """       
   # x = self.proj_m(input)
    x = input 
    if self.use_excitation: 
      x = self.excitation_layer(x)
    out = self.attention_layer(x, embedding)
    out = self.ffn_layer(out, embedding)
    #out = self.proj_o(out) 
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
  

class GCNDiffusion(nn.Module): 
  def __init__(
    self, 
    input_dim, 
    num_features, 
    model_dim = 256, 
    dropout = 0.2, 
    activation = 'GeLU', 
    num_gcn_stages = 2, 
    use_max_pooling = False,
    embedding_max_period = 10000, 
  ):
    super(GCNDiffusion, self).__init__()
    self.input_dim = input_dim
    self.num_features = num_features 
    self.dropout_p = dropout 
    self.activation = activation 
    self.time_embedding_dim = model_dim
    self.use_max_pooling = use_max_pooling 
    self.num_gcn_states = num_gcn_stages
    self.model_dim = model_dim 
    self.embedding_max_period = embedding_max_period

    def get_act(activation_fn):
      if activation_fn == 'GeLU':
        return nn.GELU(approximate = 'tanh') 
      elif activation_fn == 'ReLU':
        return nn.ReLU() 
      elif activation_fn == 'LeakyReLU':
        return nn.LeakyReLU() 
      elif activation_fn == 'ELU': 
        return nn.ELU()
      else : 
        raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')


    self.sequence_embedding = nn.Parameter(torch.randn(input_dim, num_features))

    
    self.conditional_embedding = nn.Linear(
      self.num_features * self.input_dim, 
      self.time_embedding_dim
    )

    self.time_embedding = nn.Sequential(
      nn.Linear(self.model_dim, self.time_embedding_dim), 
      nn.SiLU(), 
      nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
    )

    self.gcn_blocks = GCN(
      input_feature = self.num_features,
      hidden_feature = self.model_dim, 
      embedding_dim = self.time_embedding_dim, 
      p_dropout = self.dropout_p, 
      normalizer = ScaleShiftLayer, 
      num_stage = self.num_gcn_states,
      node_n = self.input_dim, 
      activation = get_act(self.activation), 
    ) 
    
  def forward(self, input, timesteps, memory = None):
    """
    :param input : a tensor of shape (B, num_dct_channels, num_joints * 3) 
    :param timesteps : a tensor of shape (B, ) 
    :param memory : if not None, can be the observed seqs information of shape (B, num_dct_channels, num_joints * 3)
    """

    batch_size, _, _ = input.shape

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
      total_embedding = total_embedding + self.conditional_embedding(enc_memory) 
    input = input.permute(0, 2, 1)
    x = input + self.sequence_embedding.unsqueeze(0)[:, :self.input_dim, :]
    x = x.permute(0, 2, 1)
    out = self.gcn_blocks(x, total_embedding)
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
    pre_gcn = False, 
    flatten_arc = False, 
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
    self.pre_gcn = pre_gcn
    self.flatten_arc = flatten_arc

    def get_act(activation_fn):
      if activation_fn == 'GeLU':
        return nn.GELU(approximate = 'tanh') 
      elif activation_fn == 'ReLU':
        return nn.ReLU() 
      elif activation_fn == 'LeakyReLU':
        return nn.LeakyReLU() 
      elif activation_fn == 'ELU': 
        return nn.ELU()
      else : 
        raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')


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
    # pre-gcn blocks 
    if self.pre_gcn: 
      self.pre_gcn_blocks = GCN(
        input_feature = self.num_features,
        hidden_feature = self.gcn_dim,
        embedding_dim = self.time_embedding_dim, 
        p_dropout = self.dropout_p, 
        normalizer = ScaleShiftLayer, 
        num_stage = self.num_gcn_states,
        node_n = self.input_dim, 
        activation = get_act(self.activation), 
      ) 
      self.vertices_embeddings = nn.Parameter(torch.randn(self.input_dim, self.num_features))


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
      activation = get_act(self.activation), 
    ) 

    for _ in range(self.num_layers):
      self.attn_blocks.append(
        GCNDiffusionTransformerLayer(
          input_dim = self.input_dim, 
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
        )
      )
    self.attn_blocks = nn.ModuleList(self.attn_blocks)
    self.pose_encoder = nn.Linear(input_dim, model_dim)
    self.proj_p = ConditionalTransformerLayer(
      model_dim = model_dim, 
      output_dim = input_dim, 
      embedding_dim = self.time_embedding_dim, 
    )
    
  def forward(self, input, timesteps, memory = None):
    """
    :param input : a tensor of shape (B, num_dct_channels, num_joints * 3) 
    :param timesteps : a tensor of shape (B, ) 
    :param memory : if not None, can be the observed seqs information of shape (B, num_dct_channels, num_joints * 3)
    """

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

    x = input 
    if self.pre_gcn:
      x = x.permute(0, 2, 1) + self.vertices_embeddings.unsqueeze(0)[:, :self.input_dim, :]
      x = x.permute(0, 2, 1)
      x = self.pre_gcn_blocks(x, total_embedding)  
    x = self.pose_encoder(x)
    x = x + self.sequence_embedding.unsqueeze(0)[:, :num_dct_channels, :]
    
    for i in range(self.num_layers): 
      x = self.attn_blocks[i](x, total_embedding)

    if self.flatten_arc == False: 
      enc_list = [] 
      for index, module in enumerate(self.posterior_decoder_block): 
        if index < self.num_layers // 2: 
          enc_list.append(x) 
          x = module(x, total_embedding, memory)
        else :  
          x = module(x, total_embedding, memory)
          x += enc_list[-1]  
          enc_list.pop() 
    else : 
      for _, module in enumerate(self.posterior_decoder_block): 
        x = module(x, total_embedding, memory)  

    out = self.proj_p(x, total_embedding) 
    out = self.post_gcn_blocks(out, total_embedding)
    return out

  
#'''
# Try this on 3D instead of GCN,  
# Chebyshev GCN convolution
# """
# class ChebConv(nn.Module): 
#   def __init__(self, in_channels, out_channels, K, bias = True, normalize = True):
#     super(ChebConv, self).__init__() 
#     self.normalize = normalize 
#     self.weight = torch.nn.Parameter(torch.Tensor(K + 1, 1, in_channels, out_channels))
#     torch.nn.init.normal_(self.weight)

#     if bias : 
#       self.bias = torch.nn.Parameter(torch.Tensor(1, 1, out_channels))
#       torch.nn.init.zeros_(self.bias) 
#     else: 
#       self.register_parameter("bias", None)

#     self.K = K + 1 

#   @staticmethod 
#   def get_laplacian(graph, normalize): 
#     """
#     return the Laplacian of the graph 
#     :param graph : the graph structure without self loop : [N, N] 
#     :param normalize : whether to use the normalized laplacian 
#     :return : 
#         graph laplacian 
#     """
#     if normalize:
#       D = torch.diag(torch.sum(graph, dim = -1) ** (-1 / 2))
#       L = torch.eye(graph.size(0), device = graph.device, dtype = graph.dtype) 
#     else : 
#       D = torch.diag(torch.sum(graph, dim = -1)) 
#       L = D - graph 
#     return L  

#   def cheb_polynomial(self, laplacian): 
#     """
#     Compute the Chebyshev Polynomial, according to the graph Laplacian
#     :param laplacian : the graph laplacian, [N, N]. 
#     :return : 
#       the multi order Chebyshev laplacian, [K, N, N] 
#     """
#     N = laplacian.size(0) 
#     multi_order_laplacian = torch.zeros([self.K, N, N], device = laplacian.device, dtype = torch.float32)

#     multi_order_laplacian[0] = torch.eye(N, device = laplacian.device, dtype = torch.float32)

#     if self.K == 1: 
#       return multi_order_laplacian 
#     else : 
#       multi_order_laplacian[1] = laplacian 
#       if self.K == 2: 
#         return multi_order_laplacian 
#       else : 
#         for k in range(2, self.K): 
#           multi_order_laplacian[k] = 2 * torch.matmul(laplacian, multi_order_laplacian[k - 1]) -\
#             multi_order_laplacian[k - 2]
#     return multi_order_laplacian
  
#   def forward(self, inputs, graph): 
#     """
#     :param inputs : the input data, [B, N, C] 
#     :param graph : the graph structure : [N, N] 
#     :return : 
#        convolution result, [B, N, D] 
#     """
#     laplacian = ChebConv.get_laplacian(graph, self.normalize)
#     multi_laplacian = self.cheb_polynomial(laplacian).unsqueeze(1) # [K, 1, N, N]
#     result = torch.matmul(multi_laplacian, inputs) # [K, B, N, C]  
#     result = torch.matmul(result, self.weight) # [K, B, N, D]
#     result = torch.sum(result, dim = 0) + self.bias #[B, N, D]
#     return result 

# class ChebGraphConv(nn.Module): 
#   def __init__(self, input_dim, output_dim, activation, p_dropout = 0.2): 
#     super(ChebGraphConv, self).__init__() 
#     self.gconv = ChebConv(input_dim, output_dim, K = 2) 
#     self.act = activation 
#     self.dropout = nn.Dropout(p = p_dropout)

#   def forward(self, x, adj_matrix): 
#     x = self.gconv(x, adj_matrix)
#     x = self.dropout(self.act(x))
#     return x  

# class ResidualChebGCDiff(nn.Module): 
#   def __init__(
#     self, 
#     adj_matrix, 
#     input_dim,
#     embedding_dim,  
#     output_dim, 
#     activation, 
#     hidden_dim, 
#     p_dropout = 0.2
#   ): 
#     super(ResidualChebGCDiff, self).__init__()
#     self.adj_matrix = adj_matrix 
#     self.gconv_a = ChebGraphConv(input_dim, hidden_dim, activation, p_dropout) 
#     self.gconv_b = ChebGraphConv(hidden_dim, output_dim, activation, p_dropout)
#     self.embedding_proj = nn.Sequential(
#       nn.SiLU(), 
#       nn.Linear(embedding_dim, hidden_dim),
#     )
#     self.proj_o = zero_module(nn.Linear(hidden_dim, hidden_dim))

#   def forward(self, x, cond): 
#    resi = x 
#    resi = self.gconv_a(resi, self.adj_matrix)
#    resi = resi + self.embedding_proj(cond).unsqueeze(1)
#    resi = self.gconv_b(resi, self.adj_matrix) 
#    resi = self.proj_o(resi)
#    return x + resi

# class LaplacianGraphConv(nn.Module): 
#   def __init__(self, in_features, out_features, activation): 
#     super(LaplacianGraphConv, self).__init__() 
#     self.fc = nn.Linear(in_features, out_features) 
#     self.act = activation 

#   def laplacian(self, tilde_A): 
#     tilde_D = (torch.sum(tilde_A, 0) + 1e-5) ** (-0.5) 
#     L = tilde_D * tilde_A * tilde_D 
#     return L 

#   def laplacian_batch(self, tilde_A): 
#     batch, N = tilde_A.shape[:2] 
#     tilde_D = (torch.sum(tilde_A, 1) + 1e-5) ** (-0.5) 
#     L = tilde_D.view(batch, N, 1) * tilde_A * tilde_D.view(batch, 1, N) 
#     return L 

#   def forward(self, input, A): 
#     batch = input.size(0) 
#     tilde_A = A.unsqueeze(0).repeat(batch, 1, 1)
#     x = self.fc(torch.bmm(self.laplacian_batch(tilde_A), input))
#     if self.act is not None: 
#       x = self.act(x) 
#     return x  

# class GraphNet(nn.Module): 
#   def __init__(
#     self, 
#     in_features, 
#     out_features, 
#     hidden_dim, 
#     num_points, 
#     activation
#   ): 
#     super(GraphNet, self).__init__()
#     self.A = torch.nn.Parameter(torch.eye(num_points, dtype = torch.float32), requires_grad = True)
#     self.gconv_a = LaplacianGraphConv(in_features, hidden_dim, activation = activation) 
#     self.gconv_b = LaplacianGraphConv(hidden_dim, out_features, activation = None)

#   def forward(self, input): 
#     x = self.gconv_a(input, self.A) 
#     x = self.gconv_b(x, self.A) 
#     return x
  
# class GraphAttentionLayer(nn.Module):
#   def __init__(
#     self, 
#     num_points, 
#     model_dim, 
#     ffn_dim, 
#     embedding_dim, 
#     num_head, 
#     dropout, 
#     activation,
#     use_adaptive, 
#   ):
#     super(GraphAttentionLayer, self).__init__()
#     self.model_dim = model_dim 
#     self.ffn_dim = ffn_dim 
#     self.embedding_dim = embedding_dim
#     self.num_head = num_head 
#     self.dropout = dropout
#     self.use_adaptive = use_adaptive
#     self.activation = activation 

#     self.attention_layer = (AdaptiveMultiHeadAttention if \
#       self.use_adaptive else MultiHeadAttention) (
#       model_dim = model_dim, 
#       num_head  = num_head, 
#       embedding_dim = embedding_dim, 
#       dropout = dropout, 
#     )

#     self.graph_net = GraphNet(
#       in_features = model_dim, 
#       out_features = model_dim, 
#       hidden_dim = ffn_dim,
#       num_points = num_points,  
#       activation = activation, 
#     )
 
#     self.norm = ScaleShiftLayer(
#       latent_dim = model_dim, 
#       embedding_dim = embedding_dim, 
#       dropout = dropout, 
#     )

    
#   def forward(self, input, embedding): 
#     """
#     :param input : a tensor of shape (B, T, model_dim) 
#     :param embedding : a tensor of shape (B, embedding_dim) 
#     :return : 
#       a tensor of shape (B, T, model_dim)
#     """       
#     out = self.attention_layer(input, embedding)
#     out = self.graph_net(self.norm(out, embedding))
#     return out 

# class ChebGCNDiffusionTransformer(nn.Module): 
#   def __init__(
#     self,
#     adj_matrix, 
#     num_points, 
#     joint_dim, 
#     num_features,   
#     model_dim, 
#     ffn_dim, 
#     num_layers,
#     num_heads, 
#     gcn_model_dim, 
#     dropout = 0.2, 
#     activation = 'GeLU', 
#     embedding_max_period = 10000,
#     learnable_embedding = True, 
#     pos_encoding_params = (10000, 1), 
#     use_adaptive = True,  
#   ):
#     super(ChebGCNDiffusionTransformer, self).__init__()
#     self.adj_matrix = adj_matrix 
#     self.num_points = num_points 
#     self.num_features = num_features 
#     self.model_dim = model_dim 
#     self.ffn_dim = ffn_dim 
#     self.num_layers = num_layers 
#     self.num_heads = num_heads 
#     self.dropout_p = dropout 
#     self.activation = activation 
#     self.time_embedding_dim = model_dim
#     self.embedding_max_period = embedding_max_period
#     self.learnable_embedding = learnable_embedding 
#     self._pos_encoding_params = pos_encoding_params
#     self.use_adaptive = use_adaptive
#     self.joint_dim = joint_dim
#     self.gcn_model_dim = gcn_model_dim

#     def get_act(activation_fn):
#       if activation_fn == 'GeLU':
#         return nn.GELU(approximate = 'tanh') 
#       elif activation_fn == 'ReLU':
#         return nn.ReLU() 
#       elif activation_fn == 'LeakyReLU':
#         return nn.LeakyReLU() 
#       elif activation_fn == 'ELU': 
#         return nn.ELU()
#       else : 
#         raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')


#     if self.learnable_embedding : 
#         self.sequence_embedding = nn.Parameter(torch.randn(self.num_points, model_dim))
#     else : 
#         positional_encoding = PositionalEncoding1D(
#          model_dim = self.model_dim,
#          temperature = self._pos_encoding_params[0],
#          alpha = self._pos_encoding_params[1]
#         )
#         sequence_embedding = positional_encoding(self.num_points - 1)
#         self.sequence_embedding = nn.Parameter(sequence_embedding, requires_grad = False)

#     self.conditional_embedding = nn.Linear(
#       (num_points - 1) * joint_dim * num_features,  
#       self.time_embedding_dim
#     )

#     self.time_embedding = nn.Sequential(
#       nn.Linear(self.model_dim, self.time_embedding_dim), 
#       nn.SiLU(), 
#       nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
#     )

#     self.in_features = self.joint_dim * self.num_features
#     self.out_features = self.in_features

#     self.gconv_input = ChebConv(
#       in_channels = self.in_features, 
#       out_channels = model_dim, 
#       K = 2, 
#     )

#     gconv_layers = [] 
#     attn_layers = [] 

#     for _ in range(self.num_layers): 
#       gconv_layers.append(
#         ResidualChebGCDiff(
#           adj_matrix = self.adj_matrix, 
#           input_dim = self.model_dim, 
#           embedding_dim = self.time_embedding_dim, 
#           output_dim = self.model_dim, 
#           activation = get_act(self.activation), 
#           hidden_dim = self.model_dim, 
#           p_dropout = self.dropout_p, 
#         )
#       )
#       attn_layers.append(
#         GraphAttentionLayer(
#           num_points = num_points, 
#           model_dim = model_dim,
#           ffn_dim = ffn_dim, 
#           embedding_dim = self.time_embedding_dim, 
#           num_head = num_heads, 
#           dropout = self.dropout_p, 
#           activation = get_act(activation), 
#           use_adaptive = use_adaptive, 
#         )
#       )

#     self.gconv_layers = nn.ModuleList(gconv_layers)
#     self.attn_layers = nn.ModuleList(attn_layers)

#     self.gconv_output = ChebConv(
#       in_channels = model_dim, 
#       out_channels = self.out_features, 
#       K = 2, 
#     )

#     self.proj_o = zero_module(nn.Linear((self.num_points - 1) * self.joint_dim, (self.num_points - 1) * self.joint_dim)) 

       
#   def forward(self, input, timesteps, memory = None):
#     """
#     :param input : a tensor of shape (B, num_features, num_joints * 3) 
#     :param timesteps : a tensor of shape (B, ) 
#     :param memory : if not None, can be the observed seqs information 
#     """

#     batch_size, num_dct_channels, _ = input.shape
#     input = input.contiguous().view(batch_size, num_dct_channels, self.num_points - 1, -1).permute(0, 2, 3, 1) # (b, c, n - 1, d) -> (b, n - 1, d, c) 
#     input = input.contiguous().view(batch_size, self.num_points - 1, -1) # (b, n - 1, d * c)
#     padded_root = torch.zeros((input.shape[0], 1, input.shape[2]), dtype = torch.float32).to(input.device) 
#     input = torch.cat([padded_root, input], dim = 1)  # (b, n, d * c)

#     time_embedding = self.time_embedding(
#       timestep_embedding(
#         timesteps = timesteps, 
#         dim = self.model_dim, 
#         max_period = self.embedding_max_period, 
#       )
#     )
#     total_embedding = time_embedding
#     if memory is not None : 
#       enc_memory = memory.contiguous().view(batch_size, -1)
#       total_embedding = total_embedding + self.conditional_embedding(enc_memory)

#     x = self.gconv_input(input, self.adj_matrix) # (b, n, model_dim)
#     x = x + self.sequence_embedding.unsqueeze(0)[:, :self.num_points, :]
    
#     for i in range(self.num_layers): 
#       x = self.attn_layers[i](x, total_embedding)
#       x = self.gconv_layers[i](x, total_embedding) 
#     output = self.gconv_output(x, self.adj_matrix) # (b, n, d * c)
#     output = output[:, 1:, :]  # (b, n - 1, d * c)

#     output = output.contiguous().view(batch_size, self.num_points - 1, -1, self.num_features).permute(0, 3, 1, 2) 
#     output = output.contiguous().view(batch_size, self.num_features, -1)
#     output = self.proj_o(output)
#     return output 
  

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
    pre_gcn = params['pre_gcn'], 
    flatten_arc = params['flatten_arc'], 
  )

def gcn_diffusion(params):
  return GCNDiffusion(
    input_dim = params['pose_dim'], 
    num_features = params['num_dct_channels'], 
    model_dim = params['model_dim'], 
    dropout = params['dropout'], 
    activation = params['activation'],
    use_max_pooling = params['use_max_pooling'],   
    num_gcn_stages = params['num_gcn_stages'], 
    embedding_max_period = params['embedding_max_period'], 
  )

# def cheb_gcn_diffusion_transformer(params): 
#   return ChebGCNDiffusionTransformer(
#     adj_matrix = params['adj_matrix'], 
#     joint_dim = params['joint_dim'], 
#     num_points = params['num_joints'] + 1, # included the root to match the adj_matrix 
#     num_features = params['num_dct_channels'], 
#     model_dim = params['model_dim'], 
#     ffn_dim = params['ffn_dim'], 
#     num_layers = params['num_layers'],
#     num_heads = params['num_heads'], 
#     dropout = params['dropout'], 
#     activation = params['activation'], 
#     embedding_max_period = params['embedding_max_period'], 
#     learnable_embedding = params['learnable_embedding'], 
#     pos_encoding_params = (params['pos_enc_beta'], params['pos_enc_alpha']), 
#     use_adaptive = params['use_adaptive'], 
#     gcn_model_dim = params['gcn_dim'], 
#   )