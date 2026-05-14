import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PositionalEncoding import * 

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class AdaptiveNormalization(nn.Module):
    def __init__(self, latent_dim, embedding_dim, dropout = 0.1):
        super(AdaptiveNormalization, self).__init__()
        self.latent_dim    = latent_dim 
        self.embedding_dim = embedding_dim
        self.dropout       = dropout 
        self.embedding_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embedding_dim, 2 * latent_dim),
        )
        self.layernorm = nn.LayerNorm(self.latent_dim)
        self.proj_out = nn.Sequential(
            nn.SiLU(), 
            nn.Dropout(p = dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)), 
        )
    def forward(self, latent_input, embedding_input): 
        """
        :param latent_input    : a tensor of shape (B, T, latent_dim)
        :param embedding_input : a tensor of shape (B, embedding_dim) 
        :return : 
           a tensor of shape (B, T, latent_dim)
        """

        embedding_out = self.embedding_proj(embedding_input).unsqueeze(1) # (B, 1, 2 * latent_dim) 
        scale, shift  = torch.chunk(embedding_out, 2, dim = -1) # (B, 1, latent_dim), (B, 1, latent_dim) 

        latent_out = self.layernorm(latent_input) * (1. + scale) + shift 
        out = self.proj_out(latent_out)

        return out 

class FeedForwardNetwork(nn.Module): 
    def __init__(self, model_dim, ffn_dim, embedding_dim, dropout = 0.1, activation = 'GeLU'):
        super(FeedForwardNetwork, self).__init__()
        self.model_dim     = model_dim 
        self.ffn_dim       = ffn_dim 
        self.embedding_dim = embedding_dim 

        self.adaptiveLN    = AdaptiveNormalization(latent_dim = model_dim, embedding_dim = embedding_dim, dropout = dropout) 

        self.linear_1 = nn.Linear(model_dim, ffn_dim)
        self.linear_2 = nn.Linear(ffn_dim, model_dim)

        def get_act(activation_fn):
            if activation_fn   == 'GeLU':
                return nn.GELU() 
            elif activation_fn == 'ReLU':
                return nn.ReLU() 
            elif activation_fn == 'LeakyReLU':
                return nn.LeakyReLU() 
            else : 
                raise NotImplementedError(f'Activation function {activation_fn} does not support for this class')
            
        self.act     = get_act(activation_fn = activation)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, input, embedding):
        """
        :param input     : a tensor of shape (B, T, model_dim) 
        :param embedding : a tensor of shape (B, embedding_dim) 
        :return : 
           a tensor of shape (B, T, model_dim)
        """

        ffn_out = self.linear_2(self.dropout(self.act(self.linear_1(input))))
        out     = input + self.adaptiveLN(ffn_out, embedding)
        return out 
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, model_dim, embedding_dim, num_head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.model_dim     = model_dim 
        self.embedding_dim = embedding_dim 
        self.num_head      = num_head 

        assert self.model_dim % self.num_head == 0, "Model dim must be divisible by num heads" 

        self.dropout       = nn.Dropout(dropout)
        self.layernorm     = nn.LayerNorm(model_dim)

        self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
        self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
        self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)

        self.adaptiveLN = AdaptiveNormalization(latent_dim = model_dim, embedding_dim = embedding_dim, dropout = dropout)

    def forward(self, input, embedding): 
        """
        :param input     : a tensor of shape (B, T, model_dim) 
        :param embedding : a tensor of shape (B, embedding_dim) 
        :return : 
           a tensor of shape (B, T, model_dim)
        """       

        batch_size, sequence_len, _ = input.shape
        latent_dim = self.model_dim // self.num_head
        norm_inp = self.layernorm(input) 
        query = self.query_proj(norm_inp).contiguous().view(batch_size, sequence_len, self.num_head, -1).permute(0, 2, 1, 3)
        query = query.contiguous().view(batch_size * self.num_head, sequence_len, -1)
        key   = self.key_proj(norm_inp).contiguous().view(batch_size, sequence_len,   self.num_head, -1).permute(0, 2, 1, 3)
        key   = key.contiguous().view(batch_size * self.num_head, sequence_len, -1)
        value = self.value_proj(norm_inp).contiguous().view(batch_size, sequence_len,  self.num_head, -1).permute(0, 2, 1, 3)
        value = value.contiguous().view(batch_size * self.num_head, sequence_len, -1)

        attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim) 
        attention_matrix = self.dropout(torch.softmax(attention_matrix, dim = -1))
        attention_out = torch.bmm(attention_matrix, value) 

        attention_out = attention_out.contiguous().view(batch_size, self.num_head, sequence_len, -1).permute(0, 2, 1, 3)
        attention_out = attention_out.contiguous().view(batch_size, sequence_len, -1)
        out = self.adaptiveLN(attention_out, embedding) + input
        return out 
    
# In the case I use the pretrained stuff for observed sequences.     
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, embedding_dim, num_head, dropout): 
        super(MultiHeadCrossAttention, self).__init__()
        self.model_dim     = model_dim 
        self.embedding_dim = embedding_dim
        self.num_head      = num_head 
        
        self.dropout       = nn.Dropout(dropout)
        self.layernorm     = nn.LayerNorm(model_dim)

        self.query_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
        self.key_proj   = nn.Linear(self.model_dim, self.model_dim, bias = False) 
        self.value_proj = nn.Linear(self.model_dim, self.model_dim, bias = False)
        self.adaptiveLN = AdaptiveNormalization(latent_dim = model_dim, embedding_dim = embedding_dim, dropout = dropout)


    def forward(self, input, embedding, memory): 
        """
        :param input     : a tensor of shape (B, T_{predict}, model_dim) 
        :param memory    : a tensor of shape (B, T_{observed}, model_dim)
        :param embedding : a tensor of shape (B, embedding_dim) 
        :return : 
           a tensor of shape (B, T, model_dim)
        """     
        batch_size, tgt_sequence_length, _ = input.shape
        _, src_sequence_length, _ = memory.shape

        latent_dim = self.model_dim // self.num_head

        norm_input = self.layernorm(input)
        query = self.query_proj(norm_input).contiguous().view(batch_size, tgt_sequence_length, self.num_head, -1).permute(0, 2, 1, 3)
        query = query.contiguous().view(batch_size * self.num_head, tgt_sequence_length, -1)
        key   = self.key_proj(norm_input).contiguous().view(batch_size, tgt_sequence_length,   self.num_head, -1).permute(0, 2, 1, 3)
        key   = key.contiguous().view(batch_size * self.num_head, tgt_sequence_length, -1)
        value = self.value_proj(memory).contigous().view(batch_size, src_sequence_length,  self.num_head, -1).permute(0, 2, 1, 3)
        value = value.contiguous().view(batch_size * self.num_head, src_sequence_length, -1)

        cross_attention_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(latent_dim)
        cross_attention_matrix = self.dropout(torch.softmax(cross_attention_matrix, dim = -1))
        
        cross_attention_out = torch.bmm(cross_attention_matrix, value) 

        cross_attention_out = cross_attention_out.contiguous().view(batch_size, self.num_head, tgt_sequence_length, -1).permute(0, 2, 1, 3)
        cross_attention_out = cross_attention_out.contiguous().view(batch_size, tgt_sequence_length, -1)

        out = self.adaptiveLN(cross_attention_out, embedding) + input

        return out 



class DiffusionTransformerLayer(nn.Module):
    def __init__(self, model_dim, ffn_dim, embedding_dim, num_head, dropout, activation, use_cross_attention):
        super(DiffusionTransformerLayer, self).__init__()
        self.model_dim           = model_dim 
        self.ffn_dim             = ffn_dim 
        self.embedding_dim       = embedding_dim
        self.num_head            = num_head 
        self.dropout             = dropout
        self.use_cross_attention = use_cross_attention

        self.attention_layer = MultiHeadAttention(
            model_dim = model_dim, 
            num_head  = num_head, 
            embedding_dim = embedding_dim, 
            dropout = dropout
        )
        self.ffn_layer = FeedForwardNetwork(
            model_dim = model_dim,
            ffn_dim   = ffn_dim, 
            embedding_dim = embedding_dim, 
            dropout = dropout, 
            activation = activation, 
        )
        self.cross_attention = None 
        if self.use_cross_attention : 
            self.cross_attention = MultiHeadCrossAttention(
                model_dim = model_dim, 
                embedding_dim = embedding_dim, 
                num_head = num_head, 
                dropout = dropout, 
            ) 

    def forward(self, input, embedding, memory = None): 
        """
        :param input     : a tensor of shape (B, T, model_dim) 
        :param embedding : a tensor of shape (B, embedding_dim) 
        :return : 
           a tensor of shape (B, T, model_dim)
        """       

        out = self.attention_layer(input, embedding)
        if self.cross_attention is not None : 
            assert memory is not None, "Memory is none, cannot perform Cross Attention"
            out = self.cross_attention(out, memory, embedding)
        out = self.ffn_layer(out, embedding)
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
        encoding_temperature = 10000, 
        encoding_alpha       = 1,
        diffusion_num_timesteps = 1000, 
    ):
        super(DiffusionTransformer, self).__init__()
        self.input_features          = input_features
        self.prediction_len          = prediction_len 
        self.model_dim               = model_dim 
        self.ffn_dim                 = ffn_dim 
        self.num_layers              = num_layers 
        self.use_cross_attention     = use_cross_attention 
        self.num_heads               = num_heads 
        self.dropout_p               = dropout 
        self.activation              = activation 
        self.time_embedding_dim      = model_dim
        self.diffusion_num_timesteps = diffusion_num_timesteps 

        self.pos_sinusoidal = PositionalEncoding1D(
            position_dim   = model_dim, 
            temperature    = encoding_temperature, 
            alpha = encoding_alpha, 
        )(self.prediction_len).reshape(self.prediction_len, self.model_dim)

        self.positional_encoding = nn.Parameter(self.pos_sinusoidal, requires_grad = False)

        self.proj_model = nn.Linear(self.input_features, self.model_dim)

        self.conditon_embedding = nn.Linear(self.input_features * self.prediction_len, self.time_embedding_dim)

        self.time_sinusoidal = PositionalEncoding1D(
            position_dim   = model_dim, 
            temperature    = encoding_temperature, 
            alpha          = encoding_alpha, 
        )(self.diffusion_num_timesteps).reshape(self.diffusion_num_timesteps, self.model_dim)

        self.time_encoding = nn.Parameter(self.time_sinusoidal, requires_grad = False)


        self.time_embedding = nn.Sequential(
            nn.Linear(self.model_dim, self.time_embedding_dim), 
            nn.SiLU(), 
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
        )

        self.posterior_decoder_block = nn.ModuleList()

        for _ in range(self.num_layers):
            self.posterior_decoder_block.append(
                DiffusionTransformerLayer(
                    model_dim           = self.model_dim, 
                    ffn_dim             = self.ffn_dim, 
                    embedding_dim       = self.time_embedding_dim,
                    num_head            = self.num_heads, 
                    dropout             = dropout, 
                    activation          = activation, 
                    use_cross_attention = use_cross_attention
                )
            )
        self.proj_out = zero_module(nn.Linear(self.model_dim, self.input_features))

    def forward(self, input, timesteps, memory = None):
        """
        :param input : a tensor of shape (B, T_{prediction}, input_features) 
        :param timesteps : a tensor of shape (B, ) 
        :param memory : if not None, can be the observed sequences information 
        """

        batch_size = input.shape[0]

        embedding = self.time_embedding(self.time_encoding[timesteps])

        if memory is not None : 
            enc_memory = memory.contiguous().view(batch_size, -1)
            embedding  = embedding + self.conditon_embedding(enc_memory)


        network_in = self.proj_model(input) # (B, T_{prediction}, model_dim)
        network_in = network_in + self.positional_encoding

        enc_list = [] 
        # U shape enc-dec 
        for index, module in enumerate(self.posterior_decoder_block): 
            if index < self.num_layers // 2: 
                enc_list.append(network_in) 
                network_in = module(network_in, embedding, memory)
            else : 
                network_in = module(network_in, embedding, memory)
                network_in += enc_list[-1]
                enc_list.pop()

        output = self.proj_out(network_in)
        return output 


def get_diffusion_transformer_model(
    input_features  = 51, 
    prediction_len  = 240, 
    num_heads       = 8, 
    model_dim       = 512, 
    ffn_dim         = 2048, 
    num_layers      = 6, 
    dropout         = 0.2, 
    num_diffusion_timesteps         = 100, 
    positional_encoding_temperature = 10000, 
    positional_encoding_alpha       = 1,
    use_cross_attention             = False,
    activation                      = 'GeLU', 
):
    return DiffusionTransformer(
        input_features          = input_features, 
        prediction_len          = prediction_len, 
        model_dim               = model_dim, 
        ffn_dim                 = ffn_dim, 
        num_layers              = num_layers,
        num_heads               = num_heads, 
        dropout                 = dropout, 
        activation              = activation, 
        use_cross_attention     = use_cross_attention, 
        encoding_temperature    = positional_encoding_temperature, 
        encoding_alpha          = positional_encoding_alpha,
        diffusion_num_timesteps = num_diffusion_timesteps, 
    )










    


    




