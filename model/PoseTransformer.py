import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

'''
For now, just experiment mainly on joints 
:TODO : for velocity based approach, replace the LSTM in the decoupled pose paper by this Transformer Architecture, could have make 
it learn the latent space mean and variance (an effective way to capture complex distribution of human motion ?). 
:TODO : visualize attention 
:TODO : researching on how to add information to the joint 
:TODO : reconstruct pose from PoseTransformer
'''

class PositionalEncoding1D(object):
    """
    Positional Encoding for 1D sequences 
    """

    def __init__(self, position_dim = 512, temperature = 10000, alpha = 1):
        self.position_dim = position_dim 
        self.temperature  = temperature 
        self.alpha        = alpha 
    
    def get_angles(self, position, index): 
        angle_rates = 1.0 / np.power(self.temperature, (2 * (index // 2)) / np.float32(self.position_dim))
        return self.alpha * position * angle_rates
    
    def __call__(self, sequence_length):
        angle_radians = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis], 
            np.arange(self.position_dim)[np.newaxis, :]
        ) # (sequence_length, position_dim)

        angle_radians[:, 0::2] = np.sin(angle_radians[:, 0::2])
        angle_radians[:, 1::2] = np.cos(angle_radians[:, 1::2])
        
        position_encoding = angle_radians[np.newaxis, ...] # (1, sequence_length, pos_dim) 
        position_encoding = position_encoding.astype(np.float32)

        return torch.from_numpy(position_encoding)
    
    @staticmethod
    def visualize_1d_encoding(num_visualized):
        position_encoder = PositionalEncoding1D()
        pos_encoding = position_encoder(num_visualized).numpy()

        plt.pcolormesh(pos_encoding[0], cmap = 'RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 512)) # Position dim 
        plt.ylabel('Positiom in Sequence')
        plt.colorbar() 
        plt.show()

class MultiHeadAttention(nn.Module): 
    '''
    Muli-Head Attention Layer 
    '''
    def __init__(self, num_heads, dim_model, dim_queries, dim_values, dropout, in_decoder = False, device = 'cuda'): 
        """
        :param num_heads   : number of attention heads 
        :param dim_model   : model dimension
        :param dim_queries : query dimension 
        :param dim_values  : value dimension 
        :param dropout     : dropout probability 
        :param in_decoder  : in_decoder phase 
        """

        super(MultiHeadAttention, self).__init__() 
        self.num_heads      = num_heads 
        self.dim_model      = dim_model 
        self.dim_queries    = dim_queries 
        self.dim_values     = dim_values 
        self.dropout        = nn.Dropout(dropout)
        self.in_decoder     = in_decoder
        self.proj_q         = nn.Linear(dim_model, num_heads * dim_queries) 
        self.proj_k         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_v         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_o         = nn.Linear(num_heads * dim_values, dim_model)
        self.layernorm      = nn.LayerNorm(dim_model)
        self.softmax        = nn.Softmax(dim = -1) 
        self.device         = device 

    def forward(self, q_poses, kv_poses): 
        """
        :param q_poses : input query sequences, a tensor of shape (B, query_len, dim_model) 
        :param kv_poses : input_kv   sequences, a tensor of shape (B, keyval_len, dim_model)  
        :return 
            a tensor of shape (B, query_len, dim_model)
        """
        batch_size = q_poses.shape[0] 
        query_len  = q_poses.shape[1]
        keyval_len = kv_poses.shape[1]
        
        self_attention = query_len == keyval_len
        
        curr_q_poses = q_poses.clone() # (B, J, dim_model) 
        q_poses = self.layernorm(q_poses)   
        
        # if not self attention, then no need to normalize since the encoder has already done it 
        if self_attention:
            kv_poses = self.layernorm(kv_poses) 

        query = self.proj_q(q_poses)  # (B, query_len, query_dim * num_heads) 
        key   = self.proj_k(kv_poses) # (B, query_len, query_dim * num_heads)
        val   = self.proj_v(kv_poses) # (B, query_len, value_dim * num_heads) 

        query = query.contiguous().view(batch_size, query_len, self.num_heads, -1) # (B, query_len, num_heads, query_dim) 
        key   = key.contiguous().view(batch_size, keyval_len,  self.num_heads, -1) # (B, keyval_len, num_heads, query_dim)
        val   = val.contiguous().view(batch_size, keyval_len,  self.num_heads, -1) # (B, keyval_len, num_heads, value_dim)


        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, query_len,  -1) # (B * n_heads, query_len,  query_dim) 
        key   = key.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   keyval_len, -1) # (B * n_heads, keyval_len, query_dim) 
        val   = val.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   keyval_len, -1) # (B * n_heads, keyval_len, value_dim)

        attention_weights = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.dim_queries) # (B * n_heads,  query_len, keyval_len) 
        
        if self.in_decoder and self_attention : 
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(device = self.device) # Lower triangle matrix (actually use this to filter out the future) 
            attention_weights = attention_weights.masked_fill(~not_future_mask, -1e9) 

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights) 

        attn_values = torch.bmm(attention_weights, val)  # (B * n_heads, query_len, value_dim)
        attn_values = attn_values.contiguous().view(batch_size, self.num_heads, query_len, self.dim_values)
        attn_values = attn_values.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, -1) # (B, J, n_heads * value_dim)
        out_poses = self.proj_o(attn_values) # (B, query_len, dim_model) 
        out_poses = self.dropout(out_poses) + curr_q_poses # (B, query_len, dim_model) 
        return out_poses

class PointWiseFFN(nn.Module): 
    """
    Position-wise feed forward network layer 
    """
    def __init__(self, d_model, d_inner, dropout): 
        """
        :param d_model : size of the vectors throughout the transformer model
        :param d_inner : hidden size 
        :param dropout : dropout prob ? 
        """
        super(PointWiseFFN, self).__init__()
        self.d_model = d_model 
        self.d_inner = d_inner 
        self.layer_norm = nn.LayerNorm(d_model) 
        self.fc_1       = nn.Linear(d_model, d_inner) 
        self.relu       = nn.ReLU(inplace = True) 
        self.fc_2       = nn.Linear(d_inner, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, poses): 
        ''' 
        :param poses : a tensor of shape (batch_size, query_len, dim_model)
        '''
        prev_poses = poses.clone() 
        poses      = self.layer_norm(poses) 
        
        # First layer 
        poses = self.fc_1(poses) # (batch_size, query_len, d_inner) 
        poses = self.relu(poses)
        poses = self.dropout(poses) 

        # Second Layer 
        poses = self.fc_2(poses) # (batch_size, query_len, dim_model) 
        poses = self.dropout(poses) + prev_poses 
        return poses 

class Encoder(nn.Module):
    """
    Encoder Architecture, encoded the observed sequences information,
    First step, is to try to plug it as a GRU hidden state for decoder 
    Second step, is to come up with Decoder part of Transformer 
    Third  step, extend this to latent space, get the noise sampled from the Encoder and use it to guide Diffusion Model
    """
    def __init__(self, pose_dim, positional_encoding, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout, device = 'cuda'): 
        """
        :param pose_dim            : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param num_heads           : number of attention heads 
        :param model_dim           : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim           : query dimension 
        :param value_dim           : value dimension 
        :param fc_hidden_dim       : fully connected layer hidden dim 
        :param num_layers          : number of (MultiheadAttn + FC) layers 
        :param dropout             : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(Encoder, self).__init__()
        self.pose_dim            = pose_dim 
        self.positional_encoding = positional_encoding
        self.num_heads           = num_heads 
        self.model_dim           = model_dim
        self.query_dim           = query_dim
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim 
        self.num_layers          = num_layers 
        self.dropout             = dropout
        self.apply_dropout       = nn.Dropout(dropout)
        self.embedding_fc        = nn.Linear(pose_dim, model_dim)
        self.layernorm           = nn.LayerNorm(self.model_dim) 
        self.device              = device 
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])

    
    def make_encoder_layer(self): 
        """
        Create a MultiheadAttn + FFC layer 
        """
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
                dim_model   = self.model_dim,   
                num_heads   = self.num_heads, 
                dim_queries = self.query_dim, 
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = False, 
                device      = self.device, 
            ), 
            PointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout
            )
        ])
        return encoder_layer
    
    def forward(self, observed_poses): 
        """
        For now, only support input tensor of shape (B, train_len, pose_dim = J * 3) (x, y, visibility)
        :param observed_poses : tensor of shape (B, train_len, model_dim)
        : return:
            tensor of shape (B, train_len, model_dim) 
        """
        observed_poses = self.embedding_fc(observed_poses) # (B, train_len, model_dim) 
        observed_poses = self.layernorm(observed_poses)
        # Scale to reduce the effect of positional encoding 
        observed_poses = observed_poses * math.sqrt(self.model_dim) + self.positional_encoding 
        observed_poses = self.apply_dropout(observed_poses)
        for encoder_layer in self.encoder_layers: 
            observed_poses = encoder_layer[0](observed_poses, observed_poses)
            observed_poses = encoder_layer[1](observed_poses)
        observed_poses = self.layernorm(observed_poses) # (B, train_len, dim_model)
        return observed_poses

class Decoder(nn.Module): 
    def __init__(self, pose_dim, positional_encoding, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout, device = 'cuda'): 
        """
        :param pose_dim      : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param num_heads     : number of attention heads 
        :param model_dim     : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim     : query dimension 
        :param value_dim     : value dimension 
        :param fc_hidden_dim : fully connected layer hidden dim 
        :param num_layers    : number of (MultiheadAttn + FC) layers 
        :param dropout       : dropout probability 
        :param positional_encoding : distinguish noise 
        """

        super(Decoder, self).__init__() 
        self.pose_dim            = pose_dim 
        self.positional_encoding = positional_encoding 
        self.num_heads           = num_heads 
        self.model_dim           = model_dim 
        self.query_dim           = query_dim 
        self.value_dim           = value_dim 
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers 
        self.device              = device 
        self.dropout             = dropout
        self.apply_dropout       = nn.Dropout(dropout)
        self.embedding_fc        = nn.Linear(pose_dim, model_dim)
        self.layernorm           = nn.LayerNorm(model_dim)
        self.decoder_layers      = nn.ModuleList([ 
            self.make_decoder_layer() for _ in range(num_layers)
        ])

    def make_decoder_layer(self):
        """
        Create a MultiheadAttn + FC layer for decoder layer  
        """
        decoder_layer = nn.ModuleList([
            # Cross_attention 
            MultiHeadAttention(
                num_heads   = self.num_heads, 
                dim_model   = self.model_dim, 
                dim_queries = self.query_dim, 
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = True, 
                device      = self.device, 
            ), 
            # Self_attention 
            MultiHeadAttention(
                num_heads   = self.num_heads, 
                dim_model   = self.model_dim, 
                dim_queries = self.query_dim,
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = True, 
                device      = self.device,  
            ), 
            # FC 
            PointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout, 
            )
        ])
        return decoder_layer 
    
    def forward(self, encoder_out, decoding_poses): 
        """
        For now, only support input tensor of shape (B, train_len, pose_dim = J * 3) (x, y, visibility)
        :param encoder_out    : what we get from the encoder, a tensor of shape ()
        :param decoding_poses : tensor of shape (B, train_len, pose_dim)
        : return:
            tensor of shape (B, train_len, model_dim) 
        """
        decoding_poses = self.embedding_fc(decoding_poses) # (B, decoding_len, model_dim) 
        decoding_poses = self.layernorm(decoding_poses)
        decoding_poses = decoding_poses * math.sqrt(self.model_dim) + self.positional_encoding # scale this up to reduce the effect of dim embedding 
        decoding_poses = self.apply_dropout(decoding_poses)
    
        
        for decoder_layer in self.decoder_layers: 
            # cross-attention 
            decoding_poses = decoder_layer[0](
                decoding_poses, 
                encoder_out, 
            )
            # self-attention 
            decoding_poses = decoder_layer[1]( 
                decoding_poses, 
                decoding_poses, 
            )
            decoding_poses = decoder_layer[2](
                decoding_poses, 
            )
            # (B, decoding_len, model_dim) 
        decoding_poses = self.layernorm(decoding_poses)
        return decoding_poses 

class PoseTransformer(nn.Module):
    def __init__(self, 
                 pose_dim                        = 39,  
                 encoding_length                 = 14,
                 decoding_length                 = 14, 
                 num_heads                       = 8,  
                 query_dim                       = 64, 
                 value_dim                       = 64, 
                 model_dim                       = 512, 
                 fc_hidden_dim                   = 2048,
                 num_layers                      = 6,
                 dropout                         = 0.2, 
                 positional_encoding_temperature = 10000, 
                 positional_encoding_alpha       = 1, 
                 device                          = 'cuda', 
    ): 
        """
        :param pose_dim        : input tensor, can represent either (x, y, visibility) or relative root (all - root) or velocity between frames (dx, dy) 
        :param encoding_length : the length of the encoding pose sequences 
        :param decoding_length : the length of the decoding pose sequences 
        :param num_heads       : number of attention heads 
        :param model_dim       : model dimension (the dimension that accompanied with the object through the entire network)
        :param query_dim       : query dimension 
        :param value_dim       : value dimension 
        :param fc_hidden_dim   : fully connected layer hidden dim 
        :param num_layers      : number of (MultiheadAttn + FC) layers 
        :param dropout         : dropout probability 
        :param positional_encoding : distinguish noise 
        """
        super(PoseTransformer, self).__init__()
        self.pose_dim            = pose_dim
        self.encoding_length     = encoding_length 
        self.decoding_length     = decoding_length
        self.num_heads           = num_heads
        self.query_dim           = query_dim 
        self.value_dim           = value_dim
        self.model_dim           = model_dim
        self.fc_hidden_dim       = fc_hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout 
        self.device              = device

        # Temperature and alpha for positional encoding 
        self.temperature = positional_encoding_temperature 
        self.alpha       = positional_encoding_alpha

        self.init_positional_encoding()

        self.encoder            = Encoder(
            pose_dim            = self.pose_dim, 
            positional_encoding = self.encoder_positional_encoding, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim,
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
        ) # Use Transformer encoder as the main decoder

        self.decoder            = Decoder(
            pose_dim            = self.pose_dim, 
            positional_encoding = self.decoder_positional_encoding, 
            num_heads           = self.num_heads, 
            model_dim           = self.model_dim, 
            query_dim           = self.query_dim, 
            value_dim           = self.value_dim, 
            fc_hidden_dim       = self.fc_hidden_dim, 
            num_layers          = self.num_layers, 
            dropout             = self.dropout, 
            device              = self.device, 
        )

        self.output_fc = nn.Linear(self.model_dim, self.pose_dim) # map to the output space 
        self.init_weights() 

    def init_positional_encoding(self):
        encoder_positional = PositionalEncoding1D(
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha, 
        )

        decoder_positional = PositionalEncoding1D( 
            position_dim = self.model_dim, 
            temperature  = self.temperature, 
            alpha        = self.alpha, 
        )

        self.encoder_positional_encoding = encoder_positional(sequence_length = self.encoding_length).contiguous().view(
            1, self.encoding_length, self.model_dim
        )

        self.decoder_positional_encoding = decoder_positional(sequence_length = self.decoding_length).contiguous().view(
            1, self.decoding_length, self.model_dim
        )

        # Do not train the distinguish noise 
        self.encoder_positional_encoding = nn.Parameter(self.encoder_positional_encoding, requires_grad = False) 
        self.decoder_positional_encoding = nn.Parameter(self.decoder_positional_encoding, requires_grad = False)

    def init_weights(self): 
        """
        Initialize the weights of the model 
        """
        for p in self.parameters():
            if p.dim() > 1 : 
                nn.init.xavier_uniform_(p, gain = 1.0) 

    def forward(self, encoding_poses, decoding_poses):
        """
        :param encoding_poses : tensor of shape (B, encoding_len, pose_dim)
        :param decoding_poses : tensor of shape (B, decoding_len, pose_dim)
        :return : tensor of shape (B, decoding_len, pose_dim) 
        """
        encoder_out     = self.encoder(encoding_poses)
        predicted_poses = self.decoder(encoder_out, decoding_poses)
        out             = self.output_fc(predicted_poses)
        return out

def pose_transformer_inference(model):
    """
    Reconstruct the poses with the given observed poses 
    """
    pass     

if __name__ == "__main__":
    dummy = torch.randn(1, 14, 39).to('cuda') 
    model = PoseTransformer().to('cuda')
    out = model(dummy, dummy)
    print(out.shape)