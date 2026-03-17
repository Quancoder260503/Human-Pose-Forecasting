import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

'''
So far just experiment mainly on joints 

An idea that can be implement directly : 
joint based self-attention for the encoder the RNN only captures temporal information, 
it does not directly learn the interaction between the joints. Try attention to captures 
relevant information. 

Find a way to encode joints or time embedding, or can combine both 

KEYS : first train the model on a small set of data to see if it works
'''
class MultiHeadAttention(nn.Module): 
    '''
    Muli-Head Attention Layer 
    '''
    def __init__(self, num_heads, dim_model, dim_queries, dim_values, dropout, self_attention, in_decoder = False): 
        """
        :param num_heads   : number of attention heads 
        :param dim_model   : model dimension
        :param dim_queries : query dimension 
        :param dim_values  : value dimension 
        :param dropout     : dropout probability 
        :param in_decoder  : in decoder phase 
        """

        super(MultiHeadAttention, self).__init__() 
        self.num_heads      = num_heads 
        self.dim_model      = dim_model 
        self.dim_queries    = dim_queries 
        self.dim_values     = dim_values 
        self.dropout        = nn.Dropout(dropout)
        self.in_decoder     = in_decoder
        self.self_attention = self_attention
        self.proj_q         = nn.Linear(dim_model, num_heads * dim_queries) 
        self.proj_k         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_v         = nn.Linear(dim_model, num_heads * dim_values)
        self.proj_o         = nn.Linear(num_heads * dim_values, dim_model)
        self.layer_norm     = nn.Linear(dim_model)
        self.softmax        = nn.Softmax(dim = -1) 

    def forward(self, q_poses, kv_poses): 
        """
        :param q_poses : input query sequences, a tensor of shape (B, J, dim_model) 
        :param kv_poses : input_kv   sequences, a tensor of shape (B, J, dim_model) 
        """
        batch_size = q_poses.shape[0]
        num_joints = q_poses.shape[1]

        assert torch.equal(num_joints, kv_poses.shape[1]) 
        self_attention = not self.in_decoder

        curr_q_poses = q_poses.clone() # (B, J, dim_model) 

        q_poses = self.layer_norm(q_poses)  
        
        if self.self_attention:
            kv_poses = self.layer_norm(kv_poses) 

        query = self.proj_q(q_poses)  # (B, J, query_dim * num_heads) 
        key   = self.proj_k(kv_poses) # (B, J, query_dim * num_heads)
        val   = self.proj_v(kv_poses) # (B, J, value_dim * num_heads) 

        query = query.contiguous().view(batch_size, num_joints, self.num_heads, -1) # (B, J, num_heads, query_dim) 
        key   = key.contiguous().view(batch_size, num_joints, self.num_heads,   -1) # (B, J, num_heads, query_dim)
        val   = val.contiguous().view(batch_size, num_joints, self.num_heads,   -1) # (B, J, num_heads, value_dim)

        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, num_joints, -1) # (B * n_heads, J, query_dim) 
        key   = key.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   num_joints, -1) # (B * n_heads, J, query_dim) 
        val   = val.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads,   num_joints, -1) # (B * n_heads, J, value_dim)

        attention_weights = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.dim_queries) # (B * n_heads, J, J) 
        
        ## TODO : implement the masked out for Decoder later ? 

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights) 

        attn_values = torch.bmm(attention_weights, val)  # (B * n_heads, J, value_dim)
        attn_values = attn_values.contiguous().view(batch_size, self.num_heads, num_joints, self.dim_values)
        attn_values = attn_values.permute(0, 2, 1, 3).contiguous().view(batch_size, num_joints, -1) # (B, J, n_heads * value_dim)
        out_poses = self.proj_o(attn_values) # (B, J, dim_model) 
        out_poses = self.dropout(out_poses) + curr_q_poses # (B, J, dim_model) 
        return out_poses

class JointWiseFFN(nn.Module): 
    """
    Position-wise feed forward network layer 
    """
    def __init__(self, d_model, d_inner, dropout): 
        """
        :param d_model : size of the vectors throughout the transformer model
        :param d_inner : hidden size 
        :param dropout : dropout prob ? 
        """
        super(JointWiseFFN, self).__init__()
        self.d_model = d_model 
        self.d_inner = d_inner 
        self.layer_norm = nn.LayerNorm(d_model) 
        self.fc_1       = nn.Linear(d_model, d_inner) 
        self.relu       = nn.ReLU(inplace = True) 
        self.fc_2       = nn.Linear(d_inner, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, poses): 
        ''' 
        :param poses : a tensor of shape (batch_size, J, dim_model)
        '''
        prev_poses = poses.clone() 
        poses      = self.layer_norm(poses) 
        
        # First layer 
        poses = self.fc_1(poses) # (batch_size, J, d_inner) 
        poses = self.relu(poses)
        poses = self.apply(self.dropout)

        # Second Layer 
        poses = self.fc_2(poses) # (batch_size, J, dim_model) 
        poses = self.dropout(poses) + prev_poses 
        return poses 

class Encoder(nn.Module):
    """
    Encoder Architecture, encoded the observed sequences information,
    First step, is to try to plug it as a GRU hidden state for decoder 
    Second step, is to come up with Decoder part of Transformer 
    Third  step, extend this to latent space, get the noise sampled from the Encoder and use it to guide Diffusion Model
    """
    def __init__(self, num_joints, pose_length, num_heads, model_dim, query_dim, value_dim, fc_hidden_dim, num_layers, dropout): 
        """
        :param num_joints    : number of annotated joints 
        :param pose_length   : observed pose sequences length 
        :param num_heads     : number of attention heads 
        :param query_dim     : query dimension 
        :param value_dim     : value dimension 
        :param fc_hidden_dim : fully connected layer hidden dim 
        :param num_layers    : number of (MultiheadAttn + FC) layers 
        :param dropout       : dropout probability 
        """
        super(Encoder, self).__init__()
        self.num_joints    = num_joints 
        self.pose_length   = pose_length 
        self.num_heads     = num_heads 
        self.model_dim     = model_dim
        self.query_dim     = query_dim
        self.value_dim     = value_dim 
        self.fc_hidden_dim = fc_hidden_dim 
        self.num_layers    = num_layers 
        self.dropout       = dropout
        self.layernorm     = nn.LayerNorm(self.model_dim) 
        self.embedding_fc  = nn.Linear(self.pose_length * 2, self.model_dim)
        # Embedding, a solution to adapt to archicture, is to project stuff to the embedded space
        # but for now, keep the old approach
        self.encoder_layers = nn.ModuleList([
            self.make_encoder_layer() for _ in range(self.num_layers)
        ])
    
    def make_encoder_layer(self): 
        """
        Create a MultiheadAttn + FFC layer 
        """
        encoder_layer = nn.ModuleList([
            MultiHeadAttention(
                dim_model   = self.model_dim, # pose_length * [dim(x, y) = 2] 
                num_heads   = self.num_heads, 
                dim_queries = self.query_dim, 
                dim_values  = self.value_dim, 
                dropout     = self.dropout, 
                in_decoder  = False, 
            ), 
            JointWiseFFN(
                d_model = self.model_dim, 
                d_inner = self.fc_hidden_dim, 
                dropout = self.dropout
            )
        ])
        return encoder_layer
    
    def forward(self, observed_poses): 
        """
        This is incomplete, can we plug it into the Decoder 

        :param observed_poses : tensor of shape (B, train_len, J, 2)
        : return : 
            can adjust this 
            tensor of shape (B, J, model_dim) 
        """
        batch = observed_poses.shape[0]
        observed_poses = observed_poses.permute(0, 2, 1, 3).contiguous().view(batch, self.num_joints, -1) # (B, J, train_len * 2)
        observed_poses = self.embedding_fc(observed_poses) # (B, J, dim_model)
        for encoder_layer in self.encoder_layers: 
            observed_poses = encoder_layer[0](observed_poses, observed_poses)
            observed_poses = encoder_layer[1](observed_poses)
        observed_poses = self.layernorm(observed_poses) # (B, J, dim_model)
        return observed_poses

class Decoder(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class JointAttentionLSTM(nn.Module):
    def __init__(self, 
                 num_joints     = 13, 
                 pose_length    = 15,
                 num_heads      = 8,  
                 query_dim      = 64, 
                 value_dim      = 64, 
                 model_dim      = 512, 
                 fc_hidden_dim  = 2048,
                 num_attention_layers = 6,
                 embedding_dim  = 1024,   
                 hidden_dim     = 256, 
                 num_layers     = 1, 
                 dropout        = 0.2, 
                 device         = 'cuda' 
    ): 
        super(JointAttentionLSTM, self).__init__()
        self.pose_dim       = num_joints
        self.pose_length    = pose_length 
        self.num_heads      = num_heads 
        self.query_dim      = query_dim 
        self.value_dim      = value_dim
        self.model_dim      = model_dim
        self.hidden_dim     = hidden_dim 
        self.num_layers     = num_layers
        self.dropout        = dropout 
        self.device         = device

        self.joint_attention_encoder = Encoder(
            num_joints    = num_joints, 
            pose_length   = pose_length, 
            num_heads     = num_heads, 
            model_dim     = model_dim, 
            query_dim     = query_dim,
            value_dim     = value_dim, 
            fc_hidden_dim = fc_hidden_dim, 
            num_layers    = num_attention_layers, 
            dropout       = dropout
        ) # Use Transformer encoder as the main decoder

        self.embed_proj    = nn.Linear(num_joints * model_dim, embedding_dim) 

        self.decoder = nn.LSTM(
            input_size  = embedding_dim, 
            hidden_size = hidden_dim, 
            dropout     = dropout, 
            num_layers  = num_layers, 
            batch_first = True 
        )

        self.output_fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, observed_poses, prediction_len = 14):
        """
        :param observed_poses : tensor of shape (Batch, train_len, J, 2) 
        :return : tensor of shape 
        """
        batch, _, _, _ = observed_poses.shape

        embedded_poses = self.joint_attention_encoder(observed_poses, observed_poses) # (B, J, d_model)
        embedded_poses = embedded_poses.contiguous().viet(batch, -1)                  # (B, J, d_model)
        embedded_poses = self.embed_proj(embedded_poses)                              # (B, embedding_dim)
        last_observed = observed_poses[:, -1, :].unsqueeze(1)                         # (Batch, 1, pose_dim)
        
        predicted_poses = [] 

        lstm_state = (
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32), 
            torch.zeros(self.num_layers, batch, self.hidden_dim, device = self.device, dtype = torch.float32)
        )

        for t in range(prediction_len): 
             embedded_last = self.embedding_fc(last_observed) # (batch, 1, embedding_dim)
             output, lstm_state = self.decoder(embedded_last, lstm_state) # output : (batch, 1, hidden_dim) 
             curr_poses = output.squeeze(1)  # (batch, hidden_dim)
             curr_poses = self.output_fc(curr_poses) # (batch, pose_dim)
             predicted_poses.append(curr_poses)
             last_observed = curr_poses.unsqueeze(1) # (batch, 1, pose_dim)
        
        predicted_poses = torch.stack(predicted_poses, dim = 1) # (batch, test_len, pose_dim) 

        return predicted_poses
    
    
# def LSTM_loss_function(pred_joints, gt_joints, time_step):
#     """
#     :param pred_joints : (B, T, 26)
#     :param gt_joints   : (B, T, 26)   
#     :param time_step   : prediction time step 
#     :return:
#     """
#     # Huber Loss
#     criterion  = nn.SmoothL1Loss(reduction = 'mean')
#     total_loss = 0
#     for t in range(time_step):
#         pred = pred_joints[:, t, :]
#         gt   = gt_joints[:,   t, :]
#         curr_loss = criterion(pred, gt)
#         total_loss = total_loss + curr_loss
#     return total_loss / time_step


# if __name__ == "__main__":
#     dummy = torch.randn(4, 10, 26).to('cuda') 
#     model = JointLSTM().to('cuda')
#     out = model(dummy, 15)
#     print(out.shape)