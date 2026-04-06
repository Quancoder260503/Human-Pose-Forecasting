import torch
import torch.nn as nn
import random 

class PosePredictionLSTM(nn.Module):
    def __init__(
        self,
        pose_dim      = 13,
        coordinates   = 3,           
        embedding_dim = 2048,
        hidden_dim    = 256,
        num_layers    = 2,
        dropout       = 0.2,
        residual_connection = False,
        device              = 'cuda',
    ):
        super(PosePredictionLSTM, self).__init__()

        self.pose_dim      = pose_dim * coordinates
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.device        = device

        self.residual_connection = residual_connection
        self.embedding_fc  = nn.Linear(self.pose_dim, self.embedding_dim)
        self.apply_dropout = nn.Dropout(dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size  = self.embedding_dim,
            hidden_size = self.hidden_dim,
            dropout     = lstm_dropout,
            num_layers  = self.num_layers,
            batch_first = True,
        )

        self.decoder = nn.LSTM(
            input_size  = self.embedding_dim,
            hidden_size = self.hidden_dim,
            dropout     = lstm_dropout,
            num_layers  = self.num_layers,
            batch_first = True,
        )

        self.output_fc = nn.Linear(self.hidden_dim, self.pose_dim)

    def forward(self, observed_poses, target_poses, prediction_len, teacher_forcing_rate = 0.5):
        """
        :param observed_poses      : (batch, observed_len, pose_dim)
        :param target_poses        : (batch, target_len, pose_dim) or None at inference
        :param prediction_len      : int — number of steps to predict
        :param teacher_forcing_rate: probability of using ground-truth input at each step
        :return                    : (batch, prediction_len, pose_dim)
        """
        batch, _, _ = observed_poses.shape

        lstm_state = (
            torch.zeros(self.num_layers, batch, self.hidden_dim, device=self.device, dtype=torch.float32),
            torch.zeros(self.num_layers, batch, self.hidden_dim, device=self.device, dtype=torch.float32),
        )


        embedded_poses = self.embedding_fc(observed_poses)     

        _, lstm_state = self.encoder(embedded_poses, lstm_state)

        last_observed = observed_poses[:, -1, :].unsqueeze(1)        # (B, 1, pose_dim)
        predicted_poses = []

        for t in range(prediction_len):
            if t > 0 and target_poses is not None and random.random() < teacher_forcing_rate:
                last_observed = target_poses[:, t - 1, :].unsqueeze(1)

            embedded_last = self.embedding_fc(last_observed)              # (B, 1, embed_dim)
            output, lstm_state = self.decoder(embedded_last, lstm_state)  # (B, 1, hidden_dim)
            curr_hidden = output.squeeze(1)                               # (B, hidden_dim)

            if self.residual_connection:
                curr_poses = last_observed.squeeze(1) + self.apply_dropout(self.output_fc(curr_hidden))
            else:
                curr_poses = self.output_fc(curr_hidden)             # (B, pose_dim)

            predicted_poses.append(curr_poses)
            last_observed = curr_poses.unsqueeze(1)                  # (B, 1, pose_dim)

        return torch.stack(predicted_poses, dim = 1)                   # (B, prediction_len, pose_dim)


def lstm_loss_function(pred_joints, gt_joints):
    criterion = nn.SmoothL1Loss(reduction = 'mean')
    return criterion(pred_joints, gt_joints)


def get_lstm_model(
    pose_dim      = 13,
    coordinates   = True,
    embedding_dim = 2048,
    hidden_dim    = 256,
    num_layers    = 2,
    dropout       = 0.2,
    residual_connection = False,  
    device              = 'cuda',
):
    return PosePredictionLSTM(
        pose_dim            = pose_dim,
        coordinates         = 3 if coordinates else 2,
        embedding_dim       = embedding_dim,
        hidden_dim          = hidden_dim,
        num_layers          = num_layers,
        dropout             = dropout,
        residual_connection = residual_connection,
        device              = device,
    )