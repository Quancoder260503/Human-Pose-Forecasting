# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Exp 0: Transformer with MLP pose encoder
Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model transformer --lr_scheduler step --model_type deterministic --model_version v0_mlp_enc --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --transformer_type vanilla --transformer_residual_connection True --transformer_num_heads 4 --transformer_num_layers 4 --transformer_model_dim 512 --proj_model_type mlp --pck_alpha 30 --learning_rate 1e-4 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000" `
  -RedirectStandardOutput "log\mlp_enc.txt" -RedirectStandardError "log\mlp_enc_err.txt"



# # Exp 1: Transformer with Conv1D pose encoder
# Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model transformer --lr_scheduler step --model_type deterministic --model_version v0_conv1d_enc --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --transformer_type vanilla --transformer_residual_connection True --transformer_num_heads 4 --transformer_num_layers 4 --transformer_model_dim 512 --proj_model_type conv1d --pck_alpha 30 --learning_rate 1e-4 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000" `
#   -RedirectStandardOutput "logs\conv1d_enc.txt" -RedirectStandardError "logs\conv1d_enc_err.txt"

# # Exp 2: Transformer with Joint Attention pose encoder
# Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model transformer --lr_scheduler step --model_type deterministic --model_version v0_joint_attn_enc --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --transformer_type vanilla --transformer_residual_connection True --transformer_num_heads 4 --transformer_num_layers 4 --transformer_model_dim 512 --proj_model_type attn --joint_attn_embed_dim 256 --joint_attn_num_layers 2 --joint_attn_num_heads 4 --joint_attn_fc_hidden_dim 1024 --pck_alpha 30 --learning_rate 1e-4 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000" `
#   -RedirectStandardOutput "logs\joint_attn_enc.txt" -RedirectStandardError "logs\joint_attn_enc_err.txt"

# # Exp 3: Baseline MLP
# Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model mlp --lr_scheduler step --model_type deterministic --model_version v0 --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --pck_alpha 30 --learning_rate 1e-3 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000" `
#   -RedirectStandardOutput "logs\mlp_baseline.txt" -RedirectStandardError "logs\mlp_baseline_err.txt"

# # Exp 4: Baseline LSTM
# Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model lstm --lr_scheduler step --model_type deterministic --model_version v0 --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --pck_alpha 30 --learning_rate 1e-3 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000 --lstm_residual_connection True" `
#   -RedirectStandardOutput "logs\lstm_baseline.txt" -RedirectStandardError "logs\lstm_baseline_err.txt"

# Write-Host "All 5 experiments launched. Monitor logs in the logs\ directory."