New-Item -ItemType Directory -Force -Path "log" | Out-Null

# Exp 0: Transformer with MLP pose encoder
Start-Process python3 -ArgumentList "Eval.py --dataset ntu --num_joints 17 --model transformer --lr_scheduler step --model_type deterministic --model_version v0_mlp_enc --model_input_type coordinates --mode main --num_training_frame 40 --num_testing_frame 20 --transformer_type vanilla --transformer_residual_connection True --transformer_num_heads 4 --transformer_num_layers 4 --transformer_model_dim 512 --proj_model_type mlp --pck_alpha 30 --learning_rate 1e-4 --step_per_epoch 200 --loss_type penalize --batch_size 1 --num_testing_samples 10000" `
  -RedirectStandardOutput "log\mlp_enc.txt" -RedirectStandardError "log\mlp_enc_err.txt"
