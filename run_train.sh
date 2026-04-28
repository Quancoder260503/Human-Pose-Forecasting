python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29505 Train.py \
    --dataset ntu \
    --num_joints 17 \
    --model lstm \
    --model_type deterministic  \
    --model_version v0 \
    --model_input_type root_relative \
    --mode main \
    --lr_scheduler step \
    --lr_step_size 4000 \
    --lr_gamma 0.5 \
    --num_training_frame 40 \
    --num_testing_frame  20 \
    --num_epochs 100 \
    --step_per_epoch 400 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay  1e-3 \
    --lr_eta_min 1e-6 \
    --print_step 50 \
    --teacher_forcing_rate 0.0 \
    --pck_alpha 30 \
    --loss_type penalize \
    --lstm_residual_connection True 


