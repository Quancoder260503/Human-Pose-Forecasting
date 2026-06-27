import numpy as np
import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import data.H36M as H36M
#import data.NTURGDDataset as NTURGDDataset
from model.PoseEncoderDecoder import select_pose_encoder_decoder_fn
from runner.EncoderDecoderRunner import * 

def create_dir_tree(base_dir):
  dir_tree = ['models', 'tf_logs', 'config', 'std_log']
  for dir_ in dir_tree:
    os.makedirs(os.path.join(base_dir, dir_), exist_ok = True)


def dataset_factory(params):
  if params['dataset'] == 'h36m':
    return H36M.dataset_factory(params)
#   elif params['dataset'] == 'ntu_rgb+d':
#     return NTURGDDataset.dataset_factory(params)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type = str, default = '')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--data_path', type = str)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--max_epochs', type = int, default = 500)
    parser.add_argument('--steps_per_epoch', type = int, default = 200)
    parser.add_argument('--action', nargs = '*', type = str, default = None)

    parser.add_argument('--model_type', type = str, default = 'transformer', required = True)

    parser.add_argument('--init_fn', type = str, default = 'xavier_init')
    # pose transformers related parameter
    parser.add_argument('--model_dim', type = int, default = 256)
    parser.add_argument('--num_encoder_layers', type = int, default = 4)
    parser.add_argument('--num_decoder_layers', type = int, default = 4)
    parser.add_argument('--num_heads', type = int, default = 4)
    parser.add_argument('--dim_ffn',   type = int,   default = 2048)
    parser.add_argument('--dropout',   type = float, default = 0.3)

    parser.add_argument('--source_seq_len', type = int, default = 50)                  
    parser.add_argument('--target_seq_len', type = int, default = 25)

    parser.add_argument('--max_gradient_norm', type = float, default = 0.1)

    parser.add_argument('--lr_step_size', type = int, default = 400)
    parser.add_argument('--learning_rate_fn', type = str, default = 'step')
    parser.add_argument('--step_gamma', default = 0.1, type = float)
    parser.add_argument('--exp_gamma', default = 0.95, type = float)


    parser.add_argument('--warmup_epochs', type = int, default = 100)
    parser.add_argument('--pose_format', type = str, default = 'rotmat', required = True)
    parser.add_argument('--joint_dim',  type = int, default = 9,  required = True)
    parser.add_argument('--num_joints', type = int, default = 21, required = True)

    parser.add_argument('--loss_fn', type = str, default = 'l1')
    
    parser.add_argument('--pose_embedding_type', type = str, default = 'gcn_enc')

    parser.add_argument('--dataset', type = str, default = 'h36m')
    parser.add_argument('--skip_rate', type = int, default = 5)

    parser.add_argument('--eval_num_seeds',  type = int, default = 8)
    parser.add_argument('--finetuning_ckpt', type = str, default = None)
    parser.add_argument('--pos_enc_alpha',   type = float, default = 10)
    parser.add_argument('--pos_enc_beta',    type = float, default = 500)
    parser.add_argument('--device', type = str, default = 'cuda')

    parser.add_argument('--optimizer', default = 'AdamW', type = str)
    parser.add_argument('--betas', default = (0.9, 0.999), type = tuple)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)

    parser.add_argument('--snapshot_saving_steps', default = 50, type = int)
    parser.add_argument('--include_last_obs', action = 'store_true')
    parser.add_argument('--zero_out_decoder', action = 'store_true')
    parser.add_argument('--log_frequency', default = 50, type = int)

    parser.add_argument('--mod_train', default = 0.8, type = float)

    
    parser.add_argument('--num_workers', type = int, default = 0)
    args = parser.parse_args()
    
    params = vars(args)
    train_dataset_fn, eval_dataset_fn = dataset_factory(params)

    params['pose_dim'] = train_dataset_fn.dataset._pose_dim

    pose_encoder_fn, pose_decoder_fn = select_pose_encoder_decoder_fn(params)

    for k,v in params.items():
        print('[INFO] (POTRFn@main) {}: {}'.format(k, v))

    create_dir_tree(params['model_prefix'])

    config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
    with open(config_path, 'w') as file_:
        json.dump(params, file_, indent = 4)

    model_fn = EncoderDecoderRunner(
      params = params, 
      train_dataset_fn = train_dataset_fn, 
      eval_dataset_fn  = eval_dataset_fn, 
      pose_encoder_fn  = pose_encoder_fn, 
      pose_decoder_fn  = pose_decoder_fn, 
    )
    model_fn.train()