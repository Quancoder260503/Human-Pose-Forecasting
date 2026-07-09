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
from data.MultiDataset import DatasetH36M,  DatasetH36M_Multi
#import data.NTURGDDataset as NTURGDDataset
from model.PoseEncoderDecoder import select_pose_encoder_decoder_fn
from runner.SamplingUtils import *
from runner.DiffusionRunner import *
from runner.FlowMatchingRunner import *

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def create_dir_tree(base_dir, params):
  dir_tree = ['result_dir', 'tb_dir', 'gif_dir', 'config_dir', 'log_dir', 'models']
  for dir_ in dir_tree:
    os.makedirs(os.path.join(base_dir, dir_), exist_ok = True)
    params[dir_] = os.path.join(base_dir, dir_)

def dataset_split(params):
   dataset_class_ = DatasetH36M # temporary 
   dataset_train = dataset_class_(
    mode = 'train', 
    num_observed_frames = params['num_observed_frames'], 
    num_predict_frames = params['num_predict_frames'], 
    actions = 'all', 
    dim_per_joints = 3, # (x, y, z)
   )
   dataset_test = dataset_class_(
    mode = 'test', 
    num_observed_frames = params['num_observed_frames'], 
    num_predict_frames = params['num_predict_frames'], 
    actions = 'all', 
    dim_per_joints = 3, # (x, y, z)
   )
   
   dataset_multi = DatasetH36M_Multi
   dataset_multi_test = dataset_multi(
    mode = 'test', 
    num_observed_frames = params['num_observed_frames'], 
    num_predict_frames = params['num_predict_frames'], 
    dim_per_joints = 3, # x, y, z 
    multimodal_path = params['multimodal_path'], 
    data_candidate_path = params['data_candi_path']
   )
   return { 
    'train' : dataset_train, 
    'test' : dataset_test, 
   }, dataset_multi_test 

def seed_set(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_prefix', type = str, default = '', required = True)
  parser.add_argument('--mode', default = 'train', type = str) 
  parser.add_argument('--sampling_mode', default = 'pred', type = str) # pred / switch/ control/ zero_shot')
  parser.add_argument('--algo', type = str, default = 'diffusion', required = True)
  parser.add_argument('--batch_size', type = int, default = 64)
  parser.add_argument('--learning_rate', type = float, default = 3e-4)
  parser.add_argument('--max_epochs', type = int, default = 1000)
  parser.add_argument('--action', nargs = '*', type = str, default = None)

  parser.add_argument('--init_fn', type = str, default = 'xavier_init_')
    # pose transformers related parameter

  parser.add_argument('--num_observed_frames', type = int, default = 25)                  
  parser.add_argument('--num_predict_frames',  type = int, default = 100)

  parser.add_argument('--max_gradient_norm', type = float, default = 1.)
  parser.add_argument('--lr_step_size', type = int, default = 400)
  parser.add_argument('--learning_rate_fn', type = str, default = 'milestones')
  parser.add_argument('--step_gamma', default = 0.1, type = float)
  parser.add_argument('--exp_gamma', default = 0.95, type = float)
  parser.add_argument('--milestones', type = list, default = [100, 150, 225, 275, 350, 450, 650, 850])
  parser.add_argument('--milestone_gamma', type = float, default = 0.9)
  parser.add_argument('--end_learning_rate', type = float, default = 1e-6)
  
  parser.add_argument('--pose_embedding_type', type = str, default = 'simple')

  parser.add_argument('--dataset', type = str, default = 'h36m')
  parser.add_argument('--finetuning_ckpt', type = str, default = None)
  parser.add_argument('--pos_enc_alpha',   type = float, default = 1)
  parser.add_argument('--pos_enc_beta',    type = float, default = 10000)
  parser.add_argument('--device', type = str, default = 'cuda')

  parser.add_argument('--optimizer', default = 'Adam', type = str)
  parser.add_argument('--betas', default = (0.9, 0.999), type = tuple)

  parser.add_argument('--snapshot_saving_steps', default = 10, type = int)
  parser.add_argument('--log_frequency', default = 50, type = int)

  parser.add_argument('--mod_train', default = 0.8, type = float)
  parser.add_argument('--mod_test', default = 1., type = float)

    
  parser.add_argument('--num_workers', type = int, default = 0)

  parser.add_argument('--num_train_data_samples', type = int, default = 50000)
  parser.add_argument('--num_val_data_samples', type = int, default = 5000)

  parser.add_argument('--save_gif_interval', type = int, default = 10)  
  parser.add_argument('--save_metrics_interval', type = int, default = 500)

  # Diffusion Processes
  parser.add_argument('--diffusion_num_timesteps', type = int, default = 1000)
  parser.add_argument('--ddim_num_timesteps', type = int, default = 100)
  parser.add_argument('--timesteps_sampling', type = str, default = 'uniform_time_sampling')
  parser.add_argument('--noise_scheduler', type = str, default = 'cosine')
  parser.add_argument('--rescale_timesteps', type = str2bool, default = False) 

  parser.add_argument('--num_dct_channels', type = int, default = 20)

  # Model 
  parser.add_argument('--num_joints', type = int, default = 16)
  parser.add_argument('--joint_dim', type = int, default = 3) # (x, y, z)
  parser.add_argument('--model_dim', type = int, default = 512)
  parser.add_argument('--ffn_dim', type = int, default = 2048)
  parser.add_argument('--num_layers', type = int, default = 8) 
  parser.add_argument('--num_heads', type = int, default = 8) 
  parser.add_argument('--use_gate', type = str2bool, default = False)
  parser.add_argument('--dropout', type = float, default = 0.2)
  parser.add_argument('--activation', type = str, default = 'GeLU')
  parser.add_argument('--use_cross_attention', type = str2bool, default = False)
  parser.add_argument('--embedding_max_period', type = int, default = 10000)
  parser.add_argument('--flatten_arc', type = str2bool, default = False) 
  parser.add_argument('--learnable_embedding', type = str2bool, default = True)
  parser.add_argument('--se_ratio', type = int, default = 4)
  parser.add_argument('--use_excitation', type = str2bool, default = False) 
  parser.add_argument('--use_max_pooling', type = str2bool, default = False) 
  parser.add_argument('--use_adaptive', type = str2bool, default = True)


  # Flow Matching ODE
  parser.add_argument('--path_scheduler', type = str, default = 'cosine')
  parser.add_argument('--ode_solver', type = str, default = 'rk4')
  parser.add_argument('--num_sampling_steps', type = int, default = 10)
  parser.add_argument('--path_refinement_steps', type = int, default = 1) 
  parser.add_argument('--interpolation', type = str, default = 'linear')


  # Multimodal 
  parser.add_argument('--multimodal_threshold', type = float, default = 0.5)

  parser.add_argument('--padding', type = str, default = 'last_frame')

  parser.add_argument('--multimodal_path', type = str, \
    default = 'dataset/multimodal_set/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz')
  parser.add_argument('--data_candi_path', type = str, \
    default = 'dataset/multimodal_set/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz')
  
  # EMA
  parser.add_argument('--ema_rate', type = float, default = 0.999)
  parser.add_argument('--step_start_ema', type = int, default = 2000)
  
  # visualization
  parser.add_argument('--vis_switch_num', type = int, default = 10)
  parser.add_argument('--vis_col', type = int, default = 5)
  parser.add_argument('--vis_row', type = int, default = 3)


  args = parser.parse_args()


    
  params = vars(args)

  pose_encoder_fn, pose_decoder_fn = select_pose_encoder_decoder_fn(params)

  create_dir_tree(params['model_prefix'], params)
  params['pose_dim'] = params['joint_dim'] * params['num_joints'] 

  params['idx_pad'], params['zero_index'] = generate_pad(params['padding'], params['num_observed_frames'], params['num_predict_frames'])

  config_path = os.path.join(params['model_prefix'], 'config_dir', 'config.json')        
  with open(config_path, 'w') as file_:
    json.dump(params, file_, indent = 4)
      
  dataset, dataset_multi_test = dataset_split(params)
  tb_logger = SummaryWriter(params['tb_dir'])
  logger = create_logger(os.path.join(params['log_dir'], 'log.txt'))
  
  if args.mode == 'train':
    seed_set(0)

  for k, v in params.items():
    logger.info('[INFO] (GenerativePose@main) {}: {}'.format(k, v))

  multimodal_dict = get_multimodal_dataset(
    logger = logger, 
    dataset_multi_test = dataset_multi_test, 
    params = params, 
  )
  runner = (DiffusionRunner if params['algo'] == 'diff' else FlowMatchingRunner)(
    params = params, 
    dataset = dataset, 
    multimodal_dict = multimodal_dict, 
    tblogger = tb_logger, 
    logger = logger, 
    pose_decoder_fn = pose_decoder_fn, 
    pose_encoder_fn = pose_encoder_fn, 
  )
  if args.mode == 'train':
    runner.train()
  elif args.mode == 'eval':
    runner.compute_stats(sampling_model = runner._ema_model, multimodal_dict = multimodal_dict) 
  elif args.mode == 'demo_vis':
    runner.demo_visualize(sampling_model = runner._ema_model, \
      sampling_mode = params['sampling_mode'], dataset = dataset)
  else : 
    raise not NotImplementedError