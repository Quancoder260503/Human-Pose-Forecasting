import numpy as np
import os
import sys
import argparse
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.MultiDataset import DatasetH36M, DatasetH36M_Multi
#import data.NTURGDDataset as NTURGDDataset
from runner.RunnerUtils import *
from runner.DiffusionRunner import *
from runner.FlowMatchingRunner import *
from runner.RewardModelRunner import *
from runner.PPORunner import *


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
  # NOTE: only H36M is wired up right now regardless of --dataset.
  # Kept as a single lookup point so adding new datasets is a one-line change.
  dataset_registry = {
    'h36m': DatasetH36M,
  }
  if params['dataset'] not in dataset_registry:
    raise NotImplementedError(
      f"Dataset '{params['dataset']}' is not registered. "
      f"Available: {list(dataset_registry.keys())}"
    )
  dataset_class_ = dataset_registry[params['dataset']]

  dataset_train = dataset_class_(
    mode = 'train',
    num_observed_frames = params['num_observed_frames'],
    num_predict_frames = params['num_predict_frames'],
    actions = 'all',
    dim_per_joints = 3,  # (x, y, z)
  )
  dataset_test = dataset_class_(
    mode = 'test',
    num_observed_frames = params['num_observed_frames'],
    num_predict_frames = params['num_predict_frames'],
    actions = 'all',
    dim_per_joints = 3,  # (x, y, z)
  )

  dataset_multi_test = DatasetH36M_Multi(
    mode = 'test',
    num_observed_frames = params['num_observed_frames'],
    num_predict_frames = params['num_predict_frames'],
    dim_per_joints = 3,  # x, y, z
    multimodal_path = params['multimodal_path'],
    data_candidate_path = params['data_candi_path'],
  )
  return {
    'train': dataset_train,
    'test': dataset_test,
  }, dataset_multi_test


def seed_set(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--type', type = str, default = 'diffusion',
    choices = ['diffusion', 'flow_matching', 'reward', 'ppo'],
    help = "Which pipeline to run: generative model pretraining (diffusion / "
      "flow_matching), reward model training (reward), or RL "
      "fine-tuning of a pretrained diffusion policy (ppo)."
  )
  parser.add_argument('--mode', default = 'train', type = str, choices = ['train', 'eval', 'demo_vis'])
  parser.add_argument('--sampling_mode', default = 'pred', choices = ['pred', 'switch', 'control', 'zero_shot'], type = str)  # pred / switch / control / zero_shot
  parser.add_argument('--algo', type = str, default = None,
    help = "Label used for logging/output filenames. Defaults to --type if unset."
  )

  parser.add_argument('--model_prefix', type = str, default = '', required = True)
  parser.add_argument('--model_architecture', type = str, default = 'transformer',
    choices = ['transformer', 'gcn_transformer', 'gcn'])
  parser.add_argument('--batch_size', type = int, default = 64)
  parser.add_argument('--learning_rate', type = float, default = 3e-4)
  parser.add_argument('--train_steps_per_epoch', type = int, default = -1)
  parser.add_argument('--eval_steps_per_epoch', type = int, default = -1)
  parser.add_argument('--max_epochs', type = int, default = 1000)
  parser.add_argument('--action', nargs = '*', type = str, default = None)

  parser.add_argument('--init_fn', type = str, default = 'xavier_init_')

  # pose transformer params
  parser.add_argument('--num_observed_frames', type = int, default = 25)
  parser.add_argument('--num_predict_frames', type = int, default = 100)

  parser.add_argument('--lr_step_size', type = int, default = 400)
  parser.add_argument('--learning_rate_fn', type = str, default = 'milestones',
    choices = ['step', 'exponential', 'linear', 'decay', 'milestones'])
  parser.add_argument('--step_gamma', default = 0.1, type = float)
  parser.add_argument('--exp_gamma', default = 0.95, type = float)
  parser.add_argument('--milestones', type = list, default = [150, 200, 250, 400, 650, 850])
  parser.add_argument('--milestone_gamma', type = float, default = 0.8)
  parser.add_argument('--end_learning_rate', type = float, default = 1e-6)

  parser.add_argument('--pose_embedding_type', type = str, default = 'simple')

  parser.add_argument('--dataset', type = str, default = 'h36m')
  parser.add_argument('--finetuning_ckpt', type = str, default = None)
  parser.add_argument('--pos_enc_alpha', type = float, default = 1)
  parser.add_argument('--pos_enc_beta', type = float, default = 10000)
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

  # --- Diffusion process ---
  parser.add_argument('--diffusion_num_timesteps', type = int, default = 1000)
  parser.add_argument('--ddim_num_timesteps', type = int, default = 100)
  parser.add_argument('--timesteps_sampling', type = str, default = 'uniform_time_sampling')
  parser.add_argument('--noise_scheduler', type = str, default = 'cosine')
  parser.add_argument('--rescale_timesteps', type = str2bool, default = False)
  parser.add_argument('--drop_cond_method', type = str, default = 'sample')

  # --- Backbone model ---
  parser.add_argument('--num_dct_channels', type = int, default = 20)
  parser.add_argument('--num_joints', type = int, default = 16)
  parser.add_argument('--joint_dim', type = int, default = 3)  # (x, y, z)
  parser.add_argument('--model_dim', type = int, default = 512)
  parser.add_argument('--ffn_dim', type = int, default = 1024)
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
  parser.add_argument('--pre_gcn', type = str2bool, default = True)
  parser.add_argument('--num_gcn_stages', type = int, default = 1)
  parser.add_argument('--gcn_dim', type = int, default = 128)

  # Flow matching ODE (type == flow_matching) 
  parser.add_argument('--path_scheduler', type = str, default = 'cosine')
  parser.add_argument('--ode_solver', type = str, default = 'rk4')
  parser.add_argument('--num_sampling_steps', type = int, default = 10)
  parser.add_argument('--path_refinement_steps', type = int, default = 1)
  parser.add_argument('--interpolation', type = str, default = 'linear')

  # Multimodal / eval 
  parser.add_argument('--multimodal_threshold', type = float, default = 0.5)
  parser.add_argument('--padding', type = str, default = 'last_frame')
  parser.add_argument('--multimodal_path', type = str,
    default = 'dataset/multimodal_set/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz')
  parser.add_argument('--data_candi_path', type = str,
    default = 'dataset/multimodal_set/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz')

  # EMA 
  parser.add_argument('--ema_rate', type = float, default = 0.999)
  parser.add_argument('--step_start_ema', type = int, default = 2000)

  # Visualization 
  parser.add_argument('--vis_switch_num', type = int, default = 10)
  parser.add_argument('--vis_col', type = int, default = 5)
  parser.add_argument('--vis_row', type = int, default = 3)

  # ---------------------------------------------------------------------
  # Reward model (type == reward, also consumed by ppo for scoring)
  # ---------------------------------------------------------------------
  parser.add_argument('--code_size', type = int, default = 128)
  parser.add_argument('--reward_model_architecture', type = str, default = 'transformer_encoder',
    choices = ['cross_transformer', 'gcn_encoder', 'transformer_encoder'])
  parser.add_argument('--info_nce_temperature', type = float, default = 0.07)
  parser.add_argument('--reward_use_sampler', type = str2bool, default = False)
  parser.add_argument('--num_samples', type = int, default = 5)
  parser.add_argument('--num_taken_samples', type = int, default = 10) 
  parser.add_argument('--num_select_per_samples', type = int, default = 3)
  parser.add_argument('--reward_model_path', type = str, default = None,
    help = "Path to a trained reward-model checkpoint. Required for --type ppo, "
           "and for --type reward --mode eval.")

  # ---------------------------------------------------------------------
  # PPO fine-tuning (type == ppo)
  # ---------------------------------------------------------------------
  ppo = parser.add_argument_group('ppo')
  ppo.add_argument('--diffusion_model_path', type = str, default = None,
    help = "Pretrained diffusion checkpoint to fine-tune with PPO. Required for --type ppo.")
  ppo.add_argument('--diffusion_load_ema_flag', type = int, default = 1, choices = [0, 1],
    help = "1 = load the EMA weights from --diffusion_model_path, 0 = load raw weights.")

  ppo.add_argument('--policy_optimizer', type = str, default = 'AdamW',
    choices = ['AdamW', 'Adam', 'RMSProp', 'SGD'])
  ppo.add_argument('--policy_learning_rate', type = float, default = 1e-5)
  ppo.add_argument('--value_optimizer', type = str, default = 'AdamW',
    choices = ['AdamW', 'Adam', 'RMSProp', 'SGD'])
  ppo.add_argument('--value_learning_rate', type = float, default = 1e-3)
  ppo.add_argument('--value_model_architecture', type = str, default = 'mlp',
    choices = ['mlp', 'transformer'])
  ppo.add_argument('--value_flag', type = int, default = 1, choices = [0, 1],
    help = "1 = subtract the learned value baseline from the reward to form the advantage, "
           "0 = use the raw reward as the advantage.")

  ppo.add_argument('--g_step', type = int, default = 1,
    help = "Number of rollout-collection passes per epoch before optimizing.")
  ppo.add_argument('--value_step', type = int, default = 5,
    help = "Number of value-function gradient steps per epoch.")
  ppo.add_argument('--value_batch_size', type = int, default = 512)
  ppo.add_argument('--value_network_layer_dims', type = list, default = [2048, 512, 128])
  ppo.add_argument('--value_activation', type = str, default = 'GeLU')
  ppo.add_argument('--policy_step', type = int, default = 5,
    help = "Number of PPO policy update steps per epoch.")
  ppo.add_argument('--policy_batch_size', type = int, default = 512)
  ppo.add_argument('--gradient_accumulation_steps', type = int, default = 1)
  ppo.add_argument('--buffer_size', type = int, default = 20000,
    help = "Max number of (state, next_state) transitions kept in the rollout buffer.")

  ppo.add_argument('--ratio_clip', type = float, default = 1e-4,
    help = "PPO importance-ratio clip range")
  ppo.add_argument('--reward_weight', type = float, default = 10.)
  ppo.add_argument('--kl_weight', type = float, default = 0.01)
  ppo.add_argument('--kl_warmup', type = int, default = -1,
    help = "Global step after which the KL penalty against the reference model is added to the loss.")
  ppo.add_argument('--clip_norm', type = float, default = 1.,
    help = "Gradient-norm clipping value for the policy optimizer during PPO.")
  ppo.add_argument('--max_train_steps', type = int, default = 100000,
    help = "Total policy-optimizer steps before training stops.")

  ppo.add_argument('--visual_weight', type = float, default = 2.0,
    help = "Weight on the InfoNCE-based reward term in compute_rewards.")
  ppo.add_argument('--dct_weight', type = float, default = 1.0,
    help = "Weight on the DCT-space reconstruction reward term.")
  ppo.add_argument('--three_d_weight', dest = '3d_weight', type = float, default = 1.0,
    help = "Weight on the 3D-pose-space reconstruction reward term.")

  ppo.add_argument('--eta', type = float, default = 1.0,
    help = "DDIM stochasticity parameter used during rollout collection.")
  ppo.add_argument('--cfg_scale', type = float, default = 0.7)
  ppo.add_argument('--use_classifier_free_guidance', type = str2bool, default = False)
  ppo.add_argument('--deterministic', type = str2bool, default = False)
  ppo.add_argument('--enable_complete', type = str2bool, default = False)

  args = parser.parse_args()

  params = vars(args)

  if params['algo'] is None:
    params['algo'] = params['type']

  # PPORunner.finetune_init needs to know whether this is a training run
  # or an inference-only load.
  params['train'] = (args.mode == 'train')

  params['model_prefix'] = os.path.join('exp', params['model_prefix'])
  create_dir_tree(params['model_prefix'], params)
  params['pose_dim'] = params['joint_dim'] * params['num_joints']

  params['idx_pad'], params['zero_index'] = generate_pad(
    params['padding'], params['num_observed_frames'], params['num_predict_frames'])

  # required-argument sanity checks that argparse itself can't express
  # because they're conditional on --type
  if params['type'] == 'ppo':
    if params['diffusion_model_path'] is None:
      parser.error("--diffusion_model_path is required when --type ppo")
    if params['reward_model_path'] is None:
      parser.error("--reward_model_path is required when --type ppo")
  if params['type'] == 'reward' and args.mode == 'eval' and params['reward_model_path'] is None:
    parser.error("--reward_model_path is required for --type reward --mode eval")

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

  # multimodal_dict = get_multimodal_dataset(
  #   logger = logger,
  #   dataset_multi_test = dataset_multi_test,
  #   params = params,
  # )

  if params['type'] == 'reward':
    runner = RewardRunner(
      params = params,
      dataset = dataset,
      multimodal_dict = multimodal_dict,
      tblogger = tb_logger,
      logger = logger,
    )
    if args.mode == 'train':
      runner.train()
    elif args.mode == 'eval':
    #  runner.compute_stats(multimodal_dict = multimodal_dict)
      runner.diffusion_samples_evaluate()
    else:
      raise NotImplementedError(f"--mode {args.mode} is not supported for --type reward")

  elif params['type'] == 'ppo':
    runner = PPORunner(
      params = params,
      dataset = dataset,
      multimodal_dict = None, #multimodal_dict,
      tblogger = tb_logger,
      logger = logger,
    )
    if args.mode == 'train':
      runner.train()
    elif args.mode == 'eval':
      runner.compute_stats(
        sampling_model = runner.diffusion_model,
        multimodal_dict = multimodal_dict,
      )
    elif args.mode == 'demo_vis':
      runner.demo_visualize(
        sampling_model = runner.diffusion_model,
        sampling_mode = params['sampling_mode'],
        dataset = dataset,
      )
    else:
      raise NotImplementedError(f"--mode {args.mode} is not supported for --type ppo")

  else:  # 'diffusion' or 'flow_matching'
    runner_cls = DiffusionRunner if params['type'] == 'diffusion' else FlowMatchingRunner
    runner = runner_cls(
      params = params,
      dataset = dataset,
      multimodal_dict = None,
      tblogger = tb_logger,
      logger = logger,
    )
    if args.mode == 'train':
      runner.train()
    elif args.mode == 'eval':
      runner.compute_stats(sampling_model = runner._ema_model, multimodal_dict = None)
    elif args.mode == 'demo_vis':
      runner.demo_visualize(
        sampling_model = runner._ema_model,
        sampling_mode = params['sampling_mode'],
        dataset = dataset,
      )
    else:
      raise NotImplementedError(f"--mode {args.mode} is not supported for --type {params['type']}")