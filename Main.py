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
from data.MultiDataset import DatasetH36M, DatasetH36M_Multi, DatasetHumanEva, DatasetHumanEva_Multi, DatasetAMASS
from runner.RunnerUtils import *
from runner.DiffusionRunner import *

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
  dataset_registry = {
    'h36m': DatasetH36M,
    'humaneva' : DatasetHumanEva, 
    'amass' : DatasetAMASS, 
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
    use_action = params['use_action'], 
    dim_per_joints = 3,  # (x, y, z)
  )
  dataset_test = dataset_class_(
    mode = 'test',
    num_observed_frames = params['num_observed_frames'],
    num_predict_frames = params['num_predict_frames'],
    actions = 'all',
    use_action = params['use_action'], 
    dim_per_joints = 3,  # (x, y, z)
  )

  
  params['num_classes'] = len(dataset_train.idx_to_class)
  if params['dataset'] == 'amass': 
    dataset_multi_test = dataset_test
  else: 
    dataset_multi_test = (DatasetH36M_Multi if params['dataset'] == 'h36m' else DatasetHumanEva_Multi) (
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
    choices = ['diffusion', 'motion_refine_diffusion'],
  )
  parser.add_argument('--mode', default = 'train', type = str, choices = ['train', 'eval', 'eval_per_action', 
    'demo_vis', 'dct_vis', 'inter_vis', 'superimposed_vis', 'fid_tsne_vis']
  )
  parser.add_argument('--sampling_mode', default = 'pred', choices = ['pred', 'switch', 'control', 'zero_shot'], type = str)  # pred / switch / control / zero_shot
  parser.add_argument('--algo', type = str, default = None,
    help = "Label used for logging/output filenames. Defaults to --type if unset."
  )

  parser.add_argument('--model_prefix', type = str, default = '', required = True)
  parser.add_argument('--model_architecture', type = str, default = 'gcn_transformer',
    choices = ['transformer', 'gcn_transformer'])
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
  parser.add_argument('--milestones', type = list, default = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
  parser.add_argument('--milestone_gamma', type = float, default = 0.9)
  parser.add_argument('--end_learning_rate', type = float, default = 1e-6)

  parser.add_argument('--pose_embedding_type', type = str, default = 'simple')

  parser.add_argument('--dataset', type = str, default = 'h36m', choices = ['h36m', 'humaneva', 'amass'])
  parser.add_argument('--finetuning_ckpt', type = str, default = None)
  parser.add_argument('--pos_enc_alpha', type = float, default = 1)
  parser.add_argument('--pos_enc_beta', type = float, default = 10000)
  parser.add_argument('--device', type = str, default = 'cuda')

  parser.add_argument('--optimizer', default = 'AdamW', type = str)
  parser.add_argument('--betas', default = (0.9, 0.999), type = tuple)

  parser.add_argument('--snapshot_saving_steps', default = 10, type = int)
  parser.add_argument('--log_frequency', default = 50, type = int)

  parser.add_argument('--mod_train', default = 0.8, type = float)
  parser.add_argument('--mod_test', default = 1., type = float)
  parser.add_argument('--mod_act_train', default = 0.6, type = float)
  parser.add_argument('--mod_act_test', default = 1., type = float)


  parser.add_argument('--num_workers', type = int, default = 0)

  parser.add_argument('--num_train_data_samples', type = int, default = 50000)
  parser.add_argument('--num_val_data_samples', type = int, default = 5000)

  parser.add_argument('--save_gif_interval', type = int, default = 10)
  parser.add_argument('--save_metrics_interval', type = int, default = 500)

  # --- Diffusion process ---
  parser.add_argument('--diffusion_num_timesteps', type = int, default = 1000)
  parser.add_argument('--ddim_num_timesteps', type = int, default = 10)
  parser.add_argument('--timesteps_sampling', type = str, default = 'uniform_time_sampling')
  parser.add_argument('--noise_scheduler', type = str, default = 'cosine')
  parser.add_argument('--rescale_timesteps', type = str2bool, default = False)
  parser.add_argument('--drop_cond_method', type = str, default = 'sample')
  parser.add_argument('--model_var_type', type = str, default = 'fixed_large')

  # --- Backbone model ---
  parser.add_argument('--use_action', type = str2bool, default = False) 
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
  parser.add_argument('--embedding_max_period', type = int, default = 10000)
  parser.add_argument('--learnable_embedding', type = str2bool, default = True)
  parser.add_argument('--se_ratio', type = int, default = 4)
  parser.add_argument('--use_excitation', type = str2bool, default = False)
  parser.add_argument('--use_max_pooling', type = str2bool, default = False)
  parser.add_argument('--use_adaptive', type = str2bool, default = False)


  parser.add_argument('--num_gcn_stages', type = int, default = 1)
  parser.add_argument('--gcn_dim', type = int, default = 128)
  parser.add_argument('--gcn_dropout', type = float, default = 0.3)
  parser.add_argument('--gcn_use_adaptive', type = str2bool, default = False)

  parser.add_argument('--num_samples', type = int, default = 4)
  parser.add_argument('--n_pred', type = int, default = 20)
  parser.add_argument('--n_prior', type = int, default = 20)


  parser.add_argument('-future_weight', default = 100.0, type = float)
  parser.add_argument('--past_weight', default = 1.0, type = float) 
  parser.add_argument('--diversity_weight', default = 20.0, type = float) 
  parser.add_argument('--min_threshold', default = 20.0, type = float) 
  

  # Multimodal / eval 
  parser.add_argument('--multimodal_threshold', type = float, default = 0.5)
  parser.add_argument('--padding', type = str, default = 'last_frame')
  parser.add_argument('--multimodal_path', type = str,
    default = 'dataset/multimodal_set/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz')
  parser.add_argument('--data_candi_path', type = str,
    default = 'dataset/multimodal_set/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz')
  parser.add_argument('--exp_desc', type = str, default = None)
  # EMA 
  parser.add_argument('--ema_rate', type = float, default = 0.999)
  parser.add_argument('--step_start_ema', type = int, default = 2000)

  # Visualization 
  parser.add_argument('--vis_switch_num', type = int, default = 10)
  parser.add_argument('--vis_col', type = int, default = 10)
  parser.add_argument('--vis_row', type = int, default = 3)


  parser.add_argument('--clip_norm', type = float, default = 1.0)

  parser.add_argument('--eta', type = float, default = 1.0,
    help = "DDIM stochasticity parameter used during rollout collection.")
  parser.add_argument('--cfg_scale', type = float, default = 1.0) 
  parser.add_argument('--deterministic', type = str2bool, default = False)
  parser.add_argument('--enable_complete', type = str2bool, default = False)

  # --- DPM Solver ---- 
  parser.add_argument('--use_dpm_solver', type = str2bool, default = False)
  parser.add_argument('--dpm_algorithm_type', type = str, default = 'dpmsolver++', 
    choices = ['dpmsolver', 'dpmsolver++'])
  parser.add_argument('--dpm_solver_type', type = str, default = 'dpmsolver',\
    choices = ['dpmsolver', 'taylor'])
  parser.add_argument('--dpm_solver_num_steps', type = int, default = 10)
  parser.add_argument('--dpm_solver_skip_type', type = str, default = 'logSNR', \
    choices = ['time_uniform', 'time_quadratic', 'logSNR'])
  parser.add_argument('--dpm_solver_method', type = str, default = 'singlestep_fixed', \
    choices = ['adaptive', 'multistep', 'singlestep', 'singlestep_fixed'])
  parser.add_argument('--dpm_solver_order', type = int, default = 1)
  parser.add_argument('--dpm_solver_denoise_to_zero', type = str2bool, default = True)
  
  args = parser.parse_args()

  params = vars(args)

  if params['algo'] is None:
    params['algo'] = params['type']



  params['train'] = (args.mode == 'train')

  params['model_prefix'] = os.path.join('exp', params['model_prefix'])

  if params['mode'] in ['eval', 'demo_vis', 'dct_vis', 'inter_vis', 'superimposed_vis', 'eval_per_action', 'fid_tsne_vis'] and os.path.exists(params['model_prefix']):
    config_path = os.path.join(params['model_prefix'], 'config_dir', 'config.json')
    with open(config_path, 'r') as f:
      saved_params = json.load(f)

    for key, value in saved_params.items():
      if key in [
        'enable_complete', 'finetuning_ckpt', 'deterministic', 'cfg_scale',\
        'use_classifier_free_guidance', 'exp_desc', 'mode', 'ddim_num_timesteps',\
        'use_dpm_solver', 'dpm_solver_type', 'dpm_solver_num_steps', 'dpm_solver_method', 
        'dpm_solver_order', 'dpm_solver_denoise_to_zero', 'num_val_data_samples',
        'dpm_algorithm_type', 
      ]: 
        continue
      params[key] = value

  else : 
    create_dir_tree(params['model_prefix'], params)
    params['pose_dim'] = params['joint_dim'] * params['num_joints']

    params['idx_pad'], params['zero_index'] = generate_pad(
      params['padding'], params['num_observed_frames'], params['num_predict_frames'])
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

  if params['type'] == 'diffusion':
    runner = DiffusionRunner(
      params = params,
      dataset = dataset,
      multimodal_dict = None,
      logger = logger,
      tblogger = tb_logger, 
    )
  else : 
    raise NotImplementedError(f"run type {args.type} is not supported")
  
  if args.mode == 'train':
    runner.train()
  elif args.mode == 'eval':
    multimodal_dict = get_multimodal_dataset(
     logger = logger,
     dataset_multi_test = dataset_multi_test,
     params = params,
    )
    nfes = [5, 10, 20]
    orders = [1, 2] 
    solver_methods = ['multistep', 'singlestep']
    # for nfe in nfes:
    #   for order in orders:
    #     for solver_method in solver_methods:
    #         if solver_method == 'multistep' and order == 1: 
    #           continue
    #         runner._params['dpm_solver_num_steps'] = nfe
    #         runner._params['dpm_solver_order'] = order
    #         runner._params['dpm_solver_method'] = solver_method
    #         runner._params['exp_desc'] = ("{dpm_algorithm_type}_mask_order_{dpm_solver_order}_{dpm_solver_method}"
    #           "_{dpm_solver_num_steps}_cfg_scale_{cfg_scale}").format_map(runner._params)
    logger.info(f"Evaluating {runner._params['exp_desc']}")
    runner.compute_stats(
      sampling_model = runner._ema_model, 
      multimodal_dict = multimodal_dict, 
      exp_desc = runner._params['exp_desc']
    )
  elif args.mode == 'eval_per_action': 
    actions = []
    if params['dataset'] == 'amass':
      actions = dataset_multi_test.idx_to_class
    else:
      for subject in dataset_multi_test.subjects:
        for action in dataset_multi_test.data[subject].keys():
          actions.append(action)
      actions = list(set(actions))
      for action in actions:
        if action == 'Discussion':
          action = 'Discussion 1'
        print(f"Start evaluation on {action}")
        multimodal_dict = get_multimodal_dataset(
          logger = logger,
          dataset_multi_test = dataset_multi_test,
          params = params,
          required_action = action, 
        )
        runner.compute_stats(sampling_model = runner._ema_model, multimodal_dict = multimodal_dict, exp_desc = params['exp_desc'] + '_' + action)
 
 
  elif args.mode == 'demo_vis':
    runner.demo_visualize(
      sampling_model = runner._ema_model,
      sampling_mode = params['sampling_mode'],
      dataset = dataset,
    )
  elif args.mode == 'inter_vis':
    runner.visualize_intermediate(sampling_model = runner._ema_model)
  elif args.mode == 'dct_vis': 
    runner.visualize_dct_space()
  elif args.mode == 'superimposed_vis':
    runner.visualize_superimposed(
      sampling_model = runner._ema_model,
      actions = ('DanceDB', 'GRAB', 'HUMAN4D', 'Transitions'), #('Walking 1 chunk0', 'Jog 1 chunk0', 'Box 1 chunk4', 'Gestures 1 chunk1'),
      times = (0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0),
      out_path = './superimposed_amass_ada', 
      )
  elif args.mode == 'fid_tsne_vis':
    nfes = [5, 10, 20]
    orders = [1, 2] 
    solver_methods = ['multistep', 'singlestep']
    for nfe in nfes:
      for order in orders:
        for solver_method in solver_methods:
            if solver_method == 'multistep' and order == 1: 
              continue
            runner._params['dpm_solver_num_steps'] = nfe
            runner._params['dpm_solver_order'] = order
            runner._params['dpm_solver_method'] = solver_method
            runner._params['exp_desc'] = f"{runner._params['dpm_algorithm_type']}_order_{runner._params['dpm_solver_order']}_{runner._params['dpm_solver_method']}"
            runner._params['exp_desc'] += f"_{runner._params['dpm_solver_num_steps']}_cfg_scale_{runner._params['cfg_scale']}_fid_tsne.png"
            print(f"Visualizing FID t-SNE :  {runner._params['exp_desc']}")
            runner.visualize_fid_tsne(
              sampling_model = runner._ema_model,
              num_sequences = 20000, 
              out_path = os.path.join(runner._params['model_prefix'], runner._params['exp_desc']),  
            )
  else:
    raise NotImplementedError(f"--mode {args.mode} is not supported for --type {runner._params['type']}")

