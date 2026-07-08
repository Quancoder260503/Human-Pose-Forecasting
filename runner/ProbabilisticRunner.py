import matplotlib.pyplot as plt
import numpy as np
import random 
from torch.nn.utils import clip_grad_norm_
import sys
import numpy as nn
import os
import time
from abc import abstractmethod
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
from model import ODESolver
from model.ProbabilityPath import * 
from model.VectorFieldScheduler import *
import torch.nn.functional as F
from runner.WarmupScheduler     import *
from data.Dataset3DUtils        import * 
from model.EMA                  import *
from model.DiffusionTransformer      import * 
from model.GaussianDiffusion    import *
from runner.SamplingUtils       import *
import pandas as pd 
import csv 
import copy 
from Logger import *

def skewed_timestep_sample(num_samples : int) -> torch.Tensor: 
  P_mean = -1.2 
  P_std  = 1.2 
  rnd_normal = torch.randn((num_samples, )) 
  sigma = (rnd_normal * P_std + P_mean).exp()
  time = 1 / (1 + sigma)
  time = torch.clip(time, min = 0.0001, max = 1.0) 
  return time 

class ProbabilisticRunner(object):
  def __init__(
    self, 
    params, 
    dataset, 
    multimodel_dict, 
    logger, 
    tblogger, 
    pose_encoder_fn = None, 
    pose_decoder_fn = None
  ):

    self._params = params
    self._dataset = dataset
    self._logger = logger 
    self._tblogger = tblogger
    self._multimodal_dict = multimodel_dict 

    thisname = self.__class__.__name__
    self.dct, self.idct = get_dct_matrix(\
      params['num_observed_frames'] + params['num_predict_frames'])

   
    
    self.dct = self.dct.to(params['device'], dtype = torch.float32)
    self.idct = self.idct.to(params['device'], dtype = torch.float32)
    self.init_model(pose_encoder_fn, pose_decoder_fn)
    self._model.to(self._params['device'])
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.ema_helper_init()
    self.init_algo()
    self.num_dct_channels = self._params['num_dct_channels']
    self.criterion = nn.MSELoss()

    self._params['learning_rate_type'] = 'epochwise'

  #   if self.dataset == 'ntu_rgbd':
  #       self.evaluate_fn = self.evaluate_nturgbd
  #  # elif self.dataset == 'cmu_mocap':
  #       #self.evaluate_fn = self.evaluate_amass

    m_params = filter(lambda p: p.requires_grad, self._model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])             
    
    self._logger.info('[INFO] ({}) This module has {} parameters!'.format(thisname, nparams))
    self._logger.info('[INFO] ({}) Intializing ModelFn with params'.format(thisname))
    
    
    self._global_step = 1
    self._epoch = -1
    self.finetune_init()

    self.train_losses = None 
    self.train_generator = None 
    self.val_losses = None 
    self.val_generator = None 

  @abstractmethod
  def train_one_epoch(self):
    pass 

  @abstractmethod 
  def eval_one_epoch(): 
    pass 

  @abstractmethod
  def init_algo(self): 
    pass 

  @abstractmethod
  def sampling(self, sampling_model, trajectory_dct, trajectory_dct_mod, sampling_dict): 
    pass 

  def finetune_init(self):
    if self._params['finetuning_ckpt'] is not None:
      self._logger.info(f'[INFO] (finetune_model) Finetuning from: {self._params["finetuning_ckpt"]}')
      model_dict = torch.load(self._params['finetuning_ckpt'], weights_only = False, map_location = self._params['device'])
      self._epoch = model_dict['epoch']
      self._global_steps = model_dict['global_steps']
      self._model.load_state_dict(model_dict['model'])
      self._optimizer_fn.load_state_dict(model_dict['optimizer'])
      self._lr_scheduler.load_state_dict(model_dict['lr_scheduler'])
      self._ema_model.load_state_dict(model_dict['ema_model'])

  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()

  def ema_helper_init(self):
    self._ema_helper = EMAHelper(mu = self._params['ema_rate'])
    self._ema_model = copy.deepcopy(self._model).eval().requires_grad_(False)

  def get_lr_fn(self):
    """Creates the function to be used to generate the learning rate."""
    if self._params['learning_rate_fn'] == 'step':
      return torch.optim.lr_scheduler.StepLR(
        self._optimizer_fn, 
        step_size = self._params['lr_step_size'], 
        gamma = self._params['step_gamma'], 
      )
    elif self._params['learning_rate_fn'] == 'exponential':
      return torch.optim.lr_scheduler.ExponentialLR(
        self._optimizer_fn, 
        gamma = self._params['exp_gamma'], 
      )
    elif self._params['learning_rate_fn'] == 'linear':
      lr0, T = self.learning_rate, self.max_epochs
      lrT = lr0 * 0.5
      m = (lrT - 1) / T
      lambda_fn =  lambda epoch: m * epoch + 1.0
      return torch.optim.lr_scheduler.LambdaLR(
        self._optimizer_fn, 
        lr_lambda = lambda_fn
      )
    elif self._params['learning_rate_fn'] == 'decay':
      return torch.optim.lr_scheduler.LinearLR(
        optimizer = self._optimizer_fn,
        total_iters = self._params['max_epochs'],
        start_factor = 1.0,
        end_factor = self._params['end_learning_rate'] / self._params['learning_rate'],
      )
    elif self._params['learning_rate_fn'] == 'milestones':
      return torch.optim.lr_scheduler.MultiStepLR(
        optimizer = self._optimizer_fn, 
        milestones = self._params['milestones'],
        gamma = self._params['milestone_gamma']
      ) 
    else:
      raise ValueError('Unknown learning rate function: {}'.format(
        self._params['learning_rate_fn']))

  def sampling_timesteps(self, batch_size: int) -> torch.Tensor:
    T = self._params['diffusion_num_timesteps']
    time_sampling = self._params['timesteps_sampling']
    process = self._params['algo'].lower()

    if time_sampling == 'antithetic_sampling':
      timesteps = torch.randint(
        low  = 1,
        high = T,
        size = (batch_size // 2 + 1,),  
      )
      timesteps = torch.cat([timesteps, T - timesteps - 1], dim = 0)[:batch_size]

    elif time_sampling == 'skewed_time_sampling':
      timesteps = skewed_timestep_sample(num_samples = batch_size)
    elif time_sampling == 'uniform_time_sampling':
      if process == 'diff':
        timesteps = torch.randint(low = 1, high = T, size = (batch_size, ))
      elif process == 'flow':       
        timesteps = torch.rand(batch_size)    
      else:
        raise ValueError(f"Unknown random process: {process}")

    else:
      raise ValueError(f"Unknown timesteps_sampling: {time_sampling}")

    return timesteps.to(self._params['device'])

  def prepare_train_step(self):
    self._model.train()
    self.train_generator = self._dataset['train'].sampling_generator( 
      num_samples = self._params['num_train_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time()
    self.train_losses = AverageMeter()
    self._logger.info(f'Starting training at epoch {self.curr_epoch}')

  def prepare_eval_step(self): 
    self._model.eval()
    self.val_generator = self._dataset['test'].sampling_generator( 
      num_samples = self._params['num_val_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time() 
    self.val_losses = AverageMeter()
    self._logger.info(f'Starting evaluation at epoch {self.curr_epoch}')

  def compute_train_stats(self): 
    if self.curr_epoch % self._params['save_gif_interval'] == 0: 
      generated_pose = self.pose_generator(
        sampling_model = self._model, 
        dataset = self._dataset['train'], 
        sampling_mode = 'gif', 
      )   
      render_animation(
        skeleton = self._dataset['train'].skeleton, 
        generated_poses = generated_pose, 
        algos = [self._params['algo']], 
        num_observed_frames = self._params['num_observed_frames'], 
        num_cols = self._params['vis_col'], 
        output = os.path.join(self._params['gif_dir'], f"training_{self.curr_epoch}.gif")
      )
  
  def compute_eval_stats(self): 
    if self.curr_epoch % self._params['save_gif_interval'] == 0: 
      generated_pose = self.pose_generator(
        sampling_model = self._ema_model, 
        dataset = self._dataset['train'], 
        sampling_mode = 'gif', 
      )   
      render_animation(
        skeleton = self._dataset['test'].skeleton, 
        generated_poses = generated_pose, 
        algos = [self._params['algo']], 
        num_observed_frames = self._params['num_observed_frames'], 
        num_cols = self._params['vis_col'], 
        output = os.path.join(self._params['gif_dir'], f"val_{self.curr_epoch}.gif")
      )
    
    if self.curr_epoch % self._params['save_metrics_interval'] == 0 and self.curr_epoch > 0: 
      self.compute_stats(self._ema_model, self._multimodal_dict)

  def pose_generator(self, sampling_model, dataset, action = None, num_rows = 1, sampling_mode = None): 
    trajectory_np = None
    j = None
    while True:
      poses = {}
      draw_order_indicator = -1
      for k in range(0, num_rows):
        if sampling_mode == 'switch':
          data = dataset.sample_all_action()
        elif sampling_mode == 'pred':
          data = dataset.sample_iter_action(action)
        elif sampling_mode == 'gif' or 'fix' in sampling_mode:
          data = dataset.sample()
        elif sampling_mode == 'zero_shot':
          data = dataset[np.random.randint(0, dataset.shape[0])].copy()
          data = np.expand_dims(data, axis = 0)
        else:
          raise NotImplementedError(f"unknown pose generator mode: {sampling_mode}")

        ground_truth = data[0].copy()
        ground_truth[:, :1, :] = 0
        data[:, :, :1, :] = 0

        if sampling_mode == 'switch':
          poses = {}
          trajectory_np = data[..., 1:, :].reshape([data.shape[0], \
            self._params['num_observed_frames'] + self._params['num_predict_frames'], -1])
        elif sampling_mode == 'pred' or sampling_mode == 'gif' or 'fix' in sampling_mode or sampling_mode == 'zero_shot':
          if draw_order_indicator == -1:
            poses['context'] = ground_truth
            poses['gt'] = ground_truth
          else:
            poses[f"{self._params['algo']}_{draw_order_indicator + 1}"] = ground_truth
            poses[f"{self._params['algo']}_{draw_order_indicator + 2}"] = ground_truth
          ground_truth = np.expand_dims(ground_truth, axis = 0)
          trajectory_np = ground_truth[..., 1:, :].reshape([ground_truth.shape[0],\
                self._params['num_observed_frames'] + self._params['num_predict_frames'], -1])

        traj = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32)

        mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, self._params, sampling_mode, self.dct)
        sampled_motion = self.sampling(sampling_model, traj_dct, traj_dct_mod, mode_dict)

        result_trajectory = torch.matmul(self.idct[:, :self._params['num_dct_channels']], sampled_motion)
        result_trajectory = result_trajectory.cpu().numpy()
        result_trajectory = post_process(result_trajectory, self._params)

        if k == 0:
          for j in range(result_trajectory.shape[0]):
            poses[f"{self._params['algo']}_{j}"] = result_trajectory[j]
        else:
          for j in range(result_trajectory.shape[0]):
            poses[f"{self._params['algo']}_{j + draw_order_indicator + 2 + 1}"] = result_trajectory[j]

        if draw_order_indicator == -1:
          draw_order_indicator = j
        else:
          draw_order_indicator = j + draw_order_indicator + 2 + 1

      yield poses
  
  def metrics_calc_sampling(self, sampling_model, data): 
    trajectory_np = data[..., 1:, :].transpose([0, 2, 3, 1])
    trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32)
    trajectory = trajectory.reshape([trajectory.shape[0], -1, trajectory.shape[-1]]).transpose(1, 2)


    mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(trajectory, self._params, 'metrics', self.dct)
    sampled_motion = self.sampling(sampling_model, traj_dct, traj_dct_mod, mode_dict)

    result_trajectory = torch.matmul(self.idct[:, :self._params['num_dct_channels']], sampled_motion)
    # result_trajectory.shape (K, 125, 48)
    result_trajectory = result_trajectory.cpu().numpy()
    result_trajectory = result_trajectory[None, ...]
    return result_trajectory


  def train(self):
    """Main training loop."""
    self._lr_db_curve = []
    self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]
    thisname = self.__class__.__name__
    self._logger.info('=' * 90)
    self._logger.info(f"[INFO] {thisname} | global step : {self._global_step:06d} | epoch : {(self._epoch + 1):06d} | Start training")
    self._logger.info('=' * 90)
    for e in range(self._epoch + 1, self._params['max_epochs']):
      self._scalars = {}
      self.curr_epoch = e
   
      # Train 
      self.prepare_train_step()
      self.train_one_epoch()
      self._logger.info('>>> Epoch: {} | Time: {:.2f} | Train Loss: {} | Learning Rate : {:.5f}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.train_losses.avg, self._lr_scheduler.get_last_lr()[0]))
      self.compute_train_stats()

      # Evaluate 
      self.prepare_eval_step()
      self.eval_one_epoch()
      self._logger.info('>>> Epoch: {} | Time: {:.2f} | Val Loss: {}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.val_losses.avg))
      self.compute_eval_stats()
      
      self.update_learning_rate(e, mode = 'epochwise')

      if (e + 1) % self._params['snapshot_saving_steps'] == 0:
        model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
        state = {
          'global_steps' : self._global_step,
          'epoch' : e,
          'model' : self._model.module.state_dict() if hasattr(self._model, 'module') else self._model.state_dict(),
          'optimizer' : self._optimizer_fn.state_dict(),
          'lr_scheduler' : self._lr_scheduler.state_dict(), 
          'ema_model' : self._ema_model.state_dict(), 
        }
        torch.save(state, model_path)
    # save the last one
    
    model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
    state = {
      'global_steps' : self._global_step,
      'epoch' : e,
      'model' : self._model.module.state_dict() if hasattr(self._model, 'module') else self._model.state_dict(),
      'optimizer' : self._optimizer_fn.state_dict(),
      'lr_scheduler' : self._lr_scheduler.state_dict(), 
      'ema_model' : self._ema_model.state_dict(), 
    }
    torch.save(state, model_path)

  def update_learning_rate(self, epoch_step, mode = 'stepwise'):
    """Update learning rate handler updating only when the mode matches."""
    if self._params['learning_rate_type'] == mode:
      self._lr_scheduler.step()
      self._tblogger.add_scalar('learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]

  def init_model(self, pose_encoder_fn = None, pose_decoder_fn = None):
    self._model = diffusion_transformer(
      self._params, 
      pose_encoder_fn(self._params),
    )

  def select_optimizer(self): 
    if self._params['optimizer'] == 'AdamW': 
      return torch.optim.AdamW(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
        betas = self._params['betas'], 
      )
    if self._params['optimizer'] == 'Adam':
      return torch.optim.Adam(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
        betas = self._params['betas'], 
      )
    elif self._params['optimizer'] == 'RMSProp': 
      return torch.optim.RMSprop(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
      )
    
    elif self._params['optimizer'] == 'SGD': 
      return torch.optim.SGD(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
      )
    else : 
      raise NotImplementedError(f"Optimizer {self._params['optimizer']} not supported")

  def compute_all_metrics(self, pred, gt, gt_multi):
    if pred.shape[0] == 1 : 
      diversity = 0.0 

    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()

    gt_multi = torch.from_numpy(gt_multi).to('cuda')
    gt_multi_gt = torch.cat([gt_multi, gt], dim = 0)

    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]

    diff_multi = pred - gt_multi_gt 
    dist = torch.linalg.norm(diff_multi, dim = 3)

    mmfde, _ = dist[:, :-1, -1].min(dim = 0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim = 2).min(dim = 0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim = 1).min(dim = 0)
    fde, _ = dist[:, -1, -1].min(dim = 0)
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde, mmade, mmfde
  
  def compute_stats(self, sampling_model, multimodal_dict): 
    ground_truth_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    trajectory_gt_arr = multimodal_dict['trajectory_gt_array']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in [self._params['algo']]} for x in stats_names}

    K = 50
    pred = []
    for i in tqdm.tqdm(range(0, K), position = 0):
      pred_i_nd = self.metrics_calc_sampling(sampling_model, data_group)
      pred.append(pred_i_nd)
      if i == K - 1:  # in last iteration, concatenate all candidate pred
        pred = np.concatenate(pred, axis = 0)
        pred = pred[:, :, self._params['num_observed_frames']:, :]
        ground_truth_group = torch.from_numpy(ground_truth_group).to('cuda')
        pred = torch.from_numpy(pred).to('cuda')
        for j in range(0, num_samples):
          apd, ade, fde, mmade, mmfde = self.compute_all_metrics(
            pred[:, j, :, :],
            ground_truth_group[j][np.newaxis, ...],
            trajectory_gt_arr[j],
          )
          stats_meter['APD'][self._params['algo']].update(apd)
          stats_meter['ADE'][self._params['algo']].update(ade)
          stats_meter['FDE'][self._params['algo']].update(fde)
          stats_meter['MMADE'][self._params['algo']].update(mmade)
          stats_meter['MMFDE'][self._params['algo']].update(mmfde)
        for stats in stats_names:
          str_stats = f'{stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
          self._logger.info(str_stats)

    # save stats in csv
    file_latest = f"{self._params['result_dir']}/stats_latest.csv"
    file_stat = f"{self._params['result_dir']}/stats.csv"
    with open(file_latest, 'w') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames = ['Metric'] + [self._params['algo']])
      writer.writeheader()
      for stats, meter in stats_meter.items():
        new_meter = {x: y.avg for x, y in meter.items()}
        new_meter[self._params['algo']] = new_meter[self._params['algo']].cpu().numpy()
        new_meter['Metric'] = stats
        writer.writerow(new_meter)
    
    df1 = pd.read_csv(file_latest)

    if os.path.exists(file_stat) is False:
      df1.to_csv(file_stat, index = False)
    else:
      df2 = pd.read_csv(file_stat)
      df = pd.concat([df2, df1[self._params['algo']]], axis = 1, ignore_index = True)
      df.to_csv(file_stat, index = False)

  def demo_visualize(self, sampling_mode, sampling_model, dataset):
    """
    script for drawing gifs in different modes
    """
    if self._params['dataset'] != 'h36m' and sampling_mode != 'pred':
      raise NotImplementedError(f"sorry, {sampling_mode} is currently only available in h36m setting.")
    if sampling_mode == 'switch':
      for i in range(0, self._params['vis_switch_num']):
        generated_pose = self.pose_generator(
          dataset = dataset['test'], 
          sampling_model = sampling_model,
          sampling_mode = sampling_mode
        )
        render_animation(
          skeleton = dataset['test'].skeleton, 
          generated_poses = generated_pose, 
          algos = [self._params['algo']], 
          num_observed_frames = self._params['num_observed_frames'], 
          num_cols = self._params['vis_col'], 
          output = os.path.join(self._params['gif_dir'], f'switch_{i}.gif'), 
          sampling_mode = sampling_mode
        )

    elif sampling_mode == 'pred':
      action_list = dataset['test'].prepare_iter_action()
      for i in range(0, len(action_list)):
        generated_pose = self.pose_generator(
          dataset = dataset['test'], 
          sampling_model = sampling_model, 
          sampling_mode = sampling_mode, 
          action = action_list[i], 
          num_rows = self._params['vis_row']
        )
        suffix = action_list[i]
        render_animation(
          skeleton = dataset['test'].skeleton, 
          generated_poses = generated_pose, 
          algos = [self._params['algo']], 
          num_observed_frames = self._params['num_observed_frames'], 
          num_cols = self._params['vis_col'] + 2, 
          output = os.path.join(self._params['gif_dir'], f'pred_{suffix}.gif'), 
          sampling_mode = sampling_mode
        )

    elif sampling_mode == 'control':
        # draw part-body controllable results
      fix_name = ['right_leg', 'left_leg', 'torso', 'left_arm', 'right_arm', 'fix_lower', 'fix_upper']
      for i in range(0, 7):
        mode_fix = 'fix' + '_' + str(i)
        generated_pose = self.pose_generator(
          dataset = dataset['test'], 
          sampling_model = sampling_model,
          sampling_mode = mode_fix, 
          num_rows = self._params['vis_row']
        )
        render_animation(
          skeleton = dataset['test'].skeleton, 
          generated_poses = generated_pose, 
          algos = [self._params['algo']], 
          num_observed_frames = self._params['num_observed_frames'], 
          num_cols = self._params['vis_col'] + 2, 
          output = os.path.join(self._params['gif_dir'], fix_name[i] + '.gif'), 
          mode = sampling_mode, 
          fix_index = i
        )
    elif sampling_mode == 'zero_shot':
      amass_data = np.squeeze(np.load('./data/amass_retargeted.npy'))
      for i in range(0, 15):
        generated_pose = self.pose_generator(
          dataset = amass_data, 
          sampling_model = sampling_model,
          sampling_mode = sampling_mode, 
          num_rows = self._params['vis_row']
        )
        render_animation(
          skeleton = dataset['test'].skeleton, 
          generated_poses = generated_pose, 
          algos = [self._params['algo']], 
          num_observed_frames = self._params['num_observed_frames'], 
          num_cols = self._params['vis_col'] + 2, 
          output = os.path.join(self._params['gif_dir'], f"zero_shot_{str(i)}.gif"), 
          sampling_mode = sampling_mode
        )
    else:
        raise NotImplementedError
