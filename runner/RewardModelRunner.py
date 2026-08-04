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
from tqdm import tqdm
from model.ProbabilityPath import * 
from model.VectorFieldScheduler import *
import torch.nn.functional as F
from runner.WarmupScheduler import *
from data.Dataset3DUtils import * 
from model.RewardModel import reward_transformer, reward_encoder, reward_gcn, InfoNCE, ConditioningInfoNCE
from model.DiffusionTransformer import diffusion_transformer
from model.HumanMAC.diffusion import * 
from model.HumanMAC.transformer import *
from runner.RunnerUtils import *
import pandas as pd 
import csv 
from functools import partial
import copy 
from Logger import *
import tqdm 

# Temporarily use the SOTA to samples diffusion 

"""
:TODO Temporarily use the SOTA to samples diffusion, may change to adapt with it main codebase after testing result is OK 
"""

class RewardRunner(object):
  def __init__(
    self, 
    params, 
    dataset, 
    multimodal_dict, 
    logger, 
    tblogger, 
  ):
    self._params = params
    self._dataset = dataset
    self._logger = logger 
    self._tblogger = tblogger
    self._multimodal_dict = multimodal_dict 

    thisname = self.__class__.__name__
    self.dct, self.idct = get_dct_matrix(\
      params['num_observed_frames'] + params['num_predict_frames'])


    self.dct = self.dct.to(params['device'], dtype = torch.float32)
    self.idct = self.idct.to(params['device'], dtype = torch.float32)
    self.init_model()
    self._model.to(self._params['device'])
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.num_dct_channels = self._params['num_dct_channels']
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
  
  def finetune_init(self):
    if self._params['finetuning_ckpt'] is not None:
      self._logger.info(f'[INFO] (finetune_model) Finetuning from: {self._params["finetuning_ckpt"]}')
      model_dict = torch.load(self._params['finetuning_ckpt'], weights_only = False, map_location = self._params['device'])
      self._epoch = model_dict['epoch']
      self._global_steps = model_dict['global_steps']
      self._model.prior_encoder.load_state_dict(model_dict['prior_encoder'])
      self._model.encoder.load_state_dict(model_dict['encoder'])
      self._optimizer_fn.load_state_dict(model_dict['optimizer'])
      self._lr_scheduler.load_state_dict(model_dict['lr_scheduler'])

  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()

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


  def prepare_train_step(self):
    self._model.encoder.train() 
    self._model.prior_encoder.train()
    self.train_generator = self._dataset['train'].sampling_generator( 
      num_samples = self._params['num_train_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time()
    self.train_losses = AverageMeter()
    self._logger.info(f'Starting training at epoch {self.curr_epoch}')

  def prepare_eval_step(self): 
    self._model.encoder.eval() 
    self._model.prior_encoder.eval()
    self.val_generator = self._dataset['test'].sampling_generator( 
      num_samples = self._params['num_val_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time() 
    self.val_losses = AverageMeter()
    self._logger.info(f'Starting evaluation at epoch {self.curr_epoch}')

  def train_one_epoch(self):
    for index, trajectory_np in enumerate(self.train_generator):
      self._optimizer_fn.zero_grad()
      with torch.no_grad():
        trajectory_np = trajectory_np[..., 1:, :].reshape([
          trajectory_np.shape[0], 
          self._params['num_observed_frames'] + self._params['num_predict_frames'], 
          -1, 
        ]) 
        trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32) 
        trajectory_pad = padding_trajectory(
          trajectory, 
          self._params['padding'], 
          self._params['idx_pad'], 
          self._params['zero_index'],
        )

        trajectory_dct = torch.matmul(self.dct[:self.num_dct_channels], trajectory) 
        trajectory_dct_mod = torch.matmul(self.dct[:self.num_dct_channels], trajectory_pad) 
        mode_dict, _, _ = sample_preprocessing(trajectory, self._params, 'metrics', self.dct)

    
      loss, acc = self._model(trajectory_dct, trajectory_dct_mod, mode_dict)
      loss.backward()
      self._optimizer_fn.step()
      self.update_learning_rate(self._global_step, mode = 'stepwise')
      self.train_losses.update(loss.item(), acc.item())
      self._tblogger.add_scalar('Loss/train', loss.item(), self.curr_epoch)
      self._tblogger.add_scalar('Accuracy/train', acc.item(), self.curr_epoch)
      self._global_step += 1 
      if index % 20 == 0: 
        self._logger.info(f"Train objective stats : epoch {self.curr_epoch} | step {index} | avg_loss : {self.train_losses.avg} | avg_acc : {self.train_losses.avg_acc} | learning_rate : {self._params['learning_rate']}")
      del loss, acc, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np 

  @torch.no_grad()
  def eval_one_epoch(self): 
    for trajectory_np in self.val_generator:
      with torch.no_grad():
        trajectory_np = trajectory_np[..., 1:, :].reshape([
          trajectory_np.shape[0], 
          self._params['num_observed_frames'] + self._params['num_predict_frames'], 
          -1, 
        ]) 
        trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32) 
        trajectory_pad = padding_trajectory(
          trajectory, 
          self._params['padding'], 
          self._params['idx_pad'], 
          self._params['zero_index'],
        )
        trajectory_dct = torch.matmul(self.dct[:self.num_dct_channels], trajectory) 
        trajectory_dct_mod = torch.matmul(self.dct[:self.num_dct_channels], trajectory_pad) 
        mode_dict, _, _ = sample_preprocessing(trajectory, self._params, 'metrics', self.dct)
        loss, acc = self._model(trajectory_dct, trajectory_dct_mod, mode_dict)
        self.val_losses.update(loss.item(), acc.item())
        self._tblogger.add_scalar('Loss/val', loss.item(), self.curr_epoch)
        self._tblogger.add_scalar('Accuracy/val', acc.item(), self.curr_epoch)
        del loss, acc, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np 

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
      self._logger.info('>>> Epoch: {} | Time: {:.2f} | Train Loss: {} | Train Acc: {} | Learning Rate : {:.5f}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.train_losses.avg, self.train_losses.avg_acc, self._lr_scheduler.get_last_lr()[0]))
  

      # Evaluate 
      self.prepare_eval_step()
      self.eval_one_epoch()
      self._logger.info('>>> Epoch: {} | Time: {:.2f} | Val Loss: {} | Val Acc: {}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.val_losses.avg, self.val_losses.avg_acc))
      self.update_learning_rate(e, mode = 'epochwise')


      if (e + 1) % self._params['save_metrics_interval'] == 0 and e > 0: 
        self.compute_stats(multimodal_dict = self._multimodal_dict)

      if (e + 1) % self._params['snapshot_saving_steps'] == 0:
        model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
        state = {
          'global_steps' : self._global_step,
          'epoch': e,
          'encoder' : self._model.encoder.state_dict() if hasattr(self._model, 'encoder') else None,
          'prior_encoder' : self._model.prior_encoder.state_dict() if hasattr(self._model, 'prior_encoder') else None,
          'optimizer' : self._optimizer_fn.state_dict(),
          'lr_scheduler' : self._lr_scheduler.state_dict(), 
        }
        torch.save(state, model_path)
    # save the last one
    
    model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
    state = {
      'global_steps' : self._global_step,
      'epoch' : self.curr_epoch,
      'encoder' : self._model.encoder.state_dict() if hasattr(self._model, 'encoder') else None,
      'prior_encoder' : self._model.prior_encoder.state_dict() if hasattr(self._model, 'prior_encoder') else None,
      'optimizer' : self._optimizer_fn.state_dict(),
      'lr_scheduler' : self._lr_scheduler.state_dict(), 
    }
    torch.save(state, model_path)

  def update_learning_rate(self, epoch_step, mode = 'stepwise'):
    """Update learning rate handler updating only when the mode matches."""
    if self._params['learning_rate_type'] == mode:
      self._lr_scheduler.step()
      self._tblogger.add_scalar('learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]

  def init_model(self):
    self._sample_model = None 
    self.gaussian_diffusion = None 
    if self._params['reward_use_sampler'] :
      # add diffusion model here to sample 
      self._sample_model = MotionTransformer(
        input_feats = self._params['pose_dim'], 
        num_frames = self._params['num_dct_channels'], 
      )
      model_dict = torch.load('model/HumanMAC/h36m_ckpt.pt', weights_only = False, map_location = self._params['device'])
      self._sample_model.load_state_dict(model_dict)

      # Freeze the diffusion model 
      self._sample_model.eval()
      for p in self._sample_model.parameters(): 
        p.requires_grad_(False)

      self.gaussian_diffusion = Diffusion(
        noise_steps = self._params['diffusion_num_timesteps'],
        motion_size = (self._params['num_dct_channels'], self._params['pose_dim']),  
        device = self._params['device'], 
        padding = self._params['padding'],
        EnableComplete = True,
        ddim_timesteps = self._params['ddim_num_timesteps'],
        scheduler = self._params['ddim_num_timesteps'],
        mod_test = self._params['mode_test'],
        dct = self.dct,
        idct = self.idct,
        n_pre = self._params['num_dct_channels'], 
      )
    

    if self._params['reward_model_architecture'] == 'cross_transformer': 
      enc = reward_transformer(self._params)
      self._model = ConditioningInfoNCE(
        model = enc, 
        sample_model = self._sample_model, 
        sampler = self.gaussian_diffusion, 
        num_samples = self._params['num_samples'], 
        temperature = self._params['info_nce_temperature']
      )
    elif self._params['reward_model_architecture'] == 'gcn_encoder': 
      enc, prior_enc = reward_gcn(self._params), reward_gcn(self._params)
      self._model = InfoNCE(
        encoder = enc, 
        prior_encoder = prior_enc,
        sample_model = self._sample_model, 
        sampler = self.gaussian_diffusion,
        num_samples = self._params['num_samples'],         
        temperature = self._params['info_nce_temperature']
      )
    elif self._params['reward_model_architecture'] == 'transformer_encoder': 
      enc, prior_enc = reward_encoder(self._params), reward_encoder(self._params)
      self._model = InfoNCE(
        encoder = enc, 
        prior_encoder = prior_enc,
        sample_model = self._sample_model, 
        sampler = self.gaussian_diffusion,
        num_samples = self._params['num_samples'], 
        temperature = self._params['info_nce_temperature']
      )
    else : 
      raise NotImplementedError(f"Reward Model architecture {self._params['reward_model_architecture']} not found")

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

  def compute_stats(self, multimodal_dict):
    self._model.encoder.eval()
    self._model.prior_encoder.eval()
    data_group = multimodal_dict['data_group']  # entire pose sequences
    trajectory_gt_arr = multimodal_dict['trajectory_gt_array']  # candidate futures
    num_samples = multimodal_dict['num_samples']

    stats_names = ['Accuracy', 'Probability']
    stats_meter = {x: AverageMeter() for x in stats_names}

    for j in tqdm.tqdm(range(0, num_samples), position=0):
      trajectory_np = data_group[j, :, 1:, :]
      trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32)
      trajectory = trajectory.contiguous().view(trajectory.shape[0], -1)  # [total_frames, pose_dim]
      pad_trajectory = padding_trajectory(
        trajectory = trajectory.unsqueeze(0),
        padding = self._params['padding'],
        idx_pad = self._params['idx_pad'],
        zero_index=self._params['zero_index'],
      ) # [1, total_frames, pose_dim]

      candidate_trajectory = trajectory_gt_arr[j]
      candidate_full_arr = []

      for i in range(candidate_trajectory.shape[0]):
        candidate_future = torch.tensor(candidate_trajectory[i], device = self._params['device'], dtype = torch.float32) 
        candidate_full = torch.cat([trajectory[: self._params['num_observed_frames']], candidate_future], dim = 0)
        candidate_full_arr.append(candidate_full)

      candidate_full_arr = torch.stack(candidate_full_arr, dim = 0)  # [num_candidates, total_frames, pose_dim]

      trajectory_dct_cand = torch.matmul(self.dct[: self.num_dct_channels, :], candidate_full_arr)  # [num_candidates, num_dct_channels, pose_dim]
      trajectory_dct_mod = torch.matmul(self.dct[: self.num_dct_channels, :], pad_trajectory)  # [1, num_dct_channels, pose_dim]
      trajectory_dct_main = torch.matmul(self.dct[: self.num_dct_channels, :], trajectory.unsqueeze(0))  # [1, num_dct_channels, pose_dim]

      trajectory_dct = torch.cat([trajectory_dct_main, trajectory_dct_cand], dim = 0)  # [1 + num_candidates, num_dct_channels, pose_dim]

      if isinstance(self._model, InfoNCE): 
        enc_trajectory_dct = self._model.encoder(trajectory_dct) # [1 + num_candidates, code_size]
        enc_trajectory_dct_mod = self._model.prior_encoder(trajectory_dct_mod) # [1, code_size]
      
        enc_trajectory_dct = F.normalize(enc_trajectory_dct, dim = -1) 
        enc_trajectory_dct_mod = F.normalize(enc_trajectory_dct_mod, dim = -1)
      
        all_to_prior = torch.matmul(enc_trajectory_dct, enc_trajectory_dct_mod.t()) / self._model.temperature 
        prior_to_all = torch.matmul(enc_trajectory_dct_mod, enc_trajectory_dct.t()) / self._model.temperature  
    
        all_to_prior = F.softmax(all_to_prior, dim = -1)
        prior_to_all = F.softmax(prior_to_all, dim = -1)
        prob = prior_to_all[0, 0]
        acc = int(prior_to_all.argmax(dim = -1).item() == 0)

        stats_meter['Probability'].update(prob.item())
        stats_meter['Accuracy'].update(acc)

      elif isinstance(self._model, ConditioningInfoNCE): 
        trajectory_dct_mod = trajectory_dct_mod.repeat(trajectory_dct.shape[0], 1, 1) 
        estimated_score = self._model.cond_prob_estimator(trajectory_dct, trajectory_dct_mod).squeeze(-1) # [1 + num_candidates, 1]
        estimated_score = F.normalize(estimated_score) / self._model.temperature
        estimated_score = F.softmax(estimated_score) 
        prob = estimated_score[0] 
        acc = int(estimated_score.argmax().item() == 0)
        stats_meter['Probability'].update(prob.item())
        stats_meter['Accuracy'].update(acc)
      else : 
        raise NotImplementedError('Reward model type unknown')

    for stats in stats_names:
      str_stats = f'{stats}: {stats_meter[stats].avg:.4f}'
      self._logger.info(str_stats)

    # save stats in csv
    file_latest = f"{self._params['result_dir']}/stats_latest.csv"
    file_stat = f"{self._params['result_dir']}/stats.csv"

    latest_data = {stats: stats_meter[stats].avg for stats in stats_names}
    df1 = pd.DataFrame([latest_data])
    df1.to_csv(file_latest, index = False)

    if os.path.exists(file_stat) is False:
      df1.to_csv(file_stat, index = False)
    else:
      df2 = pd.read_csv(file_stat)
      df2 = pd.concat([df2, df1], ignore_index = True)
      df2.to_csv(file_stat, index = False)