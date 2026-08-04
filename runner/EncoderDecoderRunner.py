import matplotlib.pyplot as plt
import numpy as np
import random 
from torch.nn.utils import clip_grad_norm_
import sys
import numpy as np
import json
import os
import argparse
import time
from abc import abstractmethod
import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.Deterministic.Transformer import *
from runner.WarmupScheduler import *
from data.Dataset3DUtils    import * 

class EncoderDecoderRunner(object):
  def __init__(
    self,
    params,
    train_dataset_fn = None,
    eval_dataset_fn  = None,
    pose_encoder_fn  = None,
    pose_decoder_fn  = None, 
  ):
    self._params = params
    self._train_dataset_fn = train_dataset_fn
    self._eval_dataset_fn = eval_dataset_fn
    self._visualize = False
    thisname = self.__class__.__name__
    self._norm_stats = train_dataset_fn.dataset._norm_stats
    self.init_model(pose_encoder_fn, pose_decoder_fn)
    self._loss_fn = self.layerwise_loss_fn
    self._model.to(self._params['device'])
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.finetune_init()

    self._lr_db_curve = []
    self._params['learning_rate_type'] = 'epochwise'

    self.evaluate_fn = self.evaluate_h36m 

    #   if self.dataset == 'ntu_rgbd':
    #       self.evaluate_fn = self.evaluate_nturgbd
    #  # elif self.dataset == 'cmu_mocap':
    #       #self.evaluate_fn = self.evaluate_amass

    self._writer = SummaryWriter(os.path.join(self._params['model_prefix'], 'tf_logs'))

    m_params = filter(lambda p: p.requires_grad, self._model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])             
    
    print('[INFO] ({}) This module has {} parameters!'.format(thisname, nparams))
    print('[INFO] ({}) Intializing ModelFn with params'.format(thisname))
    for k,v in self._params.items():
        print('[INFO] ({}) {}: {}'.format(thisname, k, v))
    
    self._global_step = 1
    self._epoch = -1
    self.finetune_init()


  def finetune_init(self):
    if self._params['finetuning_ckpt'] is not None:
      print('[INFO] (finetune_model) Finetuning from:', self._params['finetuning_ckpt'])
      #self._model.load_state_dict(torch.load(self._params['finetuning_ckpt'], map_location = self._params['device']))

      model_dict = torch.load(self._params['finetuning_ckpt'], weights_only = False, map_location = self._params['device'])
      
      self._epoch = model_dict['epoch']
      self._global_steps = model_dict['global_steps']
      self._model.load_state_dict(model_dict['model'])
      self._optimizer_fn.load_state_dict(model_dict['optimizer'])
      self._lr_scheduler.load_state_dict(model_dict['lr_scheduler'])



  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()
    if self._params['warmup_epochs'] > 0:
      self._lr_scheduler = GradualWarmupScheduler(
        self._optimizer_fn, 
        multiplier = 1, 
        total_epoch = self._params['warmup_epochs'], 
        after_scheduler = self._lr_scheduler
      )


  def get_lr_fn(self):
    """Creates the function to be used to generate the learning rate."""
    if self._params['learning_rate_fn'] == 'step':
      return torch.optim.lr_scheduler.StepLR(
        self._optimizer_fn, 
        step_size = self._params['lr_step_size'], 
        gamma     = self._params['step_gamma'], 
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
    else:
      raise ValueError('Unknown learning rate function: {}'.format(self._params['learning_rate_fn']))


  def print_logs(self, step_loss, current_step):
    print("[INFO] global {3:06d}; step {0:04d}; step_loss: {1:.4f}; lr: {2:.2e} {4:s}".\
      format(current_step, step_loss, self._params['learning_rate'], self._global_step, '')
    )


  def train_one_epoch(self, epoch):
    """Trains for a number of steps before evaluation."""
    epoch_loss = 0
    N = len(self._train_dataset_fn)
    for current_step, sample in enumerate(self._train_dataset_fn):
      self._optimizer_fn.zero_grad()
      for k in sample.keys():
        if k == 'actions' or k == 'decoder_outputs_euler' or k == 'action_str':
          continue
        sample[k] = sample[k].to(self._params['device'])

      decoder_pred = self._model(sample['encoder_inputs'], sample['decoder_inputs'])

      pose_loss = self.layerwise_loss_fn(
        decoder_pred = decoder_pred[0], 
        decoder_gt   = sample['decoder_outputs'], 
      )

      epoch_loss += pose_loss.item()

      pose_loss.backward()
      if self._params['max_gradient_norm'] > 0:
        clip_grad_norm_(self._model.parameters(), self._params['max_gradient_norm'] )
      
      self._optimizer_fn.step()

      if current_step % 10 == 0:
        pose_loss = pose_loss.cpu().data.numpy()
        self.print_logs(pose_loss, current_step)

      self.update_learning_rate(self._global_step, mode = 'stepwise')
      self._global_step += 1

    return epoch_loss / N

  def train(self):
    """Main training loop."""
    self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]
    thisname = self.__class__.__name__
    print('=' * 90)
    print(f"[INFO] {thisname} | global step : {self._global_step:06d} | epoch : {(self._epoch + 1):06d} | Start training")
    print('=' * 90)
    for e in range(self._epoch + 1, self._params['max_epochs']):
      self._scalars = {}
      self._model.train()
      start_time = time.time()
      epoch_loss = self.train_one_epoch(e)


      self._scalars['epoch_loss'] = epoch_loss
      print("epoch {0:04d}; epoch_loss: {1:.4f}".format(e, epoch_loss))

      _time = time.time() - start_time

      self._model.eval()
      eval_loss = self.evaluate_fn(e, _time)


      self._scalars['eval_loss'] = eval_loss
      print("[INFO] ({}) Epoch {:04d}; eval_loss: {:.4f}; lr: {:.2e}".format(thisname, e, eval_loss, self._params['learning_rate']))

      self.write_summary(e)
      model_path = os.path.join(self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
      if (e + 1) % self._params['snapshot_saving_steps'] == 0:
        state = {
          'global_steps'   : self._global_step,
          'epoch'          : e,
          'model'          : self._model.module.state_dict() if hasattr(self._model, 'module') else self._model.state_dict(),
          'optimizer'      : self._optimizer_fn.state_dict(),
          'lr_scheduler'   : self._lr_scheduler.state_dict(), 
        }
        torch.save(state, model_path)

      self.update_learning_rate(e, mode = 'epochwise')
    # save the last one
    
    model_path = os.path.join(self._params['model_prefix'], 'models', 'ckpt_epoch_%04d.pt'%e)
    state = {
      'global_steps'   : self._global_step,
      'epoch'          : e,
      'model'          : self._model.module.state_dict() if hasattr(self._model, 'module') else self._model.state_dict(),
      'optimizer'      : self._optimizer_fn.state_dict(),
      'lr_scheduler'   : self._lr_scheduler.state_dict(), 
    }
    torch.save(state, model_path)
    # self.flush_curves()

  def write_summary(self, epoch):
    self._writer.add_scalars(
        'loss/recon_loss', 
        {'train':self._scalars['epoch_loss'], 'eval': self._scalars['eval_loss']}, 
        epoch
    )

    # write scalars for H36M dataset prediction style
    action_ = self._train_dataset_fn.dataset._monitor_action
    if 'ms_eval_loss' in self._scalars.keys():
      range_len = len(self._scalars['ms_eval_loss'][action_])
      # range_len = len(self._ms_range)
      ms_dict = {str(self._ms_range[i]): self._scalars['ms_eval_loss'][action_][i] 
                for i in range(range_len)}
      ms_e = np.concatenate([np.array(v).reshape(1,range_len) 
                            for k,v in self._scalars['ms_eval_loss'].items()], axis=0)
      self._writer.add_scalars('ms_loss/eval-'+action_, ms_dict, epoch)

      ms_e = np.mean(ms_e, axis = 0)  # (n_actions)
      self._time_range_eval.append(np.expand_dims(ms_e, axis=0)) # (1, n_actions)
      all_ms = {str(self._ms_range[i]): ms_e[i] for i in range(len(ms_e))}
      self._writer.add_scalars('ms_loss/eval-all', all_ms, epoch)

      self._writer.add_scalar('MSRE/msre_eval', self._scalars['msre'], epoch)
      self._writer.add_scalars('time_range/eval', 
          {'short-term':np.mean(ms_e[:4]), 'long-term':np.mean(ms_e)}, epoch)

    if 'mAP' in self._scalars.keys():
      self._writer.add_scalar('mAP/mAP', self._scalars['mAP'], epoch)

    if 'MPJPE' in self._scalars.keys():
      self._writer.add_scalar('MPJPE/MPJPE', self._scalars['MPJPE'], epoch)

  def print_range_summary(self, action, mean_mean_errors):
    mean_eval_error = []
    # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
    print("{0: <16} |".format(action), end="")
    for ms in [1, 3, 7, 9, 13, 24]:
      if self._params['target_seq_len'] >= ms + 1:
        print(" {0:.3f} |".format(mean_mean_errors[ms]), end = "")
        mean_eval_error.append(mean_mean_errors[ms])
      else:
        print("   n/a |", end="")
    print()
    return mean_eval_error

  def print_table_header(self):
    print()
    print("{0: <16} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
      print(" {0:5d} |".format(ms), end="")
    print()

  def flush_curves(self):
    path_ = os.path.join(self._params['model_prefix'], 'loss_info')
    os.makedirs(path_, exist_ok = True)
    path_ = os.path.join(path_, 'eval_time_range.npy')
    np.save(path_, np.concatenate(self._time_range_eval, axis=0))
    path_ = os.path.join(path_, 'lr_schedule.npy')
    np.save(path_, np.array(self._lr_db_curve))

  def update_learning_rate(self, epoch_step, mode = 'stepwise'):
    """Update learning rate handler updating only when the mode matches."""
    if self._params['learning_rate_type'] == mode:
      self._lr_scheduler.step(epoch_step)
      self._writer.add_scalar('learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]


  def validation_srnn_ms(self, sample, decoder_pred):
  # the data was flatened from a sequence of size 
  # [n_actions, n_seeds, target_length, pose_size]
    n_actions = len(self._params['action_subset'])
    seq_shape = (n_actions, self._params['eval_num_seeds'], self._params['target_seq_len'], self._params['pose_dim'])
    srnn_gts_euler = sample['decoder_outputs_euler']
    decoder_pred_ = decoder_pred.cpu().numpy()
    decoder_pred_ = decoder_pred_.reshape(seq_shape)
    mean_eval_error_dict = {}

    self.print_table_header()
    eval_ms_mean = []
    for ai, action in enumerate(sample['actions']):
      action = action[0]
      decoder_pred =  decoder_pred_[ai, :, :, :]        
      srnn_pred_euler = self._eval_dataset_fn.dataset.post_process_to_euler(decoder_pred)
      mean_errors = np.zeros((self._params['eval_num_seeds'], self._params['target_seq_len']))

      for i in np.arange(self._params['eval_num_seeds']):
        # seq_len x complete_pose_dim (H36M==99)
        eulerchannels_pred = srnn_pred_euler[i]

        # n_seeds x seq_len x complete_pose_dim (H36M==96)
        action_gt = srnn_gts_euler[action]
        # seq_len x complete_pose_dim (H36M==96)
        gt_i = np.copy(action_gt.squeeze()[i].numpy())

        # Only remove global rotation. Global translation was removed before
        gt_i[:, 0:3] = 0
        

        # (seq_len, 96) - (seq_len, 96)
        idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]
        euc_error = np.power(gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)

        # shape: seq_len
        euc_error = np.sum(euc_error, 1)
        # shape: seq_len
        euc_error = np.sqrt(euc_error)
        mean_errors[i,:] = euc_error

      # This is simply the mean error over the eval_num_seeds examples
      # with shape [eval_num_seeds]
      mean_mean_errors = np.mean(mean_errors, 0)
      mean_eval_error_dict[action] = self.print_range_summary(action, mean_mean_errors)

    return mean_eval_error_dict



  @torch.no_grad()
  def evaluate_h36m(self, epoch, curr_time):
    """Evaluation loop."""
    eval_loss = 0

    sample = next(iter(self._eval_dataset_fn))
    # for j, sample in enumerate(self._eval_dataset_fn):
    for k in sample.keys():
      if (k == 'decoder_outputs_euler') or (k == 'actions'):
        continue
      sample[k] = sample[k].squeeze().to(self._params['device'])

    decoder_pred = self._model(sample['encoder_inputs'], sample['decoder_inputs'])

    srnn_loss = self.layerwise_loss_fn(
      decoder_pred = decoder_pred[0], 
      decoder_gt   = sample['decoder_outputs'], 
    )

    # [batch_size, sequence_length, pose_dim]
    decoder_pred = decoder_pred[0][-1]
    # [batch_size, sequence_length, pose_dim]
    msre_ = decoder_pred - sample['decoder_outputs']
    # [batch_size, sequence_length]
    msre_ = torch.sqrt(torch.sum(msre_ * msre_, dim = -1))
    msre_ = msre_.mean().item()

    eval_loss = srnn_loss 
    # run validation on different ranges
    mean_eval_error_dict = self.validation_srnn_ms(sample, decoder_pred)

    self._scalars['ms_eval_loss'] = mean_eval_error_dict
    self._scalars['msre'] = msre_

    return eval_loss

  def loss_l2(self, decoder_pred, decoder_gt):
    return torch.MSELoss(reduce = 'mean')(decoder_pred, decoder_gt)

  def smooth_l1(self, decoder_pred, decoder_gt):
    l1loss = torch.nn.SmoothL1Loss(reduction='mean')
    return l1loss(decoder_pred, decoder_gt)

  def loss_l1(self, decoder_pred, decoder_gt):
    return torch.nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

  def select_loss_fn(self):
    if self._params['loss_fn'] == 'mse':
      return self.loss_mse
    elif self._params['loss_fn'] == 'smoothl1':
      return self.smooth_l1
    elif self._params['loss_fn'] == 'l1':
      return self.loss_l1
    else:
      raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

  def layerwise_loss_fn(self, decoder_pred, decoder_gt):
    pose_loss = 0.0
    loss_fn = self.select_loss_fn()

    if not isinstance(decoder_pred, (list, tuple)):
      decoder_pred = [decoder_pred]

    for l in range(len(decoder_pred)):
      pose_loss += loss_fn(decoder_pred[l], decoder_gt)

    pose_loss = pose_loss/len(decoder_pred)
    return pose_loss

  def init_model(self, pose_encoder_fn = None, pose_decoder_fn = None):
    if self._params['model_type'] == 'transformer':
      self._model = get_transformer_model(
        self._params, 
        pose_encoder_fn, 
        pose_decoder_fn, 
      )
    else : 
      raise NotImplementedError('Would be added later')

  def select_optimizer(self): 
    if self._params['optimizer'] == 'AdamW': 
      return torch.optim.AdamW(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
        weight_decay = self._params['weight_decay'],
        betas = self._params['betas'], 
      )
    
    elif self._params['optimizer'] == 'RMSProp': 
      return torch.optim.RMSprop(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
        weight_decay = self._params['weight_decay'], 
      )
    
    elif self._params['optimizer'] == 'SGD': 
      return torch.optim.SGD(
        params = self._model.parameters(), 
        lr = self._params['learning_rate'], 
        weight_decay = self._params['weight_decay'], 
      )
    else : 
      raise NotImplementedError(f'Optimizer {self._params['optimizer']} not supported')
