import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import clip_grad_norm_
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
from model.GaussianDiffusion import PolicyGaussianDiffusion
from model.RewardModel import reward_transformer, reward_encoder, reward_gcn, InfoNCE, ConditioningInfoNCE
from model.DiffusionTransformer import diffusion_transformer
from model.ValueFunctionApproximator import get_linear_value_model, get_transformer_value_model
from model.GCNDiffusion import gcn_transformer_diffusion
from runner.RunnerUtils import *
import pandas as pd 
import csv 
import copy 
from Logger import *
import tqdm 
import dataclasses

@dataclasses.dataclass(frozen = False)
class TrainPolicyFunctionData:
  total_policy_loss : float = 0.
  total_ratio : float = 0.
  total_kl_divergence : float = 0.
  total_gradient_norm : float = 0.


class PPORunner(object):
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

    self.diffusion_model.to(self._params['device'])
    self.value_model.to(self._params['device'])

    self.policy_optimizer = self.select_optimizer(
      optimizer_fn = self._params['policy_optimizer'], 
      model_params = self.diffusion_model.parameters(), 
      learning_rate = self._params['policy_learning_rate']
    )

    self.value_optimizer = self.select_optimizer(
      optimizer_fn = self._params['value_optimizer'], 
      model_params = self.value_model.parameters(), 
      learning_rate = self._params['value_learning_rate'],
    )

    self.policy_lr_scheduler = self.get_lr_fn(self.policy_optimizer) 
    self.value_lr_scheduler = self.get_lr_fn(self.value_optimizer)


    self.num_dct_channels = self._params['num_dct_channels']
    self._params['learning_rate_type'] = 'epochwise'

    m_params = filter(lambda p: p.requires_grad, self.diffusion_model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])             
    self._logger.info('[INFO] ({}) Diffusion module has {} parameters!'.format(thisname, nparams))
    
    m_params = filter(lambda p: p.requires_grad, self.value_model.parameters())
    nparams = sum([np.prod(p.size()) for p in m_params])             
    self._logger.info('[INFO] ({}) Value Approximator module has {} parameters!'.format(thisname, nparams))
    
    self._global_step = 1
    self._epoch = -1

    self.train_losses = None 
    self.train_generator = None 
    self.val_losses = None 
    self.val_generator = None 

  def get_random_indices(self, num_indices, sample_size): 
   return np.random.choice(num_indices, size = sample_size, replace = False)

  def collecting_rollouts(self, trajectory_pad_dct, state_dict): 
    for _ in range(self._params['g_step']): 
      with torch.no_grad():
        trajectory_dct, latent_trajectory_dct, log_prob_lst, _ = self.policy_diffusion.forward_collect_traj_ddim(
          model = self.diffusion_model, 
          cond = trajectory_pad_dct, 
          eta = 1.,
        )
        reward_lst = [] 
        for i in range(trajectory_pad_dct.shape[0]): 
          reward_lst.append(self.compute_rewards(trajectory_pad_dct[i], trajectory_dct[i])) 
        reward_lst = torch.stack(reward_lst).detach().cpu()

        for i in range(len(latent_trajectory_dct) - 1): 
          state_dict['cond_trajectory'] = torch.cat((state_dict['cond_trajectory'], trajectory_pad_dct.detach().cpu()))
          state_dict['state'] = torch.cat((state_dict['state'], latent_trajectory_dct[i]))
          state_dict['next_state'] = torch.cat((state_dict['next_state'], latent_trajectory_dct[i + 1]))
          state_dict['timestep'] = torch.cat((state_dict['timestep'], torch.LongTensor([self.policy_diffusion.sampling_indices[i]] * trajectory_pad_dct.shape[0])))
          state_dict['next_timestep'] = torch.cat((state_dict['next_timestep'], torch.LongTensor([self.policy_diffusion.sampling_indices[i + 1]] * trajectory_pad_dct.shape[0])))
          state_dict['final_reward'] = torch.cat((state_dict['final_reward'], reward_lst))
          state_dict['log_prob'] = torch.cat((state_dict['log_prob'], log_prob_lst[i]))

        del (
          trajectory_dct, 
          latent_trajectory_dct, 
          log_prob_lst, 
          reward_lst,
        )
        torch.cuda.empty_cache()


  # TODO : may add 3d distance as another metrics 
  def compute_rewards(self, cond_trajectory, out_trajectory): 
    if isinstance(self.reward_model, InfoNCE): 
      enc_cond_trajectory = self.reward_model.prior_encoder(cond_trajectory)
      enc_out_trajectory = self.reward_model.encoder(out_trajectory) 
      enc_cond_trajectory = F.normalize(enc_cond_trajectory, dim = -1) 
      enc_out_trajectory = F.normalize(enc_out_trajectory, dim = -1) 
      prob = enc_out_trajectory  * enc_out_trajectory / self.reward_model.temperature 
      prob = F.sigmoid(prob)
    elif isinstance(self.reward_model, ConditioningInfoNCE):
      prob = self.reward_model.cond_prob_estimator(out_trajectory, cond_trajectory)
      prob = F.sigmoid(prob) 
    return np.exp(prob.detach().cpu() * self._params['visual_weight'])  
      
  def train_value_func(self, state_dict):
    indices = self.get_random_indices(state_dict['state'].shape[0], self._params['value_batch_size'])
    batch_trajectory_dct = state_dict['state'][indices]
    batch_timestep = state_dict['timestep'][indices]
    batch_final_reward = state_dict['final_reward'][indices]
    batch_trajectory_dct_mod = state_dict['cond_trajectory'][indices]

    pred_value = self.value_model(
      batch_trajectory_dct.cuda().detach(), 
      batch_trajectory_dct_mod.cuda().detach(), 
      batch_timestep.cuda().detach(), 
    )
    batch_final_reward = batch_final_reward.cuda().float() 
    value_loss = F.mse_loss(
      pred_value.float().reshape([self._params['value_batch_size'], 1]), 
      batch_final_reward.cuda().detach().reshape([self._params['value_batch_size'], 1])
    )
    (value_loss / self._params['value_step']).backward()

    del pred_value
    del batch_trajectory_dct
    del batch_timestep
    del batch_final_reward
    del batch_trajectory_dct_mod

    return (value_loss.item() / self._params['value_step']) 
  
  def train_policy_func(self, state_dict, policy_steps, tpfdata, count): 
    with torch.no_grad(): 
      indices = self.get_random_indices(state_dict['state'].shape[0], self._params['policy_batch_size'])
      batch_trajectory_dct = state_dict['state'][indices]
      batch_next_trajectory_dct = state_dict['next_state'][indices]
      batch_timestep = state_dict['timestep'][indices]
      batch_next_timestep = state_dict['next_timestep'][indices]
      batch_final_reward = state_dict['final_reward'][indices]
      batch_trajectory_dct_mod = state_dict['cond_trajectory'][indices]
      batch_log_prob = state_dict['log_prob'][indices]

    log_prob, kl_regularizer = self.policy_diffusion.forward_calculate_log_prob(
      model = self.diffusion_model, 
      cond = batch_trajectory_dct_mod.cuda(), 
      sample = batch_trajectory_dct.cuda(), 
      next_sample = batch_next_trajectory_dct.cuda(), 
      time = batch_timestep.cuda(), 
      next_time = batch_next_timestep.cuda(), 
      model_copy = self.reference_diffusion_model, 
      eta = 1., 
    )

    with torch.no_grad():
      if self._params['value_flag'] == 1:
        approximated_value = self.value_model(
          batch_trajectory_dct.cuda(),
          batch_trajectory_dct_mod.cuda(),
          batch_timestep.cuda()
        ).reshape([self._params['policy_batch_size'], 1])
        advantages = batch_final_reward.cuda().reshape([self._params['policy_batch_size'], 1]) - approximated_value
      else:
        advantages = batch_final_reward.cuda().reshape([self._params['policy_batch_size'], 1])

    ratio = torch.exp(log_prob - batch_log_prob.cuda())
    ratio = torch.clamp(ratio, 1.0 - self._params['ratio_clip'], 1.0 + self._params['ratio_clip'])

    loss = -self._params['reward_weight'] * advantages.detach().float() * ratio.float().reshape([self._params['policy_batch_size'], 1])
    loss = loss.mean()
    if count > self._params['kl_warmup']:
      loss += self._params['kl_weight'] * kl_regularizer.mean()
    loss = loss / self._params['gradient_accumulation_steps']
    loss.backward()

    tpfdata.total_ratio += ratio.mean().item() / policy_steps
    tpfdata.total_kl_divergence += kl_regularizer.mean().item() / policy_steps
    tpfdata.total_policy_loss += loss.item() / policy_steps
      

  def train_one_epoch(self, state_dict):
    max_train_steps = self._params['max_train_steps'] // self._params['policy_step']
    for index, trajectory_np in enumerate(self.train_generator):
      self.diffusion_model.eval() # freeze model weight to sample 
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

      self.collecting_rollouts(trajectory_dct_mod, state_dict)


      total_value_loss = 0.
      self.value_optimizer.zero_grad()
      for _ in range(self._params['value_step']):
        total_value_loss += self.train_value_func(state_dict)

      self.value_optimizer.step()
      print("value_loss", total_value_loss)
      self._tblogger.add_scalar('ValueLoss/value', total_value_loss, self._global_step)
      del total_value_loss
      torch.cuda.empty_cache()

      # train the actor
      self.diffusion_model.train()
      tpfdata = TrainPolicyFunctionData()
      norm = torch.tensor(0.0)
      for _ in range(self._params['policy_step']):
        self.policy_optimizer.zero_grad()
        for _ in range(int(self._params['gradient_accumulation_steps'])):
          self.train_policy_func(
            state_dict = state_dict, 
            policy_steps = self._params['policy_step'], 
            tpfdata = tpfdata,
            count = self._global_step,
          )
        norm = clip_grad_norm_(self.diffusion_model.parameters(), self._params['clip_norm'])

        tpfdata.total_gradient_norm += norm.item() / self._params['policy_step'] 

        self.policy_optimizer.step()
        print(f"count: [{self._global_step} / {max_train_steps}]")
        print("train_reward", torch.mean(state_dict['final_reward']).item())
        self._tblogger.add_scalar('Reward/train_reward', torch.mean(state_dict['final_reward']).item(), self._global_step)
        print("grad norm", tpfdata.total_gradient_norm, "ratio", tpfdata.total_ratio)
        print("kl", tpfdata.total_kl_divergence, "policy_loss", tpfdata.total_policy_loss)
        self._tblogger.add_scalar('Policy/grad_norm', tpfdata.total_gradient_norm, self._global_step)
        self._tblogger.add_scalar('Policy/ratio', tpfdata.total_ratio, self._global_step)
        self._tblogger.add_scalar('Policy/kl', tpfdata.total_kl_divergence, self._global_step)
        self._tblogger.add_scalar('Policy/policy_loss', tpfdata.total_policy_loss, self._global_step)
        torch.cuda.empty_cache()

        self._global_step += 1

  def train(self):
    state_dict = {}
    state_dict['cond_trajectory'] = torch.FloatTensor().to(torch.float32).cpu()
    state_dict['state'] = torch.FloatTensor().to(torch.float32).cpu()
    state_dict['next_state'] = torch.FloatTensor().to(torch.float32).cpu()
    state_dict['timestep'] = torch.LongTensor().cpu()
    state_dict['next_timestep'] = torch.LongTensor().cpu()
    state_dict['final_reward'] = torch.FloatTensor().to(torch.float32).cpu()
    state_dict['log_prob'] = torch.FloatTensor().to(torch.float32).cpu()

    """Main training loop."""
    thisname = self.__class__.__name__
    self._logger.info('=' * 90)
    self._logger.info(f"[INFO] {thisname} | global step : {self._global_step:06d} | epoch : {(self._epoch + 1):06d} | Start training")
    self._logger.info('=' * 90)
    for e in range(self._epoch + 1, self._params['max_epochs']):
      self.curr_epoch = e

      # Train
      self.prepare_train_step()
      self.train_one_epoch(state_dict)
      self.prepare_eval_step()
      self.compute_train_stats()
      self.compute_eval_stats()
      self._logger.info('>>> Epoch: {} | Time: {:.2f} | Policy LR : {:.6f} | Value LR : {:.6f}'.format(
        self.curr_epoch,
        time.time() - self.start_time,
        self.policy_lr_scheduler.get_last_lr()[0],
        self.value_lr_scheduler.get_last_lr()[0],
      ))

      self.update_learning_rate()

      if (e + 1) % self._params['snapshot_saving_steps'] == 0:
        self.save_checkpoint(e)
    self.save_checkpoint(e)

  def save_checkpoint(self, epoch):
    model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
    state = {
      'global_steps' : self._global_step,
      'epoch' : epoch,
      'diffusion_model' : self.diffusion_model.module.state_dict() if hasattr(self.diffusion_model, 'module') else self.diffusion_model.state_dict(),
      'value_model' : self.value_model.module.state_dict() if hasattr(self.value_model, 'module') else self.value_model.state_dict(),
      'policy_optimizer' : self.policy_optimizer.state_dict(),
      'value_optimizer' : self.value_optimizer.state_dict(),
      'policy_lr_scheduler' : self.policy_lr_scheduler.state_dict(),
      'value_lr_scheduler' : self.value_lr_scheduler.state_dict(),
    }
    torch.save(state, model_path)
    self._logger.info(f"[INFO] Saved checkpoint to {model_path} at epoch {epoch}")

  def init_model(self):
    # reward model setup 
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
      self.reward_model = InfoNCE(
        encoder = enc, 
        prior_encoder = prior_enc,
        sample_model = self._sample_model, 
        sampler = self.gaussian_diffusion,
        num_samples = self._params['num_samples'],         
        temperature = self._params['info_nce_temperature']
      )
    elif self._params['reward_model_architecture'] == 'transformer_encoder': 
      enc, prior_enc = reward_encoder(self._params), reward_encoder(self._params)
      self.reward_model = InfoNCE(
        encoder = enc, 
        prior_encoder = prior_enc,
        sample_model = self._sample_model, 
        sampler = self.gaussian_diffusion,
        num_samples = self._params['num_samples'], 
        temperature = self._params['info_nce_temperature']
      )
    else : 
      raise NotImplementedError(f"Reward Model architecture {self._params['reward_model_architecture']} not found")

    assert self._params['reward_model_path'] is not None, "Expected reward model path"
    reward_model_dict = torch.load(self._params['reward_model_path'], weights_only = False, map_location = self._params['device'])
    self.reward_model.load_state_dict(reward_model_dict['model'])

    self.reward_model.eval()
    for p in self.reward_model.parameters(): 
      p.requires_grad_(False)

    # diffusion_model 
    if self._params['model_architecture'] == 'transformer': 
      self.diffusion_model = diffusion_transformer(self._params)
    elif self._params['model_architecture'] == 'gcn_transformer': 
      self.diffusion_model = gcn_transformer_diffusion(self._params)
    else : 
      raise NotImplementedError(f"Model architecture {self._params['model_architecture']} not found")
    assert self._params['diffusion_model_path'] is not None, "Expected reward model path"
    diffusion_model_dict = torch.load(self._params['diffusion_model_path'], weights_only = False, map_location = self._params['device'])
    if self._params['diffusion_load_ema_flag'] == 1: 
      self.diffusion_model.load_state_dict(diffusion_model_dict['ema_model'])
    else : 
      self.diffusion_model.load_state_dict(diffusion_model_dict['model'])


    self.reference_diffusion_model = copy.deepcopy(self.diffusion_model)
    self.reference_diffusion_model.eval()
    for p in self.reference_diffusion_model.parameters():
      p.requires_grad_(False)

    # diffusion_policy 
    self.policy_diffusion = PolicyGaussianDiffusion(
      noise_schedule = self._params['noise_scheduler'],  
      rescale_timesteps = self._params['rescale_timesteps'], 
      num_diffusion_steps = self._params['diffusion_num_timesteps'],
      num_sampling_steps = self._params['ddim_num_timesteps'],
      cfg_scale = self._params['cfg_scale'], 
      use_classifier_free_guidance = self._params['use_classifier_free_guidance'], 
      deterministic = self._params['deterministic'], 
      enable_complete = self._params['enable_complete'], 
    )

    # value_function_approximator
    if self._params['value_model_architecture'] == 'mlp': 
      self.value_model = get_linear_value_model(self._params)
    elif self._params['value_model_architecture'] == 'transformer': 
      self.value_model = get_transformer_value_model(self._params)
    else : 
      raise NotImplementedError(f"Value Model architecture {self._params['value_model_architecture']} not found")
    
  def get_lr_fn(self, optimizer_fn):
    """Creates the function to be used to generate the learning rate."""
    if self._params['learning_rate_fn'] == 'step':
      return torch.optim.lr_scheduler.StepLR(
        optimizer = optimizer_fn, 
        step_size = self._params['lr_step_size'], 
        gamma = self._params['step_gamma'], 
      )
    elif self._params['learning_rate_fn'] == 'exponential':
      return torch.optim.lr_scheduler.ExponentialLR(
        optimizer = optimizer_fn, 
        gamma = self._params['exp_gamma'], 
      )
    elif self._params['learning_rate_fn'] == 'linear':
      lr0, T = self.learning_rate, self.max_epochs
      lrT = lr0 * 0.5
      m = (lrT - 1) / T
      lambda_fn =  lambda epoch: m * epoch + 1.0
      return torch.optim.lr_scheduler.LambdaLR(
        optimizer = optimizer_fn, 
        lr_lambda = lambda_fn
      )
    elif self._params['learning_rate_fn'] == 'decay':
      return torch.optim.lr_scheduler.LinearLR(
        optimizer = optimizer_fn,
        total_iters = self._params['max_epochs'],
        start_factor = 1.0,
        end_factor = self._params['end_learning_rate'] / self._params['learning_rate'],
      )
    elif self._params['learning_rate_fn'] == 'milestones':
      return torch.optim.lr_scheduler.MultiStepLR(
        optimizer = optimizer_fn, 
        milestones = self._params['milestones'],
        gamma = self._params['milestone_gamma']
      ) 
    else:
      raise ValueError('Unknown learning rate function: {}'.format(self._params['learning_rate_fn']))

  def prepare_train_step(self):
    self.diffusion_model.train()
    self.value_model.train()
    self.train_generator = self._dataset['train'].sampling_generator( 
      num_samples = self._params['num_train_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time()
    self.train_losses = AverageMeter()
    self._logger.info(f'Starting training at epoch {self.curr_epoch}')

  def prepare_eval_step(self): 
    self.diffusion_model.eval()
    self.value_model.eval()
    self.val_generator = self._dataset['test'].sampling_generator( 
      num_samples = self._params['num_val_data_samples'], 
      batch_size = self._params['batch_size'], 
    )
    self.start_time = time.time() 
    self.val_losses = AverageMeter()
    self._logger.info(f'Starting evaluation at epoch {self.curr_epoch}')

  def update_learning_rate(self):
    self.value_lr_scheduler.step()
    self.policy_lr_scheduler.step()
    
  def select_optimizer(self, optimizer_fn, model_params, learning_rate): 
    if optimizer_fn == 'AdamW': 
      return torch.optim.AdamW(
        params = model_params,  
        lr = learning_rate, 
        betas = self._params['betas'], 
      )
    if optimizer_fn == 'Adam':
      return torch.optim.Adam(
        params = model_params, 
        lr = learning_rate, 
        betas = self._params['betas'], 
      )
    elif optimizer_fn == 'RMSProp': 
      return torch.optim.RMSprop(
        params = model_params, 
        lr = learning_rate, 
      )
    
    elif optimizer_fn == 'SGD': 
      return torch.optim.SGD(
        params = model_params, 
        lr = learning_rate, 
      )
    else : 
      raise NotImplementedError(f"Optimizer {optimizer_fn} not supported")

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
    trajectory = trajectory.reshape([trajectory.shape[0], -1, trajectory.shape[-1]]).transpose(1, 2) # [n_samples, frame, n_joints * joint_dim]


    mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(trajectory, self._params, 'metrics', self.dct) 
    sampled_motion = self.sampling(sampling_model, traj_dct, traj_dct_mod, mode_dict) #

    result_trajectory = torch.matmul(self.idct[:, :self._params['num_dct_channels']], sampled_motion)
    result_trajectory = result_trajectory.cpu().numpy()
    result_trajectory = result_trajectory[None, ...] # [1, K, 125, 48]
    return result_trajectory


  def compute_train_stats(self): 
    if self.curr_epoch % self._params['save_gif_interval'] == 0: 
      generated_pose = self.pose_generator(
        sampling_model = self.diffusion_model, 
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
        sampling_model = self.diffusion_model, 
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
      self.compute_stats(self.diffusion_model, self._multimodal_dict)