from runner.ProbabilisticRunner import * 
from einops import rearrange, reduce
from model.DPMSolver import *
"""
Todo : construct loss to pose space similar to that of GSPS for exploration
"""

class DiffusionRunner(ProbabilisticRunner): 
  def __init__(self, params, dataset, multimodal_dict, logger, tblogger):
    super().__init__(params, dataset, multimodal_dict, logger, tblogger)
    self._logger.info("Training using Diffusion Process")
    self.criterion = nn.MSELoss()

  def init_algo(self):
    self.gaussian_diffusion = GaussianDiffusionDDIM(
      noise_schedule = self._params['noise_scheduler'],  
      rescale_timesteps = self._params['rescale_timesteps'], 
      num_diffusion_steps = self._params['diffusion_num_timesteps'],
      num_sampling_steps = self._params['ddim_num_timesteps'],
      cfg_scale = self._params['cfg_scale'], 
      deterministic = self._params['deterministic'], 
      enable_complete = self._params['enable_complete'], 
    )
    if self._params['use_dpm_solver']:
      self._logger.info("Using DPM-Solver for sampling")
      noise_scheduler = NoiseScheduleVP(
        schedule = 'discrete', 
        betas = self.gaussian_diffusion.betas, 
        alphas_cumprod = self.gaussian_diffusion.alphas_cumprod
      )
      self.dpm_solver = DPMSolver(
        noise_schedule = noise_scheduler, 
        guidance_scale = self._params['cfg_scale'],
        gaussian_process = self.gaussian_diffusion, 
        model = self._model, 
        num_dct_channels = self._params['num_dct_channels'],
        dct = self.dct, 
        idct = self.idct, 
        enable_complete = self._params['enable_complete'],
        algorithm_type = self._params['dpm_solver_type'], 
      )
  
  def train_one_epoch(self):
    """Trains for a number of steps before evaluation."""
    for index, (trajectory_np, action_np) in enumerate(self.train_generator):
      self._optimizer_fn.zero_grad()
      with torch.no_grad():
        trajectory_np = trajectory_np[..., 1:, :].reshape([
          trajectory_np.shape[0], 
          self._params['num_observed_frames'] + self._params['num_predict_frames'], 
          -1, 
        ]) 
        trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32) 

        if action_np is not None: 
          actions = torch.tensor(action_np, device = self._params['device'], dtype = torch.long)

        trajectory_pad = padding_trajectory(
          trajectory, 
          self._params['padding'], 
          self._params['idx_pad'], 
          self._params['zero_index'],
        )

        trajectory_dct = torch.matmul(self.dct[:self.num_dct_channels], trajectory) 
        trajectory_dct_mod = torch.matmul(self.dct[:self.num_dct_channels], trajectory_pad) 

        if self._params['drop_cond_method'] == 'sample': 
          batch_size = trajectory_dct.shape[0]  # batch size
          drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > self._params['mod_train']
          mask = drop_mask.view(batch_size, *([1] * (trajectory_dct.dim() - 1)))  
          null_cond = torch.zeros_like(trajectory_dct_mod)  
          trajectory_dct_mod = torch.where(mask, null_cond, trajectory_dct_mod)

          if action_np is not None: 
            act_drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > self._params['mod_act_train']
            act_mask = act_drop_mask.view(batch_size, *([1] * (actions.dim() - 1)))  
            null_act = torch.zeros_like(actions)  
            actions = torch.where(act_mask, null_act, actions)

          
        elif self._params['drop_cond_method'] == 'batch':
          if np.random.random() > self._params['mod_train']:
            trajectory_dct_mod = None 
          if np.random.random() > self._params['mod_act_train'] and action_np is not None: 
            actions = None 
        else:
          raise NotImplementedError("Unknown conditional dropping method")

      # the idea to expand APD, is to generate multi sample 
      num_samples = self._params['num_samples']

      noise = torch.randn_like(trajectory_dct).to(self._params['device'], dtype = torch.float32)

      trajectory_dct = trajectory_dct.unsqueeze(1).repeat(1, num_samples, 1, 1)
      trajectory_dct = trajectory_dct.contiguous().view(-1, trajectory_dct.shape[-2], trajectory_dct.shape[-1])


      trajectory_dct_mod = trajectory_dct_mod.unsqueeze(1).repeat(1, num_samples, 1, 1)
      trajectory_dct_mod = trajectory_dct_mod.contiguous().view(-1, trajectory_dct_mod.shape[-2], trajectory_dct_mod.shape[-1])

            
      noise = noise.unsqueeze(1).repeat(1, num_samples, 1, 1)
      noise = noise.contiguous().view(-1, noise.shape[-2], noise.shape[-1])
      
      timesteps = self.sampling_timesteps(self._params['batch_size'])
      timesteps = timesteps.unsqueeze(1).repeat(1, num_samples).view(-1) 


      x_t = self.gaussian_diffusion.q_sample(
        x_start = trajectory_dct, 
        timesteps = timesteps,
        gaussian_noise = noise
      )

      pred_noise = self._model(x_t, timesteps, trajectory_dct_mod, None if action_np is None else actions) 
      loss = F.mse_loss(pred_noise, noise, reduction = 'none') # [b * num_samples, dct_channels, pose_dim]
      loss = loss.contiguous().view(-1, num_samples, loss.shape[-2], loss.shape[-1])

      loss = reduce(loss, 'b k d c -> b k', 'mean')
      loss = reduce(loss, 'b k -> b', 'min')
      loss = loss.mean()


      loss.backward()
      self._optimizer_fn.step()
      self.update_learning_rate(self._global_step, mode = 'stepwise')
      self.train_losses.update(loss.item())
      self._tblogger.add_scalar('Loss/train', loss.item(), self.curr_epoch) 
      self._ema_helper.update_ema(self._ema_model, self._model, step_start_ema = self._params['step_start_ema'])
      self._global_step += 1 
      if index % 100 == 0: 
        self._logger.info(f"Train objective stats : epoch {self.curr_epoch} | step {index} | avg_loss : {self.train_losses.avg} | learning_rate : {self._params['learning_rate']}")
      del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np 

    self._logger.info('>>> Epoch: {} | Time: {:.2f} | Train Loss: {} | Learning Rate : {:.5f}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.train_losses.avg, self._lr_scheduler.get_last_lr()[0]))


  
  @torch.no_grad()
  def eval_one_epoch(self): 
    for (trajectory_np, action_np) in self.val_generator:
      with torch.no_grad():
        trajectory_np = trajectory_np[..., 1:, :].reshape([
          trajectory_np.shape[0], 
          self._params['num_observed_frames'] + self._params['num_predict_frames'], 
          -1, 
        ]) 
        trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32) 
        if action_np is not None:
         actions = torch.tensor(action_np, device = self._params['device'], dtype = torch.long)
        trajectory_pad = padding_trajectory(
          trajectory, 
          self._params['padding'], 
          self._params['idx_pad'], 
          self._params['zero_index'],
        )

        trajectory_dct = torch.matmul(self.dct[:self.num_dct_channels], trajectory) 
        trajectory_dct_mod = torch.matmul(self.dct[:self.num_dct_channels], trajectory_pad) 

        if self._params['drop_cond_method'] == 'sample': 
          batch_size = trajectory_dct.shape[0]  # batch size
          drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > self._params['mod_test']
          mask = drop_mask.view(batch_size, *([1] * (trajectory_dct.dim() - 1)))  
          null_cond = torch.zeros_like(trajectory_dct_mod)  
          trajectory_dct_mod = torch.where(mask, null_cond, trajectory_dct_mod)

          if action_np is not None: 
            act_drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > self._params['mod_act_test']
            act_mask = act_drop_mask.view(batch_size, *([1] * (actions.dim() - 1)))  
            null_act = torch.zeros_like(actions)  
            actions = torch.where(act_mask, null_act, actions)
    
              
        elif self._params['drop_cond_method'] == 'batch':
          if np.random.random() > self._params['mod_test']:
            trajectory_dct_mod = None 
          if np.random.random() > self._params['mod_act_test'] and action_np is not None:
            actions = None 
        else:
          raise NotImplementedError("Unknown conditional dropping method")
                

        num_samples = self._params['num_samples']

        noise = torch.randn_like(trajectory_dct).to(self._params['device'], dtype = torch.float32)
        
        trajectory_dct = trajectory_dct.unsqueeze(1).repeat(1, num_samples, 1, 1)
        trajectory_dct = trajectory_dct.contiguous().view(-1, trajectory_dct.shape[-2], trajectory_dct.shape[-1])
  
  
        trajectory_dct_mod = trajectory_dct_mod.unsqueeze(1).repeat(1, num_samples, 1, 1)
        trajectory_dct_mod = trajectory_dct_mod.contiguous().view(-1, trajectory_dct_mod.shape[-2], trajectory_dct_mod.shape[-1])
  

        noise = noise.unsqueeze(1).repeat(1, num_samples, 1, 1)
        noise = noise.contiguous().view(-1, noise.shape[-2], noise.shape[-1])
              
        timesteps = self.sampling_timesteps(self._params['batch_size'])
        timesteps = timesteps.unsqueeze(1).repeat(1, num_samples).view(-1) 

        x_t = self.gaussian_diffusion.q_sample(
          x_start = trajectory_dct, 
          timesteps = timesteps,
          gaussian_noise = noise
        )

        pred_noise = self._model(x_t, timesteps, trajectory_dct_mod, None if action_np is None else actions) 
        loss = F.mse_loss(pred_noise, noise, reduction = 'none') # [b * num_samples, dct_channels, pose_dim]
        loss = loss.contiguous().view(-1, num_samples, loss.shape[-2], loss.shape[-1])

        loss = reduce(loss, 'b k d c -> b k', 'mean')
        loss = reduce(loss, 'b k -> b', 'min')
        loss = loss.mean()

        self.val_losses.update(loss.item())
        self._tblogger.add_scalar('Loss/val', loss.item(), self.curr_epoch)
        del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np

    self._logger.info('>>> Epoch: {} | Time: {:.2f} | Val Loss: {}'.format(\
        self.curr_epoch, time.time() - self.start_time, self.val_losses.avg))



  def sampling(self, sampling_model, trajectory_dct, trajectory_dct_mod, sampling_dict):
    sampling_model.eval()
    if self._params['use_dpm_solver']: 
      with torch.no_grad():
        return self.dpm_solver.sample(
          cond = trajectory_dct_mod, 
          action = None,
          order = self._params['dpm_solver_order'],
          steps = self._params['dpm_solver_num_steps'], 
          skip_type = self._params['dpm_solver_skip_type'],
          method = self._params['dpm_solver_method'],
          solver_type = self._params['dpm_solver_type'], 
          masks = sampling_dict['mask'], 
          denoise_to_zero = self._params['dpm_solver_denoise_to_zero']
        )
    else: 
      with torch.no_grad():
        return self.gaussian_diffusion.ddim_sample_loop(
          model = sampling_model, 
          trajectory_dct = trajectory_dct, 
          trajectory_dct_mod = trajectory_dct_mod, 
          device = self._params['device'],
          actions = None,  
          dct_num_channels = self._params['num_dct_channels'], 
          dct = self.dct, 
          idct = self.idct, 
          masks = sampling_dict['mask'], 
        )
    
