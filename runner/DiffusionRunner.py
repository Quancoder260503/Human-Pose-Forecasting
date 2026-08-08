from runner.ProbabilisticRunner import * 

class DiffusionRunner(ProbabilisticRunner): 
  def __init__(self, params, dataset, multimodal_dict, logger, tblogger):
    super().__init__(params, dataset, multimodal_dict, logger, tblogger)
    self._logger.info("Training using Diffusion Process")

  def init_algo(self):
    self.gaussian_diffusion = GaussianDiffusion(
      noise_schedule = self._params['noise_scheduler'],  
      rescale_timesteps = self._params['rescale_timesteps'], 
      num_diffusion_steps = self._params['diffusion_num_timesteps'],
      num_sampling_steps = self._params['ddim_num_timesteps'],
      cfg_scale = self._params['cfg_scale'], 
      use_classifier_free_guidance = self._params['use_classifier_free_guidance'], 
      deterministic = self._params['deterministic'], 
      enable_complete = self._params['enable_complete'], 
    )
  
  def train_one_epoch(self):
    """Trains for a number of steps before evaluation."""
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

        if self._params['drop_cond_method'] == 'sample': 
          batch_size = trajectory_dct.shape[0]  # batch size
          drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > self._params['mod_train']
          mask = drop_mask.view(batch_size, *([1] * (trajectory_dct.dim() - 1)))  
          null_cond = torch.zeros_like(trajectory_dct_mod)  
          trajectory_dct_mod = torch.where(mask, null_cond, trajectory_dct_mod)
        elif self._params['drop_cond_method'] == 'batch':
          if np.random.random() > self._params['mod_train']:
            trajectory_dct_mod = None 
        else:
          raise NotImplementedError("Unknown conditional dropping method")
        
      timesteps = self.sampling_timesteps(self._params['batch_size']) 
      noise = torch.randn_like(trajectory_dct).to(self._params['device'], dtype = torch.float32)
      x_t = self.gaussian_diffusion.q_sample(x_start = trajectory_dct, timesteps = timesteps, gaussian_noise = noise)
      pred_noise = self._model(x_t, timesteps, trajectory_dct_mod)
      loss = self.criterion(pred_noise, noise)
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

        if np.random.random() > self._params['mod_train']:
          trajectory_dct_mod = None 
      
        timesteps = self.sampling_timesteps(self._params['batch_size']) 
        noise = torch.randn_like(trajectory_dct).to(self._params['device'], dtype = torch.float32)
        x_t = self.gaussian_diffusion.q_sample(x_start = trajectory_dct, timesteps = timesteps, gaussian_noise = noise)
        pred_noise = self._model(x_t, timesteps, trajectory_dct_mod)
        loss = self.criterion(pred_noise, noise)
        self.val_losses.update(loss.item())
        self._tblogger.add_scalar('Loss/val', loss.item(), self.curr_epoch)
        del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np 

  def sampling(self, sampling_model, trajectory_dct, trajectory_dct_mod, sampling_dict):
    sampling_model.eval()
    with torch.no_grad():
      return self.gaussian_diffusion.ddim_sample_loop(
        model = sampling_model, 
        trajectory_dct = trajectory_dct, 
        trajectory_dct_mod = trajectory_dct_mod, 
        device = self._params['device'], 
        dct_num_channels = self._params['num_dct_channels'], 
        dct = self.dct, 
        idct = self.idct, 
        masks = sampling_dict['mask'], 
      )[0]