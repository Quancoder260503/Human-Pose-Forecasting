from runner.ProbabilisticRunner import *
from einops import rearrange, reduce
from model.DPMSolver import *

"""
DDPM training class.  
"""

class DiffusionRunner(ProbabilisticRunner):
  def __init__(self, params, dataset, multimodal_dict, logger, tblogger):
    super().__init__(params, dataset, multimodal_dict, logger, tblogger)
    self._logger.info("Training using Diffusion Process")

  def init_algo(self):
    # set up base class for diffusion process
    self.gaussian_diffusion = GaussianDiffusionDDIM(
      noise_schedule = self._params['noise_scheduler'],
      rescale_timesteps = self._params['rescale_timesteps'],
      num_diffusion_steps = self._params['diffusion_num_timesteps'],
      num_sampling_steps = self._params['ddim_num_timesteps'],
      cfg_scale = self._params['cfg_scale'],
      deterministic = self._params['deterministic'],
      enable_complete = self._params['enable_complete'],
    )
    # Are we using DPM-Solver for sampling ? 
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
        algorithm_type = self._params['dpm_algorithm_type'],
      )
  def prepare_batch(self, trajectory_np, mod_prob):
    """
    prepare the training batch, i.e. padding and transforming the observed sequences of motions into DCT domain, also apply the conditional
    dropping strategy to allow CFG training (https://arxiv.org/pdf/2207.12598)
    :param trajectory_np (numpy.ndarray):  [batch_size, total_frames, pose_dim], the raw sequences of motions, 
    where total_frames = num_observed_frames + num_predict_frames
    :param mod_prob (np.float32) : cfg dropping probability 
    :returns: 
      - trajectory (torch.Tensor): [batch_size, total_frames, pose_dim], the raw sequences of motions, 
      where total_frames = num_observed_frames + num_predict_frames
      - trajectory_dct (torch.Tensor): [batch_size, dct_channels, pose_dim], the DCT transformed of the entire sequences of motions
      - trajectory_dct_mod (torch.Tensor): [batch_size, dct_channels, pose_dim], the DCT transformed of the padded observed sequences of motions with 
      conditional dropping applied
      - batch_size (int): the batch size
    """
    trajectory_np = trajectory_np[..., 1:, :].reshape([
      trajectory_np.shape[0],
      self._params['num_observed_frames'] + self._params['num_predict_frames'],
      -1,
    ])
    batch_size = trajectory_np.shape[0] 

    trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32)

    # padding observed sequences of motions to the same length as the entire sequences of motions and then transform into spectral domain  
    trajectory_pad = padding_trajectory(
      trajectory,
      self._params['padding'],
      self._params['idx_pad'],
      self._params['zero_index'],
    )

    trajectory_dct = torch.matmul(self.dct[:self.num_dct_channels], trajectory)
    trajectory_dct_mod = torch.matmul(self.dct[:self.num_dct_channels], trajectory_pad)

    ## Does not creating a large different empirically 
    if self._params['drop_cond_method'] == 'sample':
      drop_mask = torch.rand(batch_size, device = trajectory_dct.device) > mod_prob
      mask = drop_mask.view(batch_size, *([1] * (trajectory_dct.dim() - 1)))
      null_cond = torch.zeros_like(trajectory_dct_mod)
      trajectory_dct_mod = torch.where(mask, null_cond, trajectory_dct_mod)

    elif self._params['drop_cond_method'] == 'batch':
      if np.random.random() > mod_prob:
        trajectory_dct_mod = torch.zeros_like(trajectory_dct_mod)
    else:
      raise NotImplementedError("Unknown conditional dropping method")

    return trajectory, trajectory_dct, trajectory_dct_mod, batch_size

  # Repeating the samples for min over multiples tricks 
  def repeat_for_samples(self, tensor, num_samples):
    tensor = tensor.unsqueeze(1).repeat(1, num_samples, *([1] * (tensor.dim() - 1)))
    return tensor.contiguous().view(-1, *tensor.shape[2:])

  def train_one_epoch(self):
    """Trains for a number of steps before evaluation."""
    for index, (trajectory_np, _) in enumerate(self.train_generator):
      self._optimizer_fn.zero_grad()
      with torch.no_grad():
        trajectory, trajectory_dct, trajectory_dct_mod, batch_size = self.prepare_batch(trajectory_np, self._params['mod_train'])

      num_samples = self._params['num_samples']

      trajectory_dct = self.repeat_for_samples(trajectory_dct, num_samples)
      trajectory_dct_mod = self.repeat_for_samples(trajectory_dct_mod, num_samples)

      # sample the noise, N(0, I)
      noise = torch.randn_like(trajectory_dct)

      # sampling the timesteps
      timesteps = self.sampling_timesteps(batch_size)
      timesteps = timesteps.unsqueeze(1).repeat(1, num_samples).view(-1)

      #sample x_t from q(x_t | x_0) = N(sqrt(alpha_bar) * x_0, (1 - alpha_bar) * I)
      x_t = self.gaussian_diffusion.q_sample(
        x_start = trajectory_dct,
        timesteps = timesteps,
        gaussian_noise = noise
      )

      # model prediction and loss computation
      pred_noise = self._model(x_t, timesteps, trajectory_dct_mod)
      loss = F.mse_loss(pred_noise, noise, reduction = 'none')  # [b * num_samples, dct_channels, pose_dim]
      loss = loss.contiguous().view(batch_size, num_samples, loss.shape[-2], loss.shape[-1])

      loss = reduce(loss, 'b k d c -> b k', 'mean') # mean over dct_channels and pose_dim
      loss = reduce(loss, 'b k -> b', 'min') # min over multi samples trick, used to relax the objective function (prevent the model from hedging over a particular regions) and improve the diversity of the generations
      loss = loss.mean()

      loss.backward() 

      self._optimizer_fn.step() # update the weights 
      self.update_learning_rate(self._global_step, mode = 'stepwise') # change the learning rate in case we getting closer to the local optima
      self.train_losses.update(loss.item())
      self._tblogger.add_scalar('Loss/train', loss.item(), self.curr_epoch)
      self._ema_helper.update_ema(self._ema_model, self._model, step_start_ema = self._params['step_start_ema']) # store EMA weights for higher-quality sampling 
      self._global_step += 1
      # Logging 
      if index % 100 == 0:
        self._logger.info(f"Train objective stats : epoch {self.curr_epoch} | step {index} | avg_loss : {self.train_losses.avg} | learning_rate : {self._params['learning_rate']}")
      del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_np

    self._logger.info('>>> Epoch: {} | Time: {:.2f} | Train Loss: {} | Learning Rate : {:.5f}'.format(
        self.curr_epoch, time.time() - self.start_time, self.train_losses.avg, self._lr_scheduler.get_last_lr()[0]))

  @torch.no_grad()
  def eval_one_epoch(self):
    for (trajectory_np, _) in self.val_generator:
      trajectory, trajectory_dct, trajectory_dct_mod, batch_size = self.prepare_batch(trajectory_np, self._params['mod_test'])

      num_samples = self._params['num_samples']

      trajectory_dct = self.repeat_for_samples(trajectory_dct, num_samples)
      trajectory_dct_mod = self.repeat_for_samples(trajectory_dct_mod, num_samples)

      noise = torch.randn_like(trajectory_dct)

      timesteps = self.sampling_timesteps(batch_size)
      timesteps = timesteps.unsqueeze(1).repeat(1, num_samples).view(-1)

      x_t = self.gaussian_diffusion.q_sample(
        x_start = trajectory_dct,
        timesteps = timesteps,
        gaussian_noise = noise
      )

      pred_noise = self._model(x_t, timesteps, trajectory_dct_mod)
      loss = F.mse_loss(pred_noise, noise, reduction = 'none')  # [b * num_samples, dct_channels, pose_dim]
      loss = loss.contiguous().view(batch_size, num_samples, loss.shape[-2], loss.shape[-1])

      loss = reduce(loss, 'b k d c -> b k', 'mean')
      loss = reduce(loss, 'b k -> b', 'min')
      loss = loss.mean()

      self.val_losses.update(loss.item())
      self._tblogger.add_scalar('Loss/val', loss.item(), self.curr_epoch)
      del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_np

    self._logger.info('>>> Epoch: {} | Time: {:.2f} | Val Loss: {}'.format(
        self.curr_epoch, time.time() - self.start_time, self.val_losses.avg))


  # Sampling function for generating new samples from the trained model. 
  def sampling(self, sampling_model, padded_trajectory_dct, trajectory_dct_mod, sampling_dict, return_intermediate = False):
    sampling_model.eval()
    if self._params['use_dpm_solver']:
      with torch.no_grad():
        return self.dpm_solver.sample(
          cond = trajectory_dct_mod,
          order = self._params['dpm_solver_order'],
          steps = self._params['dpm_solver_num_steps'],
          skip_type = self._params['dpm_solver_skip_type'],
          method = self._params['dpm_solver_method'],
          solver_type = self._params['dpm_solver_type'],
          masks = sampling_dict['mask'],
          return_intermediate = return_intermediate, 
          denoise_to_zero = self._params['dpm_solver_denoise_to_zero']
        )
    else:
      with torch.no_grad():
        return self.gaussian_diffusion.ddim_sample_loop(
          model = sampling_model,
          padded_trajectory_dct = padded_trajectory_dct,
          trajectory_dct_mod = trajectory_dct_mod,
          device = self._params['device'],
          dct_num_channels = self._params['num_dct_channels'],
          dct = self.dct,
          idct = self.idct,
          masks = sampling_dict['mask'],
          return_intermediate = return_intermediate,
        )