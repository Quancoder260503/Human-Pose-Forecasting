from runner.ProbabilisticRunner import * 

class FlowMatchingRunner(ProbabilisticRunner):
  def __init__(self, params, dataset, multimodal_dict, logger, tblogger):
    super().__init__(params, dataset, multimodal_dict, logger, tblogger)
    self._logger.info(">>> Training using Flow Matching algorithm")
    self.get_ode_solver_class()

  def get_ode_solver_class(self):
    if self._params['ode_solver'] == 'euler':
      self.solver_class = ODESolver.EulerSolver
    elif self._params['ode_solver'] == 'mid_point':
      self.solver_class = ODESolver.MidPointSolver
    elif self._params['ode_solver'] == 'heun_2':
      self.solver_class = ODESolver.Heun2Solver
    elif self._params['ode_solver'] == 'heun_3':
      self.solver_class = ODESolver.Heun3Solver
    elif self._params['ode_solver'] == 'rk4':
      self.solver_class = ODESolver.RungeKutta4Solver
    else : 
      raise NotImplementedError("ODE solver not implemented")
    

  def init_algo(self):
    if self._params['path_scheduler'] == 'linear': 
      self.probability_path = AffineProbabilityPath(scheduler = ConditionalOptimalTransportScheduler())
    elif self._params['path_scheduler'] == 'cosine': 
      self.probability_path = AffineProbabilityPath(scheduler = CosineScheduler())
    else : 
      raise NotImplementedError('Path scheduler not implemented for now')
     

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
      x_1 = trajectory_dct
      x_0 = torch.randn_like(x_1).to(self._params['device'])

      path_sample = self.probability_path.sample(x_0 = x_0, x_1 = x_1, t = timesteps)
      x_t, dx_t = path_sample.x_t, path_sample.dx_t

      loss = torch.pow(self._model(x_t, timesteps, trajectory_dct_mod) - dx_t, 2).mean()
    

      loss.backward()
      # if self._params['max_gradient_norm'] > 0:
      #   clip_grad_norm_(self._model.parameters(), self._params['max_gradient_norm'])
      
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

        x_1 = trajectory_dct
        x_0 = torch.randn_like(x_1).to(self._params['device'])

        path_sample = self.probability_path.sample(x_0 = x_0, x_1 = x_1, t = timesteps)
        x_t, dx_t = path_sample.x_t, path_sample.dx_t
        loss = torch.pow(self._model(x_t, timesteps, trajectory_dct_mod) - dx_t, 2).mean()
        
        self.val_losses.update(loss.item())
        self._tblogger.add_scalar('Loss/val', loss.item(), self.curr_epoch)

        del loss, trajectory, trajectory_dct, trajectory_dct_mod, trajectory_pad, trajectory_np 

  def sampling(self, sampling_model, trajectory_dct, trajectory_dct_mod, sampling_dict):
    sampling_model.eval()
    with torch.no_grad():
      return self.ode_sample_loop(
        model = sampling_model, 
        trajectory_dct = trajectory_dct, 
        trajectory_dct_mod = trajectory_dct_mod, 
        masks = sampling_dict['mask'], 
      )

  def ode_sample_loop(self, model, trajectory_dct, trajectory_dct_mod, masks): 
    ode_solver = self.solver_class(
      velo_func = model, 
      interpolation = self._params['interpolation'],
      path_refiniment_steps = self._params['path_refinement_steps'], 
    )
    timesteps = torch.linspace(0, 1.0, self._params['num_sampling_steps'] + 1).to(trajectory_dct.device)
    return ode_solver.integrate(
      t = timesteps, 
      cond = trajectory_dct_mod, 
      mask = masks, 
      dct = self.dct, 
      idct = self.idct, 
      num_dct_channels = self.num_dct_channels, 
      probability_path = self.probability_path
    )[-1]

