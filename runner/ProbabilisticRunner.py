import os
import csv
import copy
import time
from abc import abstractmethod
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from data.Dataset3DUtils import *
from model.EMA import *
from model.FIDClassifier import *
from model.DiffusionTransformer import *
from model.GraphConvNet import *
from model.GaussianDiffusion import *
from runner.RunnerUtils import *
from Logger import *
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style("darkgrid")

def to_float(value):
  """
  :param value : a python number or a 0-d tensor (possibly on the GPU)
  :return :
    the value as a python float, so that it can be logged or written to CSV
  """
  return value.item() if torch.is_tensor(value) else float(value)


class ProbabilisticRunner(object):
  def __init__(
    self,
    params,
    dataset,
    multimodel_dict,
    logger,
    tblogger,
  ):
    """
    Base runner shared by the diffusion algorithms: builds the model, optimiser, learning-rate scheduler and EMA copy,
    and provides training, evaluation and visualisation loops. Subclasses implement the algorithm-specific parts
    (init_algo, sampling, train_one_epoch, eval_one_epoch).
    :param params : the experiment configuration
    :param dataset : a dict with the 'train' and 'test' datasets
    :param multimodel_dict : the multi-modal evaluation payload (see get_multimodal_dataset)
    :param logger   : the text logger
    :param tblogger : the tensorboard writer
    """
    self._params = params
    self.device = self._params['device']
    self._dataset = dataset
    self._logger = logger
    self._tblogger = tblogger
    self._multimodal_dict = multimodel_dict
    self.num_dct_channels = self._params['num_dct_channels']   # set before init_algo, which may need it

    thisname = self.__class__.__name__
    self.dct, self.idct = get_dct_matrix(
      params['num_observed_frames'] + params['num_predict_frames'])

    self.dct = self.dct.to(params['device'], dtype = torch.float32)
    self.idct = self.idct.to(params['device'], dtype = torch.float32)
    self.init_model()
    self._model.to(self._params['device'])
    self._optimizer_fn = self.select_optimizer()
    self.select_lr_fn()
    self.ema_helper_init()
    self.init_algo()

    self._params['learning_rate_type'] = 'epochwise'

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
    """Runs one epoch of training; implemented by the algorithm subclass."""
    pass

  @abstractmethod
  def eval_one_epoch(self):
    """Runs one epoch of validation; implemented by the algorithm subclass."""
    pass

  @abstractmethod
  def init_algo(self):
    """Builds the algorithm-specific objects (diffusion process, solver); implemented by the subclass."""
    pass

  @abstractmethod
  def sampling(self, sampling_model, padded_trajectory_dct, trajectory_dct_mod, sampling_dict, return_intermediate = False):
    """
    :param sampling_model : the network used for sampling (usually the EMA model)
    :param padded_trajectory_dct : DCT coefficients of the padded observation, shape (B, num_dct_channels, pose_dim)
    :param trajectory_dct_mod : the condition, or None for unconditional sampling
    :param sampling_dict : the sampling settings from sample_preprocessing (mask, mode, ...)
    :param return_intermediate : if True, also return the intermediate states of the reverse process
    :return :
      the sampled DCT coefficients, shape (B, num_dct_channels, pose_dim)
    """
    pass

  def finetune_init(self):
    """Restores model, optimiser, scheduler and EMA model from params['finetuning_ckpt'], if one is given."""
    if self._params['finetuning_ckpt'] is not None:
      self._logger.info(f'[INFO] (finetune_model) Finetuning from: {self._params["finetuning_ckpt"]}')
      model_dict = torch.load(self._params['finetuning_ckpt'], weights_only = False, map_location = self._params['device'])
      self._epoch = model_dict['epoch']
      self._global_step = model_dict['global_steps']          # was self._global_steps, which nothing reads
      self._model.load_state_dict(model_dict['model'])
      self._optimizer_fn.load_state_dict(model_dict['optimizer'])
      self._lr_scheduler.load_state_dict(model_dict['lr_scheduler'])
      self._ema_model.load_state_dict(model_dict['ema_model'])
      self._logger.info(f'[INFO] (finetune_model) Resuming after epoch {self._epoch}')

  def select_lr_fn(self):
    """Calls the selection of learning rate function."""
    self._lr_scheduler = self.get_lr_fn()

  def ema_helper_init(self):
    """Creates the EMA helper and a frozen copy of the model that holds the averaged weights."""
    self._ema_helper = EMAHelper(mu = self._params['ema_rate'])
    self._ema_model = copy.deepcopy(self._model).eval().requires_grad_(False)

  def get_lr_fn(self):
    """
    Creates the function to be used to generate the learning rate.

    :return :
      a torch learning-rate scheduler, selected by params['learning_rate_fn']
    """
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
      # LambdaLR multiplies the base learning rate by lambda_fn(epoch): decay linearly from 1 to 0.5 over training
      T = self._params['max_epochs']
      m = (0.5 - 1.0) / T
      lambda_fn = lambda epoch: m * epoch + 1.0
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
    """
    :param batch_size : the number of timesteps to draw
    :return :
      a tensor of shape (batch_size, ) with timesteps in {1, ..., T - 1}
    """
    T = self._params['diffusion_num_timesteps']
    time_sampling = self._params['timesteps_sampling']

    if time_sampling == 'antithetic_sampling':
      timesteps = torch.randint(
        low  = 1,
        high = T,
        size = (batch_size // 2 + 1,),
      )
      # mirror t in {1, ..., T - 1} to T - t, which stays in {1, ..., T - 1}
      timesteps = torch.cat([timesteps, T - timesteps], dim = 0)[:batch_size]
    elif time_sampling == 'uniform_time_sampling':
      timesteps = torch.randint(low = 1, high = T, size = (batch_size, ))
    else:
      raise ValueError(f"Unknown timesteps_sampling: {time_sampling}")

    return timesteps.to(self._params['device'])

  def prepare_train_step(self):
    """Switches the model to training mode and creates the training batch generator for the current epoch."""
    self._model.train()
    self.train_generator = self._dataset['train'].sampling_generator(
      num_samples = self._params['num_train_data_samples'],
      batch_size = self._params['batch_size'],
    )
    self.start_time = time.time()
    self.train_losses = AverageMeter()
    self._logger.info(f'Starting training at epoch {self.curr_epoch}')

  def prepare_eval_step(self):
    """Switches the model to evaluation mode and creates the validation batch generator for the current epoch."""
    self._model.eval()
    self.val_generator = self._dataset['test'].sampling_generator(
      num_samples = self._params['num_val_data_samples'],
      batch_size = self._params['batch_size'],
    )
    self.start_time = time.time()
    self.val_losses = AverageMeter()
    self._logger.info(f'Starting evaluation at epoch {self.curr_epoch}')

  def compute_train_stats(self):
    """Every save_gif_interval epochs, renders a GIF of predictions of the training model on training data."""
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
    """
    Every save_gif_interval epochs, renders a GIF of predictions of the EMA model on test data, and every
    save_metrics_interval epochs computes the full set of metrics with compute_stats.
    """
    if self.curr_epoch % self._params['save_gif_interval'] == 0:
      generated_pose = self.pose_generator(
        sampling_model = self._ema_model,
        dataset = self._dataset['test'],          
        sampling_mode = 'gif',
      )
      render_animation(
        skeleton = self._dataset['test'].skeleton,
        generated_poses = generated_pose,
        algos = [self._params['algo']],
        num_observed_frames = self._params['num_observed_frames'],
        num_cols = 4,
        output = os.path.join(self._params['gif_dir'], f"val_{self.curr_epoch}.gif")
      )

    if self.curr_epoch % self._params['save_metrics_interval'] == 0 and self.curr_epoch > 0:
      self.compute_stats(self._ema_model, self._multimodal_dict, self._params['exp_desc'])

  def pose_generator(self, sampling_model, dataset, action = None, num_rows = 1, sampling_mode = None):
    """
    Infinite generator of pose dictionaries for render_animation.

    :param sampling_model : the network used for sampling
    :param dataset : the dataset to draw observations from (a numpy array for 'zero_shot')
    :param action : the action to draw from, used by the 'pred' mode
    :param num_rows : the number of observations drawn per yielded dictionary
    :param sampling_mode : one of 'switch', 'pred', 'gif', 'fix_<i>' or 'zero_shot'
    :return :
      yields a dict mapping a name ('context', 'gt', '<algo>_<j>') to a pose sequence of shape (T, J, 3)
    """
    trajectory_np = None
    j = None
    while True:
      poses = {}
      draw_order_indicator = -1
      for k in range(0, num_rows):
        if sampling_mode == 'switch':
          data = dataset.sample_all_action()
        elif sampling_mode == 'pred':
          data, _ = dataset.sample_iter_action(action)
        elif sampling_mode == 'gif' or 'fix' in sampling_mode:
          data, _ = dataset.sample()
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
          trajectory_np = data[..., 1:, :].reshape([data.shape[0],
            self._params['num_observed_frames'] + self._params['num_predict_frames'], -1])
        elif sampling_mode == 'pred' or sampling_mode == 'gif' or 'fix' in sampling_mode or sampling_mode == 'zero_shot':
          if draw_order_indicator == -1:
            poses['context'] = ground_truth
            poses['gt'] = ground_truth
          else:
            poses[f"{self._params['algo']}_{draw_order_indicator + 1}"] = ground_truth
            poses[f"{self._params['algo']}_{draw_order_indicator + 2}"] = ground_truth
          ground_truth = np.expand_dims(ground_truth, axis = 0)
          trajectory_np = ground_truth[..., 1:, :].reshape([ground_truth.shape[0],
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
    """
    Draws one prediction for every observation in data.

    :param sampling_model : the network used for sampling
    :param data : an array of shape (N, T, J, 3), observed and future frames
    :return :
      an array of shape (1, N, T, (J - 1) * 3), the predictions without the root joint
    """
    trajectory_np = data[..., 1:, :].transpose([0, 2, 3, 1])
    trajectory = torch.tensor(trajectory_np, device = self._params['device'], dtype = torch.float32)
    trajectory = trajectory.reshape([trajectory.shape[0], -1, trajectory.shape[-1]]).transpose(1, 2) # [n_samples, frame, n_joints * joint_dim]

    mode_dict, padded_traj_dct, traj_dct_mod = sample_preprocessing(trajectory, self._params, 'metrics', self.dct)
    sampled_motion = self.sampling(sampling_model, padded_traj_dct, traj_dct_mod, mode_dict)

    result_trajectory = torch.matmul(self.idct[:, :self._params['num_dct_channels']], sampled_motion)
    result_trajectory = result_trajectory.cpu().numpy()
    result_trajectory = result_trajectory[None, ...] # [1, N, T, pose_dim]
    return result_trajectory

  def save_checkpoint(self, epoch):
    """
    :param epoch : the epoch just completed, stored so that training can resume after it
    :return :
      None, the checkpoint is written to <model_prefix>/models/model_ckpt.pth.tar
    """
    model_path = os.path.join(self._params['model_prefix'], 'models', 'model_ckpt.pth.tar')
    state = {
      'global_steps' : self._global_step,
      'epoch' : epoch,
      'model' : self._model.module.state_dict() if hasattr(self._model, 'module') else self._model.state_dict(),
      'optimizer' : self._optimizer_fn.state_dict(),
      'lr_scheduler' : self._lr_scheduler.state_dict(),
      'ema_model' : self._ema_model.state_dict(),
    }
    torch.save(state, model_path)

  def train(self):
    """Main training loop: train, evaluate and update the learning rate every epoch, saving checkpoints periodically."""
    self._lr_db_curve = []
    self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]
    thisname = self.__class__.__name__
    self._logger.info('=' * 90)
    self._logger.info(f"[INFO] {thisname} | global step : {self._global_step:06d} | epoch : {(self._epoch + 1):06d} | Start training")
    self._logger.info('=' * 90)
    e = self._epoch   # so the final save works even if no epoch is left to run
    for e in range(self._epoch + 1, self._params['max_epochs']):
      self._scalars = {}
      self.curr_epoch = e

      # Train
      self.prepare_train_step()
      self.train_one_epoch()
      self.compute_train_stats()

      # Evaluate
      self.prepare_eval_step()
      self.eval_one_epoch()
      self.compute_eval_stats()

      self.update_learning_rate(e, mode = 'epochwise')

      if (e + 1) % self._params['snapshot_saving_steps'] == 0:
        self.save_checkpoint(e)

    # save the last one
    self.save_checkpoint(e)

  def update_learning_rate(self, epoch_step, mode = 'stepwise'):
    """
    Update learning rate handler updating only when the mode matches.

    :param epoch_step : the epoch (or step) index, used for logging
    :param mode       : 'stepwise' or 'epochwise'; the scheduler steps only if it equals params['learning_rate_type']
    """
    if self._params['learning_rate_type'] == mode:
      self._lr_scheduler.step()
      self._tblogger.add_scalar('learning_rate/lr', self._params['learning_rate'], epoch_step)
      self._lr_db_curve.append([self._params['learning_rate'], epoch_step])
      self._params['learning_rate'] = self._lr_scheduler.get_last_lr()[0]

  def init_model(self):
    """Builds the network: a diffusion transformer predictor followed by a GCN corrector."""
    if self._params['model_architecture'] == 'gcn_transformer':
      predictor = motion_diffusion_transformer(self._params)
      corrector = graph_conv_net_v1(self._params)

      self._model = MotionRefinementNet(
        predictor = predictor,
        corrector = corrector,
      )
    else :
      raise NotImplementedError(f"Model architecture {self._params['model_architecture']} not found")

  def select_optimizer(self):
    """
    :return :
      the optimiser named by params['optimizer'] over all model parameters
    """
    if self._params['optimizer'] == 'AdamW':
      return torch.optim.AdamW(
        params = self._model.parameters(),
        lr = self._params['learning_rate'],
        betas = self._params['betas'],
      )
    elif self._params['optimizer'] == 'Adam':
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


  """
  Metrics calculation, this should be in a different files if I were a more organized with my expirment writing, unfortunately I'm not. 
  """
  def compute_all_metrics(self, pred, gt, gt_multi):
    """
    :param pred : a tensor of shape (R, F, pose_dim), the R predictions of one test example
    :param gt : a tensor of shape (1, F, pose_dim), its recorded future
    :param gt_multi : an array of shape (P, F, pose_dim), its pseudo ground truths (including the recorded future)
    :return :
      APD, ADE, FDE, MMADE, MMFDE of this example, as 0-d tensors
    """
    if pred.shape[0] == 1:
      diversity = torch.zeros((), device = pred.device)   # pdist of a single sample is empty, its mean NaN
    else:
      dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
      diversity = dist_diverse.mean()

    gt_multi = torch.from_numpy(gt_multi).to(pred.device)
    gt_multi_gt = torch.cat([gt_multi, gt], dim = 0)

    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]

    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim = 3) # (R, P + 1, F)

    mmfde, _ = dist[:, :-1, -1].min(dim = 0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim = 2).min(dim = 0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim = 1).min(dim = 0)
    fde, _ = dist[:, -1, -1].min(dim = 0)
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde, mmade, mmfde

  def set_up_fid_classifier(self):
    """Loads the pretrained Human3.6M action classifier whose features are used for FID."""
    self.classifier_for_fid = ClassifierForFID(
      input_size = self._params['pose_dim'],
      hidden_size = 128,
      hidden_layer = 2,
      output_size = 15,
      device = self._params['device'],
      use_noise = None
    ).to(self._params['device'])

    classifier_path = os.path.join("h36m_classifier.pth")
    classifier_state = torch.load(classifier_path, map_location = self._params['device'], weights_only = True)
    self.classifier_for_fid.load_state_dict(classifier_state["model"])
    self.classifier_for_fid.eval()
    self._logger.info(f'[INFO] (load_fid_classifier_model) from: {classifier_path}')

  def compute_cmd(self, sampling_model):
    """
    :param sampling_model : the model wrapped with the sampling algorithm under evaluation
    :return :
      the CMD of the predictions on random test sequences, or None if the test set has no action statistics
    """
    test_set = self._dataset['test']
    if not (hasattr(test_set, 'idx_to_class') and hasattr(test_set, 'mean_motion_per_class')):
      return None

    num_observed = self._params['num_observed_frames']
    histogram_data = []
    all_obs_classes = []
    eval_generator = test_set.sampling_generator(
      num_samples = self._params['num_val_data_samples'],
      batch_size = self._params['batch_size'],
      augment = False, 
    )
    print(f"Computing CMD for {self._params['num_val_data_samples']} samples in batches of {self._params['batch_size']}.")
    for data, actions in tqdm.tqdm(
      eval_generator, 
      total = math.ceil(self._params['num_val_data_samples'] / self._params['batch_size']), 
      position = 0,
      desc = 'eval cmd', 
      mininterval = 10
    ):
      pred = self.metrics_calc_sampling(sampling_model, data) # (1, B, T, pose_dim), numpy
      pred = torch.from_numpy(pred).permute(1, 0, 2, 3) # (B, 1, T, pose_dim), as compute_cmd_statistics expects
      pred = pred[:, :, num_observed:, :] # predicted frames only, as in Equation (CMD)
      compute_cmd_statistics(pred, actions, histogram_data, all_obs_classes)
    return calculate_cmd(test_set, histogram_data, all_obs_classes)

  def compute_stats(self, sampling_model, multimodal_dict, exp_desc):
    """
    :param sampling_model : the model wrapped with the sampling algorithm under evaluation
    :param multimodal_dict : the evaluation payload
    :param exp_desc : a string identifying this configuration, used in the output filenames
    :return :
      None, results are written to CSV under self._params['result_dir']
    """
    algo = self._params['algo']
    compute_fid = (self._params['dataset'] == 'h36m')

    if compute_fid:
      self.set_up_fid_classifier() # apperently the clasifier model is only available for Human3.6M

    ground_truth_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    trajectory_gt_arr = multimodal_dict['trajectory_gt_array']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    if compute_fid:
      stats_names = stats_names + ['FID']
    stats_meter = {name: {algo: AverageMeter()} for name in stats_names}

    # Due to technical difficulties (not enough memory), I have to split the dataset into chunks of fixed size, \
    # and compute the metrics on each chunk separately. The final metrics are then in fact a running average of the metrics on each chunk. 

    K = self._params.get('num_metric_repeats', 50)
    chunk_size = self._params.get('eval_chunk_size', 250)
    num_observed = self._params['num_observed_frames']

    all_pred_activations = []
    all_gt_activations = []
    all_pred_classes = []
    all_gt_classes = []

    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_index in tqdm.tqdm(range(num_chunks), position = 0, desc = 'eval chunks'):
      start = chunk_index * chunk_size
      end = min(start + chunk_size, num_samples)

      predictions = []
      for _ in range(K):
        predictions.append(self.metrics_calc_sampling(sampling_model, data_group[start:end]))

      predictions = np.concatenate(predictions, axis = 0) # (K, n, T, pose_dim)
      predictions = predictions[:, :, num_observed:, :]
      predictions = torch.from_numpy(predictions).to(self.device)

      gt_chunk = torch.from_numpy(ground_truth_group[start:end]).to(self.device)

      for offset in range(end - start):
        apd, ade, fde, mmade, mmfde = self.compute_all_metrics(
          predictions[:, offset, :, :],
          gt_chunk[offset][np.newaxis, ...],
          trajectory_gt_arr[start + offset],
        )
        stats_meter['APD'][algo].update(apd)
        stats_meter['ADE'][algo].update(ade)
        stats_meter['FDE'][algo].update(fde)
        stats_meter['MMADE'][algo].update(mmade)
        stats_meter['MMFDE'][algo].update(mmfde)

      if compute_fid:
        compute_fid_statistics(
          pred = predictions.permute(1, 0, 2, 3).to(torch.float32),
          gt = gt_chunk.to(torch.float32),
          classifier_for_fid = self.classifier_for_fid,
          all_pred_activations = all_pred_activations,
          all_gt_activations = all_gt_activations,
          all_pred_classes = all_pred_classes,
          all_gt_classes = all_gt_classes,
        )

      del predictions, gt_chunk
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if compute_fid:
      fid_value = calculate_fid(
        np.concatenate(all_pred_activations, axis = 0),
        np.concatenate(all_gt_activations, axis = 0),
      )
      stats_meter['FID'][algo].update(float(fid_value))

    # CMD needs action labels, so it uses its own pass over randomly drawn test sequences
    cmd_value = self.compute_cmd(sampling_model)
    if cmd_value is not None:
      stats_names = stats_names + ['CMD']
      stats_meter['CMD'] = {algo: AverageMeter()}
      stats_meter['CMD'][algo].update(float(cmd_value))

    for name in stats_names:
      summary = ' '.join([f'{key}: {to_float(meter.avg):.4f}' for key, meter in stats_meter[name].items()])
      self._logger.info(f'{name}: {summary}')

    file_latest = f"{self._params['result_dir']}/{exp_desc}_stats_latest.csv"
    file_stat = f"{self._params['result_dir']}/{exp_desc}_stats.csv"
    with open(file_latest, 'w', newline = '') as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames = ['Metric', algo])
      writer.writeheader()
      for stats, meter in stats_meter.items():
        writer.writerow({'Metric': stats, algo: to_float(meter[algo].avg)})

    df1 = pd.read_csv(file_latest)
    if not os.path.exists(file_stat):
      df1.to_csv(file_stat, index = False)
    else:
      df2 = pd.read_csv(file_stat)
      if 'Metric' not in df2.columns:           
        df2 = df2.rename(columns = {df2.columns[0]: 'Metric'})
      df1 = df1.rename(columns = {algo: f'{algo}_{df2.shape[1]}'})
      df = df2.merge(df1, on = 'Metric', how = 'outer', sort = False)
      df.to_csv(file_stat, index = False)

  def demo_visualize(self, sampling_mode, sampling_model, dataset):
    """
    Script for drawing GIFs in different modes.
    :param sampling_mode : 'switch', 'pred', 'control' or 'zero_shot'
    :param sampling_model : the network used for sampling
    :param dataset : a dict with the 'train' and 'test' datasets
    """
    if self._params['dataset'] != 'h36m' and sampling_mode != 'pred':
      raise NotImplementedError(f"sorry, {sampling_mode} is currently only available in h36m setting.")

    os.makedirs(os.path.join(self._params['gif_dir'], self._params['exp_desc']), exist_ok = True)
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
          output = os.path.join(self._params['gif_dir'], self._params['exp_desc'], f'pred_{suffix}.gif'),
          sampling_mode = sampling_mode
        )
        self.visualize_joint_trajectories(next(generated_pose))

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
          output = os.path.join(self._params['gif_dir'], self._params['exp_desc'], fix_name[i] + '.gif'),
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


  """
  Visualization functions, I should probably move them to a separate file, but these are written in a hurry since I need figures for my thesis, 
  so well they end up here for now (Hopefully I will clean this up later). 
  """

  def visualize_joint_trajectories(self, inp):
    """
    Plots the x, y and z coordinates of both feet and both wrists over time, one figure per sequence and axis.
    :param inp : a pose dictionary from pose_generator, each value of shape (T, J, 3)
    :return :
      None, figures are written to imgs/<index>_trajectory_<axis>.png
    """
    os.makedirs('imgs', exist_ok = True)
    joints = {2: 'right foot', 5: 'left foot', 12: 'left wrist', 15: 'right wrist'}   # indices without the root
    for ind, pose in enumerate(inp.values()):
      pose = pose.reshape(pose.shape[0], -1, 3)[:, 1:, :]
      for d, name in enumerate(['x', 'y', 'z']):
        fig, ax = plt.subplots(figsize = (10, 5))
        for j, jname in joints.items():
          ax.plot(range(pose.shape[0]), pose[:, j, d], label = jname)
        ax.set_xlabel('frame')
        ax.set_ylabel(f'{name} position')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'imgs/{ind}_trajectory_{name}.png', dpi = 150)
        plt.close(fig)
  
  @torch.no_grad()
  def visualize_dct_space(
    self,
    n_show = 20,
    out_path = "figures/dct_energy.img",
    diffusion_out_path = "mid_dct",
    diffusion_action = "Walking",
    n_diffusion_steps = 20,
    seed = 0):
    """
    Plots the DCT energy of one test sequence for eight actions, then the energy of one sequence along the
    forward diffusion process.

    :param n_show : the number of DCT coefficients shown
    :param out_path : the file of the eight-action energy figure
    :param diffusion_out_path : the base name of the forward-process frames, one file per timestep
    :param diffusion_action : the action of the sequence noised in the forward process
    :param n_diffusion_steps : the number of timesteps shown, equally spaced in {0, ..., T - 1}
    :param seed : the random seed
    """
    actions = [
      ("Directions", "Directions"),
      ("Discussion", "Discussion"),
      ("Walking", "Walking"),
      ("Phoning", "Phoning"),
      ("Photo", "Photo"),
      ("Posing", "Posing"),
      ("Purchases", "Purchases"),
      ("WalkDog", "Walking dog"),
    ]
    t_total = self._params["num_observed_frames"] + self._params["num_predict_frames"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    rc = {
      "font.family": "serif",
      "font.serif": ["cmr10"],
      "mathtext.fontset": "cm",
      "axes.formatter.use_mathtext": True,
      "axes.unicode_minus": False,
      "font.size": 8,
      "axes.labelsize": 8,
      "axes.titlesize": 8.5,
      "xtick.labelsize": 7,
      "ytick.labelsize": 7,
      "axes.linewidth": .5,
      "xtick.major.width": .5,
      "ytick.major.width": .5,
    }
    with matplotlib.rc_context(rc):
      fig, axes = plt.subplots(4, 2, figsize = (5.75, 5.0), sharex = True, sharey = True,
        gridspec_kw = dict(hspace = .35, wspace = .08, left = .09, right = .87, top = .955, bottom = .085)
      )

      im = None
      energy = None
      for ax, (action, title) in zip(axes.flat, actions):
        query = 'Discussion 1' if action == 'Discussion' else action
        traj, _ = self._dataset["test"].sample_iter_action(query)
        traj = traj[..., 1:, :].reshape(traj.shape[0], t_total, -1)
        traj = torch.as_tensor(traj, device = self._params["device"], dtype = torch.float32)

        energy = (torch.matmul(self.dct[:n_show], traj) ** 2)[0].cpu().numpy()
        im = ax.imshow(
          np.clip(energy, 1e-4, None), aspect = "auto", cmap = "YlGnBu",
          norm = LogNorm(1e-4, 4), interpolation = "nearest",
          extent = (.5, energy.shape[1] + .5, n_show + .5, .5)
        )
        ax.set_title(title, pad = 2.5)

      for ax in axes[:, 0]:
        ax.set_ylabel(r"DCT coefficient $l$")
      for ax in axes[-1]:
        ax.set_xlabel(r"pose dimension $k$")

      pose_dim = energy.shape[1]
      axes[0, 0].set_yticks([1] + list(range(10, n_show + 1, 10)))
      axes[0, 0].set_xticks([1] + list(range(12, pose_dim + 1, 12)))

      cax = fig.add_axes([.895, .085, .02, .87])
      fig.colorbar(im, cax = cax).set_label(r"energy $\hat{x}_{l,k}^{\,2}$")

      if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok = True)
      fig.savefig(out_path)
      plt.close(fig)

      query = 'Discussion 1' if diffusion_action == 'Discussion' else diffusion_action
      traj, _ = self._dataset["test"].sample_iter_action(query)
      traj = traj[..., 1:, :].reshape(traj.shape[0], t_total, -1)
      traj = torch.as_tensor(traj, device = self._params["device"], dtype = torch.float32)

      traj_dct = torch.matmul(self.dct[:n_show], traj)
      noise = torch.randn_like(traj_dct)

      num_diffusion_steps = self.gaussian_diffusion.num_timesteps
      t_values = np.linspace(0, num_diffusion_steps - 1, n_diffusion_steps).round().astype(int)

      diffusion_out_dir = os.path.dirname(diffusion_out_path) or "."
      os.makedirs(diffusion_out_dir, exist_ok = True)
      base_name = os.path.splitext(os.path.basename(diffusion_out_path))[0]
      ext = os.path.splitext(diffusion_out_path)[1] or ".png"

      for t in t_values:
        t_tensor = torch.tensor([t], device = traj_dct.device)
        traj_dct_t = self.gaussian_diffusion.q_sample(traj_dct, t_tensor, noise)

        energy = (traj_dct_t ** 2)[0].cpu().numpy()

        fig_t, ax_t = plt.subplots(figsize = (4.6, 1.6))
        ax_t.imshow(
          np.clip(energy, 1e-4, None),
          aspect = "auto",
          cmap = "YlGnBu",
          norm = LogNorm(1e-4, 4),
          interpolation = "nearest",
          extent = (.5, energy.shape[1] + .5, n_show + .5, .5)
        )
        ax_t.set_axis_off()
        fig_t.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)

        out_file = os.path.join(diffusion_out_dir, f"{base_name}_t{int(t)}{ext}")
        fig_t.savefig(out_file, bbox_inches = "tight", pad_inches = 0)
        plt.close(fig_t)
        self._logger.info(f'[INFO] (visualize_dct_space) wrote {out_file}')

    self._logger.info(f'[INFO] (visualize_dct_space) wrote {out_path}')

  @torch.no_grad()
  def visualize_superimposed(
    self,
    sampling_model,
    actions = ('Directions', 'Eating', 'Sitting', 'WalkDog'),
    num_samples = 50,
    n_show = 30,
    times = (0.5, 1.0, 1.5, 2.0),
    out_path = 'img_superimposed/superimposed.png',
    radius = 0.9,
    seed = 0
  ):
    """
    One row per action, columns = last observed pose, then the prediction at each time in `times` (seconds).
    Each cell overlays n_show of the num_samples predictions; the one closest to the ground truth is drawn solid.

    :param sampling_model : the network used for sampling
    :param actions : the test actions, one row each
    :param num_samples : the number of predictions drawn per sequence
    :param n_show : the number of predictions drawn in each cell
    :param times : the prediction times shown, in seconds after the last observed frame
    :param out_path : the output image
    :param radius : half-width of the plotted cube, the same for every cell
    :param seed : the random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    skeleton = self._dataset['test'].skeleton
    parents, right, left = skeleton.parents, skeleton.get_joints_right(), skeleton.get_joints_left()
    num_obs = self._params['num_observed_frames']
    total = num_obs + self._params['num_predict_frames']
    fps = 50 if self._params['dataset'] == 'h36m' else 60
    frames = [num_obs - 1] + [num_obs - 1 + int(round(t * fps)) for t in times]
    titles = ['history'] + [f'{t:g} s' for t in times]

    fig = plt.figure(figsize = (2.0 * len(frames), 2.6 * len(actions)))
    for i, action in enumerate(actions):
      query = 'Discussion 1' if action == 'Discussion' else action
      data, _ = self._dataset['test'].sample_iter_action(query)
      gt = data[0].copy()
      gt[:, :1, :] = 0                                                           # (T, J, 3), root at the origin
      traj = torch.tensor(gt[None, :, 1:, :].reshape(1, total, -1), device = self.device, dtype = torch.float32)
      traj = traj.repeat(num_samples, 1, 1)
      mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, self._params, 'metrics', self.dct)
      motion = self.sampling(sampling_model, traj_dct, traj_dct_mod, mode_dict)
      pred = torch.matmul(self.idct[:, :self._params['num_dct_channels']], motion).cpu().numpy()
      pred = post_process(pred, self._params)                                   # (R, T, J, 3)

      err = np.linalg.norm(pred[:, num_obs:] - gt[None, num_obs:], axis = -1).mean(axis = (1, 2))
      best = int(np.argmin(err))
      show = [best] + [r for r in np.random.permutation(num_samples) if r != best][:n_show - 1]

      for c, f in enumerate(frames):
        ax = fig.add_subplot(len(actions), len(frames), i * len(frames) + c + 1, projection = '3d')
        poses = [(gt[f], 1.0)] if c == 0 else [(pred[r, f], 1.0 if r == best else 0.25) for r in show]
        for pose, alpha in poses:
          for j, p in enumerate(parents):
            if p < 0:
              continue
            side = 'right' if j in right else ('left' if j in left else 'mid')
            if c == 0:   # history
              color = {'left': 'gray', 'mid': 'black', 'right': 'red'}[side]
            else:        # prediction
              color = {'left': 'purple', 'mid': 'black', 'right': 'green'}[side]
            ax.plot(*zip(pose[j], pose[p]), color = color, alpha = alpha, lw = 1.0 if alpha == 1.0 else 0.7)
        ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius); ax.set_zlim(-radius, radius)
        ax.set_box_aspect((1, 1, 1), zoom = 1.6)
        ax.view_init(elev = 15., azim = -60.)
        ax.set_axis_off()
        if i == 0:
          ax.set_title(titles[c], fontsize = 20)
        if c == 0:
          ax.text2D(-0.15, 0.5, action.split(' ')[0], transform = ax.transAxes, rotation = 90, va = 'center', fontsize = 20)
    fig.subplots_adjust(left = 0.03, right = 1, top = 0.95, bottom = 0, wspace = -0.2, hspace = -0.05)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok = True)
    plt.savefig(out_path, bbox_inches = 'tight', dpi = 300)
    plt.close(fig)
    self._logger.info(f'[INFO] (visualize_superimposed) wrote {out_path}')
  

  @torch.no_grad()
  def visualize_fid_tsne(
    self,
    sampling_model,
    num_sequences = 8000,
    out_path = 'img/fid_tsne.png',
    perplexity = 40,
    seed = 0,
    show_title = False
  ):
    """
    t-SNE of the FID classifier features of gt futures and of futures generated with the CURRENT solver settings
    (self._params['dpm_algorithm_type'], 'dpm_solver_order', 'dpm_solver_method', 'dpm_solver_num_steps'), drawn as
    one image: the closer the generated (orange) cloud is to the gt (blue) one, the lower the FID.

    :param sampling_model : the model wrapped with the sampling algorithm under evaluation
    :param num_sequences : the number of random test sequences; each gives one gt and one generated future
    :param out_path : the output image
    :param perplexity : the t-SNE perplexity
    :param seed : the random seed; the same seed gives the same test sequences in every call
    :param show_title : if True, write the solver and NFE above the plot (off for LaTeX subfigures, which caption it)
    :return :
      None, the image is written to out_path
    """
    if not hasattr(self, 'classifier_for_fid'):
      self.set_up_fid_classifier()
    self.init_algo()                                    
    num_obs = self._params['num_observed_frames']
    total = num_obs + self._params['num_predict_frames']

    def features(motion):
      """(B, F, pose_dim) array -> (B, feature_dim) FID classifier features"""
      motion = torch.as_tensor(motion, device = self.device, dtype = torch.float32).permute(0, 2, 1)
      return self.classifier_for_fid.get_fid_features(motion_sequence = motion).cpu().numpy()

    np.random.seed(seed)
    torch.manual_seed(seed)
    gt_feats, sample_feats = [], []
    for data, _ in self._dataset['test'].sampling_generator(num_samples = num_sequences,
                                                            batch_size = self._params['batch_size']):
      gt_feats.append(features(data[:, :, 1:, :].reshape(data.shape[0], total, -1)[:, num_obs:]))
      sample_feats.append(features(self.metrics_calc_sampling(sampling_model, data)[0][:, num_obs:]))
    gt_feats, sample_feats = np.concatenate(gt_feats), np.concatenate(sample_feats)
    n = len(gt_feats)

    feats = np.concatenate([gt_feats, sample_feats], axis = 0)
    feats = (feats - feats.mean(axis = 0)) / (feats.std(axis = 0) + 1e-8)
    emb = TSNE(n_components = 2, perplexity = perplexity, init = 'pca', random_state = seed).fit_transform(feats)

  
    colors = np.array(['#2a78d6'] * n + ['#eb6834'] * n)
    order = np.random.default_rng(seed).permutation(2 * n)
    fig, ax = plt.subplots(figsize = (4.6, 4.6))
    ax.scatter(*emb[order].T, s = 4, c = colors[order], alpha = 0.5, linewidths = 0)
    ax.scatter([], [], s = 20, color = '#2a78d6', label = 'ground truth')              
    ax.scatter([], [], s = 20, color = '#eb6834', label = 'generated')
    ax.legend(loc = 'lower left', fontsize = 8, frameon = True)
    ax.set_xticks([]); ax.set_yticks([])
    p = self._params
    if show_title:
      ax.set_title(f"{p['dpm_algorithm_type']}, order {p['dpm_solver_order']}, {p['dpm_solver_method']}, "
                   rf"$\mathrm{{NFE}}={p['dpm_solver_num_steps']}$", fontsize = 9)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok = True)
    fig.savefig(out_path, bbox_inches = 'tight', dpi = 300)
    plt.close(fig)
    self._logger.info(f'[INFO] (visualize_fid_tsne) {n} gt and {n} generated futures, wrote {out_path}')