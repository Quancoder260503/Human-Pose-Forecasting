import os 
import copy 
import numpy as np 
import torch 
from scipy.spatial.distance import pdist, squareform
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, writers 
from mpl_toolkits.mplot3d import Axes3D 
import pickle 
import time 
from datetime import datetime

def padding_trajectory(trajectory, padding, idx_pad, zero_index):
  if padding == 'zero':
    curr_trajectory = trajectory
    curr_trajectory[..., zero_index, :] = 0
    trajectory_pad = curr_trajectory[..., idx_pad, :]
  else : 
    trajectory_pad = trajectory[..., idx_pad, :]
  return trajectory_pad


def post_process(predictions, params): 
  predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], -1, 3)
  predictions = np.concatenate((np.tile(np.zeros((1, params['num_observed_frames'] + params['num_predict_frames'], 1, 3)),\
    (predictions.shape[0], 1, 1, 1)), predictions), axis = 2)
  predictions[..., :1, :] = 0
  return predictions

def sample_preprocessing(traj, params, sampling_mode, dct):
  if sampling_mode.split('_')[0] == 'fix':
    fix_list = [
      [0, 1, 2],  #
      [3, 4, 5],
      [6, 7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
      [0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ]

    index = int(sampling_mode.split('_')[1])

      # [ ['right_leg'], ['left_leg'], ['torso'], [‘left_arm’], ['right_arm'] ]
    joint_fix_lb = fix_list[index][0] * 3
    joint_fix_ub = fix_list[index][-1] * 3 + 3

    traj_fix = traj[:, params['idx_pad'], :]
    traj_fix[:, :, joint_fix_lb:joint_fix_ub] = traj[:, :, joint_fix_lb : joint_fix_ub]

    n = params['vis_col']
    traj = traj.repeat(n, 1, 1)

    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'],\
         traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    mask_fix = copy.deepcopy(mask)
    mask_fix[:, :, joint_fix_lb:joint_fix_ub] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
      traj_dct_mod = None

    return {
      'traj_fix': traj_fix,
      'mask': mask_fix,
      'sample_num': n,
      'mode': 'control'
    }, traj_dct, traj_dct_mod

  elif sampling_mode == 'switch':
    n = traj.shape[0]
    traj_switch = traj[:, (params['num_predict_frames'] - params['num_observed_frames']):(params['num_predict_frames'] + params['num_observed_frames']), :]
    direct_current = traj[:, (params['num_predict_frames'] - params['num_observed_frames']), :].unsqueeze(1).\
      repeat(1, params['num_predict_frames'] - params['num_observed_frames'], 1)
    traj_switch = torch.cat([direct_current, traj_switch], dim=1)

    traj_switch = traj_switch[0].unsqueeze(0).repeat(n, 1, 1)
    # traj_switch = torch.roll(traj_switch, 1, 0) # make the target traj be various

    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    mask_end = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(params['num_predict_frames'] - params['num_observed_frames'], params['num_observed_frames'] + params['num_predict_frames']):
      mask_end[:, i, :] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
        traj_dct_mod = None

    return {
      'traj_switch': traj_switch,
      'mask_end': mask_end,
      'mask': mask,
      'sample_num': n,
      'mode': 'switch'
    }, traj_dct, traj_dct_mod

  elif sampling_mode == 'gif':
    n = params['vis_col']
    traj = traj.repeat(n, 1, 1)

    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
        traj_dct_mod = None

    return {
      'mask': mask,
      'sample_num': n,
      'mode': 'gif'
    }, traj_dct, traj_dct_mod

  elif sampling_mode == 'pred':
    n = params['vis_col']
    traj = traj.repeat(n, 1, 1)

    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
        traj_dct_mod = None

    return {
      'mask': mask,
      'sample_num': n,
      'mode': 'pred'
    }, traj_dct, traj_dct_mod

  elif sampling_mode == 'zero_shot':
    n = params['vis_col']
    traj = traj.repeat(n, 1, 1)

    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
      traj_dct_mod = None

    return {
      'mask': mask,
      'sample_num': n,
      'mode': 'zero_shot'
    }, traj_dct, traj_dct_mod

  elif sampling_mode == 'metrics':
    n = traj.shape[0]
    mask = torch.zeros([n, params['num_observed_frames'] + params['num_predict_frames'], traj.shape[-1]]).to(params['device'])
    for i in range(0, params['num_observed_frames']):
      mask[:, i, :] = 1

    traj_pad = padding_trajectory(traj, params['padding'], params['idx_pad'], params['zero_index'])

    traj_dct = torch.matmul(dct[:params['num_dct_channels']], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > params['mod_test']:
      traj_dct_mod = None

    return {
      'mask': mask,
      'sample_num': n,
      'mode': 'metrics'
    }, traj_dct, traj_dct_mod
  else:
    raise NotImplementedError(f"unknown purpose for sampling: {sampling_mode}")
  
   
def get_multimodal_dataset(logger, dataset_multi_test, params):
  logger.info('=> preparing evaluation dataset ...')
  data_groups = []
  num_samples = 0
  data_generator_multi_test = dataset_multi_test.iter_generator(step = params['num_observed_frames'])
  
  for data, _ in data_generator_multi_test: 
    num_samples += 1 
    data_groups.append(data)
  
  data_groups = np.concatenate(data_groups, axis = 0) 
  all_data = data_groups[..., 1:, :].reshape(data_groups.shape[0], data_groups.shape[1], -1)
  gt_groups = all_data[:, params['num_observed_frames'] :, :]

  all_start_pose = all_data[:, params['num_observed_frames'] - 1, :]
  pairwise_distance = squareform(pdist(all_start_pose))

  trajectory_gt_arr = [] 
  num_mult = [] 

  for i in range(pairwise_distance.shape[0]):
    index = np.nonzero(pairwise_distance[i] < params['multimodal_threshold'])
    trajectory_gt_arr.append(all_data[index][:, params['num_observed_frames'] :, :])
    num_mult.append(len(index[0]))

  num_mult = np.array(num_mult)
  logger.info('=' * 80)
  logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pairwise_distance.shape[0]}')
  logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pairwise_distance.shape[0]}')
  logger.info('done...')
  logger.info('=' * 80)


  return { 
    'trajectory_gt_array' : trajectory_gt_arr, 
    'data_group' : data_groups, 
    'gt_group' : gt_groups, 
    'num_samples' : num_samples, 
  }




def get_dct_matrix(N, is_torch = True): 
  dct_m = np.eye(N)
  for k in np.arange(N): 
    for i in np.arange(N): 
      w = np.sqrt(2 / N)
      if k == 0: 
        w = np.sqrt(1 / N) 
      dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
  idct_m = np.linalg.inv(dct_m) 
  if is_torch : 
    dct_m = torch.from_numpy(dct_m) 
    idct_m = torch.from_numpy(idct_m)
  return dct_m, idct_m 
    



def render_animation(
  skeleton, 
  generated_poses, 
  algos, 
  num_observed_frames,
  fix_0 = True, 
  azim = 0.0, 
  output = None, 
  sampling_sampling_mode = 'pred', 
  size = 2, 
  num_cols = 5, 
  bitrate = 3000, 
  fix_index = None 
): 
  if sampling_sampling_mode == 'switch': 
    fix_0 = False 
  if fix_index is not None : 
    fix_list = [
      [1, 2, 3], 
      [4, 5, 6], 
      [7, 8, 9, 10], 
      [11, 12, 13], 
      [14, 15, 16], 
      [1, 2, 3, 4, 5, 6], 
      [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 
    ]
    fix_i = fix_list[fix_index]
    fix_color = 'darkblue' 
  else : 
    fix_i = None 
  
  all_poses = next(generated_poses)
  algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
  
  time_total = next(iter(all_poses.values())).shape[0]
  poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('ground_truth'), all_poses.items()))


  plt.ioff()
  num_rows = int(np.ceil(len(poses) / num_cols))

  fig = plt.figure(figsize = (size * num_cols, size * num_rows))

  axes_3d = [] 
  lines_3d = [] 
  trajectories = [] 
  radius = 1.7

  for index, (title, data) in enumerate(poses.items()):
    ax = fig.add_subplot(num_rows, num_cols, index + 1, projection = '3d')
    ax.view_init(elev = 15., azim = azim) 
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 5.0
    if sampling_sampling_mode == 'switch':
      if index == 0:
        ax.set_title('target', y = 1.0, fontsize = 12)
    if sampling_sampling_mode == 'pred' or 'fix' in sampling_sampling_mode or sampling_sampling_mode == 'control' or sampling_sampling_mode == 'zero_shot':
      if index == 0 or index == 1:
        ax.set_title(title, y = 1.0, fontsize = 12)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    axes_3d.append(ax)
    lines_3d.append([])
    trajectories.append(data[:, 0, [0, 1]])
  
  fig.tight_layout(h_pad = 15,w_pad = 15)
  fig.subplots_adjust(wspace = -0.4, hspace = 0.4)
  poses = list(poses.values())

  anim = None
  initialized = False
  animating = True
  find = 0
  hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
  pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
  tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'blue'

  parents = skeleton.parents

  def update_video(index): 
    nonlocal initialized 
    if sampling_sampling_mode == 'switch':
      if index < num_observed_frames:
        lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
      elif index > 75:
        lcol, mcol, rcol = tran_lcol, pred_mcol, tran_rcol
      else:
        lcol, mcol, rcol = pred_lcol, tran_mcol, pred_rcol
    else:
      if index < num_observed_frames:
        lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
      else:
        lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

    for n, ax in enumerate(axes_3d):
      if fix_0 and n == 0 and index >= num_observed_frames: 
        continue 
      if fix_0 and n % num_cols == 0 and index >= num_observed_frames : 
        continue 

      trajectories[n] = poses[n][:, 0, [0, 1, 2]]
      ax.set_xlim3d([-radius / 2 + trajectories[n][index, 0], radius / 2 + trajectories[n][index, 0]])
      ax.set_ylim3d([-radius / 2 + trajectories[n][index, 1], radius / 2 + trajectories[n][index, 1]])
      ax.set_zlim3d([-radius / 2 + trajectories[n][index, 2], radius / 2 + trajectories[n][index, 2]])

    if not initialized:
      for j, j_parent in enumerate(parents):
        if j_parent == -1:
          continue
        if j in skeleton.get_joints_right():
            col = rcol
        elif j in skeleton.get_joints_left():
            col = lcol
        else:
            col = mcol

        if fix_i is not None and j in fix_i:
            col = fix_color

        for n, ax in enumerate(axes_3d):
          pos = poses[n][index]
          lines_3d[n].append(
            ax.plot(
              [pos[j, 0], pos[j_parent, 0]],
              [pos[j, 1], pos[j_parent, 1]],
              [pos[j, 2], pos[j_parent, 2]], 
              zdir = 'z', 
              c = col, 
              linewidth = 3.0
            )
          )
      
      initialized = True
    else : 
      for j, j_parent in enumerate(parents):
        if j_parent == -1: 
          continue 
        if j in skeleton.get_joints_right():
          col = rcol
        elif j in skeleton.get_joints_left():
          col = lcol
        else:
          col = mcol

        if fix_i is not None and j in fix_i:
          col = fix_color

        for n, ax in enumerate(axes_3d):
          if fix_0 and n == 0 and index >= num_observed_frames:
            continue
          if fix_0 and n % num_cols == 0 and index >= num_observed_frames:
            continue

          pos = poses[n][index]
          x_array = np.array([pos[j, 0], pos[j_parent, 0]])
          y_array = np.array([pos[j, 1], pos[j_parent, 1]])
          z_array = np.array([pos[j, 2], pos[j_parent, 2]])

          lines_3d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
          lines_3d[n][j - 1][0].set_color(col)
  
  def show_animation(): 
    nonlocal anim 
    if anim is not None : 
      anim.event_source.stop() 
    anim = FuncAnimation(fig, update_video, frames = np.arange(0, poses[0].shape[0]), interval = 0, repeat = True) 
    plt.draw()
  
  def reload_poses():
    nonlocal poses
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('ground_truth'), all_poses.items()))
    for ax, title in zip(axes_3d, poses.keys()):
      ax.set_title(title, y = 1.2)
    poses = list(poses.values())
    
  
  def save_figs():
    nonlocal algo, find
    old_algo = algo
    os.makedirs('out_svg', exist_ok = True)
    suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
    os.makedirs('out_svg_' + suffix, exist_ok = True)
    for algo in algos:
      reload_poses()
      for i in range(0, time_total + 1, 10):
        if i == 0:
          update_video(0)
        else:
          update_video(i - 1)
        fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent = True)
    algo = old_algo
    find += 1
  
  def on_key(event): 
    nonlocal algo, all_poses, animating, anim
    if event.key == 'd':
      all_poses = next(generated_poses)
      reload_poses()
      show_animation()
    elif event.key == 'c':
      save()
    elif event.key == ' ':
      if animating:
          anim.event_source.stop()
      else:
          anim.event_source.start()
      animating = not animating
    elif event.key == 'v':  # save images
      if anim is not None:
        anim.event_source.stop()
        anim = None
      save_figs()
    elif event.key.isdigit():
      algo = algos[int(event.key) - 1]
      reload_poses()
      show_animation()
  
  def save(): 
    nonlocal anim 
    fps = 50 
    anim = FuncAnimation(
      fig, 
      update_video, 
      frames = np.arange(0, poses[0].shape[0]), 
      interval = 1000 / fps,
      repeat = False
    )

    os.makedirs(os.path.dirname(output), exist_ok = True)
    if output.endswith('.mp4'):
      Writer = writers['ffmpeg']
      writer = Writer(fps = fps, metadata = {}, bitrate = bitrate)
      anim.save(output, writer = writer)
    elif output.endswith('.gif'):
      anim.save(output, dpi = 80, writer = 'pillow')    
    else : 
      raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    
    print(f'video saved to {output}')

  fig.canvas.mpl_connect('key_press_event', on_key)
  save() 
  #show_animation()
  #plt.show()
  #plt.close()


def generate_pad(padding, num_observed_frames, num_predict_frames):
  zero_index = None 
  if padding == 'zero':
    idx_pad = list(range(num_observed_frames)) + [num_observed_frames - 1] * num_predict_frames # zero-velocity
    zero_index = max(idx_pad)
  elif padding == 'repeat':
    idx_pad = list(range(num_observed_frames)) * int((num_observed_frames + num_predict_frames) / num_observed_frames)  
  elif padding == 'last_frame':
    idx_pad = list(range(num_observed_frames)) + [num_observed_frames - 1] * num_predict_frames 
  else : 
    raise NotImplementedError(f"unknown padding method : {padding}")
  
  return idx_pad, zero_index

class RunningStats:
  """Tracks a running mean/std via exponential moving average, for cross-batch-consistent normalization."""
  def __init__(self, momentum = 0.99, eps=1e-6):
    self.momentum = momentum
    self.eps = eps
    self.mean = None
    self.var = None

  @torch.no_grad()
  def update(self, x):
    batch_mean = x.mean()
    batch_var = x.var(unbiased = False)
    if self.mean is None:
      self.mean = batch_mean
      self.var = batch_var
    else:
      self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
      self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

  @property
  def std(self):
    return (self.var + self.eps).sqrt()

  def normalize(self, x):
    return (x - self.mean) / (self.std + self.eps)