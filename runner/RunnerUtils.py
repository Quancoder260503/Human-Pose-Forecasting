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
from scipy import linalg
import numpy as np
from einops import rearrange

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
  
   
def get_multimodal_dataset(logger, dataset_multi_test, params, required_action = None):
  """
  Multi-modal evaluation set. The pseudo ground-truth futures of every window are searched over the WHOLE test set,
  as in the full evaluation; required_action then only selects which windows are evaluated.
  """
  logger.info('=> preparing evaluation dataset ...')
  num_obs = params['num_observed_frames']

  if params['dataset'] == 'amass':
    data_groups = dataset_multi_test.data
    req_action = [] 
    for index, data in data_groups:
      req_action.append(dataset_multi_test.get_action_category(index))
    req_action = np.array(req_action)
  else:
    actions_category = sorted({a for s in dataset_multi_test.subjects for a in dataset_multi_test.data[s].keys()})
    data_groups, req_action = [], []
    for action in actions_category:                               
      for data, _ in dataset_multi_test.iter_generator(step = num_obs, required_action = action):
        data_groups.append(data)
        req_action.extend([action] * data.shape[0])
    data_groups, req_action = np.concatenate(data_groups, axis = 0), np.array(req_action)

  all_data = data_groups[..., 1:, :].reshape(data_groups.shape[0], data_groups.shape[1], -1)
  pairwise_distance = squareform(pdist(all_data[:, num_obs - 1, :]))      
  trajectory_gt_arr = [all_data[np.nonzero(row < params['multimodal_threshold'])][:, num_obs:, :]
                       for row in pairwise_distance]

  keep = np.arange(len(all_data)) if required_action is None else np.nonzero(req_action == required_action)[0]
  num_mult = np.array([len(trajectory_gt_arr[i]) for i in keep])
  logger.info('=' * 80)
  logger.info(f'#1 future: {np.sum(num_mult == 1)}/{len(keep)}')
  logger.info(f'#<10 future: {np.sum(num_mult < 10)}/{len(keep)}')
  logger.info('=' * 80)

  return {
    'trajectory_gt_array' : [trajectory_gt_arr[i] for i in keep],
    'data_group' : data_groups[keep],
    'gt_group' : all_data[keep][:, num_obs:, :],
    'num_samples' : len(keep),
  }

# def get_multimodal_dataset(logger, dataset_multi_test, params):
#   logger.info('=> preparing evaluation dataset ...')
#   data_groups = []
#   num_samples = 0

#   if params['dataset'] == 'amass':
#     data_groups = dataset_multi_test.data
#     num_samples = data_groups.shape[0]
#   else : 
#     data_generator_multi_test = dataset_multi_test.iter_generator(step = params['num_observed_frames'])
#     for data, _ in data_generator_multi_test: 
#       num_samples += 1 
#       data_groups.append(data)
#     data_groups = np.concatenate(data_groups, axis = 0) 

#   all_data = data_groups[..., 1:, :].reshape(data_groups.shape[0], data_groups.shape[1], -1)
#   gt_groups = all_data[:, params['num_observed_frames'] :, :]

#   all_start_pose = all_data[:, params['num_observed_frames'] - 1, :]
#   pairwise_distance = squareform(pdist(all_start_pose))

#   trajectory_gt_arr = [] 
#   num_mult = [] 

#   for i in range(pairwise_distance.shape[0]):
#     index = np.nonzero(pairwise_distance[i] < params['multimodal_threshold'])
#     trajectory_gt_arr.append(all_data[index][:, params['num_observed_frames'] :, :])
#     num_mult.append(len(index[0]))

#   num_mult = np.array(num_mult)
#   logger.info('=' * 80)
#   logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pairwise_distance.shape[0]}')
#   logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pairwise_distance.shape[0]}')
#   logger.info('done...')
#   logger.info('=' * 80)


#   return { 
#     'trajectory_gt_array' : trajectory_gt_arr, 
#     'data_group' : data_groups, 
#     'gt_group' : gt_groups, 
#     'num_samples' : num_samples, 
#   }


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
  sampling_mode = 'pred', 
  size = 2, 
  num_cols = 5, 
  bitrate = 3000, 
  fix_index = None 
): 
  if sampling_mode == 'switch': 
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
    if sampling_mode == 'switch':
      if index == 0:
        ax.set_title('target', y = 1.0, fontsize = 12)
    if sampling_mode == 'pred' or 'fix' in sampling_mode or sampling_mode == 'control' or sampling_mode == 'zero_shot':
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
    if sampling_mode == 'switch':
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



def matrix_sqrt(matrix):
  """
  The principal matrix square root.

  ``linalg.sqrtm`` is called without the ``disp`` argument, which SciPy
  deprecated in 1.17 and removes in 1.18. Omitting it returns the matrix alone
  under every version; the tuple form is handled defensively regardless.

  :param matrix : an array of shape (n, n)
  :return :
    an array of shape (n, n), the principal square root of the input
  """
  result = linalg.sqrtm(matrix)
  return result[0] if isinstance(result, tuple) else result


def calculate_activation_statistics(activations):
  """
  :param activations : an array of shape (N, feature_dim)
  :return :
    mu    : an array of shape (feature_dim, )
    sigma : an array of shape (feature_dim, feature_dim)
  """
  mu = np.mean(activations, axis = 0)
  sigma = np.cov(activations, rowvar = False)
  return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps = 1e-6):
  """
  The Frechet distance between two multivariate Gaussians
  X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is

    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 * sqrt(C_1 * C_2)).

  Stable implementation following Dougal J. Sutherland.

  :param mu1    : an array of shape (feature_dim, ), mean over generated activations
  :param sigma1 : an array of shape (feature_dim, feature_dim), covariance over generated activations
  :param mu2    : an array of shape (feature_dim, ), mean over reference activations
  :param sigma2 : an array of shape (feature_dim, feature_dim), covariance over reference activations
  :param eps    : the offset added to the diagonal when the matrix product is singular
  :return :
    a scalar, the Frechet distance
  """
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)
  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
    'generated and reference mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'generated and reference covariances have different dimensions'

  diff = mu1 - mu2

  covmean = matrix_sqrt(sigma1.dot(sigma2))

  if not np.isfinite(covmean).all():
    print(
      'fid calculation produces singular product; '
      'adding %s to diagonal of cov estimates' % eps
    )
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = matrix_sqrt((sigma1 + offset).dot(sigma2 + offset))

  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
      raise ValueError('imaginary component {}'.format(np.max(np.abs(covmean.imag))))
    covmean = covmean.real

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_fid(activations_1, activations_2):
  """
  :param activations_1 : an array of shape (N_1, feature_dim), generated activations
  :param activations_2 : an array of shape (N_2, feature_dim), reference activations
  :return :
    a scalar, the Frechet distance between the two activation sets
  """
  mu1, sigma1 = calculate_activation_statistics(activations_1)
  mu2, sigma2 = calculate_activation_statistics(activations_2)
  return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def compute_fid_statistics(
  pred,
  gt,
  classifier_for_fid,
  all_pred_activations,
  all_gt_activations,
  all_pred_classes,
  all_gt_classes,
):
  """
  Accumulates classifier activations and predicted action labels for one batch,
  appending them in place to the supplied lists.
  :param pred : a tensor of shape (B, num_samples, T, pose_dim)
  :param gt : a tensor of shape (B, T, pose_dim)
  :param classifier_for_fid : the action classifier supplying FID features
  :param all_pred_activations : a list accumulating arrays of shape (B * num_samples, feature_dim)
  :param all_gt_activations : a list accumulating arrays of shape (B, feature_dim)
  :param all_pred_classes : a list accumulating arrays of shape (B, num_samples)
  :param all_gt_classes : a list accumulating arrays of shape (B, )
  :return :
    None, the lists are modified in place
  """
  batch_size, num_samples = pred.shape[0], pred.shape[1]

  pred = rearrange(pred, 'b s t d -> (b s) d t')
  gt = rearrange(gt, 'b t d -> b d t')

  with torch.no_grad():
    pred_activations = classifier_for_fid.get_fid_features(motion_sequence = pred)
    gt_activations = classifier_for_fid.get_fid_features(motion_sequence = gt)

    pred_logits = classifier_for_fid.linear2(pred_activations)
    gt_logits = classifier_for_fid.linear2(gt_activations)

  pred_activations = pred_activations.cpu().numpy()
  gt_activations = gt_activations.cpu().numpy()

  pred_classes = pred_logits.argmax(dim = 1).cpu().numpy().reshape([batch_size, num_samples])
  gt_classes = gt_logits.argmax(dim = 1).cpu().numpy()

  all_pred_activations.append(pred_activations)
  all_gt_activations.append(gt_activations)
  all_pred_classes.append(pred_classes)
  all_gt_classes.append(gt_classes)
  return


def cumulative_motion_distribution(motion_per_frame, motion_reference):
  """
  :param motion_per_frame : an array of shape (T - 1, ), the mean motion magnitude at each frame
  :param motion_reference : a scalar, the dataset mean motion for the corresponding action class
  :return :
    a scalar, the frame-weighted cumulative deviation from the reference
  """
  num_frames = len(motion_per_frame) + 1
  weights = np.arange(num_frames - 1, 0, -1)
  return np.sum(weights * np.abs(motion_per_frame - motion_reference))


def compute_cmd_statistics(pred, actions, histogram_data, all_obs_classes):
  """
  Accumulates per-frame motion magnitudes and action labels for one batch,
  appending them in place to the supplied lists.

  :param pred             : a tensor of shape (B, num_samples, T, pose_dim)
  :param actions          : a tensor or array of shape (B, ), the observed action labels
  :param histogram_data   : a list accumulating arrays of shape (B, T - 1)
  :param all_obs_classes  : a list accumulating arrays of shape (B, )
  :return :
    None, the lists are modified in place
  """
  pred_joints = rearrange(pred, '... (n c) -> ... n c', c = 3)

  displacement = pred_joints[:, :, 1:] - pred_joints[:, :, :-1]
  motion = torch.linalg.norm(displacement, dim = -1)
  motion = motion.mean(dim = 1).mean(dim = -1)
  histogram_data.append(motion.detach().cpu().numpy())
  if torch.is_tensor(actions):
    actions = actions.detach().cpu().numpy()
  all_obs_classes.append(actions)
  return

def calculate_cmd(dataset, histogram_data, all_obs_classes):
  """
  :param dataset          : the dataset, supplying idx_to_class and mean_motion_per_class
  :param histogram_data   : the accumulated per-frame motion magnitudes
  :param all_obs_classes  : the accumulated action labels
  :return :
    a scalar, the class-frequency-weighted CMD
  """
  obs_classes = np.concatenate(all_obs_classes, axis = 0)
  motion_data = np.concatenate(histogram_data, axis = 0)

  total = 0.0
  references = zip(dataset.idx_to_class, dataset.mean_motion_per_class)

  for index, (_, class_reference) in enumerate(references):
    mask = (obs_classes == index)
    if mask.sum() == 0:
      continue
    class_motion = motion_data[mask].mean(axis = 0)
    class_weight = mask.sum() / obs_classes.shape[0]
    total += cumulative_motion_distribution(class_motion, class_reference) * class_weight
  return total



