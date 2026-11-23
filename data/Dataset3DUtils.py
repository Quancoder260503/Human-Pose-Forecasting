"""Set of utility functions."""


import torch
import numpy as np
import copy
import json
import os
import cv2

import torch.nn as nn

def expmap_to_euler(action_sequence):
  rotmats = expmap_to_rotmat(action_sequence)
  eulers = rotmat_to_euler(rotmats)
  return eulers

def expmap_to_rotmat(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 3]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 9].
  """
  n_samples, n_joints, _ = action_sequence.shape
  expmap = np.reshape(action_sequence, [n_samples*n_joints, 1, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  rotmats = np.zeros([n_samples*n_joints, 3, 3])
  for i in range(rotmats.shape[0]):
    rotmats[i] = cv2.Rodrigues(expmap[i])[0]
  rotmats = np.reshape(rotmats, [n_samples, n_joints, 3*3])
  return rotmats

def rotmat_to_expmap(action_sequence):
  """Convert rotmats to expmap.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Rotation exponenital maps [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  expmaps = np.zeros([n_samples*n_joints, 3, 1])
  for i in range(rotmats.shape[0]):
    expmaps[i] = cv2.Rodrigues(rotmats[i])[0]
  expmaps = np.reshape(expmaps, [n_samples, n_joints, 3])

  return expmaps 


def rotmat_to_euler(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Euler angles for rotation maps given [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  eulers = np.zeros([n_samples*n_joints, 3])
  for i in range(eulers.shape[0]):
    eulers[i] = rotmat2euler(rotmats[i])
  eulers = np.reshape(eulers, [n_samples, n_joints, 3])
  return eulers


def rotmat2euler(R):
  """Converts a rotation matrix to Euler angles.
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args:
    R: a 3x3 rotation matrix

  Returns:
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] >= 1 or R[0,2] <= -1:
    # special case values are out of bounds for arcsinc
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;
  else:
    E2 = -np.arcsin(R[0,2])
    E1 = np.arctan2(R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2(R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def load_constants(data_path):
  offset = json.load(open(os.path.join(data_path, 'offset.json')))
  parent = json.load(open(os.path.join(data_path, 'parent.json')))
  rot_ind = json.load(open(os.path.join(data_path, 'rot_ind.json')))

  parent = np.array(parent)-1
  offset = np.array(offset).reshape(-1, 3)
  exp_map_ind = np.split(np.arange(4, 100)-1, 32)

  return parent, offset, rot_ind, exp_map_ind


def compute_forward_kinematics(angles, parent, offset, rotInd, expmapInd):
  """Computes forward kinematics from angles to 3d points.

  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """
  assert len(angles) == 99, 'Incorrect number of angles.'

  # Structure that indicates parents for each joint
  njoints = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange(njoints):
    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[rotInd[i][0]-1]
      yangle = angles[rotInd[i][1]-1]
      zangle = angles[rotInd[i][2]-1]

    r = angles[expmapInd[i]]
    thisRotation = expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz'] = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot(
          xyzStruct[parent[i]]['rotation']) + xyzStruct[parent[i]]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot(
          xyzStruct[parent[i]]['rotation'])

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array(xyz).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]

  return np.reshape( xyz, [-1] )


def revert_coordinate_space(channels, R0, T0):
  """Arrange poses to a canonical form to face the camera.

  Bring a series of poses to a canonical form so they are facing the camera 
  when they start. Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args:
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame

  Returns:
    channels_rec: The passed poses, but the first has T0 and R0, and the 
    rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  for ii in range(n):
    R_diff = expmap2rotmat(channels[ii, rootRotInd])
    R = R_diff.dot(R_prev)

    channels_rec[ii, rootRotInd] = rotmat2expmap(R)
    T = T_prev + (R_prev.T).dot(np.reshape(channels[ii,:3],[3,1])).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def rotmat2quat(R):
  """Converts a rotation matrix to a quaternion.

  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args:
    R: 3x3 rotation matrix

  Returns:
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q


def quat2expmap(q):
  """Convert quaternions to an exponential map.

  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args:
    q: 1x4 quaternion

  Returns:
    r: 1x3 exponential map

  Raises:
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r


def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) )


def expmap2rotmat(r):
  """Converts an exponential map (axis angle number) to rotation matrix.

  Converts an exponential map angle to a rotation matrix Matlab port to python 
  for evaluation purposes. This is also called Rodrigues' formula and can be
  found also implemented in opencv as cv2.Rodrigues.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args:
    r: 1x3 exponential map

  Returns:
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);

  return R


def revert_output_format(
    poses,
    data_mean,
    data_std,
    dim_to_ignore,
    actions,
    use_one_hot):
  """Transforms pose predictions to a more interpretable format.

  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization

  Args:
    poses: Sequence of pose predictions. A list with (seq_length) entries,
      each with a (batch_size, dim) output

  Returns:
    poses_out: List of tensors each of size (batch_size, seq_length, dim).
  """
  seq_len = len(poses)
  if seq_len == 0:
    return []

  batch_size, dim = poses[0].shape

  poses_out = np.concatenate(poses)
  poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  poses_out = np.transpose(poses_out, [1, 0, 2])

  poses_out_list = []
  for i in range(poses_out.shape[0]):
    poses_out_list.append(
      unnormalize_data(poses_out[i, :, :], data_mean, data_std, 
                      dim_to_ignore, actions, use_one_hot))

  return poses_out_list


def unnormalize_data(
    normalizedData, 
    data_mean, 
    data_std, 
    dimensions_to_ignore=None,
    actions=[], 
    use_one_hot=False):
  """
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    use_one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_ignore = [] if dimensions_to_ignore==None else dimensions_to_ignore
  dimensions_to_use = [i for i in range(D) if i not in dimensions_to_ignore]
  dimensions_to_use = np.array(dimensions_to_use)
  #print('Size of the normalized data', normalizedData.shape)
  #print('Size of the mean data', data_mean.shape[0])
  #print('Lenght of the dimensions to use', len(dimensions_to_use))

  if use_one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData



def create_dir_tree(base_dir):
  dir_tree = ['models', 'tf_logs', 'config', 'std_log']
  for dir_ in dir_tree:
    os.makedirs(os.path.join(base_dir, dir_), exist_ok=True)


def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
  """Generates a binary mask to prevent to use future context in a sequence."""
  if is_nonautoregressive:
    return np.zeros((seq_length, seq_length), dtype=np.float32)
  x = np.ones((seq_length, seq_length), dtype=np.float32)
  mask = np.triu(x, 1).astype(np.float32)
  return mask  # (seq_len, seq_len)


def pose_expmap2rotmat(input_pose):
  """Convert exponential map pose format to rotation matrix pose format."""
  pose_rotmat = []
  for j in np.arange(input_pose.shape[0]):
    rot_mat = [expmap2rotmat(input_pose[j, k:k+3]) for k in range(3, 97, 3)]
    pose_rotmat.append(np.stack(rot_mat).flatten())

  pose_rotmat = np.stack(pose_rotmat)
  return pose_rotmat


def expmap23d_sequence(sequence, norm_stats, params):
  viz_poses = revert_output_format(
      [sequence], norm_stats['mean'], norm_stats['std'],
      norm_stats['dim_to_ignore'], params['action_subset'],
      params['use_one_hot'])

  nframes = sequence.shape[0]
  expmap = revert_coordinate_space(
          viz_poses[0], np.eye(3), np.zeros(3))
  xyz_data = np.zeros((nframes, 96))
  for i in range(nframes):
    xyz_data[i, :] = compute_forward_kinematics(
        expmap[i, :],
        params['parent'], 
        params['offset'], 
        params['rot_ind'], 
        params['exp_map_ind']
    )

  return xyz_data

