import time
import os 
import numpy as np 
from abc import abstractmethod, ABC
import argparse

def absolute_to_relative(x, parents, invert = False, x_0 = None): 
  if not invert : 
    x_relative = x[..., 1:, :] - x[..., parents[1:], :]
    x_relative = x_relative / np.linalg.norm(x_relative, axis = -1, keepdims = True)
    return x_relative 
  else : 
    num_joints = x_0.shape[-2]
    limb_l = np.linalg.norm(x_0[..., 1:, :] - x_0[..., parents[1:], :], axis = -1, keepdims = True)
    x_relative = x * limb_l
    x_relative_root = np.zeros_like(x_relative[..., :1, :])
    x_relative = np.concatenate([x_relative_root, x_relative], axis = -2)
    for i in range(1, num_joints):
      x_relative[..., i, :] = x_relative[..., parents[i], :] + x_relative[..., i, :]
    return x_relative


class Skeleton(object): 
  def __init__(self, parents, joints_left, joints_right):
    assert len(joints_left) == len(joints_right)

    self.parents = np.array(parents)
    self.joints_left = np.array(joints_left)
    self.joints_right = np.array(joints_right)
    self.compute_metadata() 

  def get_num_joints(self):
    return len(self.parents)
  
  def get_parents(self):
    return self.parents 
  
  def has_children(self):
    return self._has_children  
  
  def get_children(self):
    return self.children 
  
  def get_joints_left(self):
    return self.joints_left 
  
  def get_joints_right(self):
    return self.joints_right 
  
  def compute_metadata(self):
    self._has_children = np.zeros(len(self.parents)).astype(bool) 
    for i, parent in enumerate(self.parents): 
      if parent != -1: 
        self._has_children[parent] = True 
    
    self.children = [] 

    for i, parent in enumerate(self.parents):
      self.children.append([])
    
    for i, parent in enumerate(self.parents):
      if parent != -1: 
        self.children[parent].append(i)


  def remove_joints(self, joints_to_remove):
    valid_joints = [] 
    for joint in range(len(self.parents)):
      if joint not in joints_to_remove : 
        valid_joints.append(joint)
    
    for i in range(len(self.parents)):
      while self.parents[i] in joints_to_remove: 
        self.parents[i] = self.parents[self.parents[i]]
      
    index_offsets = np.zeros(len(self.parents), dtype = int) 
    new_parents = [] 
    
    for i, parent in enumerate(self.parents): 
      if i not in joints_to_remove: 
        new_parents.append(parent - index_offsets[parent])
      else : 
        index_offsets[i:] += 1 
    
    self.parents = np.array(new_parents)

    if self.joints_left is not None:
      new_joints_left = []
      for joint in self.joints_left:
        if joint in valid_joints:
          new_joints_left.append(joint - index_offsets[joint])
      self.joints_left = np.array(new_joints_left)
    
    if self.joints_right is not None:
      new_joints_right = []
      for joint in self.joints_right:
        if joint in valid_joints:
          new_joints_right.append(joint - index_offsets[joint])
      self.joints_right = np.array(new_joints_right)

    self.compute_metadata()
        
    return valid_joints


class Dataset:
  def __init__(self, mode, num_observed_frames, num_predict_frames, dim_per_joints, actions = 'all'):
    self.mode = mode 
    self.num_observed_frames = num_observed_frames 
    self.num_predict_frames = num_predict_frames 
    self.total_frames = num_predict_frames + num_observed_frames 
    self.dim_per_joints = dim_per_joints
    self.actions = actions 
    self.prepare_data()
    self.std, self.mean = None, None
    self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
    self.trajectory_dim = (self.major_joints.shape[0] - 1) * dim_per_joints
    self.normalized = False  
    self.sample_index = None 

  @abstractmethod
  def prepare_data(self):
    pass 
  
  def normalize_data(self, mean = None, std = None):
    if mean is None: 
      all_seqs = [] 
      for data_sequences in self.data.values():
        for sequence in data_sequences.values():
          all_seqs.append(sequence[:, 1:])
      all_seqs = np.concatenate(all_seqs)
      self.mean = all_seqs.mean(axis = 0) 
      self.std = all_seqs.std(axis = 0) 
    else : 
      self.mean = mean 
      self.std = std 
    
    for data_sequences in self.data.values():
      for action in data_sequences.keys():
        data_sequences[action][:, 1:] = (data_sequences[action][:, 1:] - self.mean) / self.std 
    
    self.normalized = True 
  
  def sample(self): 
    subject = np.random.choice(self.subjects)
    dict_s = self.data[subject]
    action = np.random.choice(list(dict_s.keys()))
    sequences = dict_s[action]
    frame_start = np.random.randint(sequences.shape[0] - self.total_frames)
    frame_end = frame_start + self.total_frames 
    trajectory = sequences[frame_start : frame_end]
    return trajectory[None, ...]
  
  def sample_all_action(self):
    dict_s = self.data[self.test_subject]
    action_list = []
    samples = []

    for i in range(0, len(list(dict_s.keys()))):
      type = list(dict_s.keys())[i].split(' ')[0] 
      if type == 'Discussion': 
        type = 'Discussion 1'
      action_list.append(type) 
    
    action_list = list(set(action_list))

    for i in range(0, len(action_list)):
      action = action_list[i]
      sequences = dict_s[action] 
      frame_start = np.random.randint(sequences.shape[0] - self.total_frames)
      frame_end = frame_start + self.total_frames 
      trajectory = sequences[frame_start : frame_end]
      samples.append(trajectory[None, ...])

    samples = np.concatenate(samples, axis = 0) 
    return samples 
  
  def sample_iter_action(self, action_category):
    # subject = np.random.choice(self.subjects)
    dict_s = self.data[self.test_subject]
    sample = [] 
    action = action_category
    seq = dict_s[action]
    frame_start = np.random.randint(seq.shape[0] - self.total_frames)
    frame_end = frame_start + self.total_frames
    traj = seq[frame_start: frame_end]
    sample.append(traj[None, ...])
    sample = np.concatenate(sample, axis = 0)
    return sample
  


  def prepare_iter_action(self):
    dict_s = self.data[self.test_subject]
    action_list = [] 
    for i in range(0, len(list(dict_s.keys()))):
      type = list(dict_s.keys())[i].split(' ')[0] 
      if type == 'Discussion': 
        type = 'Discussion 1'
      action_list.append(type) 
    
    action_list = list(set(action_list))
    return action_list
  
  def sampling_generator(self, num_samples = 1000, batch_size = 8, augment = True):
    for i in range(num_samples // batch_size): 
      samples = [] 
      for i in range(batch_size): 
        samples.append(self.sample())
      samples = np.concatenate(samples, axis = 0)
      if augment is True:
        # data augmentation
        if np.random.uniform() > 0.5 :
          theta = np.random.uniform(0, 2 * np.pi)
          rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
          rotate_xy = np.matmul(samples.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
          samples[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
          del theta, rotate_matrix, rotate_xy 
        if np.random.uniform() > 0.5:  # x-z mirroring
          samples[..., 0] = -samples[..., 0]        
        if np.random.uniform() > 0.5:  # y-z mirroring
          samples[..., 1] = -samples[..., 1]
        yield samples

  def iter_generator(self, step = 25):
    for data_sequence in self.data.values():
      for sequence in data_sequence.values():
        sequence_len = sequence.shape[0] 
        for i in range(0, sequence_len - self.total_frames, step):
          trajectory = sequence[None, i : (i + self.total_frames)]
          yield trajectory


"""
H36M dataset class 
"""
class DatasetH36M(Dataset):
  def __init__(self, mode, num_observed_frames, num_predict_frames, dim_per_joints, actions = 'all', use_velocity = False, **kwargs):
   if 'multimodal_path' in kwargs.keys():
     self.multimodal_path = kwargs['multimodal_path']
   else : 
     self.multimodal_path = None 

   if 'data_candidate_path' in kwargs.keys():
     self.data_candidate_path = kwargs['data_candidate_path']
   else : 
     self.data_candidate_path = None 
   
   self.use_velocity = use_velocity 
   super().__init__(mode, num_observed_frames, num_predict_frames, dim_per_joints, actions)
   if self.use_velocity:
    self.trajectory_dim += self.dim_per_joints
   
   self.test_subject = 'S9'

  def prepare_data(self):
    self.data_file = os.path.join('dataset', 'multimodal_set', 'data_3d_h36m.npz')
    self.subjects_split = {'train': [1, 5, 6, 7, 8], 'test': [9, 11]}
    self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]

    self.skeleton = Skeleton(
      parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
      joints_left = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
      joints_right = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    )

    self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
    self.major_joints = np.array([x for x in range(32) if x not in self.removed_joints])
    self.skeleton.remove_joints(self.removed_joints)

    self.skeleton.parents[11] = 8 
    self.skeleton.parents[14] = 8
    
    self.process_data()
  
  def process_data(self):
    data_dicts = np.load(self.data_file, allow_pickle = True)['positions_3d'].item()
    self.s1_skeleton = data_dicts['S1']['Directions'][:1, self.major_joints].copy()
    data_filtered = dict(filter(lambda x : x[0] in self.subjects, data_dicts.items()))

    if self.actions != 'all':
      #print(data_filtered['S1'].items(), self.actions)
      for key in list(data_filtered.keys()): 
        data_filtered[key] = dict(filter(lambda x : all([a in str.lower(x[0])\
          for a in self.actions]), data_filtered[key].items()))
        if len(data_filtered[key]) == 0: 
          data_filtered.pop(key)

    for data_sequences in data_filtered.values():
      for action in data_sequences.keys():
        sequence = data_sequences[action][:, self.major_joints, :] 
        if self.use_velocity:
          velocity = (np.diff(sequence[:, :1], axis = 0) * 50).clip(-5.0, 5.0)
          velocity = np.append(velocity, velocity[[-1]], axis = 0)
          
        sequence[:, 1:] = sequence[:, 1:] - sequence[:, :1]
        if self.use_velocity : 
          sequence = np.concatenate((sequence, velocity), axis = 1)

        data_sequences[action] = sequence 
    self.data = data_filtered


class DatasetH36M_Multi(Dataset):
  def __init__(self, mode, num_observed_frames, num_predict_frames, dim_per_joints, actions = 'all', use_velocity = False, **kwargs):
   if 'multimodal_path' in kwargs.keys():
     self.multimodal_path = kwargs['multimodal_path']
   else : 
     self.multimodal_path = None 

   if 'data_candidate_path' in kwargs.keys():
     self.data_candidate_path = kwargs['data_candidate_path']
   else : 
     self.data_candidate_path = None

   self.use_velocity = use_velocity 
   super().__init__(mode, num_observed_frames, num_predict_frames, dim_per_joints, actions)
   if self.use_velocity:
    self.trajectory_dim += self.dim_per_joints
   
   self.test_subject = 'S9'

  def prepare_data(self):
    self.subjects_split = {'train': [1, 5, 6, 7, 8], 'test': [9, 11]}
    self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]

    self.skeleton = Skeleton(
      parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
      joints_left = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
      joints_right = [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
    )

    self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
    self.major_joints = np.array([x for x in range(32) if x not in self.removed_joints])
    self.skeleton.remove_joints(self.removed_joints)

    self.skeleton.parents[11] = 8 
    self.skeleton.parents[14] = 8
    
    self.process_data()
  
  def process_data(self):
    self.data_file = os.path.join('dataset', 'multimodal_set', 'data_3d_h36m.npz')
    data_dicts = np.load(self.data_file, allow_pickle = True)['positions_3d'].item()
    self.s1_skeleton = data_dicts['S1']['Directions'][:1, self.major_joints].copy()
    data_filtered = dict(filter(lambda x : x[0] in self.subjects, data_dicts.items()))

    if self.actions != 'all':
      for key in list(data_filtered.keys()): 
        data_filtered[key] = dict(filter(lambda x : all([a in str.lower(x[0])\
                                   for a in self.actions]), data_filtered[key].items()))
        if len(data_filtered[key]) == 0: 
          data_filtered.pop(key)

    assert self.multimodal_path is not None, "For now, cannot process if the multimodal path is none"
    #if self.multimodal_path is not None : 
    self.data_multimodal = np.load(self.multimodal_path, allow_pickle = True)['data_multimodal'].item()
    data_candidate = np.load(self.data_candidate_path, allow_pickle = True)['data_candidate.npy']
    
    self.data_candidate = {}
    for subject in data_filtered.keys(): 
      data_sequences = data_filtered[subject]
      for action in data_sequences.keys():
        sequence = data_sequences[action][:, self.major_joints, :] 
        if self.use_velocity:
          velocity = (np.diff(sequence[:, :1], axis = 0) * 50).clip(-5.0, 5.0)
          velocity = np.append(velocity, velocity[[-1]], axis = 0)
        
        sequence[:, 1:] = sequence[:, 1:] - sequence[:, :1]


        if self.use_velocity : 
          sequence = np.concatenate((sequence, velocity), axis = 1)

        data_sequences[action] = sequence 

        if subject not in self.data_candidate.keys():
          x_0 = np.copy(sequence[None, :1, ...])
          x_0[:, :, 0] = 0 
          self.data_candidate[subject] = absolute_to_relative(
            data_candidate, 
            parents = self.skeleton.parents, 
            invert = True, 
            x_0 = x_0 
          )      
  
    self.data = data_filtered
  
  def sample(self, n_modality = 5): 
    subject = np.random.choice(self.subjects)
    dict_sequences = self.data[subject]
    action = np.random.choice(list(dict_sequences.keys()))

    sequences = dict_sequences[action]

    frame_start = np.random.randint(sequences.shape[0] - self.total_frames)
    frame_end = frame_start + self.total_frames 
    trajectory = sequences[frame_start : frame_end]

    if n_modality > 0 and subject in self.data_multimodal.keys():
      curr_candidate = self.data_candidate[subject]
      idx_multimodal = self.data_multimodal[subject][action][frame_start]
      trajectory_multimodal = curr_candidate[idx_multimodal]

      if len(trajectory_multimodal) > 0 : 
        trajectory_multimodal[:, : self.num_observed_frames] = trajectory[None, ...][:, : self.num_observed_frames]
        if trajectory_multimodal.shape[0] > n_modality:
          state = np.random.get_state()

          curr_index = np.random.choice(np.arange(trajectory_multimodal.shape[0]), n_modality, replace = False) 
          trajectory_multimodal = trajectory_multimodal[curr_index]

          np.random.set_state(state)

      trajectory_multimodal = np.concatenate(
        [trajectory_multimodal, np.zeros_like(trajectory[None, ...][[0] * (n_modality - trajectory_multimodal.shape[0])])], 
        axis = 0
      )

      return trajectory[None, ...], trajectory_multimodal
    else : 
      return trajectory[None, ...], None 
    
  def sampling_generator(self, num_samples = 1000, batch_size = 8, n_modality = 5): 
    for _ in range(num_samples // batch_size):
      samples = [] 
      samples_multi = [] 
      for i in range(batch_size):
        curr_sample, curr_multi_sample = self.sample(n_modality = n_modality)
        samples.append(curr_sample)
        samples_multi.append(curr_multi_sample[None, ...])

      samples = np.concatenate(samples, axis = 0) 
      samples_multi = np.concatenate(samples_multi, axis = 0) 
      yield samples, samples_multi
  

  def iter_generator(self, step = 25, n_modality = 10):
    for subject in self.data.keys():
      data_sequence = self.data[subject]
      curr_candidate = self.data_candidate[subject]

      for action in data_sequence.keys():
        sequence = data_sequence[action] 
        sequence_len = sequence.shape[0]

        for i in range(0, sequence_len - self.total_frames, step): 
          trajectory = sequence[None, i : (i + self.total_frames)]
          if n_modality > 0: 
            margin_frame = 1 
            observed_threshold = 0.05 
            predict_threshold = 0.1 
            x_0 = np.copy(trajectory)
            x_0[:, :, 0] = 0

            # observed_distance 
            observed_dist = np.mean(np.linalg.norm(x_0[:, (self.num_observed_frames - margin_frame) : self.num_observed_frames, 1:] -\
             curr_candidate[:, (self.num_observed_frames - margin_frame) : self.num_observed_frames, 1:], axis = 3), axis = (1, 2))
            
            observed_index = np.where(observed_dist <= observed_threshold)[0]

            # future distance 
            predict_dist = np.mean(np.linalg.norm(x_0[:, self.num_observed_frames :, 1:] - 
              curr_candidate[observed_index, self.num_observed_frames :, 1:], axis = 3), axis = (1, 2))
            
            predict_index = np.where(predict_dist >= predict_threshold)[0]

            trajectory_multi = curr_candidate[observed_index[predict_index]]

            if len(trajectory_multi) > 0:
              trajectory_multi[:, : self.num_observed_frames] = trajectory[:, : self.num_observed_frames]

              if trajectory_multi.shape[0] > n_modality:
                trajectory_multi = trajectory_multi[: n_modality]

            trajectory_multi = np.concatenate(
              [trajectory_multi, np.zeros_like(trajectory[[0] * (n_modality - trajectory_multi.shape[0])])], 
            axis = 0)
          else : 
            trajectory_multi = None 
          
          yield trajectory, trajectory_multi
            



"""
HumanEva Dataset 
"""
class DatasetHumanEva(Dataset): 
  def __init__(self, mode, num_observed_frames, num_predict_frames, dim_per_joints, actions='all'):
    super().__init__(mode, num_observed_frames, num_predict_frames, dim_per_joints, actions)
  
  def prepare_data(self):
    self.subjects_split = {
      'train': ['Train/S1', 'Train/S2', 'Train/S3'],
      'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']
    }
    self.subjects = [x for x in self.subjects_split[self.mode]]
    self.skeleton = Skeleton(
      parents = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
      joints_left = [2, 3, 4, 8, 9, 10],
      joints_right = [5, 6, 7, 11, 12, 13]
    )
    self.major_joints = np.arange(15)
    self.test_subject = 'Validate/S2'
    self.process_data()

  def process_data(self): 
    self.data_file = os.path.join('dataset', 'multimodal_set', 'data_3d_humaneva15.npz')
    data_dicts = np.load(self.data_file, allow_pickle = True)['positions_3d'].item()
    data_filtered = dict(filter(lambda x : x[0] in self.subjects, data_dicts.items()))

    if self.mode == 'train': 
      data_filtered['Train/S3'].pop('Walking 1 chunk0')
      data_filtered['Train/S3'].pop('Walking 1 chunk2')
    else : 
      data_filtered['Validate/S3'].pop('Walking 1 chunk4')

    for key in list(data_filtered.keys()):
      data_filtered[key] = dict(
        filter(lambda x : (self.actions == 'all' or all([a in x[0] for a in self.actions])) \
        and x[1].shape[0] >= self.total_frames, data_filtered[key].items())
      )

      if len(data_filtered[key]) == 0: 
        data_filtered.pop(key)
      
    for data_sequences in data_filtered.values():
      for action in data_sequences.keys():
        sequence = data_sequences[action][:, self.major_joints, :] 
        sequence[:, 1:]  = sequence[:, 1:] - sequence[:, :1]
        data_sequences[action] = sequence
    
    self.data = data_filtered
    

class DatasetHumanEvaMulti(Dataset):
  def __init__(self, mode, num_observed_frames, num_predict_frames, dim_per_joints, actions = 'all', **kwargs):
    if 'multimodal_path' in kwargs.keys():
      self.multimodal_path = kwargs['multimodal_path']
    else : 
      self.multimodal_path = None 

    if 'data_candidate_path' in kwargs.keys():
      self.data_candidate_path = kwargs['data_candidate_path']
    else : 
      self.data_candidate_path = None

    super().__init__(mode, num_observed_frames, num_predict_frames, dim_per_joints, actions)
    self.test_subject = None

  def prepare_data(self):
    self.data_file = os.path.join('dataset', 'multimodal_set', 'data_3d_h36m.npz')
    self.subjects_split = {
      'train': ['Train/S1', 'Train/S2', 'Train/S3'],
      'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']
    }
    self.subjects = [x for x in self.subjects_split[self.mode]]
    self.skeleton = Skeleton(
      parents = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
      joints_left = [2, 3, 4, 8, 9, 10],
      joints_right = [5, 6, 7, 11, 12, 13]
    )
    self.major_joints = np.arange(15)
    self.process_data()


  def process_data(self): 
    self.data_file = os.path.join('dataset', 'multimodal_set', 'data_3d_humaneva15.npz')
    data_dicts = np.load(self.data_file, allow_pickle = True)['positions_3d'].item()
    data_filtered = dict(filter(lambda x : x[0] in self.subjects, data_dicts.items()))

    if self.mode == 'train': 
      data_filtered['Train/S3'].pop('Walking 1 chunk0')
      data_filtered['Train/S3'].pop('Walking 1 chunk2')
    else : 
      data_filtered['Validate/S3'].pop('Walking 1 chunk4')

    assert self.multimodal_path is not None, "For now, cannot process if the multimodal path is none"
    #if self.multimodal_path is not None : 
    self.data_multimodal = np.load(self.multimodal_path, allow_pickle = True)['data_multimodal'].item()
    data_candidate = np.load(self.data_candidate_path, allow_pickle = True)['data_candidate.npy']

    for key in list(data_filtered.keys()):
      data_filtered[key] = dict(
        filter(lambda x : (self.actions == 'all' or all([a in x[0] for a in self.actions])) \
        and x[1].shape[0] >= self.total_frames, data_filtered[key].items())
      )
      if len(data_filtered[key]) == 0: 
        data_filtered.pop(key)

    self.data_candi = {}
      
    for subject in data_filtered.keys():
      data_sequences = data_filtered[subject]
      for action in data_sequences.keys():
        sequence = data_sequences[action][:, self.major_joints, :] 
        sequence[:, 1:]  = sequence[:, 1:] - sequence[:, :1]
        data_sequences[action] = sequence

        if subject not in data_candidate.keys():
          x_0 = np.copy(sequence[None, :1, ...])
          x_0[:, :, 0] = 0
          self.data_candi[subject] = absolute_to_relative(
            data_candidate, 
            self.skeleton.parents, 
            True, 
            x_0 = x_0
          )
    self.data = data_filtered

  def sample(self, n_modality = 5): 
    while True:
      subject = np.random.choice(self.subjects)
      dict_seq = self.data[subject]
      action = np.random.choice(list(dict_seq.keys()))
      sequence = dict_seq[action] 
      if sequence.shape[0] > self.total_frames : 
        break 
    
    frame_start = np.random.randint(sequence.shape[0] - self.total_frames)
    frame_end = frame_start + self.total_frames 

    trajectory = sequence[frame_start : frame_end]

    if n_modality > 0: 
      curr_candidate = self.data_candi[subject]
      idx_multi = self.data_multimodal[subject][action][frame_start] 
      trajectory_multi = curr_candidate[idx_multi]

      if len(idx_multi) > 0: 
        x_0 = np.copy(trajectory[None, ...])
        x_0[:, :, 0] = 0
        
      if len(trajectory_multi) > 0: 
        trajectory_multi[: : self.num_observed_frames] = trajectory[None, ...][:, : self.num_observed_frames]
        if trajectory_multi.shape[0] > n_modality:
          state = np.random.get_state() 
          curr_idx = np.random.choice(np.arange(trajectory_multi.shape[0]), n_modality, replace = False)
          trajectory_multi = trajectory_multi[curr_idx]
          np.random.set_state(state)
      
      trajectory_multi = np.concatenate(
        [trajectory_multi, np.zeros_like(trajectory[None, ...][[0] * (n_modality - trajectory_multi.shape[0])])], 
        axis = 0
      )

      return trajectory[None, ...], trajectory_multi
    
    else : 
      return trajectory[None, ...], None 


  def sampling_generator(self, num_samples = 1000, batch_size = 8, n_modality = 5):
    for _ in range(num_samples // batch_size):
      samples = [] 
      samples_multi = [] 
      for i in range(batch_size):
        curr_sample, curr_multi_sample = self.sample(n_modality = n_modality)
        samples.append(curr_sample)
        samples_multi.append(curr_multi_sample[None, ...])

      samples = np.concatenate(samples, axis = 0) 
      samples_multi = np.concatenate(samples_multi, axis = 0) 
      yield samples, samples_multi 

  def iter_generator(self, step = 25):
    for data_sequences in self.data.values():
      for sequence in data_sequences.values():
        seq_len = sequence.shape[0]
        for i in range(0, seq_len - self.total_frames, step): 
          trajectory = sequence[None, i : i + self.total_frames]
          yield trajectory, None


    

"""
Multimodal NTURGB+D
"""


# if __name__ == '__main__': 
#   # generate the candidate 
#   parser = argparse.ArgumentParser() 
#   parser.add_argument('--subject', default = ['S1', 'S5', 'S6', 'S7', 'S8'])
#   parser.add_argument('--action', default = 'all')
#   parser.add_argument('--num_observed_frames', default = 25, type = int) 
#   parser.add_argument('--num_predict_frames', default = 100, type = int)
#   parser.add_argument('--skip_rate', default = 20, type = int)
#   args = parser.parse_args()
#   # change this manually for NTURGB+D and HumanEVA
#   dataset = 'h36M'
#   train_dataset = DatasetH36M(
#     mode = 'train', 
#     num_observed_frames = 0, 
#     num_predict_frames = 1, 
#     actions = 'all',
#     dim_per_joints = 3,  
#     use_velocity = False, 
#   )

#   test_dataset = DatasetH36M(
#     mode = 'test', 
#     num_observed_frames = 0, 
#     num_predict_frames = 1, 
#     actions = 'all', 
#     dim_per_joints = 3, 
#     use_velocity = False
#   )


#   data = train_dataset.data
#   parents = train_dataset.skeleton.get_parents()
#   margin_frame = 1 
#   observed_threshold = 0.05 
#   predict_threshold = 0.1 

#   start_time = time.time()

#   data_candidates = []
#   for subject in train_dataset.subjects: 
#     for key in data[subject].keys():
#       curr_data = np.copy(data[subject][key])
#       curr_data[:, 0] = 0
#       num_frames = curr_data.shape[0]
#       indexes = np.arange(0, num_frames - args.num_observed_frames - args.num_predict_frames, args.skip_rate)[:, None]\
#           + np.arange(args.num_observed_frames + args.num_predict_frames)[None, :]
#       curr_data = curr_data[indexes]
#       curr_data = absolute_to_relative(curr_data, parents = parents)
#       data_candidates.append(curr_data)
  
#   data_candidates = np.concatenate(data_candidates, axis = 0)
#   np.savez_compressed(
#    f'dataset/multimodal_set/data_candi_observed_frames{args.num_observed_frames:d}_predict_frames{args.num_predict_frames:d}_skiprate{args.skip_rate}.npz', 
#    data_candidate = data_candidates
#   )

#   data_multimodal = {} 
#   for subject in train_dataset.subjects:
#     data_subject = {}
#     if subject not in args.subject : 
#       continue

#     for key in data[subject].keys():
#       start_time = time.time() 
#       data_key = {}
#       curr_data = np.copy(data[subject][key])
#       curr_data[:, 0] = 0
#       num_frames = curr_data.shape[0] 

#       curr_candidates = absolute_to_relative(
#         data_candidates,
#         parents = parents, 
#         invert = True, 
#         x_0 = curr_data[None, ...][:, :1], 
#       )

#       indexs = np.arange(0, num_frames - args.num_observed_frames - args.num_predict_frames + 1)[:, None] +\
#           np.arange(args.num_observed_frames + args.num_predict_frames)[None, :]

#       # observation distance
#       observed_distance = np.mean(
#         np.linalg.norm(
#           curr_data[indexs][:, (args.num_observed_frames - margin_frame) : args.num_observed_frames, 1:][:, None, ...] -
#           curr_candidates[:,(args.num_observed_frames - margin_frame) : args.num_observed_frames, 1:][None, ...], axis = 4
#         ),
#         axis = (2, 3)
#       ) 

#       for idx in np.arange(0, num_frames - args.num_observed_frames - args.num_predict_frames + 1):
#         obs_dist = observed_distance[idx]
#         index_observed = np.where(obs_dist <= observed_threshold)[0]
#         curr_candi = curr_candidates[index_observed]
#         trajectory = curr_data[idx : (idx + args.num_observed_frames + args.num_predict_frames)]
#         x_0 = np.copy(trajectory[None, ...])
#         x_0[:, :, 0] = 0

#         dist_pred = np.mean(np.linalg.norm(x_0[:, args.num_observed_frames:, 1:] \
#            - curr_candi[:, args.num_observed_frames:, 1:], axis = 3), axis = (1, 2))
        
#         idx_pred = np.where(dist_pred >= predict_threshold)[0]
#         idx_cand = index_observed[idx_pred]

#         data_key[idx] = idx_cand
      
#       data_subject[key] = data_key
#       print(f'>>> time used for {subject}_{key}: {time.time() - start_time:.3f}')

#     data_multimodal[subject] = data_subject
  
#   np.savez_compressed(
#     f'./dataset/multimodal_set/data_multimodal_observed_frames{args.num_observed_frames:d}_{margin_frame:d}_threshold_observed{observed_threshold:.3f}\
#       _prediction_frames{args.num_predict_frames:d}_predict_threshold{predict_threshold:.3f}_index_subject{args.subject}.npz',
#     data_multimodal = data_multimodal, 
#   )










  
  



  
