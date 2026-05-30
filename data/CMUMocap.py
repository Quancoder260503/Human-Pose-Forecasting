"""
Borrowed from 
https://github.com/wei-mao-2019/LearnTrajDep/tree/master/utils
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.DatasetUtil import *
from torch.autograd.variable import Variable
from data.Visualization import * 
from data import Forward_kinematics
from torch.utils.data import Dataset
from data import Dataset3DUtils, DatasetUtil
import os 


def normalize_dataset_path(path):
    path = os.path.expanduser(path)
    if os.name == 'nt' and (path.startswith('/') or path.startswith('\\')) and not os.path.splitdrive(path)[0]:
        path = path.lstrip('/\\')
    return os.path.abspath(os.path.normpath(path))


class CMU_Motion3D(Dataset):
    def __init__(self, path_to_data, actions, input_n = 100, output_n = 100, split = 0, data_mean = 0, data_std = 0, dim_used = 0):
        self.path_to_data = path_to_data
        self.split = split
        path_to_data = normalize_dataset_path(path_to_data)
        actions = Dataset3DUtils.define_actions_cmu(actions) 
        if split == 0:
            path_to_data = os.path.join(path_to_data, 'train')
            is_test = False
        else:
            path_to_data = os.path.join(path_to_data, 'test')
            is_test = True

        all_seqs, dim_ignore, dim_use, data_mean, data_std = Dataset3DUtils.load_data_cmu_3d(path_to_data, actions, input_n, output_n, data_std=data_std, data_mean = data_mean, is_test = is_test)
        

        if not is_test:
            dim_used = dim_use

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used] 
        self.all_seqs = all_seqs

        self.data_mean = data_mean
        self.data_std  = data_std

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        return self.all_seqs[item]
    
