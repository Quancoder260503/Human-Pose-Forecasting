"""
Input processed and trained as Coordinates
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from abc import ABC, abstractmethod
import tqdm 
import random

class Dataset2D(Dataset, ABC):
    def __init__(self,
        num_training_frame, 
        num_testing_frame, 
        frame_input_w, 
        frame_input_h, 
        box_size, 
        heatmap_size, 
        num_joints,
        gaussian_sigma,
        data_dir,
        train,
        require_action, 
        on_debug_mode,
        input_representation, 
        dataset_name, 
    ):
        """
        :param num_training_frame : frames extracted for training
        :param num_testing_frame  : frames extracted for testing  
        :param frame_h            : original frame height 
        :param frame_w            : original frame width 
        :param box_size           : width of the normalized image 
        :param num_joints         : number of joints for the dataset 
        :param gaussian_sigma     : gaussian s.t.d for drawing heatmap 
        :param data_dir           : root to dataset 
        :param train              : is this training set ? 
        :param require_action     : including action label ? 
        :param on_debug_mode      : extract only one instance ? used only for debugging model   
        """

        self.frame_input_w        = frame_input_w
        self.frame_input_h        = frame_input_h
        self.num_training_frame   = num_training_frame 
        self.num_testing_frame    = num_testing_frame 
        self.box_size             = box_size
        self.heatmap_size         = heatmap_size
        self.num_joints           = num_joints
        self.gaussian_sigma       = gaussian_sigma
        self.train                = train
        self.input_representation = input_representation
        self.dataset_name         = dataset_name
        self.min_frame            = num_training_frame + num_testing_frame

        if self.train is True :
            self.data_dir = data_dir + 'train/'
        else :
            self.data_dir = data_dir + 'test/'

        #self.frames_data         = os.listdir(self.data_dir)
        self.on_debug_mode       = on_debug_mode
        self.require_action      = require_action 
        self.load_data()

    def __len__(self):
        return len(self.frames_data)

    def __getitem__(self, index):
        if self.on_debug_mode : 
            index = 0
        
        frames = self.frames_data[index]
        data   = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()
        return self.transformation(data, self.num_joints) 

        
    @abstractmethod    
    def transformation(self, storage, num_joints):
        pass 

    def load_data(self):
        frame_names = os.listdir(self.data_dir)
        self.frames_data = [] 
        if self.dataset_name == 'ntu':
            for index in range(len(frame_names)):
                frame_name = frame_names[index]
                storage   = np.load(os.path.join(self.data_dir, frame_name), allow_pickle = True).item()
                name          = storage['name']
                human_id_list = list(storage['data'].keys())

                assert len(human_id_list), f"video {name} has no person in the video"

                human_id_list.sort(key = lambda x : storage['data'][x]['colors'].shape[0], reverse = True) 
                human_id    = human_id_list[0]
                data        = storage['data'][human_id]
                label       = data['colors'] # projected 2D locations
                n_frames    = label.shape[0]     

                if n_frames >= self.min_frame : 
                    self.frames_data.append(frame_name)

        elif self.dataset_name == 'pennaction':
            for index in range(len(self.frame_names)):
                frame_name = frame_names[index]
                storage   = np.load(os.path.join(self.data_dir, frame_name), allow_pickle = True).item()
                n_frames    = storage['nframes']   
                if n_frames >= self.min_frame : 
                    self.frames_data.append(frame_name)
        else : 
            raise NotImplementedError('Dataset unknown')