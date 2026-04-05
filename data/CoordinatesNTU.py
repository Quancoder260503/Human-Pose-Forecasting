"""
Input from NTU processed and trained as Coordinates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset import Dataset2D
from torchvision import transforms
import random

class NTUCoordinates(Dataset2D):
    def __init__(self,
        num_training_frame = 14, 
        num_testing_frame  = 14, 
        frame_input_w      = 1920, 
        frame_input_h      = 1080, 
        box_size           = 640, 
        heatmap_size       = 64,  
        num_joints         = 25,
        gaussian_sigma     = 21, 
        data_dir           =  'dataset/NanyangTechRGB/', 
        train              = True,
        on_debug_mode      = False, 
        require_action     = False, 
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

        super(NTUCoordinates, self).__init__(
            num_training_frame   = num_training_frame, 
            num_testing_frame    = num_testing_frame, 
            frame_input_w        = frame_input_w, 
            frame_input_h        = frame_input_h, 
            box_size             = box_size, 
            heatmap_size         = heatmap_size, 
            num_joints           = num_joints, 
            gaussian_sigma       = gaussian_sigma, 
            data_dir             = data_dir, 
            train                = train, 
            on_debug_mode        = on_debug_mode,
            require_action       = require_action,  
            input_representation = 'coordinates', 
        )
        
    def transformation(self, storage, num_joints): 
        name          = storage['name']
        human_id_list = list(storage['data'].keys())

        assert len(human_id_list), f"video {name} has no person in the video"

        human_id_list.sort(key = lambda x : storage['data'][x]['colors'].shape[0], reverse = True) 
        human_id    = human_id_list[0]

        data        = storage['data'][human_id]
        label       = data['colors'] # projected 2D locations
       
        n_frames    = label.shape[0]     
        total_frame   = self.num_training_frame + self.num_testing_frame


        assert n_frames >= total_frame, "num frames must be more than total frame"
        
        num_joints  = label.shape[1] 
        dim         = label.shape[2]

        assert num_joints == self.num_joints and dim == 2, "Does not match expected information"
        
        visibility  = np.ones((label.shape[0], label.shape[1], 1))
        
        for time_ind in range(label.shape[0]): 
            for joint_ind in range(label.shape[1]): 
                if np.isnan(label[time_ind, joint_ind, 0]) or np.isnan(label[time_ind, joint_ind, 1]): 
                    visibility[time_ind, joint_ind] = 0

        label[:, :, 0] = label[:, :, 0] / self.frame_input_w
        label[:, :, 1] = label[:, :, 1] / self.frame_input_h
        
        label      = torch.tensor(label) 
        visibility = torch.tensor(visibility)
        label  = torch.cat([label, visibility], dim = -1) # (x, y, visibility)
        
        if self.train : 
            start_index = np.random.randint(0, n_frames - total_frame + 1)

            observed_poses  = label[start_index : (start_index + self.num_training_frame), :, :]
            observed_poses  = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 3)

            targeted_poses  = label[(start_index + self.num_training_frame): (start_index + total_frame), :, :]
            targeted_poses  = targeted_poses.contiguous().view(self.num_testing_frame, self.num_joints * 3) 
            
            observed_poses[torch.isnan(observed_poses)] = 0.5 
            targeted_poses[torch.isnan(targeted_poses)] = 0.5
            
            return observed_poses, targeted_poses, start_index, name
        
        else : 
            """
            For testing we are going to test the generation based on all samples
            """
            observed_poses_lst = [] 
            targeted_poses_lst = [] 

            for start_index in range(0, n_frames - total_frame + 1):
                observed_poses  = label[start_index : (start_index + self.num_training_frame), :, :]
                observed_poses  = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 3)

                targeted_poses  = label[(start_index + self.num_training_frame): (start_index + total_frame), :, :]
                targeted_poses  = targeted_poses.contiguous().view(self.num_testing_frame, self.num_joints * 3) 

                observed_poses[torch.isnan(observed_poses)] = 0.5 
                targeted_poses[torch.isnan(targeted_poses)] = 0.5

                observed_poses_lst.append(observed_poses) 
                targeted_poses_lst.append(targeted_poses)

            observed_poses_lst = torch.stack(observed_poses_lst, dim = 0)
            targeted_poses_lst = torch.stack(targeted_poses_lst, dim = 0)

            return observed_poses_lst, targeted_poses_lst, n_frames, name
        
    