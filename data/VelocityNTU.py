"""
Input from NTU processed and trained as Coordinates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset import Dataset2D
from data.DatasetUtil import NTU_MAJOR_JOINTS_INDS_LST
from torchvision import transforms
import random

class NTUVelocity(Dataset2D):
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

        super(NTUVelocity, self).__init__(
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
            input_representation = 'velocity', 
            dataset_name         = 'ntu', 
        )
        self.major_joint_inds = NTU_MAJOR_JOINTS_INDS_LST
        
    def transformation(self, storage): 
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
        label       = label[:, self.major_joint_inds, :]
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

        #if self.train : 
        start_index = np.random.randint(0, n_frames - total_frame + 1)
        observed_poses = label[start_index : (start_index + self.num_training_frame), :, :2] # (train_len, J, 2)
        observed_masks = label[start_index : (start_index + self.num_training_frame), :,  2] # (train_len, J) 
        
        observed_poses = observed_poses.contiguous().view(self.num_training_frame, -1)       # (train_len, J * 2)
        observed_poses[torch.repeat_interleave(observed_masks.bool(), 2, dim = -1)] = torch.nan
        
        observed_velocity = observed_poses[1:, :] - observed_poses[:-1, :]


        targeted_poses  = label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :2] # (test_len, J, 2) 
        targeted_masks  = label[(start_index + self.num_training_frame) : (start_index + total_frame), :,  2] # (test_len, J)

        targeted_poses  = targeted_poses.contiguous().view(self.num_testing_frame, -1)                          # (test_len, J * 2)
        targeted_poses[torch.repeat_interleave(targeted_masks.bool(), 2, dim = -1)] = torch.nan 
        
        targeted_velocity = torch.cat([(targeted_poses[0, :] - observed_poses[-1, :]).unsqueeze(0), targeted_poses[1:, :] - targeted_poses[:-1, :]], dim = 0) 


        observed_poses[torch.isnan(observed_poses)] = 0 
        targeted_poses[torch.isnan(targeted_poses)] = 0

        observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
        targeted_poses = torch.cat([targeted_poses.contiguous().view(-1, num_joints, 2), targeted_masks.unsqueeze(-1)], dim = -1)
        observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
        targeted_poses = targeted_poses.contiguous().view(-1, num_joints * 3)
        
        observed_velocity[torch.isnan(observed_velocity)] = 0 
        targeted_velocity[torch.isnan(targeted_velocity)] = 0

        return observed_poses, targeted_poses, observed_velocity, targeted_velocity, start_index, name
        # else :     

        #     observed_pose_lst = []
        #     targeted_pose_lst = []
        #     observed_velocity_lst = [] 
        #     targeted_velocity_lst = []

        #     for start_index in range(n_frames - total_frame + 1):
        #         observed_poses = label[start_index : (start_index + self.num_training_frame), :, :2] # (train_len, J, 2)
        #         observed_masks = label[start_index : (start_index + self.num_training_frame), :,  2] # (train_len, J) 
                
        #         observed_poses = observed_poses.contiguous().view(self.num_training_frame, -1)       # (train_len, J * 2)
        #         observed_poses[torch.repeat_interleave(observed_masks.bool(), 2, dim = -1)] = torch.nan
                
        #         observed_velocity = observed_poses[1:, :] - observed_poses[:-1, :]


        #         targeted_poses  = label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :2] # (test_len, J, 2) 
        #         targeted_masks  = label[(start_index + self.num_training_frame) : (start_index + total_frame), :,  2] # (test_len, J)

        #         targeted_poses  = targeted_poses.contiguous().view(self.num_testing_frame, -1)                          # (test_len, J * 2)
        #         targeted_poses[torch.repeat_interleave(targeted_masks.bool(), 2, dim = -1)] = torch.nan 
                
        #         targeted_velocity = torch.cat([(targeted_poses[0, :] - observed_poses[-1, :]).unsqueeze(0), targeted_poses[1:, :] - targeted_poses[:-1, :]], dim = 0) 

        #         observed_poses[torch.isnan(observed_poses)] = 0
        #         targeted_poses[torch.isnan(targeted_poses)] = 0


        #         observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
        #         targeted_poses = torch.cat([targeted_poses.contiguous().view(-1, num_joints, 2), targeted_masks.unsqueeze(-1)],  dim = -1)
        #         observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
        #         targeted_poses = targeted_poses.contiguous().view(-1, num_joints * 3)
                
        #         observed_velocity[torch.isnan(observed_velocity)] = 0 
        #         targeted_velocity[torch.isnan(targeted_velocity)] = 0


        #         observed_pose_lst.append(observed_poses)
        #         targeted_pose_lst.append(targeted_poses)
        #         observed_velocity_lst.append(observed_velocity)
        #         targeted_velocity_lst.append(targeted_velocity)

        #     observed_pose_lst = torch.stack(observed_pose_lst, dim = 0) 
        #     targeted_pose_lst = torch.stack(targeted_pose_lst, dim = 0) 
        #     observed_velocity_lst = torch.stack(observed_velocity_lst, dim = 0) 
        #     targeted_velocity_lst = torch.stack(targeted_velocity_lst, dim = 0)

        #     return observed_pose_lst, targeted_pose_lst, observed_velocity_lst, targeted_velocity_lst, n_frames, name
    