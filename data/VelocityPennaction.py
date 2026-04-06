import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset import Dataset2D
from torchvision import transforms
import random

class PennActionVelocity(Dataset2D):
    def __init__(self,
        num_training_frame = 7, 
        num_testing_frame  = 7, 
        frame_input_w      = 640, 
        frame_input_h      = 480, 
        box_size           = 480, 
        heatmap_size       = 64, 
        num_joints         = 13,
        gaussian_sigma     = 21,
        data_dir           = 'dataset/PennAction/',
        train              = True,
        require_action     = False, 
        on_debug_mode      = False,
    ):
        super(PennActionVelocity, self).__init__(
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
            require_action       = require_action,  
            on_debug_mode        = on_debug_mode,
            input_representation = 'velocity',  
        )

    def transformation(self, storage, num_joints): 
        n_frames    = storage['nframes']      
        frames_path = storage['framepath']
        dim         = storage['dimensions']     

        label = np.zeros((n_frames, num_joints, 3)) 

        label[:, :, 0] = storage['x'].astype(np.float32)  
        label[:, :, 1] = storage['y'].astype(np.float32)
        label[:, :, 2] = storage['visibility']

        h, w = dim 

        label[:, :, 0] = label[:, :, 0] / w
        label[:, :, 1] = label[:, :, 1] / h
        
        total_frame = self.num_training_frame + self.num_testing_frame
        
        if self.train : 
            start_index = np.random.randint(0, n_frames - total_frame + 1)

            observed_poses = torch.tensor(label[start_index : (start_index + self.num_training_frame), :, :2]) 
            observed_masks = torch.tensor(label[start_index : (start_index + self.num_training_frame), :,  2])

            observed_poses = observed_poses.contiguous().view(self.num_training_frame, num_joints * 2) 
            observed_poses[torch.repeat_interleave(observed_masks, 2, dim = -1) == 0] = torch.nan

            observed_velocity = observed_poses[1 :, :] - observed_poses[:-1, :] 

            # Target pose transformation 
            targeted_poses = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :2]) 
            targeted_masks = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :,  2])

            targeted_poses = targeted_poses.contiguous().view(self.num_testing_frame, num_joints * 2) 
            targeted_poses[torch.repeat_interleave(targeted_masks, 2, dim = -1) == 0] = torch.nan

            targeted_velocity = torch.cat([(targeted_poses[0, :] - observed_poses[-1, :]).unsqueeze(0),  targeted_poses[1:, :] -  targeted_poses[:-1, :]], dim = 0)

            observed_poses[torch.isnan(observed_poses)] = 0
            targeted_poses[torch.isnan(targeted_poses)] = 0

            observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
            targeted_poses = torch.cat([targeted_poses.contiguous().view(-1, num_joints, 2), targeted_masks.unsqueeze(-1)], dim = -1)
           
           
            observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
            targeted_poses = targeted_poses.contiguous().view(-1, num_joints * 3)

            observed_velocity[torch.isnan(observed_velocity)] = 0 
            targeted_velocity[torch.isnan(targeted_velocity)] = 0

            return observed_poses, targeted_poses, observed_velocity, targeted_velocity, start_index, frames_path
        
        else : 

            observed_pose_lst = []
            targeted_pose_lst = []
            observed_velocity_lst = [] 
            targeted_velocity_lst = []

            for start_index in range(n_frames - total_frame + 1):
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


                observed_pose_lst.append(observed_poses)
                targeted_pose_lst.append(targeted_poses)
                observed_velocity_lst.append(observed_velocity)
                targeted_velocity_lst.append(targeted_velocity)

            observed_pose_lst = torch.stack(observed_pose_lst, dim = 0) 
            targeted_pose_lst = torch.stack(targeted_pose_lst, dim = 0) 
            observed_velocity_lst = torch.stack(observed_velocity_lst, dim = 0) 
            targeted_velocity_lst = torch.stack(targeted_velocity_lst, dim = 0)

            return observed_pose_lst, targeted_pose_lst, observed_velocity_lst, targeted_velocity_lst, n_frames, frames_path