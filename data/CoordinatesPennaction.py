import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset import Dataset2D
from torchvision import transforms
import random

class PennActionCoordinates(Dataset2D):
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
        super(PennActionCoordinates, self).__init__(
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
            input_representation = 'coordinates', 
            dataset_name         = 'pennaction', 
        )

    def transformation(self, storage): 
        n_frames    = storage['nframes']      
        frames_path = storage['framepath']
        dim         = storage['dimensions']     
        num_joints = self.num_joints
        label = np.zeros((n_frames, num_joints, 3)) 

        label[:, :, 0] = storage['x'].astype(np.float32)  
        label[:, :, 1] = storage['y'].astype(np.float32)
        label[:, :, 2] = storage['visibility']

        h, w = dim 

        label[:, :, 0] = label[:, :, 0] / w 
        label[:, :, 1] = label[:, :, 1] / h        
        
        total_frame = self.num_training_frame + self.num_testing_frame
        
        if self.train : 
        
            start_index = 0 #np.random.randint(0, n_frames - total_frame + 1) 
            # zero out these terms 
            observed_poses = torch.tensor(label[start_index : (start_index + self.num_training_frame), :, :2]) 
            observed_masks = torch.tensor(label[start_index : (start_index + self.num_training_frame), :,  2])

            observed_poses = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 2) 
            observed_poses[torch.repeat_interleave(observed_masks, 2, dim = -1) == 0] = torch.nan

            # Target pose transformation 
            targeted_poses = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :2]) 
            targeted_masks = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :,  2])

            targeted_poses = targeted_poses.contiguous().view(self.num_testing_frame, self.num_joints * 2) 
            targeted_poses[torch.repeat_interleave(targeted_masks, 2, dim = -1) == 0] = torch.nan

            # Replace nan by 0.5
            observed_poses[torch.isnan(observed_poses)] = 0.5
            targeted_poses[torch.isnan(targeted_poses)] = 0.5

            observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
            targeted_poses = torch.cat([targeted_poses.contiguous().view(-1, num_joints, 2), targeted_masks.unsqueeze(-1)], dim = -1)

            observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
            targeted_poses = targeted_poses.contiguous().view(-1, num_joints * 3)

            return observed_poses, targeted_poses, start_index, frames_path
        
        else : 
            
            observed_pose_lst = [] 
            targeted_pose_lst = [] 

            for start_index in range(0, n_frames - total_frame + 1): 
                observed_poses = torch.tensor(label[start_index : (start_index + self.num_training_frame), :, :2]) 
                observed_masks = torch.tensor(label[start_index : (start_index + self.num_training_frame), :,  2])

                observed_poses = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 2) 
                observed_poses[torch.repeat_interleave(observed_masks, 2, dim = -1) == 0] = torch.nan

                # Target pose transformation 
                targeted_poses = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :2]) 
                targeted_masks = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :,  2])

                targeted_poses = targeted_poses.contiguous().view(self.num_testing_frame, self.num_joints * 2) 
                targeted_poses[torch.repeat_interleave(targeted_masks, 2, dim = -1) == 0] = torch.nan

                # Replace nan by 0.5
                observed_poses[torch.isnan(observed_poses)] = 0.5
                targeted_poses[torch.isnan(targeted_poses)] = 0.5

                observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
                targeted_poses = torch.cat([targeted_poses.contiguous().view(-1, num_joints, 2), targeted_masks.unsqueeze(-1)], dim = -1)

                observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
                targeted_poses = targeted_poses.contiguous().view(-1, num_joints * 3)

                observed_pose_lst.append(observed_poses) 
                targeted_pose_lst.append(targeted_poses)
            
            observed_pose_lst = torch.stack(observed_pose_lst, dim = 0)
            targeted_pose_lst = torch.stack(targeted_pose_lst, dim = 0)

            return observed_pose_lst, targeted_pose_lst, n_frames, frames_path
