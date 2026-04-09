"""
Input from NTU processed and trained as Coordinates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset import Dataset2D
from torchvision import transforms
import random
import os 

MIN_STD    = 1e-4
MIN_FRAMES = 60 

class NTURootRelative(Dataset2D):
    def __init__(self,
        num_training_frame = 14, 
        num_testing_frame  = 14, 
        frame_input_w      = 1920, 
        frame_input_h      = 1080, 
        box_size           = 640, 
        heatmap_size       = 64,  
        num_joints         = 25,
        gaussian_sigma     = 21, 
        data_dir           = 'dataset/NanyangTechRGB/', 
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

        super(NTURootRelative, self).__init__(
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
            input_representation = 'root_relative', 
            dataset_name         = 'ntu', 
        )
        self.compute_stats()
        self.eps = 1e-7

    def compute_stats(self):    
        
        root_distance_x  = [[] for _ in range(self.num_joints)]
        root_distance_y  = [[] for _ in range(self.num_joints)]

        for index in range(len(self.frames_data)):
            frames        = self.frames_data[index]
            storage       = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()
            name          = storage['name']
            human_id_list = list(storage['data'].keys())

            assert len(human_id_list), f"video {name} has no person in the video"

            human_id_list.sort(key = lambda x : storage['data'][x]['colors'].shape[0], reverse = True) 
            human_id    = human_id_list[0]

            data        = storage['data'][human_id]
            label       = data['colors'] # projected 2D locations
       
            n_frames    = label.shape[0]     
            num_joints  = label.shape[1] 
            dim         = label.shape[2]
        
            visibility  = np.ones((label.shape[0], label.shape[1]))
        
            for time_ind in range(label.shape[0]): 
                for joint_ind in range(label.shape[1]): 
                    if np.isnan(label[time_ind, joint_ind, 0]) or np.isnan(label[time_ind, joint_ind, 1]): 
                       visibility[time_ind, joint_ind] = 0
            
            root_position = 0

            for time_step in range(n_frames):
                if visibility[time_step, root_position] == 0:
                    continue

                for child_index in range(num_joints): 
                    if child_index == root_position or visibility[time_step, child_index] == 0: 
                        continue
    
                    root_distance_x[child_index].append(label[time_step, child_index, 0] - label[time_step, root_position, 0]) 
                    root_distance_y[child_index].append(label[time_step, child_index, 1] - label[time_step, root_position, 1])
        

        self.mean_x = np.mean(root_distance_x, axis = 0) # (num_joints, 1) 
        self.mean_y = np.mean(root_distance_y, axis = 0) # (num_joints, 1)

        self.std_x  = np.std(root_distance_x, axis = 0) + self.eps 
        self.std_y  = np.std(root_distance_y, axis = 0) + self.eps 

        self.std_x[self.std_x < MIN_STD] = 1 
        self.std_y[self.std_y < MIN_STD] = 1


        
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


        
        label      = torch.tensor(label) 
        visibility = torch.tensor(visibility)
        label      = torch.cat([label, visibility], dim = -1) # (x, y, visibility)

        
        root_position = 0

        median_root_pos_x, median_root_pos_y = np.median(label[:, root_position, 0]), np.median(label[:, root_position, 1]) 
            
        assert (not np.isnan(median_root_pos_x)) and (not np.isnan(median_root_pos_y)), f"cannot extract root in set {name} and {storage}" 

        root_vectors = np.zeros((n_frames, num_joints, 2))

        for time_step in range(n_frames):
            if visibility[time_step, root_position] == 0: 
                label[time_step, root_position, 0], label[time_step, root_position, 1] = median_root_pos_x, median_root_pos_y
        

            root_vectors[time_step, root_position, 0] = label[time_step, root_position, 0] / self.frame_input_w 
            root_vectors[time_step, root_position, 1] = label[time_step, root_position, 1] / self.frame_input_h

            for child_index in range(num_joints): 
                if child_index == root_position : 
                    continue
                root_vectors[time_step, child_index, 0] = (label[time_step, child_index, 0] - label[time_step, root_position, 0]) 
                root_vectors[time_step, child_index, 1] = (label[time_step, child_index, 1] - label[time_step, root_position, 1]) 

                root_vectors[time_step, child_index, 0] = (root_vectors[time_step, child_index, 0] - self.mean_x[child_index]) / self.std_x[child_index]
                root_vectors[time_step, child_index, 1] = (root_vectors[time_step, child_index, 1] - self.mean_y[child_index]) / self.std_y[child_index]  


        if self.train : 
            start_index = np.random.randint(0, n_frames - total_frame + 1)
  
            observed_poses        = torch.tensor(       label[start_index : (start_index + self.num_training_frame), :, :]) 
            observed_root_vectors = torch.tensor(root_vectors[start_index : (start_index + self.num_training_frame), :, :])
            observed_root_vectors = observed_root_vectors.contiguous().view(self.num_training_frame, -1) 

            observed_root_vectors[torch.isnan(observed_root_vectors)] = 0  
            observed_poses[torch.isnan(observed_poses)]               = 0
        
            targeted_poses        = torch.tensor(       label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :])
            targeted_root_vectors = torch.tensor(root_vectors[(start_index + self.num_training_frame) : (start_index + total_frame), :, :]) 
            targeted_root_vectors = targeted_root_vectors.contiguous().view(self.num_testing_frame, -1) 

            targeted_root_vectors[torch.isnan(targeted_root_vectors)] = 0 
            targeted_poses[torch.isnan(targeted_poses)]               = 0

            return observed_poses, targeted_poses, observed_root_vectors, targeted_root_vectors, start_index, name
        
        else : 
            observed_pose_lst = [] 
            targeted_pose_lst = []
            observed_root_vector_lst = []
            targeted_root_vector_lst = []

            for start_index in range(n_frames - total_frame + 1):
                observed_poses        = torch.tensor(       label[start_index : (start_index + self.num_training_frame), :, :]) 
                observed_root_vectors = torch.tensor(root_vectors[start_index : (start_index + self.num_training_frame), :, :])
                observed_root_vectors = observed_root_vectors.contiguous().view(self.num_training_frame, -1) 

                observed_root_vectors[torch.isnan(observed_root_vectors)] = 0  
                observed_poses[torch.isnan(observed_poses)]               = 0
            
                targeted_poses        = torch.tensor(       label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :])
                targeted_root_vectors = torch.tensor(root_vectors[(start_index + self.num_training_frame) : (start_index + total_frame), :, :]) 
                targeted_root_vectors = targeted_root_vectors.contiguous().view(self.num_testing_frame, -1) 

                targeted_root_vectors[torch.isnan(targeted_root_vectors)] = 0 
                targeted_poses[torch.isnan(targeted_poses)]               = 0

                observed_pose_lst.append(observed_poses)
                targeted_pose_lst.append(targeted_poses)
                observed_root_vector_lst.append(observed_root_vectors)
                targeted_root_vector_lst.append(targeted_root_vectors)   
            
            observed_pose_lst = torch.stack(observed_pose_lst, dim = 0) 
            targeted_pose_lst = torch.stack(targeted_pose_lst, dim = 0) 
            observed_root_vector_lst = torch.stack(observed_root_vector_lst, dim = 0) 
            targeted_root_vector_lst = torch.stack(targeted_root_vector_lst, dim = 0)

            return observed_pose_lst, targeted_pose_lst, observed_root_vector_lst, targeted_root_vector_lst, start_index, name
