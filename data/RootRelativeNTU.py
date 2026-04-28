"""
Input from NTU processed and trained as Coordinates
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data.Dataset     import Dataset2D
from data.DatasetUtil import *
import os 


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
        data_dir           = 'dataset/NanyangTechRGB', 
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
        self.major_joint_inds = NTU_MAJOR_JOINTS_INDS_LST
        self.load_compute_norm_stats(data_dir)

    def load_compute_norm_stats(self, data_dir):
        mean_path = os.path.join(data_dir, 'mean.npy') # mean path 
        std_path  = os.path.join(data_dir, 'std.npy') # std  path 

        if os.path.exists(mean_path):
            self.mean = np.load(mean_path) 
            self.std  = np.load(std_path)
        elif self.train : 
            self.compute_stats() 
            np.save(mean_path, self.mean)
            np.save(std_path,   self.std)
        else : 
            raise ValueError('Cannot compute statistics in non-training mode') 

    def compute_stats(self):    
        root_distance = []
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
            label       = label[:, self.major_joint_inds, :]            
            
            # label : np.array of shape (num_frames, J, 2) 
    
            root_sequences   = np.expand_dims(label[:, 0, :], axis = 1) 
            action_sequences = label - root_sequences # (num_frames, J, 2)
            root_distance.append(action_sequences[:, 1:, :]) # (num_frames, J - 1, 2)
        
        root_distance = np.concatenate(root_distance, axis = 0) # (all_frames, J - 1, 2)
        self.mean = np.nanmean(root_distance, axis = 0)
        self.std  = np.nanstd(root_distance,  axis = 0)
        self.std[np.where(self.std < MIN_STD)] = 1.0 

        self.mean = self.mean.ravel()
        self.std  = self.std.ravel()


    def transformation(self, storage): 
        name          = storage['name']
        human_id_list = list(storage['data'].keys())

        assert len(human_id_list), f"video {name} has no person in the video"

        human_id_list.sort(key = lambda x : storage['data'][x]['colors'].shape[0], reverse = True) 
        human_id    = human_id_list[0]

        data        = storage['data'][human_id]
        label       = data['colors'].copy() # projected 2D locations
       
        n_frames    = label.shape[0]     
        total_frame   = self.num_training_frame + self.num_testing_frame


        assert n_frames >= total_frame, "num frames must be more than total frame"
        
        label       = label[:, self.major_joint_inds, :]
        num_joints  = label.shape[1] 
        dim         = label.shape[2]


        assert num_joints == self.num_joints and dim == 2, "Does not match expected information"


        root_position = 0

        median_root_pos_x, median_root_pos_y = np.nanmedian(label[:, root_position, 0]), np.nanmedian(label[:, root_position, 1]) 
            
        assert (not np.isnan(median_root_pos_x)) and (not np.isnan(median_root_pos_y)), f"cannot extract root in set {name} and {storage}" 

        # we do stuff a little bit differently here 
        root_sequences = label[:, 0, :] # (n_frames, 2)
        root_sequences[np.isnan(root_sequences[:, 0]), 0] = median_root_pos_x
        root_sequences[np.isnan(root_sequences[:, 1]), 1] = median_root_pos_y
        root_sequences = np.expand_dims(root_sequences, axis = 1) # (n_frames, 1, 2)
        
        action_sequences = label - root_sequences # (n_frames, J, 2)
        action_sequences[:, 0, :] = root_sequences[:, 0, :]  

        action_sequences = action_sequences.reshape((n_frames, -1))
        action_sequences[:, 2:] = (action_sequences[:, 2:] - self.mean) / self.std 
        action_sequences[:, 0]  = action_sequences[:, 0] / self.frame_input_w
        action_sequences[:, 1]  = action_sequences[:, 1] / self.frame_input_h


       # if self.train: 
        start_index = np.random.randint(0, n_frames - total_frame + 1)
        observed_poses    = torch.tensor(label[start_index : (start_index + self.num_training_frame), :, :]) 
        observed_act_seqs = torch.tensor(action_sequences[start_index : (start_index + self.num_training_frame), :]) 

        observed_poses[torch.isnan(observed_poses)]       = 0
        observed_act_seqs[torch.isnan(observed_act_seqs)] = 0

        targeted_poses    = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :])
        targeted_act_seqs = torch.tensor(action_sequences[(start_index + self.num_training_frame) : (start_index + total_frame), :]) 
        
        targeted_poses[torch.isnan(targeted_poses)]       = 0
        targeted_act_seqs[torch.isnan(targeted_act_seqs)] = 0

        return observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, start_index, name
    
        # else : 
        #     observed_pose_lst = [] 
        #     targeted_pose_lst = []
        #     observed_act_seqs_lst = []
        #     targeted_act_seqs_lst = []

        #     for start_index in range(n_frames - total_frame + 1):
        #         observed_poses    = torch.tensor(label[start_index : (start_index + self.num_training_frame), :, :]) 
        #         observed_act_seqs = torch.tensor(action_sequences[start_index : (start_index + self.num_training_frame), :]) 

        #         observed_poses[torch.isnan(observed_poses)]       = 0
        #         observed_act_seqs[torch.isnan(observed_act_seqs)] = 0

        #         targeted_poses    = torch.tensor(label[(start_index + self.num_training_frame) : (start_index + total_frame), :, :])
        #         targeted_act_seqs = torch.tensor(action_sequences[(start_index + self.num_training_frame) : (start_index + total_frame), :]) 
    
        #         targeted_poses[torch.isnan(targeted_poses)]       = 0
        #         targeted_act_seqs[torch.isnan(targeted_act_seqs)] = 0


        #         observed_pose_lst.append(observed_poses)
        #         targeted_pose_lst.append(targeted_poses)
        #         observed_act_seqs_lst.append(observed_act_seqs)
        #         targeted_act_seqs_lst.append(targeted_act_seqs)   
            
        #     observed_pose_lst = torch.stack(observed_pose_lst, dim = 0) 
        #     targeted_pose_lst = torch.stack(targeted_pose_lst, dim = 0) 
        #     observed_act_seqs_lst = torch.stack(observed_act_seqs_lst, dim = 0) 
        #     targeted_act_seqs_lst = torch.stack(targeted_act_seqs_lst, dim = 0)

        #     return observed_pose_lst, observed_act_seqs_lst, targeted_pose_lst, targeted_act_seqs_lst, start_index, name
