import torch
import numpy as np
import matplotlib.pyplot as plt
from data.MultiDataset import Dataset2D, Dataset3D
from data.DatasetUtil import *
from data.Visualization import * 
import os 


class NTU2D(Dataset2D):
    def __init__(self,
        num_training_frame = 14, 
        num_testing_frame  = 14, 
        frame_input_w      = 1920, 
        frame_input_h      = 1080, 
        box_size           = 640, 
        heatmap_size       = 64,  
        num_joints         = 17,
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

        super(NTU2D, self).__init__(
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
        print("2D representation")

    def load_compute_norm_stats(self, data_dir):
        mean_path = os.path.join(data_dir, 'mean_2d_root_relative.npy') # mean path 
        std_path  = os.path.join(data_dir, 'std_2d_root_relative.npy') # std  path 

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
            label       = data['colors'] 
            label       = label[:, self.major_joint_inds, :]            
            
            # label : np.array of shape (num_frames, J, 2) 
    
            root_sequences   = np.expand_dims(label[:, 0, :], axis = 1) 
            action_sequences = label - root_sequences # (num_frames, J, 2)
            root_distance.append(action_sequences) # (num_frames, J, 2)
        
        root_distance = np.concatenate(root_distance, axis = 0) # (all_frames, J, 2)
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

        action_sequences = action_sequences.reshape((n_frames, -1))
        root_sequences   = root_sequences.reshape((n_frames, -1))

        action_sequences[:, :] = (action_sequences[:, :] - self.mean) / self.std

        root_sequences[:, 0]  = (root_sequences[:, 0] / self.frame_input_w * 2.0 - 1.0) 
        root_sequences[:, 1]  = (root_sequences[:, 1] / self.frame_input_h * 2.0 - 1.0) 

        start_index = np.random.randint(0, n_frames - total_frame + 1)
        observed_act_seqs  = torch.tensor(action_sequences[start_index : (start_index + self.num_training_frame), :]) 
        observed_root_seqs = torch.tensor(root_sequences[start_index : (start_index + self.num_training_frame),   :])

        observed_act_seqs[torch.isnan(observed_act_seqs)]   = 0.0
        observed_root_seqs[torch.isnan(observed_root_seqs)] = 0.0

        
        targeted_act_seqs  = torch.tensor(action_sequences[(start_index + self.num_training_frame) : (start_index + total_frame), :])
        targeted_root_seqs = torch.tensor(root_sequences[(start_index + self.num_training_frame) : (start_index + total_frame), :])
        
        targeted_act_seqs[torch.isnan(targeted_act_seqs)]   = 0.0 
        targeted_root_seqs[torch.isnan(targeted_root_seqs)] = 0.0 

        # merge the root and action sequences 
        observed_act_seqs = torch.cat([observed_root_seqs, observed_act_seqs[:, 2:]], dim = -1)
        targeted_act_seqs = torch.cat([targeted_root_seqs, targeted_act_seqs[:, 2:]], dim = -1)


        return observed_act_seqs, targeted_act_seqs, start_index, name
    


class NTU3D(Dataset3D):
    def __init__(self,
        num_training_frame = 14, 
        num_testing_frame  = 14, 
        num_joints         = 17, 
        data_dir           = 'dataset/NanyangTechRGB', 
        train              = True,
        on_debug_mode      = False, 
        require_action     = False, 
        frame_input_w      = 1920, 
        frame_input_h      = 1080, 
        box_size           = 640, 
        heatmap_size       = 64, 
    ):
        """
        :param num_training_frame : frames extracted for training
        :param num_testing_frame  : frames extracted for testing  
        :param num_joints         : number of joints for the dataset 
        :param data_dir           : root to dataset 
        :param train              : is this training set ? 
        :param require_action     : including action label ? 
        :param on_debug_mode      : extract only one instance ? used only for debugging model   
        """

        super(NTU3D, self).__init__(
            num_training_frame   = num_training_frame, 
            num_testing_frame    = num_testing_frame, 
            num_joints           = num_joints, 
            data_dir             = data_dir, 
            train                = train, 
            on_debug_mode        = on_debug_mode,
            require_action       = require_action,   
            dataset_name         = 'ntu', 
        )
        self.major_joint_inds = NTU_MAJOR_JOINTS_INDS_LST
        self.load_compute_norm_stats(data_dir)

        print("3D representation")

    def load_compute_norm_stats(self, data_dir):
        mean_path = os.path.join(data_dir, 'mean_3d.npy') # mean path 
        std_path  = os.path.join(data_dir, 'std_3d.npy') # std  path
        root_mean_path = os.path.join(data_dir, 'root_mean_3d.npy')
        root_std_path  = os.path.join(data_dir, 'root_std_3d.npy') 

        if os.path.exists(mean_path):
            self.mean = np.load(mean_path) 
            self.std  = np.load(std_path)
            self.root_mean = np.load(root_mean_path) 
            self.root_std  = np.load(root_std_path)
            
        elif self.train : 
            self.compute_stats() 
            np.save(mean_path,  self.mean)
            np.save(std_path,   self.std)
            np.save(root_mean_path, self.root_mean)
            np.save(root_std_path,  self.root_std)
        else : 
            raise ValueError('Cannot compute statistics in non-training mode') 


    def compute_stats(self):    
        root_distance = []
        root_position = [] 
        for index in range(len(self.frames_data)):
            frames        = self.frames_data[index]
            storage       = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()
            name          = storage['name']
            human_id_list = list(storage['data'].keys())
            assert len(human_id_list), f"video {name} has no person in the video"

            human_id_list.sort(key = lambda x : storage['data'][x]['joints'].shape[0], reverse = True) 
            human_id    = human_id_list[0]

            data        = storage['data'][human_id]
            label       = data['joints']
            label       = label.reshape(-1, 25, 3)
            label       = label[:, self.major_joint_inds, :]            
            
            # label : np.array of shape (num_frames, J - 1, 3) 
            root_sequences   = np.expand_dims(label[:, 0, :], axis = 1) 
            action_sequences = label - root_sequences        # (num_frames, J, 3)
            
            root_distance.append(action_sequences) # (num_frames, J, 3)
            root_position.append(root_sequences)    
        
        root_distance = np.concatenate(root_distance, axis = 0) # (all_frames, J, 3)
        root_position = np.concatenate(root_position, axis = 0) # (all_frames, 1, 3)
        self.mean = np.nanmean(root_distance, axis = 0)
        self.std  = np.nanstd(root_distance,  axis = 0) # (1, 3 )

        self.root_mean = np.nanmean(root_position, axis = 0) 
        self.root_std  = np.nanstd(root_position, axis = 0) 

        self.std[np.where(self.std < MIN_STD)] = 1.0 
        self.root_std[np.where(self.root_std < MIN_STD)] = 1.0 

        self.mean = self.mean.ravel()
        self.std  = self.std.ravel()
        self.root_mean = self.root_mean.ravel() 
        self.root_std  = self.root_std.ravel()


    def transformation(self, storage): 
        name          = storage['name']
        human_id_list = list(storage['data'].keys())

        assert len(human_id_list), f"video {name} has no person in the video"

        human_id_list.sort(key = lambda x : storage['data'][x]['joints'].shape[0], reverse = True) 
        human_id    = human_id_list[0]

        data        = storage['data'][human_id]
        label       = data['joints'].copy() # 3D locations
        label       = label.reshape(-1, 25, 3)

        n_frames      = label.shape[0]      
       
        total_frame   = self.num_training_frame + self.num_testing_frame

        assert n_frames >= total_frame, "num frames must be more than total frame"
    
        label       = label[:, self.major_joint_inds, :]
        num_joints  = label.shape[1] 
        dim         = label.shape[2]

        assert num_joints == self.num_joints and dim == 3, "Does not match expected information"


        root_position = 0

        median_root_pos_x, median_root_pos_y, median_root_pos_z = np.nanmedian(label[:, root_position, 0]), np.nanmedian(label[:, root_position, 1]), np.nanmedian(label[:, root_position, 2]) 

        assert (not np.isnan(median_root_pos_x)) and (not np.isnan(median_root_pos_y)) and (not np.isnan(median_root_pos_z)), f"cannot extract root in set {name} and {storage}" 

        # we do stuff a little bit differently here 
        root_sequences = label[:, 0, :] # (n_frames, 3)

        root_sequences[np.isnan(root_sequences[:, 0]), 0] = median_root_pos_x
        root_sequences[np.isnan(root_sequences[:, 1]), 1] = median_root_pos_y
        root_sequences[np.isnan(root_sequences[:, 2]), 2] = median_root_pos_z

        root_sequences = np.expand_dims(root_sequences, axis = 1) # (n_frames, 1, 3)
        
        action_sequences = label - root_sequences
        action_sequences = action_sequences.reshape((n_frames, -1)) 
        root_sequences   = root_sequences.reshape((n_frames, -1))


        action_sequences[:, :] = (action_sequences[:, :] - self.mean) / self.std 
        root_sequences[:, :]   = (root_sequences[:, :]   - self.root_mean) / self.root_std

        start_index = np.random.randint(0, n_frames - total_frame + 1)

        observed_act_seqs  = torch.tensor(action_sequences[  start_index : (start_index + self.num_training_frame), :]) 
        observed_root_seqs = torch.tensor(root_sequences[start_index : (start_index + self.num_training_frame), :])

        observed_act_seqs[torch.isnan(observed_act_seqs)]   = 0.0
        observed_root_seqs[torch.isnan(observed_root_seqs)] = 0.0

        targeted_act_seqs  = torch.tensor(action_sequences[(start_index + self.num_training_frame) : (start_index + total_frame), :]) 
        targeted_root_seqs = torch.tensor(root_sequences[(start_index + self.num_training_frame) : (start_index + total_frame),   :])
        

        targeted_act_seqs[torch.isnan(targeted_act_seqs)]   = 0.0
        targeted_root_seqs[torch.isnan(targeted_root_seqs)] = 0.0 

         # merge the root and action sequences 
        observed_act_seqs = torch.cat([observed_root_seqs, observed_act_seqs[:, 3:]], dim = -1)
        targeted_act_seqs = torch.cat([targeted_root_seqs, targeted_act_seqs[:, 3:]], dim = -1)

        return observed_act_seqs, targeted_act_seqs, start_index, name
    