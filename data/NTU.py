import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .NTUmeta import *
import random 

''' 
For Nanyang Tech dataset, 

RGB frame exists but very heavy, around 133GB, 
hence only the joints and heat maps are 
used for training and testing. 

TODO : 
1. Visualizing the joint using video 
2. Extracting joints and action class for classification task

'''

NTU_VIDEO_PATH    = 'dataset/NanyangTechRGB/frames/nturgb+d_rgb'

class NTU(Dataset):
    def __init__(self,
        num_training_frame = 7, 
        num_testing_frame  = 7, 
        ntu_input_w = 1920, 
        ntu_input_h = 1080, 
        input_w = 640,
        input_h = 640,
        map_h   = 46,
        map_w   = 46,
        num_joints     = 25,
        gaussian_sigma = 21,
        data_dir = 'dataset/NanyangTechRGB/',
        train = True,
        require_action  = False, 
        root_relative   = False,
        parent_relative = False,  
        velocity        = False, 
        coordinates     = False, 
        on_debug_mode   = False,
    ):
        """
        :param num_training_frame : frames extracted for training
        :param num_testing_frame  : frames extracted for testing  
        :param input_w            : width of the normalized image 
        :param input_h            : height of the normalized image 
        :param num_joints         : number of joints for the dataset 
        :param gaussian_sigma     : gaussian s.t.d for drawing heatmap 
        :param data_dir           : root to dataset 
        :param train              : is this training set ? 
        :param require_action     : including action label ? 
        :param root_relative      : input processed as normalized root relative (center hip)? 
        :param parent_relative    : input processed as parent_relative 
        :param velocity           : input processed as velocity (displacment) of key joints on different frames ?
        :param coordinates        : input processed as local coordinate of the image ? 
        :param on_debug_mode      : extract only one instance ? used only for debugging model   
        """

        super(NTU, self).__init__()
        self.ntu_input_w        = ntu_input_w
        self.ntu_input_h        = ntu_input_h 
        self.num_training_frame = num_training_frame 
        self.num_testing_frame  = num_testing_frame 
        self.input_w = input_w
        self.input_h = input_h
        self.map_h = map_h
        self.map_w = map_w
        self.num_joints     = num_joints
        self.gaussian_sigma = gaussian_sigma
        self.transform      = self.get_default_transform() # Just in case I might use image for training 
        self.train          = train
        self.require_action = require_action
        if self.train is True :
            self.data_dir = data_dir + 'train/'
        else :
            self.data_dir = data_dir + 'test/'

        self.frames_data     = os.listdir(self.data_dir)
        self.root_relative   = root_relative 
        self.parent_relative = parent_relative
        self.velocity        = velocity 
        self.coordinates     = coordinates
        self.on_debug_mode   = on_debug_mode
        self.ntu_parents     = get_ntu_parent()
        self.joint_names     = get_ntu_joint_names()
        self.joint_inds      = get_ntu_joint_ids()
        self.action_names    = ACTION_CLASSES 

    def __len__(self):
        return len(self.frames_data)

    def __getitem__(self, index):
        '''
        At least one out of coordinates, root_relative, keypoint_value are true 
        At most  one out of coordinates, root_relative, keypoint_value are true 
        :param index: parameter index
        :return: the required objects 
        '''
        # Load data 
        if self.on_debug_mode : 
            index = 0 # focus on only one sample in debug mode 
        
        frames = self.frames_data[index]
        data   = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()

        flags = [self.parent_relative, self.coordinates, self.velocity, self.root_relative]
        assert sum(flags) == 1, "Exactly one of root_relative, coordinates, or velocity, root_relative must be True"
        
        if self.coordinates : 
            observed_poses, target_poses, start_index , name = self.transformation(data)
            return observed_poses, target_poses, start_index, name 
        
        elif self.velocity :
            observed_poses, target_poses, observed_velocity, target_velocity, start_index, name = self.transformation(data) 
            return observed_poses, target_poses, observed_velocity, target_velocity, start_index, name 
        
        elif self.root_relative: 
            observed_root_vectors, targeted_root_vectors, observed_poses, target_poses, scale_factor, start_index, name = self.transformation(data) 
            return observed_root_vectors, targeted_root_vectors, observed_poses, target_poses, scale_factor, start_index, name 
         
        elif self.parent_relative : 
            observed_bones, targeted_bones, observed_poses, target_poses, scale_factor, start_index, name = self.transformation(data)
            return observed_bones, targeted_bones, observed_poses, target_poses, scale_factor, start_index, name 
        
        else : 
            raise NotImplementedError("Must be one of the listed inputs")
            
    def transformation(self, storage, num_joints = 25): 
        '''
        return :
          observed_poses : (train_len, 25 * 2)
          target_poses   : (test_len,  25 * 2)
        '''
        name          = storage['name']
        human_id_list = list(storage['data'].keys())
        total_frame   = self.num_training_frame + self.num_testing_frame
        assert len(human_id_list), f"video {name} has no person in the video"
       
        human_id_list.sort(key = lambda x : storage['data'][x]['colors'].shape[0], reverse = True) 
        human_id    = human_id_list[0] # For now, only support spotting the first person in the video, idea may extened later 
       
        data        = storage['data'][human_id]
        label       = data['colors'] # projected 2D locations
       
        n_frames    = label.shape[0]     
        assert n_frames >= total_frame, "num frames must be more than total frame"
        num_joints  = label.shape[1] 
        dim         = label.shape[2]
        assert num_joints == self.num_joints and dim == 2, "Does not match expected information"
        visibility  = np.ones((label.shape[0], label.shape[1], 1))
        
        start_index    = np.random.randint(0, n_frames - 1 - total_frame + 1)
       
        #assert torch.isnan(label).sum() == 0, "NaN values found in label!"

        for time_ind in range(label.shape[0]): 
            for joint_ind in range(label.shape[1]): 
                if np.isnan(label[time_ind, joint_ind, 0]) or np.isnan(label[time_ind, joint_ind, 1]): 
                    visibility[time_ind, joint_ind] = 0

        if self.coordinates : 
            label[:, :, 0] = label[:, :, 0] / self.ntu_input_w
            label[:, :, 1] = label[:, :, 1] / self.ntu_input_h
            
            label      = torch.tensor(label) 
            visibility = torch.tensor(visibility)
            
            label  = torch.cat([label, visibility], dim = -1) # (x, y, visibility)

            observed_poses  = label[start_index : start_index + self.num_training_frame, :, :]
            observed_poses  = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 3)

            target_poses  = label[start_index + self.num_training_frame : start_index + total_frame, :, :]
            target_poses  = target_poses.contiguous().view(self.num_testing_frame, self.num_joints * 3) 

           # assert len(torch.isnan(observed_poses)) == 0 and len(torch.isnan(target_poses)) == 0
            
            observed_poses[torch.isnan(observed_poses)] = 0.5 
            target_poses[torch.isnan(target_poses)]     = 0.5
            
            return observed_poses, target_poses, start_index, name
        
        elif self.velocity : 
            label      = torch.tensor(label) 
            visibility = torch.tensor(visibility) 
            label      = torch.cat([label, visibility], dim = -1) # (x, y, visibility)
            
            observed_poses = label[start_index : start_index + self.num_training_frame, :, :2] # (train_len, J, 2)
            observed_masks = label[start_index : start_index + self.num_training_frame, :,  2] # (train_len, J) 
            observed_poses = observed_poses.contiguous().view(self.num_training_frame, -1)   # (train_len, J * 2)
            observed_poses[torch.repeat_interleave(observed_masks.bool(), 2, dim = -1)] = torch.nan
            
            observed_velocity = observed_poses[1:, :] - observed_poses[:-1, :]


            target_poses  = label[start_index + self.num_training_frame : start_index + total_frame, :, :2] # (test_len, J, 2) 
            target_masks  = label[start_index + self.num_training_frame : start_index + total_frame, :,  2] # (test_len, J)
            target_poses  = target_poses.contiguous().view(self.num_testing_frame, -1) # (test_len, J * 2)
            target_poses[torch.repeat_interleave(target_masks.bool(), 2, dim = -1)] = torch.nan 
            
            target_velocity  = torch.cat([(target_poses[0, :] - observed_poses[-1, :]).unsqueeze(0), target_poses[1:, :] - target_poses[:-1, :]], dim = 0) 

            observed_poses[torch.isnan(observed_poses)] = 0.5 
            target_poses[torch.isnan(target_poses)]     = 0.5

            observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
            target_poses   = torch.cat([target_poses.contiguous().view(-1, num_joints, 2),   target_masks.unsqueeze(-1)],   dim = -1)
            observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
            target_poses   = target_poses.contiguous().view(-1,   num_joints * 3)
            
            observed_velocity[torch.isnan(observed_velocity)] = 0 
            target_velocity[torch.isnan(target_velocity)]     = 0

            return observed_poses, target_poses, observed_velocity, target_velocity, start_index, name 
        
        elif self.parent_relative: 
            head_position = self.joint_inds['Head']

            median_head_pos_x, median_head_pos_y = np.median(label[:, head_position, 0]), np.median(label[:, head_position, 1]) 
            
            assert (not np.isnan(median_head_pos_x)) and (not np.isnan(median_head_pos_y)), f"cannot extract head set {name} and {storage}" 
            
            min_x, max_x, min_y, max_y = self.input_w, 0, self.input_h, 0

            for time_step in range(start_index, start_index + total_frame): 
              for joint in range(self.num_joints): 
                if visibility[time_step, joint] : 
                    max_x = max(max_x, label[time_step, joint, 0])
                    min_x = min(min_x, label[time_step, joint, 0])
                    max_y = max(max_y, label[time_step, joint, 1])
                    min_y = min(min_y, label[time_step, joint, 1])

            assert max_x >= min_x and max_y >= min_y, "No human ever appears in the scenes"

            scale_x = 1. * (max_x - min_x) / self.ntu_input_w * self.input_w 
            scale_y = 1. * (max_y - min_y) / self.ntu_input_h * self.input_h 
            scale_factor = np.array([scale_x, scale_y]) 

            vector_label = np.zeros((total_frame, self.num_joints, 2))

            for time_step in range(start_index, start_index + total_frame):
                if visibility[time_step, head_position] == 0: 
                   label[time_step, head_position, 0], label[time_step, head_position, 1] = median_head_pos_x, median_head_pos_y
                
                curr_index = time_step - start_index

                x_norm = label[time_step, :, 0] / self.ntu_input_w
                y_norm = label[time_step, :, 1] / self.ntu_input_h

                vector_label[curr_index, head_position, 0], vector_label[curr_index, head_position, 1] = x_norm[head_position], y_norm[head_position] 
                for child_index, childname in enumerate(self.joint_names): 
                    parent_index = self.ntu_parents[child_index]
                    if parent_index == child_index : 
                        continue 
                    if visibility[time_step, child_index] and visibility[time_step, parent_index]:
                        vector_label[curr_index, child_index, 0] = (x_norm[parent_index] - x_norm[child_index]) * self.input_w / scale_factor[0]
                        vector_label[curr_index, child_index, 1] = (y_norm[parent_index] - y_norm[child_index]) * self.input_h / scale_factor[1]


            observed_poses = torch.tensor(label[start_index : start_index + self.num_training_frame, :, :]) 
            observed_bones = torch.tensor(vector_label[: self.num_training_frame, :, :]) 
            observed_bones = observed_bones.contiguous().view(self.num_training_frame, -1) 


            target_poses   = torch.tensor(label[start_index + self.num_training_frame : start_index + total_frame, :, :])
            targeted_bones = torch.tensor(vector_label[self.num_training_frame :, :, :]) 
            targeted_bones = targeted_bones.contiguous().view(self.num_testing_frame, -1) 

            return observed_bones, targeted_bones, observed_poses, target_poses, scale_factor, start_index, name 
        
        elif self.root_relative:
            # Re-process later, we can further compute the statistics of all videos, instead of a single frame 
            # For NTU, we can use spine hip as the root and find the relative position of other joints to the spine base 

            root_position = 0

            median_root_pos_x, median_root_pos_y = np.median(label[:, root_position, 0]), np.median(label[:, root_position, 1]) 
            
            assert (not np.isnan(median_root_pos_x)) and (not np.isnan(median_root_pos_y)), f"cannot extract root in set {name} and {storage}" 
            
            min_x, max_x, min_y, max_y = self.input_w, 0, self.input_h, 0

            for time_step in range(start_index, start_index + total_frame): 
              for joint in range(self.num_joints): 
                if visibility[time_step, joint] : 
                    max_x = max(max_x, label[time_step, joint, 0])
                    min_x = min(min_x, label[time_step, joint, 0])
                    max_y = max(max_y, label[time_step, joint, 1])
                    min_y = min(min_y, label[time_step, joint, 1])

            assert max_x >= min_x and max_y >= min_y, "No human ever appears in the scenes"

            scale_x = 1. * (max_x - min_x) / self.ntu_input_w * self.input_w 
            scale_y = 1. * (max_y - min_y) / self.ntu_input_h * self.input_h 
            scale_factor = np.array([scale_x, scale_y]) 

            vector_label = np.zeros((total_frame, self.num_joints, 2))

            for time_step in range(start_index, start_index + total_frame):
                if visibility[time_step, root_position] == 0: 
                   label[time_step, root_position, 0], label[time_step, root_position, 1] = median_root_pos_x, median_root_pos_y
                
                curr_index = time_step - start_index

                x_norm = label[time_step, :, 0] / self.ntu_input_w
                y_norm = label[time_step, :, 1] / self.ntu_input_h

                vector_label[curr_index, root_position, 0], vector_label[curr_index, root_position, 1] = x_norm[root_position], y_norm[root_position] 

                for child_index, childname in enumerate(self.joint_names): 
                    if child_index == root_position : 
                        continue 
                    vector_label[curr_index, child_index, 0] = (x_norm[child_index] - x_norm[root_position]) * self.input_w / scale_factor[0]
                    vector_label[curr_index, child_index, 1] = (y_norm[child_index] - y_norm[root_position]) * self.input_h / scale_factor[1]


            observed_poses        = torch.tensor(label[start_index : start_index + self.num_training_frame, :, :]) 
            observed_root_vectors = torch.tensor(vector_label[: self.num_training_frame, :, :]) 
            observed_root_vectors = observed_root_vectors.contiguous().view(self.num_training_frame, -1) 

            target_poses   = torch.tensor(label[start_index + self.num_training_frame : start_index + total_frame, :, :])
            targeted_root_vectors = torch.tensor(vector_label[self.num_training_frame :, :, :]) 
            targeted_root_vectors = targeted_root_vectors.contiguous().view(self.num_testing_frame, -1) 

            return observed_root_vectors, targeted_root_vectors, observed_poses, target_poses, scale_factor, start_index, name 
        else : 
            raise NotImplementedError("Method not supported for now")

    def get_center_map(self, x, y, sigma, size_w, size_h):
       '''
       Generate Gaussian Heat Map
       :param x: center point x
       :param y: center point y
       :param sigma: gaussian sigma
       :param size_w: size of the heat map
       :param size_h: size of the heat map
       :return: numpy array of shape (size_w, size_h)
       '''
       gridy, gridx = np.mgrid[0 : size_h, 0 : size_w]
       L2_distance = (gridx - x) ** 2 + (gridy - y) ** 2
       variance = sigma ** 2
       return np.exp(-1.0 * L2_distance / 2.0 / variance)

    def is_not_on_plane(self, x, y, width, height):
        return x < 0.001 or y < 0.001 or x > width or y > height

    def gen_heat_map(self, total_frame, label, box_size, stride, sigma):
        '''
        generate label heat map for each joints
        :param label: 3 * parts_num * sequence_train
        :param box_size: size of the box
        :param stride:
        :param sigma: gaussian sigma
        :param img_old_dim : original image dimension
        :return: tensor of shape (sequence_train, joints_num + 1, heatmap_size, heatmap_size)
        '''

        heatmap_size = box_size // stride
        frames_heat_map = torch.zeros(total_frame, self.num_joints + 1, heatmap_size, heatmap_size)
        for f_idx in range(total_frame):
            for j_idx in range(self.num_joints):
                if label[2, j_idx, f_idx] > 0:
                    cx, cy   = label[0, j_idx, f_idx], label[1, j_idx, f_idx]
                    cx = cx * self.map_w / self.input_w
                    cy = cy * self.map_h / self.input_h
                    heat_map = self.get_center_map(x = cx, y = cy, sigma = sigma, size_w = self.map_w, size_h = self.map_h)
                else :
                    heat_map = np.zeros((heatmap_size, heatmap_size))
                frames_heat_map[f_idx, j_idx, :, :] = torch.from_numpy(heat_map)
            background_map = np.ones((heatmap_size, heatmap_size))


            for m in range(heatmap_size):
                for n in range(heatmap_size):
                    max_value = max(frames_heat_map[f_idx, :, m, n])
                    background_map[m, n] = max(1. - max_value, 0)

            frames_heat_map[f_idx, self.num_joints, :, :] = torch.from_numpy(background_map)
        return frames_heat_map

    @staticmethod
    def get_default_transform():
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
#            normalize,
        ])
        return transform

    @staticmethod 
    def get_activity_from_file(path_file):
        # The pattern is SsssCcccPpppRrrrAaaa.skeleton
        pattern        = path_file.split('/')[-1].split('.')[0]
        setup_id       = int(pattern[1 : 4])
        camera_id      = int(pattern[5 : 8])
        subject_id     = int(pattern[9 : 12])
        replication_id = int(pattern[13 : 16])
        activity_id    = int(pattern[17 :])
        return (setup_id, camera_id, subject_id, replication_id, activity_id)
    
    @staticmethod
    def capture_video_frame(video_file_path, video_name):
      cap = cv2.VideoCapture(os.path.join(video_file_path, video_name.split('.')[0] + '_rgb.avi'))
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
           break  # End of video
        frames.append(frame) 
      return np.array(frames)
    
    @staticmethod 
    def get_video_frames(name, start_index, num_training_frame, num_testing_frame, input_w, input_h):
        video_frames = NTU.capture_video_frame(NTU_VIDEO_PATH, video_name = name) 
        total_frames = num_training_frame + num_testing_frame
        if video_frames is not None :
            resized_frames = []  
            for image in video_frames : 
                resized_frames.append(cv2.resize(image, (input_w, input_h)))
                video_frames = np.array(resized_frames) 
                observed_images = video_frames[start_index : start_index + num_training_frame, :, :, :] 
                target_images   = video_frames[start_index + num_training_frame : start_index + total_frames, :, :, :] 
                return observed_images, target_images
        return None  

#if __name__ == '__main__': 
    # ntu_set = NTU(train = False, coordinates = True)
    # observed_poses, target_poses, observed_images, target_images, name = ntu[62]
    # print(name) 
    # fig, axes = plt.subplots(1, 3, figsize = (24, 24))
    # for index in range(min(3, len(observed_poses))):
    #     curr_frame_pose = observed_poses[index, :].contiguous().view(25, 3) 
    #     curr_frame_pose[:, 0] = curr_frame_pose[:, 0] * ntu.input_w
    #     curr_frame_pose[:, 1] = curr_frame_pose[:, 1] * ntu.input_h
    #     background_img  = observed_images[index, :, :, :]
    #     annotated_img   = draw_skeleton(background_img, curr_frame_pose.detach().cpu().numpy()) 
    #     axes[index].imshow(annotated_img)    
    # plt.tight_layout()
    # plt.show()