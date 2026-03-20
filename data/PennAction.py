import os
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, dtype, floating
from torch import Tensor
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import math 


def get_pennaction_parents(): 
    return [0, 0, 0, 1, 2, 3, 4, 1, 2, 7, 8, 9, 10]

class PennAction(Dataset):
    def __init__(self,
        num_training_frame = 7, 
        num_testing_frame  = 7, 
        input_w = 480,
        input_h = 480,
        map_h   = 46,
        map_w   = 46,
        num_joints     = 13,
        gaussian_sigma = 21,
        data_dir = 'dataset/PennAction/',
        train = True,
        require_image   = False, 
        require_action  = False, 
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
        :param parent_relative    : input processed as normalized root bone_vector ?
        :param velocity           : input processed as velocity (displacment) of key joints on different frames ?
        :param coordinates        : input processed as local coordinate of the image ? 
        :param on_debug_mode      : extract only one instance ? used only for debugging model   
        """

        super(PennAction, self).__init__()
        self.num_training_frame = num_training_frame 
        self.num_testing_frame  = num_testing_frame 
        self.input_w = input_w
        self.input_h = input_h
        self.map_h = map_h
        self.map_w = map_w
        self.num_joints = num_joints
        self.gaussian_sigma = gaussian_sigma
        self.transform = self.get_default_transform() # Just in case I might use image for training 
        self.train          = train
        self.require_image  = require_image 
        self.require_action = require_action
        if self.train is True :
            self.data_dir = data_dir + 'train/'
        else :
            self.data_dir = data_dir + 'test/'

        self.frames_data        = os.listdir(self.data_dir)
        self.parent_relative    = parent_relative
        self.velocity           = velocity 
        self.coordinates        = coordinates
        self.on_debug_mode      = on_debug_mode
        self.pennaction_parents = get_pennaction_parents()

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

        flags = [self.bone_vector, self.coordinates, self.velocity]
        assert sum(flags) == 1, "Exactly one of root_relative, coordinates, or velocity must be True"
        
        if self.coordinates : 
            observed_poses, target_poses, observed_images, target_images = self.transformation(data)
            if self.require_image : 
               return observed_poses, target_poses, observed_images, target_images
            else : 
              return observed_poses, target_poses
        elif self.parent_relative : 
            observed_bones, target_bones, observed_poses, target_poses, scaled_factor, observed_images, target_images = self.transformation(data) 
            if self.require_image : 
                return observed_bones, target_bones, observed_poses, target_poses, scaled_factor, observed_images, target_images
            else : 
                return observed_bones, target_bones, observed_poses, target_poses, scaled_factor
        else : 
            observed_poses, target_poses, observed_velocity, target_velocity,  observed_images, target_images = self.transformation(data) 
            if self.require_image : 
                return observed_poses, target_poses, observed_velocity, target_velocity, observed_images, target_images
            else : 
                return observed_poses, target_poses, observed_velocity, target_velocity 
            
    def transformation(self, data, num_joints = 13): 
        '''
        return :
          observed_poses : (train_len, 39) # coordinates 
          target_poses   : (test_len,  39) # coordinates 
        '''
        n_frames    = data['nframes']      
        frames_path = data['framepath']
        dim         = data['dimensions']     
        
        x = data['x'].astype(np.float64)
        y = data['y'].astype(np.float64)

        visibility = data['visibility']
        total_frame = self.num_training_frame + self.num_testing_frame

        start_index = np.random.randint(0, n_frames - 1 - total_frame + 1) 

        label        = np.zeros((3, num_joints, total_frame))  
        images       = torch.zeros(total_frame, 3, self.input_h, self.input_w)
        vector_label = np.zeros((2, num_joints, total_frame))

        # get median head pos
        h, w = dim 
        median_head_pos_x, median_head_pos_y = np.median(x[:, 0]), np.median(y[:, 0])
        scale_x      = (np.max(x[start_index : start_index + total_frame, :]) - np.min(x[start_index : start_index + total_frame, :])) / w * self.input_w 
        scale_y      = (np.max(y[start_index : start_index + total_frame, :]) - np.min(y[start_index : start_index + total_frame, :])) / h * self.input_h
        scale_factor = np.array([scale_x, scale_y]) 
        
        for i in range(total_frame):
            img_path = os.path.join(frames_path, '%06d' % (start_index + i + 1) + '.jpg')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_w, self.input_h), interpolation = cv2.INTER_CUBIC)
            images[i, :, :, :] = self.transform(img)
            
            label[0, :, i] = x[start_index + i, :] / w
            label[1, :, i] = y[start_index + i, :] / h
            label[2, :, i] = visibility[start_index + i]

            if visibility[start_index + i, 0] == 0: 
                   # head is not visible somehow 
                x[start_index + i, 0], y[start_index + i, 0] = median_head_pos_x, median_head_pos_y
                # if bone_vector is True, then the third dimension will store the scale
                # storing head locations first, normalize its pixel location in the image
                
            x_norm = x[start_index + i, :] / w
            y_norm = y[start_index + i, :] / h
                
            vector_label[0, 0, i], vector_label[1, 0, i]  = x_norm[0], y_norm[0]

            for joint, parent in enumerate(self.pennaction_parents[1:], start=1):
                if visibility[start_index + i, joint] and visibility[start_index + i, parent] : 
                    vector_label[0, joint, i] = (x_norm[parent] - x_norm[joint]) * self.input_w / scale_factor[0]
                    vector_label[1, joint, i] = (y_norm[parent] - y_norm[joint]) * self.input_h / scale_factor[1]
                
    

        observed_img = images[: self.num_training_frame, :, :, :] 
        target_img   = images[self.num_training_frame :, :, :, :]
        
        if self.coordinates : 
            # Observed pose transformation
            observed_poses = torch.tensor(label[:2, :, : self.num_training_frame]) 
            observed_masks = torch.tensor(label[2,  :, : self.num_training_frame])
            observed_poses = observed_poses.permute(2, 1, 0) 
            observed_masks = observed_masks.permute(1, 0) 
            observed_poses = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 2) 
            observed_poses[torch.repeat_interleave(observed_masks, 2, dim = -1) == 0] = np.nan

            # Target pose transformation 
            target_poses = torch.tensor(label[:2, :, self.num_training_frame : ]) 
            target_masks = torch.tensor(label[2,  :, self.num_training_frame : ])
            target_poses = target_poses.permute(2, 1, 0) 
            target_masks = target_masks.permute(1, 0) 
            target_poses = target_poses.contiguous().view(self.num_testing_frame, self.num_joints * 2) 
            target_poses[torch.repeat_interleave(target_masks, 2, dim = -1) == 0] = np.nan

            # Replace nan by 0.5
            observed_poses[torch.isnan(observed_poses)] = 0.5
            target_poses[torch.isnan(target_poses)]     = 0.5

            observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
            target_poses   = torch.cat([target_poses.contiguous().view(-1, num_joints, 2),   target_masks.unsqueeze(-1)],   dim = -1)

            observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
            target_poses   = target_poses.contiguous().view(-1,   num_joints * 3)

            return observed_poses, target_poses, observed_img, target_img 

        elif self.parent_relative: 
           observed_poses = torch.tensor(label[:, :, : self.num_training_frame])
           observed_poses = observed_poses.permute(2, 1, 0) # (Time, J, 3) 
           observed_bones = torch.tensor(vector_label[:, :, : self.num_training_frame]) # (2, J, train_len)
           observed_bones = observed_bones.permute(2, 1, 0)
           observed_bones = observed_bones.contiguous().view(self.num_training_frame, -1) 


           target_poses   = torch.tensor(label[:, :, self.num_training_frame : ])
           target_poses   = target_poses.permute(2, 1, 0)
           targeted_bones = torch.tensor(vector_label[:, :, self.num_training_frame :]) # (2, J, test_len)
           targeted_bones = targeted_bones.permute(2, 1, 0) 
           targeted_bones = targeted_bones.contiguous().view(self.num_testing_frame, -1) 

           return observed_bones, targeted_bones, observed_poses, target_poses, scale_factor, observed_img, target_img
        
        else : # Velocity implementation 

            # This is poor software practice, but given the scale of the experiment, this can be neglected 

            # Observed pose transformation 

            observed_poses = torch.tensor(label[:2, :, : self.num_training_frame]) 
            observed_masks = torch.tensor(label[2,  :, : self.num_training_frame])
            observed_poses = observed_poses.permute(2, 1, 0) 
            observed_masks = observed_masks.permute(1, 0) 
            observed_poses = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 2) 
            observed_poses[torch.repeat_interleave(observed_masks, 2, dim = -1) == 0] = np.nan

            observed_velocity = observed_poses[1 :, :] - observed_poses[:-1, :] 

            # Target pose transformation 
            target_poses = torch.tensor(label[:2, :, self.num_training_frame : ]) 
            target_masks = torch.tensor(label[2,  :, self.num_training_frame : ])
            target_poses = target_poses.permute(2, 1, 0) 
            target_masks = target_masks.permute(1, 0) 
            target_poses = target_poses.contiguous().view(self.num_testing_frame, self.num_joints * 2) 
            target_poses[torch.repeat_interleave(target_masks, 2, dim = -1) == 0] = np.nan

            target_velocity = torch.cat([target_poses[0, :] - observed_poses[-1, :],  target_poses[1:, :] -  target_poses[:-1, :]], dim = 0)

            observed_poses[torch.isnan(observed_poses)] = 0.5
            target_poses[torch.isnan(target_poses)]     = 0.5

            observed_poses = torch.cat([observed_poses.contiguous().view(-1, num_joints, 2), observed_masks.unsqueeze(-1)], dim = -1) 
            target_poses   = torch.cat([target_poses.contiguous().view(-1, num_joints, 2),   target_masks.unsqueeze(-1)],   dim = -1)
            observed_poses = observed_poses.contiguous().view(-1, num_joints * 3) 
            target_poses   = target_poses.contiguous().view(-1,   num_joints * 3)

            observed_velocity[torch.isnan(observed_velocity)] = 0 
            target_velocity[torch.isnan(target_velocity)]     = 0

            return observed_poses, target_poses, observed_velocity, target_velocity, observed_img, target_img 


        # For now, skip these twos 
        # heat_maps  =  self.gen_heat_map(
        #     total_frame = total_frame, 
        #     label       = label, 
        #     box_size    = min(self.input_w, self.input_h),
        #     stride      = 8,
        #     sigma       = self.gaussian_sigma / 3,
        # )

        # center_map =  self.get_center_map(
        #     x      = self.input_w / 2.0,
        #     y      = self.input_h / 2.0,
        #     sigma  = self.gaussian_sigma,
        #     size_w = self.input_w,
        #     size_h = self.input_h
        # )
        #center_map = torch.from_numpy(center_map)

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
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
         #   normalize,
        ])
        return transform


if __name__ == "__main__":
    test_penn_action = PennAction(bone_vector = True, train = False, require_image = False) 
    observed_bones, target_bones, scaled_factor = test_penn_action[0]
    scaled_factor = torch.tensor(scaled_factor)
    og_poses = PennAction.reconstruct_pose(observed_bones.unsqueeze(0), scaled_factor.unsqueeze(0))
 #   print(og_poses.shape)
    