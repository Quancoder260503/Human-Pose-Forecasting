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

May add Gaussian Filter to the joint coordinates

TODO : 
1. Visualizing the joint using video 
2. Implementing RootRelative, Bone Vector + Velocity
'''

class NTU(Dataset):
    def __init__(self,
        num_training_frame = 7, 
        num_testing_frame  = 7, 
        ntu_input_w = 1920, 
        ntu_input_h = 1080, 
        input_w = 480,
        input_h = 480,
        map_h   = 46,
        map_w   = 46,
        num_joints     = 25,
        gaussian_sigma = 21,
        data_dir = 'dataset/NanyangTechRGB/',
        train = True,
        require_image  = False,  # For now, does not support require image due to limited capability 
        require_action = False, 
        bone_vector    = False, 
        velocity       = False, 
        coordinates    = False, 
        on_debug_mode  = False,
        require_name   = False, 
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
        :param bone_vector        : input processed as normalized root bone_vector ?
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
        self.num_joints = num_joints
        self.gaussian_sigma = gaussian_sigma
        self.transform      = self.get_default_transform() # Just in case I might use image for training 
        self.train          = train
        self.require_image  = require_image    # for now, this is not supported
        self.require_action = require_action
        if self.train is True :
            self.data_dir = data_dir + 'train/'
        else :
            self.data_dir = data_dir + 'test/'

        self.frames_data   = os.listdir(self.data_dir)
        self.bone_vector   = bone_vector 
        self.velocity      = velocity 
        self.coordinates   = coordinates
        self.on_debug_mode = on_debug_mode
        self.require_name  = require_name
        self.ntu_parents   = JOINT_PARENT_NAMES
        self.joint_names   = JOINT_NAMES 
        self.action_names  = ACTION_CLASSES

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
            observed_poses, target_poses, name = self.transformation(data)
            if self.require_name : 
               return observed_poses, target_poses, name  
            else : 
               return observed_poses, target_poses
        elif self.bone_vector : 
            pass # to implement 
            
        else : 
            pass # to implement 
            
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
       
        # scale the image, may smooth it later 
        label[:, :, 0] = label[:, :, 0] / self.ntu_input_w
        label[:, :, 1] = label[:, :, 1] / self.ntu_input_h
        n_frames    = label.shape[0]     
        assert n_frames >= total_frame, "num frames must be more than total frame"
        num_joints  = label.shape[1] 
        dim         = label.shape[2]
        assert num_joints == self.num_joints and dim == 2, "Does not match expected information"
        visibility  = np.ones((label.shape[0], label.shape[1], 1))
        # Mark up the visibility 
        
        start_index = np.random.randint(0, n_frames - 1 - total_frame + 1)
        label          = torch.tensor(label) 
        visibility     = torch.tensor(visibility)
        
        
        #assert torch.isnan(label).sum() == 0, "NaN values found in label!"

        for time_ind in range(label.shape[0]): 
            for joint_ind in range(label.shape[1]): 
                if torch.isnan(label[time_ind, joint_ind, 0]) or torch.isnan(label[time_ind, joint_ind, 1] == 0) : 
                    visibility[time_ind, joint_ind] = 0

        label  = torch.cat([label, visibility], dim = -1) # (x, y, visibility)

        observed_poses = label[start_index : start_index + self.num_training_frame, :, :] 
        observed_poses = observed_poses.contiguous().view(self.num_training_frame, self.num_joints * 3) 

        target_poses = label[start_index + self.num_training_frame : start_index + total_frame, :, :]
        target_poses = target_poses.contiguous().view(self.num_testing_frame, self.num_joints * 3) 

        #assert len(torch.isnan(observed_poses)) == 0 and len(torch.isnan(target_poses)) == 0
        observed_poses[torch.isnan(observed_poses)] = 0.5 
        target_poses[torch.isnan(target_poses)]     = 0.5
        return observed_poses, target_poses, name

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
    
def get_colors():
    colors = {
        'pink' : np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors

def draw_skeleton(image, kp_2d, thickness = 1, shadow = 0.3):
    kp_2d = np.array(kp_2d, dtype=int)
    image = image.copy()
   # image = (image * shadow).astype(np.uint8)
    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = get_ntu_skeleton()
    for idx, pt in enumerate(kp_2d):
        if pt[0] > 0 and pt[1] > 0: 
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)

    for i, (j1, j2) in enumerate(skeleton):
        if kp_2d[j1, 0] > 0 and kp_2d[j1, 1] > 0 and kp_2d[j2, 0] > 0 and kp_2d[j2, 1] > 0:
            color = rcolor 
            pt1 = tuple(kp_2d[j1, :2])
            pt2 = tuple(kp_2d[j2, :2])
            cv2.line(image, pt1, pt2, color, thickness= 2)
    return image

if __name__ == '__main__': 
    ntu = NTU(train = True, coordinates = True) 
    observed_poses, target_poses, name = ntu[0]
    print(name) 
    fig, axes = plt.subplots(1, 3, figsize = (24, 8))
    for index in range(min(3, len(observed_poses))):
        curr_frame_pose = observed_poses[index, :].contiguous().view(25, 3) 
        curr_frame_pose[:, 0] = curr_frame_pose[:, 0] * 1080
        curr_frame_pose[:, 1] = curr_frame_pose[:, 1] * 1080
        background_img  = (np.ones((1080, 1080, 3)) * 255).astype(np.uint8) 
        annotated_img   = draw_skeleton(background_img, curr_frame_pose.detach().cpu().numpy()) 
        axes[index].imshow(annotated_img)    
    plt.tight_layout()
    plt.show()