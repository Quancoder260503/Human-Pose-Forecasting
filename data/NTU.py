import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


''' 
For Nanyang Tech dataset, 
RGB frame exists but very heavy, around 133GB, 
hence only the joints and heat maps are 
used for training and testing. 

May add Gaussian Filter to the joint coordinates
'''

class NTU(Dataset):
    def __init__(self,
        map_h   = 46,
        map_w   = 46,
        num_joints = 25, # 25 joints for NTU
        total_frame = 30, 
        gaussian_sigma = 21,
        data_dir = 'dataset/NanyangTechRGB/',
        train = True,
    ):
        super(NTU, self).__init__()
        # Standard resolution for NTU 
        self.input_h = 1920 
        self.input_w = 1080
        self.map_h = map_h
        self.map_w = map_w
        self.num_joints     = num_joints
        self.total_frame    = total_frame
        self.gaussian_sigma = gaussian_sigma
        self.train = train
        if self.train is True :
            self.data_dir = data_dir + 'train/'
        else :
            self.data_dir = data_dir + 'test/'
        self.frames_data = os.listdir(self.data_dir)

    def __len__(self):
        # number of videos in train or test set
        return len(self.frames_data)

    def __getitem__(self, index):
        '''
        :param index: parameter index
        :return:
          observed_poses : tensor of shape (total_frame, num_joints * 2) 
        '''
        frames = self.frames_data[index]
        storage   = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()
        observed_poses, target_poses = self.transformation_on_joint(
            storage = storage, 
            num_joints = self.num_joints
        )  
        return observed_poses, target_poses
    
    def transformation_on_joint(self, storage, num_joints = 25): 
        '''
        return :
          observed_poses : (train_len, 25 * 2)
          target_poses   : (test_len,  25 * 2)
        '''
        name        = storage['name']
        human_id    = list(storage['data'].keys())[0] # So far only support spotting the first person in the video, idea may extened later 
        data        = storage['data'][human_id]
        label       = data['colors'] # projected 2D locations
        # scale the image, may smooth it later 
        label[:, :, 0] = label[:, :, 0] / self.input_w 
        label[:, :, 1] = label[:, :, 1] / self.input_h 
        print(label.shape)
        n_frames    = label.shape[0]     
        num_joints  = label.shape[1] 
        dim         = label.shape[2]

        assert num_joints == self.num_joints and dim == 2, "Does not match expected information"
        
        total_frame = self.total_frame

        num_training_frame = total_frame // 2 + total_frame % 2 
        num_testing_frame  = total_frame - num_training_frame
        
        start_index = np.random.randint(0, n_frames - 1 - total_frame + 1)

        observed_poses = torch.tensor(label[start_index : start_index + num_training_frame, :, :]) 
        observed_poses = observed_poses.contiguous().view(num_training_frame, self.num_joints * 2) 

        target_poses = torch.tensor(label[start_index + num_training_frame : start_index + total_frame, :, :]) 
        target_poses = target_poses.contiguous().view(num_testing_frame, self.num_joints * 2) 

        observed_poses[torch.isnan(observed_poses)] = 0
        target_poses[torch.isnan(target_poses)]     = 0

        return observed_poses, target_poses


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


    def gen_heat_map(self, label, box_size, stride, sigma, img_old_dim):
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
        frames_heat_map = torch.zeros(self.sequence_train, self.num_joints + 1, heatmap_size, heatmap_size)
        for f_idx in range(self.sequence_train):
            for j_idx in range(self.num_joints):
                cx, cy   = label[0, j_idx, f_idx], label[1, j_idx, f_idx]
                cx = cx * self.map_w / self.input_w
                cy = cy * self.map_h / self.input_h
                #print(cx, cy)
                heat_map = self.get_center_map(x = cx, y = cy, sigma = sigma, size_w = self.map_w, size_h = self.map_h)
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

# if __name__ == '__main__': 
#     ntu = NTU(train = False)
#     print(len(ntu))