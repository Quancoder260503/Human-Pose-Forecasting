import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class JHMDB(Dataset):
    def __init__(self,
        input_w = 368,
        input_h = 368,
        map_h   = 46,
        map_w   = 46,
        num_joints = 15,
        sequence_train = 5,
        gaussian_sigma = 21,
        data_dir = 'dataset/JHMDB/',
        train = True,
        transform = None
    ):
        super(JHMDB, self).__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.map_h = map_h
        self.map_w = map_w
        self.num_joints = num_joints
        self.sequence_train = sequence_train
        self.gaussian_sigma = gaussian_sigma
        self.transform = transform
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
           images     : Tensor of shape (sequence_train, 3, width, height)
           label_map  : Tensor of shape (46, 46, class + 1, sequence_train)
           center_map : Tensor of shape  (1, 368, 368)
        '''
        frames = self.frames_data[index]
        data   = np.load(os.path.join(self.data_dir, frames), allow_pickle = True).item()

        images, label, heat_maps, center_map = self.transformation_jhmdb(
            data = data,
            box_size = self.input_w,
            num_joints = 15,
        )
        # (1, 368, 368) -> (1, 1, 368, 368)
        center_map = center_map.unsqueeze_(0)
        return images, label, heat_maps, center_map

    def transformation_jhmdb(self, data, box_size = 368, num_joints = 15):
        '''
        :param data: data dictionary
        :param box_size: the box size
        :param num_joints: number of joints
        :param train: training or testing data
        :return: images : Tensor of shape (sequence_train, 3, width, height)
        '''
        n_frames    = data['nframes']
        frames_path = data['framepath']
        pose        = data['pos_img']
        pose        = pose.transpose(2, 1, 0)

        x = pose[:, :, 0]
        y = pose[:, :, 1]

        # sample a random starting frame index so that the number of frames trained is at least 5
        start_index = np.random.randint(0, n_frames - 1 - self.sequence_train + 1)
        images = torch.zeros(self.sequence_train, 3, self.input_h, self.input_w)
        label  = np.zeros((2, num_joints, self.sequence_train)) # (x, y)
        H, W = -1, -1
        for i in range(self.sequence_train):
            img_path = os.path.join(frames_path, '%05d' % (start_index + i + 1) + '.png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]  # numpy arrays are (h, w, c)
            if H > -1 and W > -1:
                assert H == h and W == w, "Image size must matched previous frames"
            else :
                W, H = w, h

            img = cv2.resize(img, (self.input_w, self.input_h), interpolation = cv2.INTER_LINEAR)
            images[i, :, :, :] = self.transform(img)
            # Read the label
            ratio_x = self.input_w / W
            ratio_y = self.input_h / H
            label[0, :, i] = x[start_index + i] * ratio_x
            label[1, :, i] = y[start_index + i] * ratio_y

        heat_maps  =  self.gen_heat_map(
            label,
            box_size = box_size,
            stride = 8,
            sigma = self.gaussian_sigma / 3,
            img_old_dim = [H, W]
        )

        center_map =  self.get_center_map(
            x = box_size / 2.0,
            y = box_size / 2.0,
            sigma = self.gaussian_sigma,
            size_w = box_size,
            size_h = box_size
        )
        center_map = torch.from_numpy(center_map)
        return images, label, heat_maps, center_map

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

# test_jhmdb_action = JHMDB(transform=JHMDB.get_default_transform(368, 368))
# image, label, heat_map, center_map = test_jhmdb_action[10]
#
# num_joints = heat_map.shape[1]  # includes background channel
# num_cols   = 6
# num_rows   = (num_joints + 1 + num_cols - 1) // num_cols  # +1 for the image
#
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
# axes = axes.flatten()
#
# # Show image in first cell
# axes[0].imshow(JHMDB.torch2numpy(image[0]))
# axes[0].set_title("Image")
# axes[0].axis("off")
#
# # Show all joint heatmaps
# for j in range(num_joints):
#     axes[j + 1].imshow(heat_map[0][j], cmap="inferno")
#     title = f"Joint {j}" if j < num_joints - 1 else "Background"
#     axes[j + 1].set_title(title)
#     axes[j + 1].axis("off")
#
# # Hide any unused subplots
# for k in range(num_joints + 1, len(axes)):
#     axes[k].axis("off")
#
# plt.tight_layout()
# plt.show()