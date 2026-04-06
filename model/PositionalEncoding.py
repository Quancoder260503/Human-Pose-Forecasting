import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

class PositionalEncoding1D(object):
    """
    Positional Encoding for 1D sequences 
    """
    def __init__(self, position_dim = 512, temperature = 10000, alpha = 1):
        self.position_dim = position_dim 
        self.temperature  = temperature 
        self.alpha        = alpha 
    
    def get_angles(self, position, index): 
        angle_rates = 1.0 / np.power(self.temperature, (2 * (index // 2)) / np.float32(self.position_dim))
        return self.alpha * position * angle_rates
    
    def __call__(self, sequence_length):
        angle_radians = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis], 
            np.arange(self.position_dim)[np.newaxis, :]
        ) # (sequence_length, position_dim)

        angle_radians[:, 0::2] = np.sin(angle_radians[:, 0::2])
        angle_radians[:, 1::2] = np.cos(angle_radians[:, 1::2])
        
        position_encoding = angle_radians[np.newaxis, ...] # (1, sequence_length, pos_dim) 
        position_encoding = position_encoding.astype(np.float32)

        return torch.from_numpy(position_encoding)
    
    @staticmethod
    def visualize_1d_encoding(num_visualized):
        position_encoder = PositionalEncoding1D()
        pos_encoding = position_encoder(num_visualized).numpy()

        plt.pcolormesh(pos_encoding[0], cmap = 'RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 512)) # Position dim 
        plt.ylabel('Positiom in Sequence')
        plt.colorbar() 
        plt.show()