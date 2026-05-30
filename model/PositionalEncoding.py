import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt

class PositionalEncoding1D(object):
    def __init__(self, model_dim = 512, temperature=10000, alpha=1):
        self._model_dim = model_dim
        self._temperature    = temperature
        self._alpha          = alpha

    def __call__(self, seq_length):
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(self._model_dim)[np.newaxis, :])

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = pos_encoding.astype(np.float32)

        return torch.from_numpy(pos_encoding)

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(self._temperature, (2 *  (i//2)) / np.float32(self._model_dim))
        return self._alpha*pos * angle_rates

    def visualize(self, num_visualized):
        pos_encoding = self(num_visualized).numpy()[0]
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(pos_encoding, cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, self.d_model))
        plt.ylabel('Position in Sequence')
        plt.title('1D Positional Encoding')
        plt.colorbar()
        plt.tight_layout()
        plt.show()