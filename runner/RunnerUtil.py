import matplotlib.pyplot as plt
import numpy as np
import torch 
import math 
from data.DatasetUtil         import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def save_checkpoint(epoch, model_a, model_b, optimizer_a, optimizer_b, lr_scheduler_a, lr_scheduler_b, save_path):
    """
    Checkpoint saver. Each save overwrites the previous one.
    :param epoch: epoch number
    :param model_a: transformer model A (handles DDP-wrapped models transparently)
    :param model_b: transformer model B (handles DDP-wrapped models transparently)
    :param optimizer_a: optimizer state for model A
    :param optimizer_b: optimizer state for model B
    :param lr_scheduler_a: learning rate scheduler state for model A
    :param lr_scheduler_b: learning rate scheduler state for model B
    """

    state = {
        'epoch'          : epoch,
        'model'          : model_a.module.state_dict() if hasattr(model_a, 'module') else model_a.state_dict(),
        'model_b'        : None if model_b is None else (model_b.module.state_dict() if hasattr(model_b, 'module') else model_b.state_dict()),
        'optimizer'      : optimizer_a.state_dict(),
        'optimizer_b'    : None if optimizer_b is None else optimizer_b.state_dict(),
        'lr_scheduler'   : lr_scheduler_a.state_dict() if lr_scheduler_a is not None else None,
        'lr_scheduler_b' : None if lr_scheduler_b is None else lr_scheduler_b.state_dict(),
    }

    torch.save(state, save_path)
    
"""
Metrics calculation
"""

def pck_joints(prediction, ground_truth, prediction_dict, joint_names, alpha, eps=1e-3):
    """
    pred: tensor of shape (num_joints, 2) or (num_joints, 3)
    gt: tensor of same shape
    predictions: dict containing counts for each joint and total
    joint_names: list of joint names
    eps: threshold to consider a joint as visible (ignore near-zero coords)
    """
    num_joints = prediction.shape[0]
    joint_dim  = prediction.shape[1]

    visible_ground_truth = (ground_truth.abs() > eps).all(dim=-1)  # (num_joints,)
    visible_prediction   = (prediction.abs()   > eps).all(dim=-1)  # (num_joints,)

    for joint_ind in range(num_joints):
        prediction_dict['total']['all'] += 1
        prediction_dict[joint_names[joint_ind]]['all'] += 1

        if not visible_ground_truth[joint_ind]:
            if not visible_prediction[joint_ind]:
                prediction_dict['total']['correct'] += 1
                prediction_dict[joint_names[joint_ind]]['correct'] += 1
        else:
            dist = np.linalg.norm(
                prediction[joint_ind, :joint_dim].cpu().detach().numpy() -
                ground_truth[joint_ind, :joint_dim].cpu().detach().numpy()
            )
            if dist < alpha:
                prediction_dict['total']['correct'] += 1
                prediction_dict[joint_names[joint_ind]]['correct'] += 1

    return prediction_dict
    


def compute_joints_metric(prediction_sequences, ground_truth_sequences, timesteps, alpha, dataset): 
    '''
    :param pred      : tensor of shape (B, T, J, D)
    :param gt        : tensor of shape (B, T, J, D)  
    :param threshold : size of the initial image 
    :param time_step : number of time steps  
    :return : 
       predictions = result tracker 
    '''

    predictions = DATASET_PCK_DICT[dataset]()
    joint_names = JOINT_NAMES[dataset]

    for batch_idx in range(ground_truth_sequences.shape[0]): 
        for time_idx in range(timesteps):
            curr_prediction, curr_ground_truth = prediction_sequences[batch_idx, time_idx, :, :], ground_truth_sequences[batch_idx, time_idx, :, :]  
            predictions = pck_joints(
                prediction      = curr_prediction, 
                ground_truth    = curr_ground_truth, 
                prediction_dict = predictions, 
                joint_names     = joint_names, 
                alpha           = alpha,
            )
    return predictions


def reconstruct_pose(input_sequences, mean, std, frame_input_h, frame_input_w, root_mean, root_std, joint_dim):
    """
    :param : 
       input_sequences : tensor of shape (batch_size, n_frames, J * joint_dim)
       mean : tensor of shape (J * joint_dim), mean for each keypoints coordinates 
       std  : tensor of shape (J * joint_dim), std  for each keypoints coordinates
       root_mean : tensor of shape (1. joint_dim) # mean for global position of the root 
       root_std  : tensor of shape (1, joint_dim) # std for root position of the root 
       frame_input_h : input frame height
       frame_input_w : input frame width 
       joint_dim : 2D or 3D ? 
    :return : 
       original_poses : tensor of shape (batch_size, n_frames, J * join_dim )
    """
    batch_size, time_step, _  = input_sequences.shape  
    original_poses         = torch.zeros_like(input_sequences)

    if joint_dim == 2: 
        original_poses[:, :, 0]  = (input_sequences[:, :, 0] + 1.0) / 2.0 * frame_input_w
        original_poses[:, :, 1]  = (input_sequences[:, :, 1] + 1.0) / 2.0 * frame_input_h
        original_poses[:, :, 2:] = input_sequences[:, :, joint_dim:] * std[joint_dim:] + mean[joint_dim:] 
    elif joint_dim == 3: 
        original_poses[:, :, :3] = input_sequences[:, :,:joint_dim] * root_std + root_mean
        original_poses[:, :, 3:] = input_sequences[:, :, joint_dim:] * std[joint_dim:] + mean[joint_dim:]
    else : 
        raise NotImplementedError('Only support 2D and 3D')

    original_poses = original_poses.contiguous().view(batch_size, time_step, -1, joint_dim)
    original_poses[:, :, 1:, :] = original_poses[:, :, 1:, :] + original_poses[:, :, 0, :].unsqueeze(2)
    original_poses = original_poses.contiguous().view(batch_size, time_step, -1)
    return original_poses


def get_acc(predictions):
    acc = {}
    for key in predictions :
        acc[key] = predictions[key]['correct'] / predictions[key]['all']
    return acc

def calc_mpjpe(predicted_poses, target_poses, num_joints, joint_dim):
    if len(predicted_poses.shape) == 3:
        predicted_poses = predicted_poses.contiguous().view(predicted_poses.shape[0], predicted_poses.shape[1], num_joints, -1) 
        target_poses    = target_poses.contiguous().view(target_poses.shape[0], target_poses.shape[1], num_joints, -1) 
    diff = predicted_poses[:, :, :, :joint_dim] - target_poses[:, :, :, :joint_dim]
    per_joint_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # (B, T, J)
    return per_joint_dist.mean().item()


def calc_mpjpe_per_timestep(predicted_poses, target_poses, num_joints, joint_dim):
    if len(predicted_poses.shape) == 3:
        predicted_poses = predicted_poses.contiguous().view(predicted_poses.shape[0], predicted_poses.shape[1], num_joints, -1) 
        target_poses    = target_poses.contiguous().view(target_poses.shape[0], target_poses.shape[1], num_joints, -1) 
    diff = predicted_poses[:, :, :, : joint_dim] - target_poses[:, :, :, : joint_dim]
    per_joint_dist = torch.sqrt(torch.sum(diff ** 2, dim = -1))  # (B, T, J)
    return per_joint_dist.mean(dim = (0, 2))  # (T,) — one value per future frame


def compute_dct_matrix(num):
    dct_m = np.eye(num)
    for k in range(num): 
        for i in range(num): 
            w = np.sqrt(2 / num) if k > 0 else np.sqrt(1 / num)
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / num)
    idct_m = np.linalg.inv(dct_m)   
    dct_m  = torch.from_numpy(dct_m.astype(np.float32))
    idct_m = torch.from_numpy(idct_m.astype(np.float32))
    return dct_m, idct_m
            
    