import matplotlib.pyplot as plt
import numpy as np
import torch 
import math 
from data.DatasetUtil         import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from runner.CoordinatesRunner  import * 
from runner.RootRelativeRunner import *
from runner.VelocityRunner     import * 
from runner.Losses             import *

"""
MODEL UTILS 
"""

PCK_ALPHA = 0.05

def save_checkpoint(epoch, model, optimizer, scheduler, save_path):
    """
    Checkpoint saver. Each save overwrites the previous one.
    :param epoch: epoch number
    :param model: transformer model (handles DDP-wrapped models transparently)
    :param optimizer: optimizer state
    :param scheduler: lr scheduler state (pass None if not used)
    """

    state = {
        'epoch'     : epoch,
        'model'     : model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict() if scheduler is not None else None,
    }

    torch.save(state, save_path)
    
"""
Metrics calculation
"""

def pck_joints(pred, gt, predictions, joint_names, alpha, eps = 1e-3, prob_thres = 0.5):
    """
    Compute PCK (Percentage of Correct Keypoints) for visible joints only.
    pred: tensor of shape (num_joints, 2) or (num_joints, 3)
    gt: tensor of same shape
    predictions: dict containing counts for each joint and total
    joint_names: list of joint names
    eps: threshold to consider a joint as visible (ignore near-zero coords)
    prob_thres : in the case also including visibility, use prob_thres to make decision whether a joints appears or not 
    """
    last_dim = pred.shape[-1] 
    num_joints = pred.shape[0]
    if last_dim == 2: 
        visible_gt   = (gt[:, 0].abs()   > eps) & (gt[:, 1].abs()   > eps)
        visible_pred = (pred[:, 0].abs() > eps) & (pred[:, 1].abs() > eps)
        for joint_ind in range(num_joints):
            predictions['total']['all'] += 1
            predictions[joint_names[joint_ind]]['all'] += 1

            if not visible_gt[joint_ind]:
                if not visible_pred[joint_ind]: 
                    predictions['total']['correct'] += 1 
                    predictions[joint_names[joint_ind]]['correct'] += 1 
                continue 

            dist = np.linalg.norm(pred[joint_ind][:2].cpu().detach().numpy() - gt[joint_ind][:2].cpu().detach().numpy())
            if dist < alpha:
               predictions['total']['correct'] += 1
               predictions[joint_names[joint_ind]]['correct'] += 1

        return predictions
    elif last_dim == 3: 
        for joint_ind in range(num_joints): 
            predictions['total']['all'] += 1
            predictions[joint_names[joint_ind]]['all'] += 1
            
            if gt[joint_ind, 2] == 0: 
                if pred[joint_ind, 2] < prob_thres : 
                    predictions['total']['correct'] += 1 
                    predictions[joint_names[joint_ind]]['correct'] += 1 
            else : 
                dist = np.linalg.norm(pred[joint_ind][:2].cpu().detach().numpy() - gt[joint_ind][:2].cpu().detach().numpy())
                if dist < alpha:
                  predictions['total']['correct'] += 1
                  predictions[joint_names[joint_ind]]['correct'] += 1
        return predictions
    else : 
        raise NotImplementedError("Method not supported for now")
    


def compute_joints_metric(pred, gt, time_step, alpha, dataset): 
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

    for b in range(gt.shape[0]): 
        for t in range(time_step):
            curr_pred = pred[b, t, :, :]  
            curr_gt   = gt[b, t,   :, :]  
            predictions = pck_joints(
                pred        = curr_pred, 
                gt          = curr_gt, 
                predictions = predictions, 
                joint_names = joint_names, 
                alpha       = alpha
            )
    return predictions


def reconstruct_pose_from_root(root_vectors, mean, std, frame_input_h, frame_input_w):
    """
    :param : 
       root_vectors : tensor of shape (batch_size, n_frames, J * 2)
       mean : tensor of shape ((J - 1) * 2) 
       std  : tensor of shape ((J - 1) * 2)
       frame_input_h : input frame height
       frame_input_w : input frame width 
    :return : 
       original_poses : tensor of shape (batch_size, n_frames, J * 2)
    """
    batch_size, time_step, _    = root_vectors.shape  
    original_poses         = torch.zeros_like(root_vectors)
    
    original_poses[:, :, 0] = root_vectors[:, :, 0] * frame_input_w 
    original_poses[:, :, 1] = root_vectors[:, :, 1] * frame_input_h

    original_poses[:, :, 2:] = root_vectors[:, :, 2:] * std + mean 

    original_poses = original_poses.contiguous().view(batch_size, time_step, -1, 2)
    original_poses[:, :, 1:, :] = original_poses[:, :, 1:, :] + original_poses[:, :, 0, :].unsqueeze(2)
    original_poses = original_poses.contiguous().view(batch_size, time_step, -1)
    return original_poses


def speed_to_pos(prediction_velocities, observed_poses, device = 'cuda'):
    batch_size, pose_len, pose_dim = prediction_velocities.shape
    predicted_poses = torch.zeros(batch_size, pose_len, pose_dim).to(device) 
    current = observed_poses[:, -1, :].clone().detach()
    for t in range(pose_len):
        predicted_poses[:, t, :] = current + prediction_velocities[:, t, :]
        current = predicted_poses[:, t, :].clone().detach()
    return predicted_poses

def get_acc(predictions):
    acc = {}
    for key in predictions :
        acc[key] = predictions[key]['correct'] / predictions[key]['all']
    return acc

def calc_rmse(predicted_poses, target_poses, num_joints):
    if len(predicted_poses.shape) == 3:
        predicted_poses = predicted_poses.contiguous().view(predicted_poses.shape[0], predicted_poses.shape[1], num_joints, -1) 
        target_poses    = target_poses.contiguous().view(target_poses.shape[0], target_poses.shape[1], num_joints, -1) 

    diff = predicted_poses[:, :, :, :2] - target_poses[:, :, :, :2] 
    per_joint_rmse = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim = -1), dim = (0, 1)))
    return per_joint_rmse.mean().item()



def KL_annealing(epoch, total_epochs, schedule_type = 'linear', start_beta = 0.0, end_beta = 1.0):
    progress = epoch / total_epochs
    if schedule_type == "linear":
        return start_beta + (end_beta - start_beta) * progress
    elif schedule_type == "cyclical":
        return end_beta * (1 + math.cos(math.pi * (1 - progress))) / 2
    elif schedule_type == "sigmoid":
        return start_beta + (end_beta - start_beta) / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_runner(args):
    if   args.model_input_type == 'coordinates' : 
        return CoordinatesRunner
    elif args.model_input_type == 'velocity' : 
        return VelocityRunner 
    elif args.model_input_type == 'root_relative' :
        return RootRelativeRunner 
    else : 
        raise NotImplementedError('representation is not supported for dataset')