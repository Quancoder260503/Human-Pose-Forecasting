import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil         import *
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split


"""
MODEL UTILS 
"""

PCK_ALPHA = 0.05


def save_checkpoint(epoch, model, optimizer, model_name, prefix_dir_name):
    """
    Checkpoint saver. Each save overwrites the previous one.
    :param epoch: epoch number
    :param model: transformer model weight
    :param optimizer: optimizer state
    :param model_name : model version 
    :param prefix_dir_name: check point filename prefix
    """
    # FIX 1: Save model.module.state_dict() when wrapped in DDP, so inference
    # can load without needing DDP (avoids 'module.' prefix mismatch)
    state = {
        'epoch'     : epoch,
        'model'     : model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer' : optimizer.state_dict(), 
    }
    filename = prefix_dir_name + f'{model_name}.pth.tar'
    torch.save(state, filename)

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
        visible_gt   = (gt[:, 0].abs() > eps)   & (gt[:, 1].abs() > eps)
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


def reconstruct_pose_from_root(root_vectors, mean, std, dataset, frame_input_h, frame_input_w, eps = 1e-2):
    batch, time_step, _    = root_vectors.shape 
    root_vectors           = root_vectors.contiguous().view(batch, time_step, -1, 2) # (B, T, J, 2)
    original_poses         = torch.zeros_like(root_vectors)
    joint_ids              = JOINT_INDS[dataset]()
    root_position          = 0

    mean_x, mean_y = mean 
    std_x,  std_y  = std 

    for b in range(batch): 
        for t in range(time_step):
            curr_vectors = root_vectors[b, t, :, :] 
            
            original_poses[b, t, root_position, 0] = curr_vectors[root_position, 0] * frame_input_w
            original_poses[b, t, root_position, 1] = curr_vectors[root_position, 1] * frame_input_h
            
            for child_index in range(len(joint_ids)): 
                if child_index == root_position : 
                    continue 
                if abs(curr_vectors[child_index, 0]) > eps and abs(curr_vectors[child_index, 1]) > eps: 
                    curr_vectors[child_index, 0] = curr_vectors[child_index, 0] * std_x + mean_x 
                    curr_vectors[child_index, 1] = curr_vectors[child_index, 1] * std_y + mean_y 
                    original_poses[b, t, child_index, :] = original_poses[b, t, root_position, :] + curr_vectors
        
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

def calc_rmse(predicted_poses, target_poses):
    diff = predicted_poses[:, :, :, :2] - target_poses[:, :, :, :2] 
    per_joint_rmse = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim = -1), dim = (0, 1)))
    return per_joint_rmse.mean().item()


