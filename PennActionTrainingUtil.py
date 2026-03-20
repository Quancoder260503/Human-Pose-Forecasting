import matplotlib.pyplot as plt
import numpy as np
from data.KeyPointUtil import get_pennaction_joint_names, get_pennaction_parents
import torch 
from model.LSTM import *

def get_pennaction_predictions_dict():
    return {
        'total'       :  { 'all' : 0, 'correct' : 0 },
        'headtop'     :  { 'all' : 0, 'correct' : 0 },
        'rshoulder'   :  { 'all' : 0, 'correct' : 0 },
        'lshoulder'   :  { 'all' : 0, 'correct' : 0 },
        'lelbow'      :  { 'all' : 0, 'correct' : 0 },
        'relbow'      :  { 'all' : 0, 'correct' : 0 },
        'lwrist'      :  { 'all' : 0, 'correct' : 0 },
        'rwrist'      :  { 'all' : 0, 'correct' : 0 },
        'lhip'        :  { 'all' : 0, 'correct' : 0 },
        'rhip'        :  { 'all' : 0, 'correct' : 0 },
        'lknee'       :  { 'all' : 0, 'correct' : 0 },
        'rknee'       :  { 'all' : 0, 'correct' : 0},
        'lankle'      :  { 'all' : 0, 'correct' : 0},
        'rankle'      :  { 'all' : 0, 'correct' : 0},
    }


def pck(pred, gt, predictions, joint_names):
    """
     An estimation is considered correct if it lies within \alpha * max(h, w) from the true position, H_map, W_map is the height and width of the heat map 
     :param pred: tensor of shape (J, H_map, W_map) 
     :param gt: tensor of shape (J, H_map, W_map) 
     :param threshold: threshold to consider the boundary 
     :param predictions: previous average meters of joints 
     :return: predictions : next average meters of joints 
    """
    alpha = 0.05
    for i in range(pred.shape[0] - 1):  # all joints
        predictions['total']['all'] += 1
        predictions[joint_names[i]]['all'] += 1

        # Skip joints not visible
        if np.max(gt[i]) == 0:
            if np.max(pred[i]) == 0:
                predictions['total']['correct'] += 1
                predictions[joint_names[i]]['correct'] += 1
        else:
            pred_x, pred_y = np.where(pred[i] == np.max(pred[i]))
            pred_x, pred_y = pred_x[0], pred_y[0]

            gt_x, gt_y = np.where(gt[i] == np.max(gt[i]))
            gt_x, gt_y = gt_x[0], gt_y[0]

            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if dist < alpha :  # threshold in pixels
                predictions['total']['correct'] += 1
                predictions[joint_names[i]]['correct'] += 1

    return predictions


def pck_joints(pred, gt, predictions, joint_names, eps=1e-3):
    """
    Compute PCK (Percentage of Correct Keypoints) for visible joints only.
    pred: tensor of shape (num_joints, 2) or (num_joints, 3)
    gt: tensor of same shape
    predictions: dict containing counts for each joint and total
    joint_names: list of joint names
    eps: threshold to consider a joint as visible (ignore near-zero coords)
    """
    alpha = 0.05  # fraction of reference length

    # visibility mask: gt has non-zero coordinates
    visible = (gt[:, 0].abs() > eps) & (gt[:, 1].abs() > eps)

    for i in range(pred.shape[0]):
        if not visible[i]:
            continue  # skip invisible joints

        predictions['total']['all'] += 1
        predictions[joint_names[i]]['all'] += 1

        dist = np.linalg.norm(pred[i][:2].cpu().detach().numpy() - gt[i][:2].cpu().detach().numpy())
        if dist < alpha:
            predictions['total']['correct'] += 1
            predictions[joint_names[i]]['correct'] += 1

    return predictions

def compute_metric(pred_maps, gt_maps, time_step):
    '''
    :param pred_maps: tensor of shape (B, T, J, H_map, W_map)
    :param gt_maps: tensor of shape (B, T, J, H_map, W_map)
    :param threshold: size of the heatmap
    :param time_step: number of time steps
    :param joint_names : joint names of the dataset 
    :return:
      predictions : results tracker
    '''
    predictions = get_pennaction_predictions_dict()
    joint_names = get_pennaction_joint_names()
    for b in range(gt_maps.shape[0]):
        for t in range(time_step):
            gt = gt_maps[b, t, :, :, :]
            pred = np.array(pred_maps[b, t, :, :, :].cpu().data)
            predictions = pck(pred, gt, predictions, joint_names)
    return predictions



def compute_joints_metric(pred, gt, time_step): 
    '''
    :param pred      : tensor of shape (B, T, J, 2)
    :param gt        : tensor of shape (B, T, J, 2)  
    :param threshold : size of the initial image 
    :param time_step : number of time steps  
    :return : 
       predictions = result tracker 
    '''

    predictions = get_pennaction_predictions_dict()
    joint_names = get_pennaction_joint_names()
    for b in range(gt.shape[0]): 
        for t in range(time_step):
            curr_pred = pred[b, t, :, :] # (J, 2)
            curr_gt   = gt[b, t, :, :]   # (J, 2)
            predictions = pck_joints(curr_pred, curr_gt, predictions, joint_names)
    return predictions

""" 
Only for bone vectors 
"""
def reconstruct_pose(compressed_poses, scaled_factor, input_h = 480, input_w = 480, eps = 1e-2):
        """
        :param compressed_poses : a tensor of shape (B, T, 2 * J) where B = batch, T = time_step, J = num_joints
        :param scaled_factor : a tensor of shapa (B, ) needed to reconstruct the pose 
        :return : original poses (B, T, J, 2)
        """ 
        batch, time_step, _  = compressed_poses.shape 
        compressed_poses = compressed_poses.contiguous().view(batch, time_step, -1, 2) # (B, T, J, 2)
        original_poses = torch.zeros_like(compressed_poses)
        pennaction_parents = get_pennaction_parents()
        for b in range(batch): 
            for t in range(time_step):
                curr_vectors = compressed_poses[b, t, :, :] 
                original_poses[b, t, 0, 0] = curr_vectors[0, 0] * input_w
                original_poses[b, t, 0, 1] = curr_vectors[0, 1] * input_h
                for joint, parent in enumerate(pennaction_parents[1:], start = 1):
                        if abs(curr_vectors[parent, 0]) > eps and abs(curr_vectors[parent, 1]) > eps: 
                            offset = curr_vectors[joint, :] * scaled_factor[b]
                            original_poses[b, t, joint, :] = original_poses[b, t, parent, :] - offset 
            
        return original_poses / max(input_h, input_w)

def get_acc(predictions):
    acc = {}
    for key in predictions :
        acc[key] = predictions[key]['correct'] / predictions[key]['all']
    return acc

def plot_loss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label = "Train Loss")
    ax.plot(loss_list[:, 1], label = "Validation Loss")
    ax.set_title("Loss Curves - " + title, fontsize = 12)
    ax.set_xlabel("Epoch", fontsize = 10)
    ax.set_ylabel("Loss", fontsize = 10)
    ax.legend(prop = {'size' : 10})

def plot_metric(ax1, ax2, metric_list, title):
    # Prepare metrics
    head_acc  = [[x[0]['headtop'], x[1]['headtop']] for x in metric_list] 
    sho_acc   = [[(x[0]['lshoulder'] + x[0]['rshoulder']) / 2.0, 
                  (x[1]['lshoulder'] + x[1]['rshoulder']) / 2.0] for x in metric_list]
    elb_acc   = [[(x[0]['lelbow'] + x[0]['relbow']) / 2.0, 
                  (x[1]['lelbow'] + x[1]['relbow']) / 2.0] for x in metric_list]
    wri_acc   = [[(x[0]['lwrist'] + x[0]['rwrist']) / 2.0, 
                  (x[1]['lwrist'] + x[1]['rwrist']) / 2.0] for x in metric_list]
    hip_acc   = [[(x[0]['lhip'] + x[0]['rhip']) / 2.0, 
                  (x[1]['lhip'] + x[1]['rhip']) / 2.0] for x in metric_list]
    knee_acc  = [[(x[0]['lknee'] + x[0]['rknee']) / 2.0, 
                  (x[1]['lknee'] + x[1]['rknee']) / 2.0] for x in metric_list]
    ank_acc   = [[(x[0]['lankle'] + x[0]['rankle']) / 2.0, 
                  (x[1]['lankle'] + x[1]['rankle']) / 2.0] for x in metric_list]
    total_acc = [[x[0]['total'], x[1]['total']] for x in metric_list]

    # Training set plots
    ax1.plot([x[0] for x in total_acc], label="Total")
    ax1.plot([x[0] for x in head_acc], label="Head")
    ax1.plot([x[0] for x in sho_acc], label="Shoulders")
    ax1.plot([x[0] for x in elb_acc], label="Elbows")
    ax1.plot([x[0] for x in wri_acc], label="Wrists")
    ax1.plot([x[0] for x in hip_acc], label="Hips")
    ax1.plot([x[0] for x in knee_acc], label="Knees")
    ax1.plot([x[0] for x in ank_acc], label="Ankles")
    ax1.set_title("PCK accuracy on Training Set - " + title, fontsize=12)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.legend(prop={'size': 10})

    # Validation set plots
    ax2.plot([x[1] for x in total_acc], label="Total")
    ax2.plot([x[1] for x in head_acc], label="Head")
    ax2.plot([x[1] for x in sho_acc], label="Shoulders")
    ax2.plot([x[1] for x in elb_acc], label="Elbows")
    ax2.plot([x[1] for x in wri_acc], label="Wrists")
    ax2.plot([x[1] for x in hip_acc], label="Hips")
    ax2.plot([x[1] for x in knee_acc], label="Knees")
    ax2.plot([x[1] for x in ank_acc], label="Ankles")
    ax2.set_title("PCK accuracy on Validation Set - " + title, fontsize=12)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.legend(prop={'size': 10})

def plot(loss_list, metric_list, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 8))
    plot_loss(ax1, loss_list, title)
    plot_metric(ax2, ax3, metric_list, title)
    plt.show()

def save_checkpoint(epoch, model, optimizer, model_name, prefix_dir_name):
    """
    Checkpoint saver. Each save overwrites the previous one
    :param epoch: epoch number
    :param model: transformer model weight
    :param optimizer: optimizer state
    :param model_name : model version 
    :param prefix_dir_name: check point filename prefix
    """
    state = {
        'epoch' : epoch,
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(), 
    }
    filename = prefix_dir_name + f'{model_name}.pth.tar'
    torch.save(state, filename)
