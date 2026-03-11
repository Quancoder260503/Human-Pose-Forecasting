import matplotlib.pyplot as plt
import numpy as np 
from data.NTUmeta import * 
import torch 

"""
So far only for upenn, my change to adapt with other dataset after. 
"""
def get_ntu_predictions_dict():
    return {
     'total'         : { 'all' : 0, 'correct' : 0}, 
     'SpineBase'     : { 'all' : 0, 'correct' : 0 }, 
     'SpineMid'      : { 'all' : 0, 'correct' : 0 }, 
     'Neck'          : { 'all' : 0, 'correct' : 0 }, 
     'Head'          : { 'all' : 0, 'correct' : 0 }, 
     'ShoulderLeft'  : { 'all' : 0, 'correct' : 0 }, 
     'ShoulderLeft'  : { 'all' : 0, 'correct' : 0 }, 
     'ElbowLeft'     : { 'all' : 0, 'correct' : 0 },
     'WristLeft'     : { 'all' : 0, 'correct' : 0 }, 
     'HandLeft'      : { 'all' : 0, 'correct' : 0 }, 
     'ShoulderRight' : { 'all' : 0, 'correct' : 0 }, 
     'ElbowRight'    : { 'all' : 0, 'correct' : 0 }, 
     'WristRight'    : { 'all' : 0, 'correct' : 0 },
     'HandRight'     : { 'all' : 0, 'correct' : 0 }, 
     'HipLeft'       : { 'all' : 0, 'correct' : 0 }, 
     'KneeLeft'      : { 'all' : 0, 'correct' : 0 }, 
     'AnkleLeft'     : { 'all' : 0, 'correct' : 0 }, 
     'FootLeft'      : { 'all' : 0, 'correct' : 0 }, 
     'HipRight'      : { 'all' : 0, 'correct' : 0 },
     'KneeRight'     : { 'all' : 0, 'correct' : 0 }, 
     'AnkleRight'    : { 'all' : 0, 'correct' : 0 }, 
     'FootRight'     : { 'all' : 0, 'correct' : 0 },
     'FootRight'     : { 'all' : 0, 'correct' : 0 },
     'SpineShoulder' : { 'all' : 0, 'correct' : 0 }, 
     'HandTipLeft'   : { 'all' : 0, 'correct' : 0 },
     'ThumbLeft'     : { 'all' : 0, 'correct' : 0 }, 
     'HandTipRight'  : { 'all' : 0, 'correct' : 0 }, 
     'ThumbRight'    : { 'all' : 0, 'correct' : 0 }
    }


def pck(pred, gt, predictions):
    """
     An estimation is considered correct if it lies within \alpha * max(h, w) from the true position, H_map, W_map is the height and width of the heat map 
     :param pred: tensor of shape (J, H_map, W_map) 
     :param gt: tensor of shape (J, H_map, W_map) 
     :param predictions: previous average meters of joints 
     :return: predictions : next average meters of joints 
    """
    alpha = 0.01
    joint_names = JOINT_NAMES
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


def pck_joints(pred, gt, predictions): 
    """
    Compute PCK (Percentage of Correct Keypoints) for each joint and overall.
    
    pred: tensor of shape (num_joints, 2) or (num_joints, 3)
    gt: tensor of same shape
    predictions: dict containing counts for each joint and total
    joint_names: list of joint names
    """
    alpha = 0.01  # fraction of threshold
    joint_names = JOINT_NAMES
    for i in range(pred.shape[0]): 
        predictions['total']['all'] += 1 
        predictions[joint_names[i]]['all'] += 1 
    
        dist = np.linalg.norm(pred[i].cpu().detach().numpy() - gt[i].cpu().detach().numpy())
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
    :return:
      predictions : results tracker
    '''
    predictions = get_ntu_predictions_dict()
    for b in range(gt_maps.shape[0]):
        for t in range(time_step):
            gt = gt_maps[b, t, :, :, :]
            pred = np.array(pred_maps[b, t, :, :, :].cpu().data)
            predictions = pck(pred, gt, predictions)
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

    predictions = get_ntu_predictions_dict()
    for b in range(gt.shape[0]): 
        for t in range(time_step):
            curr_pred = pred[b, t, :, :] # (J, 2)
            curr_gt   = gt[b, t, :, :]   # (J, 2)
            predictions = pck_joints(curr_pred, curr_gt, predictions)
    return predictions

def get_acc(predictions):
    acc = {}
    for key in predictions :
        acc[key] = predictions[key]['correct'] / predictions[key]['all']
    return acc

# TODO : adapt with NTU 

# def plot_loss(ax, loss_list, title):
#     ax.plot(loss_list[:, 0], label = "Train Loss")
#     ax.plot(loss_list[:, 1], label = "Validation Loss")
#     ax.set_title("Loss Curves - " + title, fontsize = 12)
#     ax.set_xlabel("Epoch", fontsize = 10)
#     ax.set_ylabel("Loss", fontsize = 10)
#     ax.legend(prop = {'size' : 10})

# def plot_metric(ax1, ax2, metric_list, title):
#     # Prepare metrics
#     head_acc  = [[x[0]['headtop'], x[1]['headtop']] for x in metric_list] 
#     sho_acc   = [[(x[0]['lshoulder'] + x[0]['rshoulder']) / 2.0, 
#                   (x[1]['lshoulder'] + x[1]['rshoulder']) / 2.0] for x in metric_list]
#     elb_acc   = [[(x[0]['lelbow'] + x[0]['relbow']) / 2.0, 
#                   (x[1]['lelbow'] + x[1]['relbow']) / 2.0] for x in metric_list]
#     wri_acc   = [[(x[0]['lwrist'] + x[0]['rwrist']) / 2.0, 
#                   (x[1]['lwrist'] + x[1]['rwrist']) / 2.0] for x in metric_list]
#     hip_acc   = [[(x[0]['lhip'] + x[0]['rhip']) / 2.0, 
#                   (x[1]['lhip'] + x[1]['rhip']) / 2.0] for x in metric_list]
#     knee_acc  = [[(x[0]['lknee'] + x[0]['rknee']) / 2.0, 
#                   (x[1]['lknee'] + x[1]['rknee']) / 2.0] for x in metric_list]
#     ank_acc   = [[(x[0]['lankle'] + x[0]['rankle']) / 2.0, 
#                   (x[1]['lankle'] + x[1]['rankle']) / 2.0] for x in metric_list]
#     total_acc = [[x[0]['total'], x[1]['total']] for x in metric_list]

#     # Training set plots
#     ax1.plot([x[0] for x in total_acc], label="Total")
#     ax1.plot([x[0] for x in head_acc], label="Head")
#     ax1.plot([x[0] for x in sho_acc], label="Shoulders")
#     ax1.plot([x[0] for x in elb_acc], label="Elbows")
#     ax1.plot([x[0] for x in wri_acc], label="Wrists")
#     ax1.plot([x[0] for x in hip_acc], label="Hips")
#     ax1.plot([x[0] for x in knee_acc], label="Knees")
#     ax1.plot([x[0] for x in ank_acc], label="Ankles")
#     ax1.set_title("PCK accuracy on Training Set - " + title, fontsize=12)
#     ax1.set_ylabel("Score", fontsize=10)
#     ax1.set_xlabel("Epoch", fontsize=10)
#     ax1.legend(prop={'size': 10})

#     # Validation set plots
#     ax2.plot([x[1] for x in total_acc], label="Total")
#     ax2.plot([x[1] for x in head_acc], label="Head")
#     ax2.plot([x[1] for x in sho_acc], label="Shoulders")
#     ax2.plot([x[1] for x in elb_acc], label="Elbows")
#     ax2.plot([x[1] for x in wri_acc], label="Wrists")
#     ax2.plot([x[1] for x in hip_acc], label="Hips")
#     ax2.plot([x[1] for x in knee_acc], label="Knees")
#     ax2.plot([x[1] for x in ank_acc], label="Ankles")
#     ax2.set_title("PCK accuracy on Validation Set - " + title, fontsize=12)
#     ax2.set_ylabel("Score", fontsize=10)
#     ax2.set_xlabel("Epoch", fontsize=10)
#     ax2.legend(prop={'size': 10})

# def plot(loss_list, metric_list, title):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 8))
#     plot_loss(ax1, loss_list, title)
#     plot_metric(ax2, ax3, metric_list, title)
#     plt.show()

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
