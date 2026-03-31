import matplotlib.pyplot as plt
import numpy as np
from data.KeyPointUtil import get_pennaction_joint_names, get_pennaction_parents
from data.NTUmeta import *
from model.Lstm               import *
from model.Mlp                import *
from model.LstmMotionVae      import * 
import torch 

"""
1. TODO implement plot function for predictions accuracy when frames goes on (TO THINK) 
"""

def get_model(args):
    if args.model == 'mlp': 
        model = get_mlp_model(
            pose_dim      = args.num_joints, 
            coordinates   = args.coordinates, 
            train_len     = args.num_training_frame - (1 if args.velocity else 0), 
            predict_len   = args.num_testing_frame, 
            hidden_layers = args.mlp_hidden_layers, 
            dropout       = args.dropout, 
            device        = args.device, 
        )
        criterion = mlp_loss_function
        return model, criterion 
    elif args.model == 'lstm': 
        model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.coordinates, 
            embedding_dim  = args.lstm_embedding_dim,
            hidden_dim     = args.lstm_hidden_dim, 
            num_layers     = args.lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
            residual_connection = args.lstm_residual_connection,
        )
        criterion = lstm_loss_function 
        return model, criterion 
    
    elif args.model == 'motion_vae': 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.motion_vae_hidden_dim, 
            latent_dim      = args.motion_vae_latent_dim,
            encoding_layers = args.motion_vae_encoding_layers, 
            decoding_layers = args.motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        return vae_model, criterion_vae


    elif args.model == 'decouple_motion_vae': 
        # Motion Vae Comprises of two models, one for the root, and one for others 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.decouple_motion_vae_hidden_dim, 
            latent_dim      = args.decouple_motion_vae_latent_dim,
            encoding_layers = args.decouple_motion_vae_encoding_layers, 
            decoding_layers = args.decouple_motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        lstm_model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.coordinates, 
            embedding_dim  = args.decouple_lstm_embedding_dim,
            hidden_dim     = args.decouple_lstm_hidden_dim, 
            num_layers     = args.decouple_lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
        )

        criterion_lstm = lstm_loss_function

        return (vae_model, lstm_model), (criterion_vae, criterion_lstm)
    else : 
        raise NotImplementedError('Model not supported for now') 


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

def get_ntu_predictions_dict():
    return {
     'total'         : { 'all' : 0, 'correct' : 0}, 
     'SpineBase'     : { 'all' : 0, 'correct' : 0 }, 
     'SpineMid'      : { 'all' : 0, 'correct' : 0 }, 
     'Neck'          : { 'all' : 0, 'correct' : 0 }, 
     'Head'          : { 'all' : 0, 'correct' : 0 }, 
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
     'SpineShoulder' : { 'all' : 0, 'correct' : 0 }, 
     'HandTipLeft'   : { 'all' : 0, 'correct' : 0 },
     'ThumbLeft'     : { 'all' : 0, 'correct' : 0 }, 
     'HandTipRight'  : { 'all' : 0, 'correct' : 0 }, 
     'ThumbRight'    : { 'all' : 0, 'correct' : 0 }
    }


DATASET_NAME = { 
    'pennaction' : get_pennaction_joint_names(), 
    'ntu'        : get_ntu_joint_names(), 
}

DATASET_PARENT = { 
    'pennaction' : get_pennaction_parents(), 
    'ntu'        : get_ntu_parent(), 
}

DATASET_HEAD_POS = { 
    'pennaction' : 0, # Head Top is 0 for Penn Action 
    'ntu'        : 3, # Head position for NTU 
}

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

    # FIX 2: Use fresh dict each call instead of shared global (was accumulating across calls)
    predictions = get_ntu_predictions_dict() if dataset == 'ntu' else get_pennaction_predictions_dict()
    joint_names = DATASET_NAME[dataset]

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


def reconstruct_pose(compressed_poses, scaled_factor, dataset, box_size = 480, eps = 1e-2):
    
    """
    :param compressed_poses : a tensor of shape (B, T, J * 2) 
    :param scaled_factor : a tensor of shapa (B, ) needed to reconstruct the pose 
    :return : original poses (B, T, J, 2)
    """ 

    batch, time_step, _  = compressed_poses.shape 
    compressed_poses     = compressed_poses.contiguous().view(batch, time_step, -1, 2) # (B, T, J, 2)
    original_poses       = torch.zeros_like(compressed_poses)
    parents              = DATASET_PARENT[dataset]
    head_position        = DATASET_HEAD_POS[dataset]
    for b in range(batch): 
        for t in range(time_step):
            curr_vectors = compressed_poses[b, t, :, :] 
            original_poses[b, t, head_position, 0] = curr_vectors[head_position, 0] * box_size
            original_poses[b, t, head_position, 1] = curr_vectors[head_position, 1] * box_size
            for joint, parent in enumerate(parents):
                if joint == parent : 
                    continue
                if abs(curr_vectors[parent, 0]) > eps and abs(curr_vectors[parent, 1]) > eps: 
                    offset = curr_vectors[joint, :] * scaled_factor[b]
                    original_poses[b, t, joint, :] = original_poses[b, t, parent, :] - offset 
    
    return original_poses / box_size

def reconstruct_pose_from_root(compressed_poses, scaled_factor, box_size = 480, eps = 1e-2):
    """
    This is only used for NTU 
    :param compressed_poses : a tensor of shape (B, T, J * 2) where B = batch, T = time_step, J = num_joints
    :param scaled_factor : a tensor of shapa (B, ) needed to reconstruct the pose 
    :return : original poses (B, T, J, 2)
    """ 
    batch, time_step, _  = compressed_poses.shape 
    compressed_poses     = compressed_poses.contiguous().view(batch, time_step, -1, 2) # (B, T, J, 2)
    original_poses       = torch.zeros_like(compressed_poses)
    ntu_joint_ids        = get_ntu_joint_ids()
    root_position        = ntu_joint_ids['SpineBase']

    for b in range(batch): 
        for t in range(time_step):
            curr_vectors = compressed_poses[b, t, :, :] 
            original_poses[b, t, root_position, 0] = curr_vectors[root_position, 0] * box_size
            original_poses[b, t, root_position, 1] = curr_vectors[root_position, 1] * box_size
            
            for child_index in range(len(ntu_joint_ids)): 
                if child_index == root_position : 
                    continue 
                if abs(curr_vectors[child_index, 0]) > eps and abs(curr_vectors[child_index, 1]) > eps: 
                    offset = curr_vectors[child_index, :] * scaled_factor[b]
                    original_poses[b, t, child_index, :] = original_poses[b, t, root_position, :] + offset 
        
    return original_poses / box_size


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

'''
PENN ACTION PLOT 
'''


def plot_metric_pennaction(ax1, ax2, metric_list, title, plot_training = True):
    if plot_training : 
        head_acc  = [[x[0]['headtop'], x[1]['headtop']] for x in metric_list] 
        sho_acc   = [[(x[0]['lshoulder'] + x[0]['rshoulder']) / 2.0, 
                      (x[1]['lshoulder'] + x[1]['rshoulder']) / 2.0] for x in metric_list]
        elb_acc   = [[(x[0]['lelbow'] + x[0]['relbow']) / 2.0, 
                      (x[1]['lelbow'] + x[1]['relbow']) / 2.0]       for x in metric_list]
        wri_acc   = [[(x[0]['lwrist'] + x[0]['rwrist']) / 2.0, 
                      (x[1]['lwrist'] + x[1]['rwrist']) / 2.0]       for x in metric_list]
        hip_acc   = [[(x[0]['lhip'] + x[0]['rhip']) / 2.0, 
                      (x[1]['lhip'] + x[1]['rhip']) / 2.0]           for x in metric_list]
        knee_acc  = [[(x[0]['lknee'] + x[0]['rknee']) / 2.0, 
                      (x[1]['lknee'] + x[1]['rknee']) / 2.0]         for x in metric_list]
        ank_acc   = [[(x[0]['lankle'] + x[0]['rankle']) / 2.0, 
                      (x[1]['lankle'] + x[1]['rankle']) / 2.0]       for x in metric_list]
        
        total_acc = [[x[0]['total'], x[1]['total']] for x in metric_list]
        
        ax1.plot([x[0] for x in total_acc], label="Total")
        ax1.plot([x[0] for x in head_acc],  label="Head")
        ax1.plot([x[0] for x in sho_acc],   label="Shoulders")
        ax1.plot([x[0] for x in elb_acc],   label="Elbows")
        ax1.plot([x[0] for x in wri_acc],   label="Wrists")
        ax1.plot([x[0] for x in hip_acc],   label="Hips")
        ax1.plot([x[0] for x in knee_acc],  label="Knees")
        ax1.plot([x[0] for x in ank_acc],   label="Ankles")
        ax1.set_title("PCK accuracy on Training Set - " + title, fontsize=12)
        ax1.set_ylabel("Score", fontsize = 10)
        ax1.set_xlabel("Epoch", fontsize = 10)
        ax1.legend(prop={'size': 10})
    

        ax2.plot([x[1] for x in total_acc], label="Total")
        ax2.plot([x[1] for x in head_acc],  label="Head")
        ax2.plot([x[1] for x in sho_acc],   label="Shoulders")
        ax2.plot([x[1] for x in elb_acc],   label="Elbows")
        ax2.plot([x[1] for x in wri_acc],   label="Wrists")
        ax2.plot([x[1] for x in hip_acc],   label="Hips")
        ax2.plot([x[1] for x in knee_acc],  label="Knees")
        ax2.plot([x[1] for x in ank_acc],   label="Ankles")
        ax2.set_title(f"PCK accuracy on Validation Set - " + title, fontsize=12)
        ax2.set_ylabel("Score", fontsize=10)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.legend(prop={'size': 10})
    else : 
        head_acc  = [x['headtop'] for x in metric_list] 
        sho_acc   = [(x['lshoulder'] + x['rshoulder']) / 2.0 for x in metric_list]
        elb_acc   = [(x['lelbow']    + x['relbow'])    / 2.0 for x in metric_list]
        wri_acc   = [(x['lwrist']    + x['rwrist'])    / 2.0 for x in metric_list]
        hip_acc   = [(x['lhip']      + x['rhip'])      / 2.0 for x in metric_list]
        knee_acc  = [(x['lknee']     + x['rknee'])     / 2.0 for x in metric_list]
        ank_acc   = [(x['lankle']    + x['rankle'])    / 2.0 for x in metric_list]
        total_acc = [x['total'] for x in metric_list]
        
        ax2.plot([x for x in total_acc], label="Total")
        ax2.plot([x for x in head_acc],  label="Head")
        ax2.plot([x for x in sho_acc],   label="Shoulders")
        ax2.plot([x for x in elb_acc],   label="Elbows")
        ax2.plot([x for x in wri_acc],   label="Wrists")
        ax2.plot([x for x in hip_acc],   label="Hips")
        ax2.plot([x for x in knee_acc],  label="Knees")
        ax2.plot([x for x in ank_acc],   label="Ankles")
        ax2.set_title("PCK accuracy on Inference Time - " + title, fontsize=12)
        ax2.set_ylabel("Score",        fontsize = 10)
        ax2.set_xlabel("Time horizon (Num Frames)", fontsize = 10)
        ax2.legend(prop={'size': 10})

'''
NTU PLOT
'''
def plot_metric_ntu(ax1, ax2, metric_list, title, plot_training = True): 
    if plot_training : 
    # Training plots
        head_acc     = [[x[0]['Head'],      x[1]['Head']]      for x in metric_list]
        shoulder_acc = [[(x[0]['ShoulderLeft']  + x[0]['ShoulderRight']) / 2.0, 
                     (x[1]['ShoulderLeft']  + x[1]['ShoulderRight']) / 2.0] for x in metric_list]
        elbow_acc    = [[(x[0]['ElbowLeft'] + x[0]['ElbowRight']) / 2.0, 
                     (x[1]['ElbowLeft'] + x[1]['ElbowRight']) / 2.0]        for x in metric_list]
        wrist_acc    = [[(x[0]['WristLeft'] + x[0]['WristRight']) / 2.0, 
                     (x[1]['WristLeft'] + x[1]['WristRight']) / 2.0]        for x in metric_list]
        hip_acc      = [[(x[0]['HipLeft']   + x[0]['HipRight'])   / 2.0, 
                     (x[1]['HipLeft']   + x[1]['HipRight'])   / 2.0]        for x in metric_list]
        knee_acc     = [[(x[0]['KneeLeft']  + x[0]['KneeRight'])  / 2.0, 
                     (x[1]['KneeLeft']  + x[1]['KneeRight'])  / 2.0]        for x in metric_list]
        ank_acc      = [[(x[0]['AnkleLeft'] + x[0]['AnkleRight']) / 2.0, 
                     (x[1]['AnkleLeft'] + x[1]['AnkleRight']) / 2.0]        for x in metric_list]
        spine_base   = [[x[0]['SpineBase'], x[1]['SpineBase']] for x in metric_list]
    
        spine_mid    = [[x[0]['SpineMid'],  x[1]['SpineMid']]  for x in metric_list]
        neck         = [[x[0]['Neck'],      x[1]['Neck']]      for x in metric_list]

        total_acc    = [[x[0]['total'],     x[1]['total']]     for x in metric_list]
        ax1.plot([x[0] for x in total_acc],    label = "Total")
        ax1.plot([x[0] for x in head_acc],     label = "Head")
        ax1.plot([x[0] for x in neck],         label = "Neck")
        ax1.plot([x[0] for x in spine_base],   label = "Spine Base")
        ax1.plot([x[0] for x in spine_mid],    label = "Spine Mid")
        ax1.plot([x[0] for x in shoulder_acc], label = "Shoulders")
        ax1.plot([x[0] for x in elbow_acc],    label = "Elbows")
        ax1.plot([x[0] for x in wrist_acc],    label = "Wrists")
        ax1.plot([x[0] for x in hip_acc],      label = "Hips")
        ax1.plot([x[0] for x in knee_acc],     label = "Knees")
        ax1.plot([x[0] for x in ank_acc],      label = "Ankles")
        ax1.set_title("PCK accuracy on Training Set - " + title, fontsize=12)
        ax1.set_ylabel("Score", fontsize=10)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.legend(prop={'size': 10})
  
        ax2.plot([x[1] for x in total_acc],    label = "Total")
        ax2.plot([x[1] for x in head_acc],     label = "Head")
        ax2.plot([x[1] for x in neck],         label = "Neck")
        ax2.plot([x[1] for x in spine_base],   label = "Spine Base")
        ax2.plot([x[1] for x in spine_mid],    label = "Spine Mid")
        ax2.plot([x[1] for x in shoulder_acc], label = "Shoulders")
        ax2.plot([x[1] for x in elbow_acc],    label = "Elbows")
        ax2.plot([x[1] for x in wrist_acc],    label = "Wrists")
        ax2.plot([x[1] for x in hip_acc],      label = "Hips")
        ax2.plot([x[1] for x in knee_acc],     label = "Knees")
        ax2.plot([x[1] for x in ank_acc],      label = "Ankles")
        ax2.set_title(f"PCK accuracy on Validation Set - " + title, fontsize = 12)
        ax2.set_ylabel("Score",       fontsize = 10)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.legend(prop={'size': 10})
    else : 
        shoulder_acc = [(x['ShoulderLeft']  + x['ShoulderRight']) / 2.0 for x in metric_list]
        elbow_acc    = [(x['ElbowLeft']     + x['ElbowRight'])    / 2.0 for x in metric_list]
        wrist_acc    = [(x['WristLeft']     + x['WristRight'])    / 2.0 for x in metric_list]
        hip_acc      = [(x['HipLeft']       + x['HipRight'])      / 2.0 for x in metric_list]
        knee_acc     = [(x['KneeLeft']      + x['KneeRight'])     / 2.0 for x in metric_list]
        ank_acc      = [(x['AnkleLeft']     + x['AnkleRight'])    / 2.0 for x in metric_list]
        head_acc     = [x['Head']      for x in metric_list]
        spine_base   = [x['SpineBase'] for x in metric_list]
        spine_mid    = [x['SpineMid']  for x in metric_list]
        neck         = [x['Neck']      for x in metric_list]
        total_acc    = [x['total']     for x in metric_list]

        ax2.plot([x for x in total_acc],    label = "Total")
        ax2.plot([x for x in head_acc],     label = "Head")
        ax2.plot([x for x in neck],         label = "Neck")
        ax2.plot([x for x in spine_base],   label = "Spine Base")
        ax2.plot([x for x in spine_mid],    label = "Spine Mid")
        ax2.plot([x for x in shoulder_acc], label = "Shoulders")
        ax2.plot([x for x in elbow_acc],    label = "Elbows")
        ax2.plot([x for x in wrist_acc],    label = "Wrists")
        ax2.plot([x for x in hip_acc],      label = "Hips")
        ax2.plot([x for x in knee_acc],     label = "Knees")
        ax2.plot([x for x in ank_acc],      label = "Ankles")
        ax2.set_title(f"PCK accuracy on Inference Time " + title, fontsize = 12)
        ax2.set_ylabel("Score",        fontsize = 10)
        ax2.set_xlabel("Time horizon (Num Frames)", fontsize = 10)
        ax2.legend(prop={'size': 10})
    


PLOT_METRIC_DICT = { 
    'pennaction' : plot_metric_pennaction, 
    'ntu'        : plot_metric_ntu, 
} 


def plot_loss(ax1, loss_list, title):
    ax1.plot(loss_list[:, 0], label="Train Loss")
    ax1.plot(loss_list[:, 1], label="Validation Loss")
    ax1.set_title("Loss Curves - " + title, fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss",  fontsize=10) 
    ax1.legend(prop={'size': 10})

def plot(loss_list, metric_list, title, dataset, save_path = None):
    plot_metric_func     = PLOT_METRIC_DICT[dataset] 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 8))
    plot_loss(ax1, loss_list, title)
    plot_metric_func(ax2, ax3, metric_list, title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_inference(metric_list, title, dataset, save_path = None): 
    plot_metric_func = PLOT_METRIC_DICT[dataset]
    fig, ax = plt.subplots(1, 1, figsize = (24, 24))
    plot_metric_func(None, ax, metric_list, title, plot_training = False)
    if save_path : 
        plt.savefig(save_path) 
        plt.close() 
    else : 
        plt.show()

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

