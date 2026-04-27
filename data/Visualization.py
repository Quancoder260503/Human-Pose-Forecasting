import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from data.ImgUtil import torch2numpy
from data.DatasetUtil import *



"""
Train time visualization 
"""


def plot_metric_pennaction(ax1, ax2, metric_list, plot_training = True):
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
        ax1.set_title("PCK accuracy on Training Set", fontsize=12)
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
        ax2.set_title(f"PCK accuracy on Validation Set", fontsize=12)
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
        ax2.set_title("PCK accuracy on Inference Time", fontsize=12)
        ax2.set_ylabel("Score",        fontsize = 10)
        ax2.set_xlabel("Time horizon (Num Frames)", fontsize = 10)
        ax2.legend(prop={'size': 10})

'''
NTU PLOT
'''
def plot_metric_ntu(ax1, ax2, metric_list, plot_training = True): 
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
        ax1.set_title("PCK accuracy on Training Set", fontsize=12)
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
        ax2.set_title(f"PCK accuracy on Validation Set", fontsize = 12)
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
        ax2.set_title(f"PCK accuracy on Inference Time" , fontsize = 12)
        ax2.set_ylabel("Score",        fontsize = 10)
        ax2.set_xlabel("Time horizon (Num Frames)", fontsize = 10)
        ax2.legend(prop={'size': 10})


PLOT_METRIC_DICT = { 
    'pennaction' : plot_metric_pennaction, 
    'ntu'        : plot_metric_ntu, 
} 


def plot_loss(ax1, loss_list):
    ax1.plot(loss_list[:, 0], label="Train Loss")
    ax1.plot(loss_list[:, 1], label="Validation Loss")
    ax1.set_title("Loss Curves", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss",  fontsize=10) 
    ax1.legend(prop={'size': 10})

def plot(loss_list, metric_list, dataset, save_path = None):
    plot_metric_func     = PLOT_METRIC_DICT[dataset] 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 8))
    plot_loss(ax1, loss_list)
    plot_metric_func(ax2, ax3, metric_list)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_inference(metric_list, dataset, save_path = None): 
    plot_metric_func = PLOT_METRIC_DICT[dataset]
    fig, ax = plt.subplots(1, 1, figsize = (24, 24))
    plot_metric_func(None, ax, metric_list, plot_training = False)
    if save_path : 
        plt.savefig(save_path) 
        plt.close() 
    else : 
        plt.show()



"""
Test time visualization
"""
def get_colors():
    colors = {
        'pink'         : np.array([197, 27, 125]),  # L lower leg
        'light_pink'   : np.array([233, 163, 201]),  # L upper leg
        'light_green'  : np.array([161, 215, 106]),  # L lower arm
        'green'        : np.array([77, 146, 33]),  # L upper arm
        'red'          : np.array([215, 48, 39]),  # head
        'light_red'    : np.array([252, 146, 114]),  # head
        'light_orange' : np.array([252, 141, 89]),  # chest
        'purple'       : np.array([118, 42, 131]),  # R lower leg
        'light_purple' : np.array([175, 141, 195]),  # R upper
        'light_blue'   : np.array([145, 191, 219]),  # R lower arm
        'blue'         : np.array([69, 117, 180]),  # R upper arm
        'gray'         : np.array([130, 130, 130]),  #
        'white'        : np.array([255, 255, 255]),  #
    }
    return colors


def draw_skeleton(
        image, 
        poses, 
        prob          = None, 
        prob_draw     = 0.65, 
        dataset       = 'ntu', 
        shadow        = 0.3, 
        eps           = 1.0, 
        unnormalize   = False, 
        frame_input_w = 1920, 
        frame_input_h = 1080,
    ):

    if unnormalize : 
        poses[:, 0] = poses[:, 0] * frame_input_w
        poses[:, 1] = poses[:, 1] * frame_input_h
        
    if prob is None : 
        prob  = np.ones(poses.shape[0])

    poses = np.array(poses, dtype=int)

    image = image.copy()
    image = (image * shadow).astype(np.uint8)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()

    skeleton = get_skeleton(dataset)

    for idx, pt in enumerate(poses):
        if prob[idx] > prob_draw and pt[0] > eps and pt[1] > eps:
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)

    for i, (j1, j2) in enumerate(skeleton):
        if prob[j1] > prob_draw and prob[j2] > prob_draw and poses[j1, 0] > eps and poses[j1, 1] > eps and poses[j2, 0] > eps and poses[j2, 1] > eps:
            color = rcolor
            pt1 = tuple(poses[j1, :2])
            pt2 = tuple(poses[j2, :2])
            cv2.line(image, pt1, pt2, color, thickness = 2)

    return image

def show_video(video, output_path = "output.mp4", fps = 25):
    height, width, _ = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Video", frame_bgr)
        out.write(frame_bgr)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()


"""
Visualize Attention  
"""
import matplotlib.pyplot as plt
import torch

def visualize_attention(attn_weights, cmap='inferno'):
    if isinstance(attn_weights[0], list):
        attn_weights = [a[1] for a in attn_weights]

    num_mats = len(attn_weights)
    cols = (num_mats + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    axes = axes.flatten()

    for i, attn in enumerate(attn_weights):
        attn = attn.detach().cpu()

        if attn.dim() == 3:
            attn = attn.mean(dim = 0)

        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        im = axes[i].imshow(attn.numpy(), cmap = cmap, aspect='auto')

        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key')
        axes[i].set_ylabel('Query')

        fig.colorbar(im, ax = axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()