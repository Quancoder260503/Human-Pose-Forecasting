import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from data.ImgUtil import torch2numpy
from data.DatasetUtil import *
from data import Forward_kinematics
from matplotlib.animation import FuncAnimation
import data.Dataset3DUtils
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

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

def plot_rmse_over_horizon(rmse_per_timestep, save_path = None):
    """
    rmse_per_timestep: np.array of shape (prediction_len, )
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rmse_per_timestep) + 1), rmse_per_timestep, marker='o')
    plt.xlabel("Prediction Time Step")
    plt.ylabel("RMSE (pixels)")
    plt.title("RMSE over Prediction Horizon")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
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
    dataset       = 'ntu', 
    shadow        = 0.3, 
    eps           = 1.0, 
):
    poses = np.array(poses, dtype=int)

    image = image.copy()
    image = (image * shadow).astype(np.uint8)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()

    skeleton = get_skeleton(dataset)

    for idx, pt in enumerate(poses):
        if pt[0] > eps and pt[1] > eps:
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)

    for i, (j1, j2) in enumerate(skeleton):
        if poses[j1, 0] > eps and poses[j1, 1] > eps and poses[j2, 0] > eps and poses[j2, 1] > eps:
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

_CAMERA_AZIM = {
    "C001": -90 + 45,   # -45   left-45°  camera: orbit viewer right to match
    "C002": -90,        # -90   front camera: straight on
    "C003": -90 - 45,   # -135  right-45° camera: orbit viewer left to match
}

def animate_single_poses3d(sequence, camera_id, fps=30, output_path='pose_prediction_3d.gif'):
    assert sequence.ndim == 3 and sequence.shape[1:] == (17, 3)

    seq_mpl   = sequence[:, :, [0, 2, 1]]
    T         = seq_mpl.shape[0]

    mins      = seq_mpl.min(axis = (0, 1))
    maxs      = seq_mpl.max(axis = (0, 1))
    max_range = (maxs - mins).max() / 2.0
    mids      = (mins + maxs) / 2.0

    skeleton  = get_skeleton("ntu")

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
    ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
    ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (left/right)")
    ax.set_ylabel("Z (depth, m)")
    ax.set_zlabel("Y (up, m)")
    ax.view_init(elev=15, azim=_CAMERA_AZIM[camera_id])

    pose0   = seq_mpl[0]
    scatter = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2], s=40)

    bone_lines = []
    for i, j in skeleton:
        (line,) = ax.plot(
            [pose0[i, 0], pose0[j, 0]],
            [pose0[i, 1], pose0[j, 1]],
            [pose0[i, 2], pose0[j, 2]],
            linewidth=2,
        )
        bone_lines.append((line, i, j))

    title = fig.suptitle("Frame 0")

    def update(frame):
        pose = seq_mpl[frame]
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, i, j in bone_lines:
            line.set_data_3d(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]],
            )
        title.set_text(f"Frame {frame} / {T}")
        return [scatter, *[ln for ln, _, _ in bone_lines]]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)

    ani.save(output_path, writer='pillow', fps=fps)
    print(f"Saved GIF => {output_path}")
    plt.tight_layout()
    plt.show()
    return ani

def animate_ntu_pose_3d(observed_sequence, predicted_sequence, target_sequence, camera_id, fps=30, output_path='pose_prediction_3d.gif'):
    """
    observed_sequence  : (T_obs,  17, 3) — NTU RGB+D raw Kinect v2 camera-space coordinates
    target_sequence    : (T_pred, 17, 3)
    predicted_sequence : (T_pred, 17, 3)

      col 0  X : horizontal, positive = camera-right (subject's left)
      col 1  Y : vertical,   positive = up
      col 2  Z : depth,      positive = away from sensor (always >= 0)

    Kinect v2 Y is up-positive — no axis flip needed.
    We map directly: mpl-X = Kinect-X, mpl-Y = Kinect-Z, mpl-Z = Kinect-Y
    so that matplotlib's Z-up convention shows the skeleton upright, and
    depth runs along matplotlib's Y axis (into the scene).
    """

    assert observed_sequence.ndim  == 3 and observed_sequence.shape[1:]  == (17, 3)
    assert target_sequence.ndim    == 3 and target_sequence.shape[1:]    == (17, 3)
    assert predicted_sequence.ndim == 3 and predicted_sequence.shape[1:] == (17, 3)

    # Remap axes to match matplotlib Z-up convention (same as before)
    obs_mpl  = observed_sequence[:, :, [0, 2, 1]]
    tgt_mpl  = target_sequence[:, :, [0, 2, 1]]
    pred_mpl = predicted_sequence[:, :, [0, 2, 1]]

    # Concatenate for consistent axis limits across all frames
    all_mpl = np.concatenate([obs_mpl, tgt_mpl, pred_mpl], axis=0)
    T_obs   = obs_mpl.shape[0]
    T_pred  = tgt_mpl.shape[0]
    T_total = T_obs + T_pred

    mins      = all_mpl.min(axis = (0, 1))
    maxs      = all_mpl.max(axis = (0, 1))
    max_range = (maxs - mins).max() / 2.0
    mids      = (mins + maxs) / 2.0

    skeleton = get_skeleton("ntu")

    fig, (ax_tgt, ax_pred) = plt.subplots(1, 2, figsize = (16, 8), subplot_kw = {"projection": "3d"})

    def setup_ax(ax, title_str):
        ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
        ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
        ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X (left/right)")
        ax.set_ylabel("Z (depth, m)")
        ax.set_zlabel("Y (up, m)")
        ax.view_init(elev = 15, azim = _CAMERA_AZIM[camera_id])
        ax.set_title(title_str)

    setup_ax(ax_tgt,  "Ground Truth")
    setup_ax(ax_pred, "Prediction")

    def make_artists(ax, pose0):
        scatter = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2], s = 40)
        bone_lines = []
        for i, j in skeleton:
            (line,) = ax.plot(
                [pose0[i, 0], pose0[j, 0]],
                [pose0[i, 1], pose0[j, 1]],
                [pose0[i, 2], pose0[j, 2]],
                linewidth=2,
            )
            bone_lines.append((line, i, j))
        return scatter, bone_lines

    pose0_tgt  = obs_mpl[0]   # both start on first observed frame
    pose0_pred = obs_mpl[0]

    scatter_tgt,  bones_tgt  = make_artists(ax_tgt,  pose0_tgt)
    scatter_pred, bones_pred = make_artists(ax_pred, pose0_pred)

    title = fig.suptitle("Frame 0")

    def get_pose(frame):
        """Return (tgt_pose, pred_pose) for a given global frame index."""
        if frame < T_obs:
            # Observed window: both panels show the same observed pose
            return obs_mpl[frame], obs_mpl[frame]
        else:
            pred_idx = frame - T_obs
            return tgt_mpl[pred_idx], pred_mpl[pred_idx]

    def update_artists(scatter, bone_lines, pose):
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, i, j in bone_lines:
            line.set_data_3d(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]],
            )

    def update(frame):
        tgt_pose, pred_pose = get_pose(frame)
        update_artists(scatter_tgt,  bones_tgt,  tgt_pose)
        update_artists(scatter_pred, bones_pred, pred_pose)
        label = "Observed" if frame < T_obs else f"Predicted {frame - T_obs} / {T_pred}"
        title.set_text(f"Frame {frame} / {T_total}  [{label}]")
        return [scatter_tgt, scatter_pred, *[ln for ln, _, _ in bones_tgt], *[ln for ln, _, _ in bones_pred]]

    ani = FuncAnimation(fig, update, frames = T_total, interval = 1000 // fps, blit = False)

    ani.save(output_path, writer = 'pillow', fps = fps)
    print(f"Saved GIF => {output_path}")
    plt.tight_layout()
    plt.show()
    return ani


def animate_ntu_pose_2d(
    observe_sequences, 
    target_sequences,
    predict_sequences, 
    frame_input_w, 
    frame_input_h, 
    name, 
    start_index, 
    num_training_frame, 
    num_testing_frame,
    num_joints,
    fps = 30,
    output_path = 'pose_prediction.gif'
):
    observed_imgs, target_imgs = get_video_frames_ntu(
        name               = name, 
        start_index        = start_index, 
        num_training_frame = num_training_frame, 
        num_testing_frame  = num_testing_frame, 
        frame_input_w      = frame_input_w,
        frame_input_h      = frame_input_h, 
    )

    frames_real = []
    frames_pred = []


    for index in range(num_training_frame):
        curr_observed_image = observed_imgs[index, :, :, :]
        curr_observed_poses = observe_sequences[index]
        curr_observed_poses = curr_observed_poses.contiguous().view(num_joints, -1)
        frames_real.append(draw_skeleton(curr_observed_image, curr_observed_poses.detach().cpu().numpy()))
        frames_pred.append(draw_skeleton(curr_observed_image, curr_observed_poses.detach().cpu().numpy()))


    for index in range(num_testing_frame):
        curr_target_image = target_imgs[index, :, :, :]

        curr_target_pose = target_sequences[index]
        curr_target_pose = curr_target_pose.contiguous().view(num_joints, -1)

        curr_predict_pose = predict_sequences[index]
        curr_predict_pose = curr_predict_pose.contiguous().view(num_joints, -1)

        frames_real.append(draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy()))
        frames_pred.append(draw_skeleton(curr_target_image, curr_predict_pose.detach().cpu().numpy()))


    fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize = (10, 5))
    
    ax_real.set_title("Ground Truth")
    ax_pred.set_title("Prediction")

    for ax in (ax_real, ax_pred):
        ax.axis('off')

    im_real = ax_real.imshow(frames_real[0])
    im_pred = ax_pred.imshow(frames_pred[0])
    title   = fig.suptitle("Frame 0")

    def update(frame):
        im_real.set_data(frames_real[frame])
        im_pred.set_data(frames_pred[frame])
        title.set_text(f"Frame {frame} / {num_testing_frame}")
        return [im_real, im_pred]

    ani = FuncAnimation(fig, update, frames = num_testing_frame, interval = 1000 // fps, blit = False)

    ani.save(output_path, writer = 'pillow', fps = fps)
    print(f"Saved GIF => {output_path}")
    plt.tight_layout()
    plt.show()
    return ani


def animate_cmu_pose_3d(
    observed_sequence,
    predicted_sequence,
    target_sequence,
    fps=30,
    output_path='pose_prediction_cmu_3d.gif'
):
    """
    Animate CMU MoCap 3D pose predictions vs ground truth.

    observed_sequence  : (T_obs,  38, 3)
    target_sequence    : (T_pred, 38, 3)
    predicted_sequence : (T_pred, 38, 3)

    Skeleton uses 1-indexed I/J arrays (converted to 0-indexed internally).
    Left/right colouring is preserved from the original Ax3DPoseCMU definition.
    """
    assert observed_sequence.ndim  == 3 and observed_sequence.shape[1:]  == (38, 3)
    assert target_sequence.ndim    == 3 and target_sequence.shape[1:]    == (38, 3)
    assert predicted_sequence.ndim == 3 and predicted_sequence.shape[1:] == (38, 3)

    # ── Skeleton definition (from Ax3DPoseCMU, converted to 0-indexed) ──────
    I = np.array([1,2,3,4,5,6,1,8,9,10,11,12,1,14,15,16,17,18,19,16,
                  21,22,23,25,26,24,28,16,30,31,32,33,34,35,33,37]) - 1
    J = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,
                  22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38]) - 1
    LR = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,
                   1,1,1,0,0,0,0,0,0,0,0,0], dtype=bool)

    LCOLOR = "#3498db"
    RCOLOR = "#e74c3c"

    all_data  = np.concatenate([observed_sequence, target_sequence, predicted_sequence], axis=0)
    T_obs     = observed_sequence.shape[0]
    T_pred    = target_sequence.shape[0]
    T_total   = T_obs + T_pred

    mins      = all_data.min(axis=(0, 1))
    maxs      = all_data.max(axis=(0, 1))
    max_range = (maxs - mins).max() / 2.0
    mids      = (mins + maxs) / 2.0

    fig, (ax_tgt, ax_pred) = plt.subplots(
        1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"}
    )

    def setup_ax(ax, title_str):
        ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
        ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
        ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title_str)

    setup_ax(ax_tgt,  "Ground Truth")
    setup_ax(ax_pred, "Prediction")

    def make_artists(ax, pose0):
        scatter = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2], s=40)
        bone_lines = []
        for idx in range(len(I)):
            i, j = I[idx], J[idx]
            color = LCOLOR if LR[idx] else RCOLOR
            (line,) = ax.plot(
                [pose0[i, 0], pose0[j, 0]],
                [pose0[i, 1], pose0[j, 1]],
                [pose0[i, 2], pose0[j, 2]],
                linewidth=2, color=color,
            )
            bone_lines.append((line, i, j, color))
        return scatter, bone_lines

    scatter_tgt,  bones_tgt  = make_artists(ax_tgt,  observed_sequence[0])
    scatter_pred, bones_pred = make_artists(ax_pred, observed_sequence[0])

    title = fig.suptitle("Frame 0")

    def get_pose(frame):
        if frame < T_obs:
            return observed_sequence[frame], observed_sequence[frame]
        pred_idx = frame - T_obs
        return target_sequence[pred_idx], predicted_sequence[pred_idx]

    def update_artists(scatter, bone_lines, pose):
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, i, j, _ in bone_lines:
            line.set_data_3d(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]],
            )

    def update(frame):
        tgt_pose, pred_pose = get_pose(frame)
        update_artists(scatter_tgt,  bones_tgt,  tgt_pose)
        update_artists(scatter_pred, bones_pred, pred_pose)
        label = "Observed" if frame < T_obs else f"Predicted {frame - T_obs} / {T_pred}"
        title.set_text(f"Frame {frame} / {T_total}  [{label}]")
        return [
            scatter_tgt, scatter_pred,
            *[ln for ln, _, _, _ in bones_tgt],
            *[ln for ln, _, _, _ in bones_pred],
        ]

    ani = FuncAnimation(fig, update, frames=T_total, interval=1000 // fps, blit=False)

    ani.save(output_path, writer='pillow', fps=fps)
    print(f"Saved GIF => {output_path}")
    plt.tight_layout()
    plt.show()
    return ani


def animate_h36m_pose_3d(
    observed_sequence,
    predicted_sequence,
    target_sequence,
    fps = 30,
    output_path='pose_prediction_h36m_3d.gif'
):
    """
    Animate Human3.6M 3D pose predictions vs ground truth.

    observed_sequence  : (T_obs,  17, 3)
    target_sequence    : (T_pred, 17, 3)
    predicted_sequence : (T_pred, 17, 3)

    Skeleton uses 1-indexed I/J arrays (converted to 0-indexed internally).
    Left/right colouring is preserved from the original Ax3DPoseH36M definition.
    """
    assert observed_sequence.ndim  == 3 and observed_sequence.shape[1:]  == (17, 3)
    assert target_sequence.ndim    == 3 and target_sequence.shape[1:]    == (17, 3)
    assert predicted_sequence.ndim == 3 and predicted_sequence.shape[1:] == (17, 3)

    # 
    I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype = bool)

    LCOLOR = "#3498db"
    RCOLOR = "#e74c3c"

    all_data  = np.concatenate([observed_sequence, target_sequence, predicted_sequence], axis=0)
    T_obs     = observed_sequence.shape[0]
    T_pred    = target_sequence.shape[0]
    T_total   = T_obs + T_pred

    mins      = all_data.min(axis=(0, 1))
    maxs      = all_data.max(axis=(0, 1))
    max_range = (maxs - mins).max() / 2.0
    mids      = (mins + maxs) / 2.0

    fig, (ax_tgt, ax_pred) = plt.subplots(
        1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"}
    )

    def setup_ax(ax, title_str):
        ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
        ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
        ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev = 120, azim = -90) 
        ax.set_title(title_str) 


    setup_ax(ax_tgt,  "Ground Truth")
    setup_ax(ax_pred, "Prediction")

    def make_artists(ax, pose0):
        scatter = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2], s = 40, marker = '.', zorder = 5)
        bone_lines = []
        for idx in range(len(I)):
            i, j = I[idx], J[idx]
            color = LCOLOR if LR[idx] else RCOLOR
            (line, ) = ax.plot(
                [pose0[i, 0], pose0[j, 0]],
                [pose0[i, 1], pose0[j, 1]],
                [pose0[i, 2], pose0[j, 2]],
                linewidth=3, marker='.', markersize=10, color=color,
            )
            bone_lines.append((line, i, j, color))
        return scatter, bone_lines

    scatter_tgt,  bones_tgt  = make_artists(ax_tgt,  observed_sequence[0])
    scatter_pred, bones_pred = make_artists(ax_pred, observed_sequence[0])

    title = fig.suptitle("Frame 0")


    def get_pose(frame):
        if frame < T_obs:
            return observed_sequence[frame], observed_sequence[frame]
        pred_idx = frame - T_obs
        return target_sequence[pred_idx], predicted_sequence[pred_idx]

    def update_artists(scatter, bone_lines, pose):
        scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        for line, i, j, _ in bone_lines:
            line.set_data_3d(
                [pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]],
            )

    def update(frame):
        tgt_pose, pred_pose = get_pose(frame)
        update_artists(scatter_tgt,  bones_tgt,  tgt_pose)
        update_artists(scatter_pred, bones_pred, pred_pose)
        label = "Observed" if frame < T_obs else f"Predicted {frame - T_obs} / {T_pred}"
        title.set_text(f"Frame {frame} / {T_total}  [{label}]")
        return [
            scatter_tgt, scatter_pred,
            *[ln for ln, _, _, _ in bones_tgt],
            *[ln for ln, _, _, _ in bones_pred],
        ]

    ani = FuncAnimation(fig, update, frames=T_total, interval=1000 // fps, blit=False)

    ani.save(output_path, writer='pillow', fps=fps)
    print(f"Saved GIF => {output_path}")
    plt.tight_layout()
    plt.show()
    return ani