'% Pytorch dependencies%'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import cv2
from data.PennAction import PennAction
import numpy as np
from data.JHMDB import JHMDB
from ultralytics import YOLO
from util.KeyPointUtil import convert_kps, get_pennaction_joint_names
from util.Visualization import draw_skeleton
from util.ImgUtil import torch2numpy
import math 

YOLO_IMAGE_H = 640
YOLO_IMAGE_W = 640
BATCH_SIZE = 32

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing = 'xy')
    coords = torch.stack(coords[::-1], dim = 0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, flows, mode = 'bilinear', mask = False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    flows = flows.permute(0, 2, 3, 1)
    B, _, H, W = img.shape
    dxgrid, dygrid = flows.split([1, 1], dim = -1)
    coords = coords_grid(B, H, W)
    coords = coords.permute(0, 2, 3, 1).to('cuda')
    ygrid, xgrid = coords.split([1, 1], dim = -1)
    xgrid = xgrid + dxgrid
    ygrid = ygrid + dygrid
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim = -1)
    img = F.grid_sample(img, grid, align_corners = True, mode = mode)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def get_predictions_dict():
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

def calc_PCK(
        pred_joints,
        gt_joints,
        predictions_dict,
        map_size = 640,
        sigma = 0.04
):
    '''
    :param predictions_dict: Average Meter for PCK
    :param pred_joints: tensor of shape (B, J, 3)
    :param gt_joints: tensor of shape (B, J, 3)
    :param map_size: size of the box
    :param sigma: scale factor
    :return:
       the average PCK scores across different joints
    '''
    joints_name = get_pennaction_joint_names()
    for b in range(pred_joints.shape[0]):
        for j in range(pred_joints.shape[1]):
          if gt_joints[b, j, 2] < 1 :
              continue

          predictions_dict['total']['all'] += 1
          predictions_dict[joints_name[j]]['all'] += 1
          dist = torch.sqrt((pred_joints[b, j, 0] - gt_joints[b, j, 0]) ** 2 + (pred_joints[b, j, 1] - gt_joints[b, j, 1]) ** 2)
          if dist < map_size * sigma :
              predictions_dict['total']['correct'] += 1
              predictions_dict[joints_name[j]]['correct'] += 1

    return predictions_dict

def visualize_frame(
        list_of_flows,
        previous_frame,
        current_frame,
        next_frame,
        predicted_frame,
        predicted_frame_with_yolo_joints
    ):

    flow_vis = flow_to_image(list_of_flows[-1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    axes[0, 0].imshow(torch2numpy(previous_frame))
    axes[0, 0].set_title('Previous frame')
    axes[0, 1].imshow(torch2numpy(current_frame))
    axes[0, 1].set_title('Current frame')
    axes[0, 2].imshow(torch2numpy(next_frame))  # you had current_frame here too
    axes[0, 2].set_title('Next frame')
    axes[1, 0].imshow(torch2numpy(predicted_frame))
    axes[1, 0].set_title('Predicted frame')
    axes[1, 1].imshow(flow_vis[0].permute(1, 2, 0).detach().numpy())
    axes[1, 1].set_title('Optical flow')
    """
    Visualize the vector field
    """
    coords = coords_grid(1, YOLO_IMAGE_H, YOLO_IMAGE_W)[0].cpu().numpy()  # (2, H, W)
    opt_flow = list_of_flows[-1][0].detach().cpu().numpy()  # (2, H, W)
    X, Y, U, V = coords[0], coords[1], opt_flow[0], opt_flow[1]
    step = 40
    axes[1, 2].imshow(torch2numpy(current_frame[0]))
    axes[1, 2].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        color='black',
        scale=0.2,
        scale_units='xy',
        angles='xy',
    )
    axes[1, 2].set_title('Flow direction')

    pred = torch2numpy(predicted_frame)
    real = torch2numpy(next_frame)  # (H, W, 3) float

    pred_gray = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.
    real_gray = cv2.cvtColor((real * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.

    superimposed = np.stack([
        pred_gray,
        real_gray,
        np.zeros_like(pred_gray),  # B
    ], axis=-1)

    axes[1, 2].imshow(superimposed)
    axes[1, 2].set_title('Predicted (R) vs Actual (G)')

    plt.show()

def baseline_trainer():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # YOLO on GPU
    yolo_model = YOLO('yolo11n-pose.pt')

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    penn_action = PennAction(sequence_train=7, transform=PennAction.get_default_transform())

    dataloader = DataLoader(
        dataset=penn_action,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    predictions_dict = get_predictions_dict()
    losses = []
    scaled_dist = math.sqrt(YOLO_IMAGE_H ** 2 + YOLO_IMAGE_W ** 2)
    resize = transforms.Resize((YOLO_IMAGE_H, YOLO_IMAGE_W))

    with torch.no_grad():

        for batch_idx, (frames, labels, _, _) in enumerate(dataloader):

            labels = labels.permute(0, 3, 2, 1)

            predicted_frames = []
            target_joints_all = []

            orig_h, orig_w = frames.shape[-2:]
            scale_x = 1. * YOLO_IMAGE_W / orig_w
            scale_y = 1. * YOLO_IMAGE_H / orig_h

            for b in range(frames.shape[0]):
                for t in range(1, labels.shape[1] - 1):

                    previous_frame = resize(frames[b, t-1].unsqueeze(0)).to(device)
                    current_frame  = resize(frames[b, t].unsqueeze(0)).to(device)

                    target_joints  = labels[b, t+1].unsqueeze(0).float()

                    target_joints_scaled = target_joints.clone()
                    target_joints_scaled[:, :, 0] *= scale_x
                    target_joints_scaled[:, :, 1] *= scale_y

                    list_of_flows = model(current_frame, previous_frame)
                    predicted_frame = bilinear_sampler(current_frame, list_of_flows[-1])

                    predicted_frames.append(predicted_frame)
                    target_joints_all.append(target_joints_scaled)

            # Stack everything
            predicted_frames = torch.cat(predicted_frames, dim=0)
            target_joints_all = torch.cat(target_joints_all, dim=0).to(device)

            # ---- YOLO BATCH INFERENCE ----
            results = yolo_model(predicted_frame.cpu(), device = 'cpu', verbose=False)

            pjmse_loss = 0.0
            num_samples = 0

            for i, result in enumerate(results):

                if result.keypoints is None:
                    continue

                predicted_joints = result.keypoints.xy

                if predicted_joints.shape[0] == 0:
                    continue

                predicted_joints = predicted_joints[0].unsqueeze(0)

                predicted_joints = torch.cat(
                    [predicted_joints,
                     torch.ones(*predicted_joints.shape[:2], 1, device=predicted_joints.device)],
                    dim=2
                )

                predicted_joints = torch.tensor(
                    convert_kps(predicted_joints, src='coco', dst='pennaction')
                ).to(device)

                target_joints_scaled = target_joints_all[i].unsqueeze(0)

                per_joint_error = torch.sqrt(((predicted_joints[:, :, :2] - target_joints_scaled[:, :, :2]) ** 2).sum(dim=-1)) / scaled_dist

                visibility = target_joints_scaled[:, :, 2]

                loss = (per_joint_error * visibility).sum() / visibility.sum()

                pjmse_loss += loss.item() 
                num_samples += 1

                predictions_dict = calc_PCK(predicted_joints, target_joints_scaled, predictions_dict)

            if num_samples > 0:
                losses.append(pjmse_loss / num_samples)

            total_acc = predictions_dict['total']['correct'] / predictions_dict['total']['all']
            loss = np.mean(losses)
            print(f'Batch : {batch_idx}/{len(dataloader)} | PJMSE : {loss:.3f} | PCK : {total_acc:.3f}')

if __name__ == '__main__':
    baseline_trainer()