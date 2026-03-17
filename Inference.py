"""
Script that investigate the actual performance of models 
"""
import torch 
import os 
from PennActionTrainingUtil import get_lstm_model
from data.PennAction import PennAction
import cv2 
import matplotlib.pyplot as plt 
from util.KeyPointUtil import convert_kps, get_pennaction_joint_names
from util.Visualization import draw_skeleton
from util.ImgUtil import torch2numpy
from PennActionTrainingUtil import *
from model.LSTM import *
import math 
import random 

def inference_by_coordinates(checkpoint = 'checkpoint/baseline/joint_lstm_pennaction.pth.tar'):
    device     = 'cuda'
    assert os.path.exists(checkpoint) 
    model_dict = torch.load(checkpoint, weights_only = False) 
    start_epoch = model_dict['epoch'] + 1 
    model       = get_lstm_model(device, pose_dim = 13)
    model.load_state_dict(model_dict['model'])
    total_frame = 14
    penn_action = PennAction(
      num_training_frame = 7, 
      num_testing_frame  = 7, 
      train = False, 
      require_image = True, 
      coordinates   = True, 
      on_debug_mode = True, 
    )     
    observed_poses, target_poses, observed_images, target_images = penn_action[random.randint(0, len(penn_action))] 
    observed_poses = observed_poses.to(device, dtype = torch.float32).unsqueeze(0)
    target_poses   = target_poses.to(device,   dtype = torch.float32).unsqueeze(0)
    predicted_poses = model(observed_poses, prediction_len = total_frame // 2)
    loss            = LSTM_loss_function(predicted_poses, target_poses, total_frame // 2)
    print(f"RMSE Loss : {math.sqrt(loss):.4f}")
     
    fig, axes = plt.subplots(3, 7, figsize = (24, 8))
    for index in range(0, 7): 
       feed_pose  = observed_poses[:, index, :].squeeze(0)
       feed_pose, feed_prob  = torch.split(feed_pose.contiguous().view(13, 3), [2, 1], dim = -1)
       curr_pose  = target_poses[:, index, :].squeeze(0)
       curr_pose, curr_prob  = torch.split(curr_pose.contiguous().view(13, 3), [2, 1], dim = -1)  # Scale factor
       pred_pose = predicted_poses[:, index, :].squeeze(0)
       pred_pose, pred_prob  = torch.split(pred_pose.contiguous().view(13, 3), [2, 1], dim = -1)
       curr_image = target_images[index, :, :, :] 
       pred_prob  = torch.sigmoid(pred_prob)
       curr_image = np.ascontiguousarray(torch2numpy(curr_image))
       draw_image = draw_skeleton(curr_image, curr_pose.detach().cpu().numpy() * penn_action.input_w,  curr_prob.detach().cpu().numpy())
       axes[0, index].imshow(draw_image)
       pred_image = draw_skeleton(curr_image, pred_pose.detach().cpu().numpy() * penn_action.input_h , pred_prob.detach().cpu().numpy())
       axes[1, index].imshow(pred_image)

       feed_image = observed_images[index, :, :, :] 
       feed_image = np.ascontiguousarray(torch2numpy(feed_image))
       feed_image = draw_skeleton(feed_image, feed_pose.detach().cpu().numpy() * penn_action.input_w, feed_prob.detach().cpu().numpy())
       axes[2, index].imshow(feed_image)
       #diffs = torch.mean(torch.norm(curr_pose - pred_pose, dim = 1) * 368)
       #print(diffs) 

    plt.tight_layout()
    plt.show() 

def inference_by_bone_vector(checkpoint = 'checkpoint/baseline/relative_lstm_pennaction.pth.tar'):
    device = 'cuda'
    assert os.path.exists(checkpoint) 
   
    model_dict = torch.load(checkpoint, weights_only = False) 
    start_epoch = model_dict['epoch'] + 1 

    model       = get_lstm_model(device, pose_dim = 13, parent_relative = True)
    model.load_state_dict(model_dict['model'])
   
    pennaction = PennAction(
      num_training_frame = 7, 
      num_testing_frame  = 7, 
      train = False, 
      require_image = True, 
      bone_vector   = True,
      on_debug_mode = False,   
    )     

    observed_bones, target_bones, observed_poses, target_poses, scaled_factor, observed_images, target_images = pennaction[random.randint(0, len(pennaction))]
    
    scaled_factor  = torch.tensor(scaled_factor)


    observed_bones = observed_bones.to(device, dtype = torch.float32).unsqueeze(0)
    target_bones   = target_bones.to(device,   dtype = torch.float32).unsqueeze(0)
    scaled_factor  = scaled_factor.to(device,  dtype = torch.float32).unsqueeze(0) 
    predicted_bones = model(observed_bones, 7)

    predicted_poses = reconstruct_pose(compressed_poses = predicted_bones, scaled_factor = scaled_factor)

    fig, axes = plt.subplots(2, 7, figsize = (24, 8))
    for index in range(0, 7):     
       curr_image = target_images[index, :, :, :]
       curr_image = np.ascontiguousarray(torch2numpy(curr_image))
       curr_predict_pose = predicted_poses[:, index, :].squeeze(0)
       curr_pose  = target_poses[index, :, :].squeeze(0)
       curr_pose, curr_prob  = torch.split(curr_pose.contiguous().view(13, 3), [2, 1], dim = -1) 
       draw_image = draw_skeleton(curr_image, curr_pose.detach().cpu().numpy() * pennaction.input_w,  curr_prob.detach().cpu().numpy())
       axes[0, index].imshow(draw_image)
       pred_image = draw_skeleton(curr_image, curr_predict_pose.detach().cpu().numpy() * pennaction.input_w)
       axes[1, index].imshow(pred_image)
       # TODO : add rmse 
    plt.tight_layout()
    plt.show() 


if __name__ == '__main__':
    inference_by_bone_vector()