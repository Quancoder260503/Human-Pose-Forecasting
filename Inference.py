"""
Script that investigate the actual performance of models 
"""
import torch 
import os 
from PennActionTrainingUtil import get_lstm_model
from data.PennAction import PennAction
from data.NTU import *
import cv2 
import matplotlib.pyplot as plt 
from data.KeyPointUtil import convert_kps, get_pennaction_joint_names
from data.ImgUtil import torch2numpy
from PennActionTrainingUtil import *
from NTUTrainingUtil import *
from model.LSTM import *
import math 
import random 

def inference_by_coordinates(num_joints, checkpoint = 'checkpoint/baseline/joint_lstm_pennaction.pth.tar'):
    device     = 'cuda'
    assert os.path.exists(checkpoint) 
    model_dict = torch.load(checkpoint, weights_only = False) 
    start_epoch = model_dict['epoch'] + 1 
    model       = get_lstm_model(device, pose_dim = num_joints)
    model.load_state_dict(model_dict['model'])
    total_frame = 14
    ntu_set = NTU(
      num_training_frame = 7, 
      num_testing_frame  = 7, 
      train = False,  
      coordinates   = True,  
      require_image = True, 
    )     
    observed_poses, target_poses, observed_images, target_images, name = ntu_set[35]
    print(observed_images.shape)  
    observed_poses = observed_poses.to(device, dtype = torch.float32).unsqueeze(0)
    target_poses   = target_poses.to(device,   dtype = torch.float32).unsqueeze(0)
    predicted_poses = model(observed_poses, prediction_len = total_frame // 2)
    loss            = LSTM_loss_function(predicted_poses, target_poses, total_frame // 2)
    print(f"RMSE Loss : {math.sqrt(loss):.4f}")
     
    fig, axes = plt.subplots(3, 7, figsize = (24, 8))
    for index in range(0, 7): 
       feed_pose  = observed_poses[:, index, :].squeeze(0)
       feed_pose, feed_prob  = torch.split(feed_pose.contiguous().view(num_joints, 3), [2, 1], dim = -1)
       
       curr_pose  = target_poses[:, index, :].squeeze(0)
       curr_pose, curr_prob  = torch.split(curr_pose.contiguous().view(num_joints, 3), [2, 1], dim = -1)  # Scale factor
       
       pred_pose = predicted_poses[:, index, :].squeeze(0)
       pred_pose, pred_prob  = torch.split(pred_pose.contiguous().view(num_joints, 3), [2, 1], dim = -1)
       
       background_img  = (np.ones((ntu_set.input_h, ntu_set.input_w, 3)) * 255).astype(np.uint8) 
       
       pred_prob  = torch.sigmoid(pred_prob)

       curr_observed_image = observed_images[index, :, :, :]
       curr_target_image   = target_images[index, :, :, :]

       gt_image   = draw_skeleton(curr_target_image, curr_pose.detach().cpu().numpy() * ntu_set.input_w,  curr_prob.detach().cpu().numpy())
       pred_image = draw_skeleton(curr_target_image, pred_pose.detach().cpu().numpy() * ntu_set.input_h , pred_prob.detach().cpu().numpy())
       feed_image = draw_skeleton(curr_observed_image, feed_pose.detach().cpu().numpy() * ntu_set.input_w, feed_prob.detach().cpu().numpy())
       
       axes[0, index].imshow(feed_image)
       axes[0, index].axis('off')
       axes[1, index].imshow(gt_image)
       axes[1, index].axis('off')
       axes[2, index].imshow(pred_image)
       axes[2, index].axis('off')
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
    plt.tight_layout()
    plt.show() 


if __name__ == '__main__':
  #  inference_by_bone_vector()
  inference_by_coordinates(
    num_joints = 25, 
    checkpoint = 'checkpoint/baseline/joint_lstm_ntu.pth.tar'
  )
