"""
Script that investigate the actual performance of models, dataset : NTU 
:TODO add root_relative 
"""
import torch 
import os 
from data.NTU import *
import matplotlib.pyplot as plt 
from data.Visualization import *
from data.KeyPointUtil  import * 
from data.ImgUtil       import *
from Util       import * 
from model.Lstm               import *
from model.Mlp                import *
from model.RnnMotionVae       import *
from model.AttentionMotionVae import * 
import math 
import random 
import argparse

def inference_by_coordinates(args, model, criterion, ntu):
   # Sample a random point from testing data set 
   observed_poses, target_poses, start_index, name = ntu[random.randint(0, len(ntu))] 

   observed_poses  = observed_poses.to(args.device, dtype = torch.float32).unsqueeze(0)
   target_poses    = target_poses.to(args.device,   dtype = torch.float32).unsqueeze(0)

   predicted_poses = model(observed_poses = observed_poses, prediction_len = args.num_testing_frame)
   loss            = criterion(predicted_poses, target_poses, args.num_testing_frame)
   
   print(f"RMSE Loss : {math.sqrt(loss.item()):.4f}")

   fig, axes = plt.subplots(2, args.num_testing_frame, figsize = (24, 8))

   videos_real = []
   videos_pred = []

   pck_prediction = get_ntu_predictions_dict()

   observed_images, target_images = NTU.get_video_frames(name, start_index, args.num_training_frame, args.num_testing_frame)

   for index in range(0, args.num_testing_frame): 
      curr_target_pose                    = target_poses[:, index, :].squeeze(0)
      curr_target_pose, curr_target_prob  = torch.split(curr_target_prob.contiguous().view(args.num_joints, 3), [2, 1], dim = -1)  # Scale factor
      
      curr_pred_pose                  = predicted_poses[:, index, :].squeeze(0)
      curr_pred_pose, curr_pred_prob  = torch.split(curr_pred_pose.contiguous().view(args.num_joints, 3), [2, 1], dim = -1)

      pred_prob  = torch.sigmoid(pred_prob)

      curr_target_image = torch2numpy(target_images[index, :, :, :])

      pck_prediction = pck_joints(curr_pred_pose, curr_target_pose, pck_prediction, get_ntu_joint_names(), alpha = 0.05)


      target_annotated_image  = draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy() * ntu.box_size, curr_target_prob.detach().cpu().numpy())
      target_prediction_image = draw_skeleton(curr_target_image, curr_pred_pose.detach().cpu().numpy()   * ntu.box_size, curr_pred_prob.detach().cpu().numpy())

      axes[0, index].imshow(target_annotated_image)
      axes[1, index].axis('off')
      axes[2, index].imshow(target_prediction_image)
      axes[2, index].axis('off')

      videos_real.append(target_annotated_image)
      videos_pred.append(target_prediction_image)

      acc = get_acc(pck_prediction)
      print(acc)

   plt.tight_layout()
   plt.show() 


def inference_by_parent_relative(args, model, criterion, ntu):

   observed_bones, target_bones, observed_poses, target_poses, scaled_factor, start_index, name = ntu[random.randint(0, len(ntu))]
    
   scaled_factor  = torch.tensor(scaled_factor)

   observed_bones = observed_bones.to(args.device, dtype = torch.float32).unsqueeze(0)
   target_bones   = target_bones.to(args.device,   dtype = torch.float32).unsqueeze(0)
   scaled_factor  = scaled_factor.to(args.device,  dtype = torch.float32).unsqueeze(0) 
   
   predicted_bones = model(observed_bones, args.num_testing_frame)

   predicted_poses = reconstruct_pose(compressed_poses = predicted_bones, scaled_factor = scaled_factor, dataset = 'pennaction')
   target_poses    = reconstruct_pose(compressed_poses = target_bones,    scaled_factor = scaled_factor, dataset = 'pennaction')
    
   fig, axes = plt.subplots(2, 7, figsize = (24, 8))

   pck_prediction = get_ntu_predictions_dict()
    
   loss            = criterion(predicted_bones, target_bones, args.num_testing_frame)
   print(f"RMSE Loss : {math.sqrt(loss.item()):.4f}")

   videos_real = []
   videos_pred = [] 

   observed_images, target_images = NTU.get_video_frames(name, start_index, args.num_training_frame, args.num_testing_frame)

   for index in range(0, args.num_testing_frame):     
      curr_target_image = target_images[index, :, :, :]
      curr_target_image = np.ascontiguousarray(torch2numpy(curr_target_image))
      
      curr_predict_pose = predicted_poses[:, index, :].squeeze(0)
      curr_target_pose  = target_poses[:, index, :].squeeze(0)

      pck_prediction = pck_joints(curr_predict_pose, curr_target_pose, pck_prediction, get_pennaction_joint_names(), alpha = 0.05)

      target_annotated_image = draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy()  * ntu.box_size) 
      axes[0, index].imshow(target_annotated_image)
      target_predicted_image = draw_skeleton(curr_target_image, curr_predict_pose.detach().cpu().numpy() * ntu.box_size)
      axes[1, index].imshow(target_predicted_image)

      videos_real.append(target_annotated_image)
      videos_pred.append(target_predicted_image)

      acc = get_acc(pck_prediction)
      print(acc)
    
   plt.tight_layout()
   plt.show() 


def inference_by_velocity(args, model, criterion, ntu):
   observed_poses, target_poses, observed_velocity, target_velocity, start_index, name = ntu[random.randint(0, len(ntu))]
   observed_poses    = observed_poses.unsqueeze(0).to(args.device,    dtype = torch.float32)
   observed_velocity = observed_velocity.unsqueeze(0).to(args.device, dtype = torch.float32)
   target_poses      = target_poses.unsqueeze(0).to(args.device,      dtype = torch.float32)
   target_velocity   = target_velocity.unsqueeze(0).to(args.device,   dtype = torch.float32)

   observed_poses   = observed_poses.contiguous().view(1, 7, -1, 3)[:, :, :, :2]
   observed_poses   = observed_poses.contiguous().view(1, 7, -1) 

   predicted_velocity, _, _  = model(observed_velocity, args.num_testing_frame)

   predicted_poses = speed_to_pos(observed_poses = observed_poses, prediction_velocities = predicted_velocity, device = args.device)

   fig, axes = plt.subplots(2, args.num_testing_frame, figsize = (24, 8))  

   videos_real = []
   videos_pred = [] 

   pck_prediction = get_ntu_predictions_dict()

   observed_images, target_images = NTU.get_video_frames(name, start_index, args.num_training_frame, args.num_testing_frame)
   
   for index in range(0, args.num_testing_frame):     
      curr_target_image = target_images[index, :, :, :]
      curr_target_image = np.ascontiguousarray(torch2numpy(curr_target_image))

      curr_target_pose  = target_poses[:, index, :].squeeze(0)     
      curr_target_pose  = curr_target_pose.contiguous().view(13, 3)[:, :2]
      
      curr_predict_pose = predicted_poses[:, index, :].squeeze(0) 
      curr_predict_pose = curr_predict_pose.contiguous().view(13, 2) 

      pck_prediction = pck_joints(curr_predict_pose, curr_target_pose, pck_prediction, get_pennaction_joint_names(), alpha = 0.05)

      
      target_annotated_image = draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy()  * pennaction.box_size)
      axes[0, index].imshow(target_annotated_image)
      target_predicted_image = draw_skeleton(curr_target_image, curr_predict_pose.detach().cpu().numpy() * pennaction.box_size)
      axes[1, index].imshow(target_predicted_image)
   
     
      videos_real.append(target_annotated_image)
      videos_pred.append(target_predicted_image)
      
      acc = get_acc(pck_prediction)
      print(acc)
     
   plt.tight_layout()
   plt.show() 
   

REPRESENTATION_INFERENCE = { 
   'coordinates'      : inference_by_coordinates, 
   'parent_relatives' : inference_by_parent_relative,
   'velocity'         : inference_by_velocity,  
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',                                 type = str,   required = True)
    parser.add_argument('--dataset',            default = 'ntu',        type = str,   required = False)
    parser.add_argument('--num_joints',         default = 25,           type = int,   required = False)
    parser.add_argument('--num_training_frame', default = 15,           type = int,   required = False)
    parser.add_argument('--num_testing_frame',  default = 15,           type = int,   required = False) 
    parser.add_argument('--learning_rate',      default = 1e-4,         type = float, required = False)
    parser.add_argument('--device',             default = 'cuda',       type = str,   required = False) 
    parser.add_argument('--model' ,             default = 'lstm',       type = str,   required = True) 
    parser.add_argument('--coordinates',        default = False,        type = bool,  required = False)
    parser.add_argument('--velocity',           default = False,        type = bool,  required = False)
    parser.add_argument('--decouple_velocity',  default = False,        type = bool,  required = False)
    parser.add_argument('--parent_relative',    default = False,        type = bool,  required = False) 
    parser.add_argument('--root_relative',      default = False,        type = bool,  required = False)
    parser.add_argument('--pck_alpha',          default = 0.05,         type = float, required = False)
    parser.add_argument('--boxsize',            default = 640,          type = float, required = False)
    parser.add_argument('--dropout',            default = 0.2,          type = float, required = False)

    # MLP 
    parser.add_argument('--mlp_hidden_layers',  default = [1024, 2048, 1024], type = list[int], required = False) 
    
    # LSTM 
    parser.add_argument('--lstm_embedding_dim', default = 2048, type = int, required = False) 
    parser.add_argument('--lstm_hidden_dim'   , default = 256,  type = int, required = False)
    parser.add_argument('--lstm_num_layers',    default = 2,    type = int, required = False) 

    # MOTION VAE 
    parser.add_argument('--motion_vae_hidden_dim',       default = 512, type = int, required = False) 
    parser.add_argument('--motion_vae_latent_dim',       default = 128, type = int, required = False) 
    parser.add_argument('--motion_vae_encoding_layers',  default = 2,   type = int, required = False) 
    parser.add_argument('--motion_vae_decoding_layers',  default = 1,   type = int, required = False) 

    # DECOUPLE MOTION VAE
    parser.add_argument('--decouple_motion_vae_hidden_dim',       default = 32,  type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_latent_dim',       default = 16,  type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_encoding_layers',  default = 2,   type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_decoding_layers',  default = 1,   type = int, required = False) 
    parser.add_argument('--decouple_lstm_embedding_dim',          default = 8,   type = int, required = False)
    parser.add_argument('--decouple_lstm_hidden_dim',             default = 16,  type = int, required = False)
    parser.add_argument('--decouple_lstm_num_layers',             default = 1,   type = int, required = False)
   
    args = parser.parse_args()
    
    assert os.path.exists(args.checkpoint), "checkpoint does not exist"

    model, criterion = get_model(args)
    
    model_dict = torch.load(args.checkpoint, weights_only = False)  
    model.load_state_dict(model_dict['model']) 
    model.to(args.device) 

    ntu_dataset = NTU(
      num_training_frame = args.num_training_frame, 
      num_testing_frame  = args.num_testing_frame, 
      box_size           = args.box_size,
      num_joints         = args.num_joints, 
      train              = False, 
      root_relative      = args.root_relative, 
      parent_relative    = args.parent_relative, 
      velocity           = args.velocity or args.decouple_velocity,
      coordinates        = args.coordinates, 
    )
    inference_func = None 

    if args.parent_relative : 
      inference_func = REPRESENTATION_INFERENCE['parent_relatives']
    elif args.coordinates : 
      inference_func = REPRESENTATION_INFERENCE['coordinates']
    elif args.velocity : 
      inference_func = REPRESENTATION_INFERENCE['velocity']
    else : 
       raise NotImplementedError('Not implemented for now')
    
    inference_func(args, model, criterion, ntu_dataset)
