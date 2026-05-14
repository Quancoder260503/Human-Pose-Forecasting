from data.NTU import NTU3D
from data.Visualization import *
from runner.RunnerUtil  import *
from model.DiffusionTransformer import *
import random 

if __name__ == '__main__':

   #model = get_diffusion_transformer_model().to('cuda')

   dataset = NTU3D(num_training_frame = 100, num_testing_frame = 50)
   observed_act_seqs, targeted_act_seqs,  _, name = dataset[random.randint(0, 100)]


   dct, idct = compute_dct_matrix(num = 100)
   print(name)
   
   n_channels = 20

   observed_poses = reconstruct_pose(
      input_sequences = observed_act_seqs.unsqueeze(0),
      mean = dataset.mean, 
      std  = dataset.std, 
      frame_input_h = 1080, 
      frame_input_w = 1920, 
      root_mean = dataset.root_mean, 
      root_std  = dataset.root_std, 
      joint_dim = 3, 
   )
   animate_single_poses3d(observed_poses.detach().cpu().numpy().reshape(-1, 17, 3), name[4:8], fps = 30, output_path = 'sandbox_vis_3d.gif')

   observed_act_seqs = torch.matmul(dct[:n_channels], observed_act_seqs) # transform into frequency spaces 
   observed_act_seqs = torch.matmul(idct[:,:n_channels], observed_act_seqs) # revert back 
   observed_poses = reconstruct_pose(
      input_sequences = observed_act_seqs.unsqueeze(0),
      mean = dataset.mean, 
      std  = dataset.std, 
      frame_input_h = 1080, 
      frame_input_w = 1920, 
      root_mean = dataset.root_mean, 
      root_std  = dataset.root_std, 
      joint_dim = 3, 
   )
   animate_single_poses3d(observed_poses.detach().cpu().numpy().reshape(-1, 17, 3), name[4:8], fps = 30, output_path = 'sandbox_vis_3d_dct.gif')



