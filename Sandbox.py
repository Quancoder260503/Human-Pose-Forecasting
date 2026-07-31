import random 
import torch
from data.MultiDataset import *
from data.Visualization import *
from runner.SamplingUtils import *
from model.Deterministic.GraphConvNet import SimpleEncoder

if __name__ == '__main__':
   model = SimpleEncoder(
      n_nodes = 16, 
      input_features = 3, 
      hidden_dim = 512, 
      model_dim = 512, 
   ).to('cuda')
   inp = torch.randn((10, 20, 48)).cuda()
   out = model(inp)
   # dataset_class_ = DatasetH36M # temporary 
   # dataset_train = dataset_class_(
   #  mode = 'train', 
   #  num_observed_frames = 25, 
   #  num_predict_frames = 200, 
   #  actions = ['walking'], 
   #  dim_per_joints = 3, 
   # )
   # dct, idct = get_dct_matrix(225, False)
   # trajectory_np = next(dataset_train.iter_generator())  
   # trajectory_np = trajectory_np[0]                    

   # T, J, C = trajectory_np.shape
   # traj_flat = trajectory_np.reshape(T, -1)              
   # traj_dct  = np.matmul(dct[:20, :],  traj_flat)        
   # traj_idct = np.matmul(idct[:, :20], traj_dct)        
   # trajectory_np = traj_idct.reshape(T, J, C)      
   # trajectory_np[:, 1:, :] += trajectory_np[:, :1, :]    

   # animate_h36m_pose_3d(trajectory_np, dataset_train.skeleton.parents)