import random 
import torch

if __name__ == '__main__':
    
   print(torch.load('transformer_checkpoint/models/ckpt_epoch_0149.pt', weights_only = False, map_location = "cuda:0")['model'])

   
   # model = get_lstm_model(
   #    pose_dim = 16, 
   #    joint_dim = 3, 
   #    embedding_dim = 2048, 
   #    hidden_dim = 256, 
   #    num_layers = 2, 
   #    residual_connection = True, 
   #    dropout = 0.2, 
   #    device = 'cuda', 
   # )



   model.load_state_dict(model_dict['model'])
   model = model.to('cuda')
   model.eval()

   enc_inp = curr_observed.unsqueeze(0)[:, :, 3:].to('cuda')
   dec_inp = curr_padded.unsqueeze(0)[:, :, 3:].to('cuda')
   out = model(enc_inp, dec_inp)[-1].detach().cpu()
   print(np.std(out.numpy(), axis=1).mean())
   

   pad_root = torch.zeros((out.shape[0], out.shape[1], 3))
   curr_predicted = torch.cat([pad_root, out], dim = -1)
   

   # dct, idct = compute_dct_matrix(num = 100)
   
   # n_channels = 20

   # curr_observed_dct = torch.matmul(dct[:n_channels], curr_observed) # transform into frequency spaces 
   # curr_observed_dct = torch.matmul(idct[:,:n_channels], curr_observed_dct) # revert back 

   # dct, idct = compute_dct_matrix(num = 50)

   # curr_targeted_dct = torch.matmul(dct[:n_channels], curr_targeted) # transform into frequency spaces 
   # curr_targeted_dct = torch.matmul(idct[:,:n_channels], curr_targeted_dct) # revert back
   # 

   curr_targeted  = dataset.denormalize_data(curr_targeted)
   curr_observed  = dataset.denormalize_data(curr_observed)
   curr_predicted = dataset.denormalize_data(curr_predicted)
   curr_padded    = dataset.denormalize_data(curr_padded)



   

   animate_h36m_pose_3d(curr_observed.reshape(-1, 17, 3), curr_predicted.reshape(-1, 17, 3), curr_targeted.reshape(-1, 17, 3), fps = 10)



   # # model = get_diffusion_transformer_model().to('cuda')

   # # dataset = NTU3D(num_training_frame = 50, num_testing_frame = 50)
   # # observed_act_seqs, targeted_act_seqs,  _, name = dataset[random.randint(0, len(dataset))]


   # # dct, idct = compute_dct_matrix(num = 50)
   # # print(name)
   
   # # n_channels = 20

   # # observed_act_seqs_2 = torch.matmul(dct[:n_channels], observed_act_seqs) # transform into frequency spaces 
   # # observed_act_seqs_2 = torch.matmul(idct[:,:n_channels], observed_act_seqs_2) # revert back 
   # # observed_poses = reconstruct_pose(
   # #    input_sequences = observed_act_seqs_2.unsqueeze(0),
   # #    mean = dataset.mean, 
   # #    std  = dataset.std, 
   # #    frame_input_h = 1080, 
   # #    frame_input_w = 1920, 
   # #    root_mean = dataset.root_mean, 
   # #    root_std  = dataset.root_std, 
   # #    joint_dim = 3, 
   # # )
   # # animate_single_poses3d(observed_poses.detach().cpu().numpy().reshape(-1, 17, 3), name[4:8], fps = 30, output_path = 'sandbox_vis_3d_dct.gif')



