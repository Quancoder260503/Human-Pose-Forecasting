'''
Autoencoder runner used to construct the latent presentation of the observed pose sequences, 
this can be used as a signal that would be used to guide diffusion model 
'''

import matplotlib.pyplot as plt
import numpy as np
import random 
from data.DatasetUtil   import *
from runner.RunnerUtil  import *
from runner.Runner      import * 
from torch.nn.utils     import clip_grad_norm_
from model.GaussianDiffusion import * 
from model.EMA import *

class DiffusionRunner(Runner):
    def __init__(self, 
            pck_dict_fn, 
            rmse_dict_fn,
            args, 
            model_checkpoint, 
            eval, 
        ):
        super(DiffusionRunner, self).__init__(
            pck_dict_fn      = pck_dict_fn, 
            rmse_dict_fn     = rmse_dict_fn,  
            args             = args, 
            model_checkpoint = model_checkpoint, 
            eval             = eval, 
        )

        self.mean = torch.tensor(np.array(self.mean), device = self.args.device, dtype = torch.float32)
        self.std  = torch.tensor(np.array(self.std),  device = self.args.device, dtype = torch.float32)
        if self.root_mean is not None : 
            self.root_mean = torch.tensor(np.array(self.root_mean), device = self.args.device, dtype = torch.float32)
        if self.root_std is not None : 
            self.root_std  = torch.tensor(np.array(self.root_std),  device = self.args.device, dtype = torch.float32)

        self.dct, self.idct = compute_dct_matrix(num = self.args.num_training_frame + self.args.num_testing_frame)
        self.num_dct_channels = self.args.num_dct_channels

    
    def prepare_dataset(self): 
        return NTU2D if self.args.joint_dim == 2 else NTU3D
    

    def predict_diffusion(self, observed_sequences, target_sequences, epoch = None):
        # for this it is better to train, diffusion model not conditioned on the obsevered ones 
        if self.args.pose_decoupling is False : 
            batch_size = observed_sequences.shape[0]
            total_sequences = torch.cat([observed_sequences, target_sequences], dim = 1) # (B, observed_len + tgt_len, pose_dim)
            dct_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], total_sequences) 
            # padded the last observed_frames
            padded_part = observed_sequences[:, -1, :].clone().unsqueeze(1).repeat(1, self.args.num_testing_frame, 1)
            padded_sequences = torch.cat([observed_sequences, padded_part], dim = 1) 

            dct_padded_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], padded_sequences)

            timesteps = torch.randint(low  = 0, high = self.args.diffusion_num_time_steps, size = (batch_size // 2 + 1, ),)
            timesteps = timesteps.to(self.args.device)
            timesteps = torch.cat([timesteps, self.args.diffusion_num_time_steps - timesteps - 1], dim = 0)[: batch_size]

            loss = self.gaussian_diffusion.training_losses(
                model           = self.model_a, 
                x_start         = dct_sequences, 
                timesteps       = timesteps, 
                condition_guide = dct_padded_sequences, 
            )
            return loss, None
        
        else : 
            batch_size = observed_sequences.shape[0]

            observed_root_sequences = observed_sequences[:, :, : self.args.joint_dim]
            targeted_root_sequences = target_sequences[:, :, : self.args.joint_dim]

            observed_sequences = observed_sequences[:, :, self.args.joint_dim : ]
            target_sequences   = target_sequences[:, :, self.args.joint_dim : ]


            total_sequences = torch.cat([observed_sequences, target_sequences], dim = 1) # (B, observed_len + tgt_len, pose_dim)
            dct_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], total_sequences) 
            # padded the last observed_frames
            padded_part = observed_sequences[:, -1, :].clone().unsqueeze(1).repeat(1, self.args.num_testing_frame, 1)
            padded_sequences = torch.cat([observed_sequences, padded_part], dim = 1) 

            dct_padded_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], padded_sequences)

            timesteps = torch.randint(low  = 0, high = self.args.diffusion_num_time_steps, size = (batch_size // 2 + 1, ),)
            timesteps = timesteps.to(self.args.device)
            timesteps = torch.cat([timesteps, self.args.diffusion_num_time_steps - timesteps - 1], dim = 0)[: batch_size]

            loss_a = self.gaussian_diffusion.training_losses(
                model           = self.model_a, 
                x_start         = dct_sequences, 
                timesteps       = timesteps, 
                condition_guide = dct_padded_sequences, 
            )

            if self.model_b_variational:
                prediction_roots, mean, log_var = self._forward(self.model_b, self.args.model_b, observed_root_sequences, prediction_len = self.args.num_testing_frame)
                beta = KL_annealing(epoch, self.args.num_epochs, self.args.kl_annealing_schedule, self.args.kl_start_beta, self.args.kl_end_beta) if epoch is not None else self.args.kl_end_beta
                loss_roots = self._compute_loss(self.criterion_b, self.args.loss_fn_b, prediction_roots, targeted_root_sequences, mean, log_var, beta, True)
            else :
                prediction_roots = self._forward(self.model_b, self.args.model_b, observed_root_sequences, )
                loss_roots = self._compute_loss(self.criterion_b, self.args.loss_fn_b, prediction_roots, targeted_root_sequences)
            
            return loss_a, loss_roots


    def sampling_diffusion(self, observed_sequences): 
        # USE DDIM to guide reverse sampling process from Gaussian 
        if self.args.pose_decoupling is False : 
            padded_part = observed_sequences[:, -1, :].clone().unsqueeze(1).repeat(1, self.args.num_testing_frame, 1)
            padded_sequences = torch.cat([observed_sequences, padded_part], dim = 1)

            masks = torch.zeros_like(padded_sequences) 
            for i in range(self.args.num_training_frame):
                masks[:, i, :] = 1

            dct_padded_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], padded_sequences)

            return self.gaussian_diffusion.ddim_sample_loop(
                model             = self.model_a, 
                shape             = dct_padded_sequences.shape, 
                device            = self.args.device,
                dct               = self.dct, 
                idct              = self.idct, 
                dct_num_channels  = self.num_dct_channels,  
                conditional_guide = dct_padded_sequences, 
                masks             = masks, 
            )[0]
        
        else :

            observed_root_sequences = observed_sequences[:, :, : self.args.joint_dim]
            observed_sequences = observed_sequences[:, :, self.args.joint_dim : ]



            # padded the last observed_frames
            padded_part = observed_sequences[:, -1, :].clone().unsqueeze(1).repeat(1, self.args.num_testing_frame, 1)
            padded_sequences = torch.cat([observed_sequences, padded_part], dim = 1) 

            masks = torch.zeros_like(padded_sequences) 
            for i in range(self.args.num_training_frame):
                masks[:, i, :] = 1

            dct_padded_sequences = torch.matmul(self.dct[: self.num_dct_channels, :], padded_sequences)

            predicted_sequences   = self.gaussian_diffusion.ddim_sample_loop(
                model             = self.model_a, 
                shape             = dct_padded_sequences.shape, 
                device            = self.args.device,
                dct               = self.dct, 
                idct              = self.idct, 
                dct_num_channels  = self.num_dct_channels,  
                conditional_guide = dct_padded_sequences, 
                masks             = masks, 
            )[0]

            
            if self.model_b_variational:
                prediction_roots, _, _ = self._forward(self.model_b, self.args.model_b, observed_root_sequences, self.args.num_testing_frame)
            else :
                prediction_roots = self._forward(self.model_b, self.args.model_b, observed_root_sequences, self.args.num_testing_frame)
            
            output = torch.cat([prediction_roots, predicted_sequences], dim = -1)
            return output 
     
    
    # For now, only support masked diffusion 
    def process_per_epoch(self, epoch):
        """
        :param model        : model to train 
        :param train_loader : train dataloader 
        :param val_loader   : validation dataloader
        :param optimizer    : set default to ADAM 
        :param criterion    : loss function
        :param epoch        : current epoch
        """

        train_losses = []
        validation_losses = [] 

        self.model_a.train()
        if self.model_b is not None: 
            self.model_b.train()

        for batch_idx, (observed_act_seqs, targeted_act_seqs,  _, _) in enumerate(self.train_loader):
            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

            batch_size            = targeted_act_seqs.shape[0]
            prediction_len        = targeted_act_seqs.shape[1]
            
            loss = 0.0 

            if self.args.pose_decoupling : 
                loss_a, loss_b = self.predict_diffusion(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                loss = loss_a.item() + loss_b.item()
            
                train_losses.append(loss)

                self.optimizer_a.zero_grad()
                self.optimizer_b.zero_grad()
            
                loss_a.backward()
                loss_b.backward()

                if self.args.max_norm > 0 : 
                    clip_grad_norm_(self.model_a.parameters(), self.args.max_norm)
                    clip_grad_norm_(self.model_b.parameters(), self.args.max_norm)
            
                self.optimizer_a.step()
                self.optimizer_b.step()
                self.lr_scheduler_a.step()
                self.lr_scheduler_b.step()

            else : 
                loss, _  = self.predict_diffusion(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                loss = loss_a.item() 

                train_losses.append(loss)

                self.optimizer_a.zero_grad()
                loss_a.backward()

                if self.args.max_norm > 0 : 
                    clip_grad_norm_(self.model_a.parameters(), self.args.max_norm)
             
                self.optimizer_a.step()
                self.lr_scheduler_a.step()
            
            if batch_idx == self.args.step_per_epoch : 
                break 

        
        self.model_a.eval()
        if self.model_b is not None : 
            self.model_b.eval()

        validation_per_joint_rmse = []
        validation_total_pck = self.pck_dict_fn()
        
        with torch.no_grad():
            for batch_idx, (observed_act_seqs, targeted_act_seqs,  _, _) in enumerate(self.val_loader):
                observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
                targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

                batch_size            = targeted_act_seqs.shape[0]
                prediction_len        = targeted_act_seqs.shape[1]
            
                loss = 0.0 
                if self.args.pose_decoupling: 
                    loss_a, loss_b = self.predict_diffusion(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                    predicted_act_seqs = self.sampling_diffusion(observed_act_seqs, epoch = epoch)
                    loss = loss_a.item() + loss_b.item()
                else : 
                    loss_a, _ = self.predict_diffusion(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                    predicted_act_seqs = self.sampling_diffusion(observed_act_seqs, epoch = epoch)
                    loss = loss_a.item()

                validation_losses.append(loss)

                predicted_poses  = reconstruct_pose(
                    input_sequences        = predicted_act_seqs, 
                    mean                   = self.mean, 
                    std                    = self.std,   
                    root_mean              = self.root_mean, 
                    root_std               = self.root_std, 
                    frame_input_h          = self.args.frame_input_h, 
                    frame_input_w          = self.args.frame_input_w,
                    joint_dim              = self.args.joint_dim,  
                )

                targeted_poses = reconstruct_pose(
                    input_sequences        = targeted_act_seqs, 
                    mean                   = self.mean, 
                    std                    = self.std,   
                    root_mean              = self.root_mean, 
                    root_std               = self.root_std, 
                    frame_input_h          = self.args.frame_input_h, 
                    frame_input_w          = self.args.frame_input_w,
                    joint_dim              = self.args.joint_dim,  
                )

                predictions = compute_joints_metric(
                    prediction_sequences   = predicted_poses.contiguous().view(batch_size, prediction_len, -1, self.args.joint_dim),  
                    ground_truth_sequences = targeted_poses.contiguous().view(batch_size, prediction_len,  -1, self.args.joint_dim),  
                    timesteps = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha,  
                )

                validation_per_joint_rmse.append(calc_mpjpe(predicted_poses, targeted_poses, self.args.num_joints, self.args.joint_dim))

                for metric in predictions:
                    for key in predictions[metric]:
                        validation_total_pck[metric][key] += predictions[metric][key]
            
            return np.mean(train_losses), np.mean(validation_losses), np.mean(validation_per_joint_rmse), validation_total_pck 


    def evaluation(self):
        test_losses         = [] 
        test_per_joint_rmse = []
        test_total_pck      = self.pck_dict_fn()
        
        self.model_a.eval()
        if self.model_b is not None : 
            self.model_b.eval()
        with torch.no_grad():
            for _, (observed_act_seqs, targeted_act_seqs, _, _) in enumerate(self.test_loader): 
                observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
                targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)


                batch_size            = targeted_act_seqs.shape[0]
                prediction_len        = targeted_act_seqs.shape[1]

                loss = 0.0 

                loss = 0.0 
                if self.args.pose_decoupling: 
                    loss_a, loss_b = self.predict_diffusion(observed_act_seqs, targeted_act_seqs)
                    predicted_act_seqs = self.sampling_diffusion(observed_act_seqs)
                    loss = loss_a.item() + loss_b.item()
                else : 
                    loss_a, _ = self.predict_diffusion(observed_act_seqs, targeted_act_seqs)
                    predicted_act_seqs = self.sampling_diffusion(observed_act_seqs)
                    loss = loss_a.item()

                test_losses.append(loss)

                if not isinstance(predicted_act_seqs, list): 
                    predicted_act_seqs = [predicted_act_seqs]

                predicted_act_seqs = predicted_act_seqs[-1]

                predicted_poses  = reconstruct_pose(
                    input_sequences        = predicted_act_seqs, 
                    mean                   = self.mean, 
                    std                    = self.std,   
                    root_mean              = self.root_mean, 
                    root_std               = self.root_std, 
                    frame_input_h          = self.args.frame_input_h, 
                    frame_input_w          = self.args.frame_input_w,
                    joint_dim              = self.args.joint_dim,  
                )

                targeted_poses  = reconstruct_pose(
                    input_sequences        = targeted_act_seqs, 
                    mean                   = self.mean, 
                    std                    = self.std,   
                    root_mean              = self.root_mean, 
                    root_std               = self.root_std, 
                    frame_input_h          = self.args.frame_input_h, 
                    frame_input_w          = self.args.frame_input_w,
                    joint_dim              = self.args.joint_dim,  
                )

                predictions = compute_joints_metric(
                    prediction_sequences   = predicted_poses.contiguous().view(batch_size, prediction_len, -1, self.args.joint_dim),  
                    ground_truth_sequences = targeted_poses.contiguous().view(batch_size, prediction_len,  -1, self.args.joint_dim),  
                    timesteps = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha,  
                )


                test_per_joint_rmse.append(calc_mpjpe_per_timestep(predicted_poses, targeted_poses, self.args.num_joints, self.args.joint_dim))
                
                for metric in predictions:
                    for key in predictions[metric]:
                        test_total_pck[metric][key] += predictions[metric][key]

        
            test_per_joint_rmse = np.stack(test_per_joint_rmse, axis = 0).mean(axis = 0)  # (prediction_len,)
            test_total_pck_acc  = get_acc(test_total_pck)  

            print(
                f"Test Loss: {np.mean(test_losses):.4f} | "
                f"PCK: {test_total_pck_acc['total']:.4f} | "
                f"Per Joint RMSE: {test_per_joint_rmse.mean():.4f}"
            )

            plot_rmse_over_horizon(test_per_joint_rmse, save_path = 'root_relative_rmse_horizon.png')

    def inference(self):
        self.model_a.eval()
        if self.model_b is not None : 
            self.model_b.eval()
        with torch.no_grad():
            observed_act_seqs, targeted_act_seqs, start_index, name = next(iter(self.val_loader))
            
            name = name[0]

            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

            if self.args.pose_decoupling: 
                loss_a, loss_b = self.predict_diffusion(observed_act_seqs, targeted_act_seqs)
                predicted_act_seqs = self.sampling_diffusion(observed_act_seqs)
                loss = loss_a.item() + loss_b.item()
            else : 
                loss_a, _ = self.predict_diffusion(observed_act_seqs, targeted_act_seqs)
                predicted_act_seqs = self.sampling_diffusion(observed_act_seqs)
                loss = loss_a.item()


            predicted_poses  = reconstruct_pose(
                input_sequences        = predicted_act_seqs, 
                mean                   = self.mean, 
                std                    = self.std,   
                root_mean              = self.root_mean, 
                root_std               = self.root_std, 
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w,
                joint_dim              = self.args.joint_dim,  
            )

            targeted_poses  = reconstruct_pose(
                input_sequences        = targeted_act_seqs, 
                mean                   = self.mean, 
                std                    = self.std,   
                root_mean              = self.root_mean, 
                root_std               = self.root_std, 
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w,
                joint_dim              = self.args.joint_dim,  
            )

            observed_poses = reconstruct_pose(   
                input_sequences        = observed_act_seqs, 
                mean                   = self.mean, 
                std                    = self.std,   
                root_mean              = self.root_mean, 
                root_std               = self.root_std, 
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w,
                joint_dim              = self.args.joint_dim,  

            )

            self.visualize_pose(
                predicted_poses = predicted_poses.reshape(-1, self.args.num_joints, self.args.joint_dim).detach().cpu().numpy(), 
                target_poses    = targeted_poses.reshape(-1,  self.args.num_joints, self.args.joint_dim).detach().cpu().numpy(), 
                observed_poses  = observed_poses.reshape(-1,  self.args.num_joints, self.args.joint_dim).detach().cpu().numpy(), 
                start_index     = start_index, 
                name            = name, 
            )