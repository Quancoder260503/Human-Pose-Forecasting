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


class AutoencoderRunner(Runner):
    def __init__(self, 
            pck_dict_fn, 
            rmse_dict_fn,
            args, 
            model_checkpoint, 
            eval, 
        ):
        super(AutoencoderRunner, self).__init__(
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

    def prepare_dataset(self): 
        return NTU2D if self.args.joint_dim == 2 else NTU3D
    
    def process_per_epoch(self, epoch):
        train_losses = []
        valid_losses = [] 
        train_per_joint_rmse  = []
        valid_per_joint_rmse  = []

        train_total_pck = self.pck_dict_fn()
        valid_total_pck = self.pck_dict_fn()

        self.model_a.train()
        
        for batch_idx, (observed_act_seqs, _, _, _) in enumerate(self.train_loader):
            observed_act_seqs = observed_act_seqs.to(self.args.device, dtype = torch.float32)

            batch_size, prediction_len = observed_act_seqs.shape[0], observed_act_seqs.shape[1]

            sample_indices = torch.randperm(batch_size * prediction_len)[: self.args.autoencoder_batch_size]

            observed_act_seqs = observed_act_seqs.contiguous().view(batch_size * prediction_len, -1)[sample_indices]

            predicted_act_seqs, (loss_a, _) = self.predict_autoencoder(observed_act_seqs, epoch = epoch)
            
            train_losses.append(loss_a.item())

            self.optimizer_a.zero_grad()
            loss_a.backward()

            if self.args.max_norm > 0 : 
                clip_grad_norm_(self.model_a.parameters(), self.args.max_norm)
                
            self.optimizer_a.step()
            self.lr_scheduler_a.step()

            
            if isinstance(predicted_act_seqs, list): 
                predicted_act_seqs = predicted_act_seqs[-1]

            predicted_act_seqs = predicted_act_seqs.contiguous().view(self.args.autoencoder_batch_size, -1, self.args.joint_dim)

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

            observed_poses  = reconstruct_pose(
                input_sequences        = observed_act_seqs, 
                mean                   = self.mean, 
                std                    = self.std,   
                root_mean              = self.root_mean, 
                root_std               = self.root_std, 
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w,
                joint_dim              = self.args.joint_dim,  
            )

            predictions = compute_joints_metric(
                pred      = predicted_poses.contiguous().view(batch_size, prediction_len, -1, self.args.joint_dim),  
                gt        = observed_poses.contiguous().view(batch_size, prediction_len,  -1, self.args.joint_dim),  
                time_step = prediction_len, 
                dataset   = self.args.dataset, 
                alpha     = self.args.pck_alpha, 
            )


            train_per_joint_rmse.append(calc_mpjpe(predicted_poses, observed_poses, self.args.num_joints))

            for metric in predictions:
                for key in predictions[metric]:
                    train_total_pck[metric][key] += predictions[metric][key]

            if batch_idx % self.args.print_step == 0 :
                acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{self.args.step_per_epoch} |'
                    f'Loss = {np.mean(train_losses):.6f} |' 
                    f'PJRMSE  = {np.mean(train_per_joint_rmse)} |' 
                    f'PCK accuracy = {acc:.4f}'
                )
            
            if batch_idx == self.args.step_per_epoch : 
                break 
        

        self.model_a.eval()
        with torch.no_grad():
            for _, (observed_act_seqs, _, _, _)  in enumerate(self.val_loader):
                observed_act_seqs = observed_act_seqs.to(self.args.device, dtype = torch.float32)
                batch_size, prediction_len = observed_act_seqs.shape[0], observed_act_seqs.shape[1]
                sample_indices = torch.randperm(batch_size * prediction_len)[: self.args.autoencoder_batch_size]
                observed_act_seqs = observed_act_seqs.contiguous().view(batch_size * prediction_len, -1)[sample_indices]
                predicted_act_seqs, (loss_a, _) = self.predict_autoencoder(observed_act_seqs, epoch = epoch)
                valid_losses.append(loss_a.item())

            
                if isinstance(predicted_act_seqs, list): 
                    predicted_act_seqs = predicted_act_seqs[-1]

                predicted_act_seqs = predicted_act_seqs.contiguous().view(self.args.autoencoder_batch_size, -1, self.args.joint_dim)

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

                
                observed_poses  = reconstruct_pose(
                    input_sequences        = observed_act_seqs, 
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
                    ground_truth_sequences = observed_poses.contiguous().view(batch_size, prediction_len,  -1, self.args.joint_dim),  
                    timesteps              = prediction_len, 
                    dataset                = self.args.dataset, 
                    alpha                  = self.args.pck_alpha, 
                )  

                valid_per_joint_rmse.append(calc_mpjpe(predicted_poses, observed_poses, self.args.num_joints))
                
                for metric in predictions:
                    for key in predictions[metric]:
                        valid_total_pck[metric][key] += predictions[metric][key]

        return np.mean(train_losses), np.mean(train_per_joint_rmse), train_total_pck, np.mean(valid_losses), np.mean(valid_per_joint_rmse), valid_total_pck
    

    def evaluation(self):
        self.model_a.eval()
        with torch.no_grad():
            test_losses         = [] 
            test_per_joint_rmse = []
            test_total_pck      = self.pck_dict_fn()

            for index, (observed_act_seqs, _, _, _) in enumerate(self.test_loader): 
                observed_act_seqs = observed_act_seqs.to(self.args.device, dtype = torch.float32)

                batch_size, prediction_len = observed_act_seqs.shape[0], observed_act_seqs.shape[1]

                sample_indices = torch.randperm(batch_size * prediction_len)[: self.args.autoencoder_batch_size]

                observed_act_seqs = observed_act_seqs.contiguous().view(batch_size * prediction_len, -1)[sample_indices]
    

                predicted_act_seqs, (loss_a, _) = self.predict_autoencoder(observed_act_seqs)
                
                test_losses.append(loss_a.item())

                if isinstance(predicted_act_seqs, list): 
                    predicted_act_seqs = predicted_act_seqs[-1]

                predicted_act_seqs = predicted_act_seqs.contiguous().view(self.args.autoencoder_batch_size, -1, self.args.joint_dim)


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

                observed_poses  = reconstruct_pose(
                    input_sequences        = observed_act_seqs, 
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
                    ground_truth_sequences = observed_poses.contiguous().view(batch_size, prediction_len,  -1, self.args.joint_dim),  
                    timesteps = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha, 
                )         


                test_per_joint_rmse.append(calc_mpjpe(predicted_poses, observed_poses, self.args.num_joints))

                for metric in predictions:
                    for key in predictions[metric]:
                        test_total_pck[metric][key] += predictions[metric][key]


                if index == self.args.num_testing_samples : 
                    break 

            test_per_joint_rmse = np.stack(test_per_joint_rmse, axis = 0).mean(axis = 0)  # (prediction_len,)

            print(
                f"Test Loss: {np.mean(test_losses):.4f} | "
                f"Per Joint RMSE: {test_per_joint_rmse.mean():.4f}"
            )



    def inference(self):
        self.model_a.eval()
        if self.model_b is not None : 
            self.model_b.eval()
        with torch.no_grad():
            observed_act_seqs, _, start_index, name = next(iter(self.val_loader))
            
            name = name[0]

            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)

            predicted_act_seqs, _ = self.predict_autoencoder(observed_act_seqs)

            if isinstance(predicted_act_seqs, list): 
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
                target_poses    = None, 
                observed_poses  = observed_poses.reshape(-1,  self.args.num_joints, self.args.joint_dim).detach().cpu().numpy(), 
                start_index     = start_index, 
                name            = name, 
            )

    def predict_autoencoder(self, pose_sequences, epoch = None):
        prediction = self.model_a(pose_sequences) 
        loss = self._compute_loss(self.criterion_a, self.args.loss_fn_a, prediction, pose_sequences)
        return prediction, loss



      
      
      