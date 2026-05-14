import matplotlib.pyplot as plt
import numpy as np
import random 
from data.DatasetUtil   import *
from runner.RunnerUtil  import *
from runner.Runner      import * 
from torch.nn.utils     import clip_grad_norm_

"""
Comparison for root relative is on absolute coordinates (without norm), hence  
make sure to set pck threshold as pixel distance
"""

class NeuralNetworkRunner(Runner):
    def __init__(self, 
            pck_dict_fn, 
            rmse_dict_fn,
            args, 
            model_checkpoint, 
            eval, 
        ):
        super(NeuralNetworkRunner, self).__init__(
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
        """
        :param model        : model to train 
        :param train_loader : train dataloader 
        :param val_loader   : validation dataloader
        :param optimizer    : set default to ADAM 
        :param criterion    : loss function
        :param epoch        : current epoch
        """

        train_losses = []
        valid_losses = [] 

        train_total_pck = self.pck_dict_fn()
        valid_total_pck = self.pck_dict_fn()

        """
        :param model        : model to train 
        :param train_loader : train dataloader 
        :param val_loader   : validation dataloader
        :param optimizer    : set default to ADAM 
        :param criterion    : loss function
        :param epoch        : current epoch
        :param device       : 'cuda' 
        """

        train_losses = []
        valid_losses = [] 
        train_per_joint_rmse  = []
        valid_per_joint_rmse  = []


        train_total_pck = self.pck_dict_fn()
        valid_total_pck = self.pck_dict_fn()

    
        self.model_a.train()
        if self.model_b is not None : 
            self.model_b.train()

        for batch_idx, (observed_act_seqs, targeted_act_seqs,  _, _) in enumerate(self.train_loader):
            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)


            batch_size            = targeted_act_seqs.shape[0]
            prediction_len        = targeted_act_seqs.shape[1]
            
            loss = 0.0 

            if self.args.pose_decoupling : 
                predicted_act_seqs, (loss_a, loss_b) = self.predict(observed_act_seqs, targeted_act_seqs, epoch = epoch)
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
                predicted_act_seqs, (loss_a, _) = self.predict(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                loss = loss_a.item() 

                train_losses.append(loss)

                self.optimizer_a.zero_grad()
                loss_a.backward()

                if self.args.max_norm > 0 : 
                    clip_grad_norm_(self.model_a.parameters(), self.args.max_norm)
             
                self.optimizer_a.step()
                self.lr_scheduler_a.step()


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

            train_per_joint_rmse.append(calc_mpjpe(predicted_poses, targeted_poses, self.args.num_joints, self.args.joint_dim))

            for metric in predictions:
                for key in predictions[metric]:
                    train_total_pck[metric][key] += predictions[metric][key]

            if batch_idx % self.args.print_step == 0:
                acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
            
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{self.args.step_per_epoch} |'
                    f'Loss = {np.mean(train_losses):.6f} |' 
                    f'PJRMSE  = {np.mean(train_per_joint_rmse)} |' 
                    f'PCK accuracy = {acc:.4f}'
                )
            
            if batch_idx == self.args.step_per_epoch : 
                break 


        self.model_a.eval()
        if self.model_b is not None : 
            self.model_b.eval()

        with torch.no_grad():
            for batch_idx, (observed_act_seqs, targeted_act_seqs,  _, _) in enumerate(self.val_loader):
                observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
                targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

                batch_size            = targeted_act_seqs.shape[0]
                prediction_len        = targeted_act_seqs.shape[1]
            
                loss = 0.0 
                if self.args.pose_decoupling: 
                    predicted_act_seqs, (loss_a, loss_b) = self.predict(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                    loss = loss_a.item() + loss_b.item()
                else : 
                    predicted_act_seqs, (loss_a, _) = self.predict(observed_act_seqs, targeted_act_seqs, epoch = epoch)
                    loss = loss_a.item() 

                valid_losses.append(loss)

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


                valid_per_joint_rmse.append(calc_mpjpe(predicted_poses, targeted_poses, self.args.num_joints, self.args.joint_dim))

                for metric in predictions:
                    for key in predictions[metric]:
                        valid_total_pck[metric][key] += predictions[metric][key]
                
    
        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)    

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

                if self.args.pose_decoupling: 
                    predicted_act_seqs, (loss_a, loss_b) = self.predict(observed_act_seqs, targeted_act_seqs)
                    loss = loss_a.item() + loss_b.item()
                else : 
                    predicted_act_seqs, (loss_a, _) = self.predict(observed_act_seqs, targeted_act_seqs)
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
            predicted_act_seqs, _ = self.predict(observed_act_seqs, targeted_act_seqs)

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

    

    def predict(self, src_seq, tgt_seq, epoch = None):
        if self.args.pose_decoupling : 
            batch_size, src_length, _ = src_seq.shape
            _, tgt_length, _ = tgt_seq.shape
            src_poses = src_seq.contiguous().view(batch_size, src_length, -1, self.args.joint_dim)
            tgt_poses = tgt_seq.contiguous().view(batch_size, tgt_length, -1, self.args.joint_dim)
            
            # Assumption that source root is at 0 
            src_roots, src_poses  = src_poses[:, :, 0, :], src_poses[:, :, 1:, :]
            tgt_roots, tgt_poses  = tgt_poses[:, :, 0, :], tgt_poses[:, :, 1:, :]

            src_poses = src_poses.contiguous().view(batch_size, src_length, -1)
            tgt_poses = tgt_poses.contiguous().view(batch_size, tgt_length, -1)

            if self.model_a_variational:
                prediction_poses, mean, log_var = self._forward(self.model_a, self.args.model_a, src_poses, tgt_poses)
                beta = KL_annealing(epoch, self.args.num_epochs, self.args.kl_annealing_schedule, self.args.kl_start_beta, self.args.kl_end_beta) if epoch is not None else self.args.kl_end_beta
                loss_poses = self._compute_loss(self.criterion_a, self.args.loss_fn_a, prediction_poses, tgt_poses, mean, log_var, beta, True)
            else :
                prediction_poses = self._forward(self.model_a, self.args.model_a, src_poses, tgt_poses)
                loss_poses = self._compute_loss(self.criterion_a, self.args.loss_fn_a, prediction_poses, tgt_poses)
            
            if self.model_b_variational:
                prediction_roots, mean, log_var = self._forward(self.model_b, self.args.model_b, src_roots, tgt_roots)
                beta = KL_annealing(epoch, self.args.num_epochs, self.args.kl_annealing_schedule, self.args.kl_start_beta, self.args.kl_end_beta) if epoch is not None else self.args.kl_end_beta
                loss_roots = self._compute_loss(self.criterion_b, self.args.loss_fn_b, prediction_roots, tgt_roots, mean, log_var, beta, True)
            else :
                prediction_roots = self._forward(self.model_b, self.args.model_b, src_roots, tgt_roots)
                loss_roots = self._compute_loss(self.criterion_b, self.args.loss_fn_b, prediction_roots, tgt_roots)

            prediction_poses = prediction_poses.contiguous().view(batch_size, tgt_length, -1, self.args.joint_dim)
            prediction_roots = prediction_roots.unsqueeze(2)

            prediction = torch.cat([prediction_roots, prediction_poses], dim = 2)
            prediction = prediction.contiguous().view(batch_size, tgt_length, -1)
            return prediction, (loss_poses, loss_roots)

        else : 
            if self.model_a_variational:
                prediction, mean, log_var = self._forward(self.model_a, self.args.model_a, src_seq, tgt_seq)
                beta = KL_annealing(epoch, self.args.num_epochs, self.args.kl_annealing_schedule, self.args.kl_start_beta, self.args.kl_end_beta) if epoch is not None else self.args.kl_end_beta
                loss    = self._compute_loss(self.criterion_a, self.args.loss_fn_a, prediction, tgt_seq, mean, log_var, beta, True)
            else :
                prediction = self._forward(self.model_a, self.args.model_a, src_seq, tgt_seq)
                loss    = self._compute_loss(self.criterion_a, self.args.loss_fn_a, prediction, tgt_seq)
    
            if self._last_attn is not None:
                self_attn, cross_attn = self._last_attn
                return prediction, loss, self_attn, cross_attn
            return prediction, (loss, None) 

    
    

      
      
      