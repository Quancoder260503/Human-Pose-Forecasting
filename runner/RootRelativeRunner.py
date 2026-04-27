import matplotlib.pyplot as plt
import numpy as np
import random 
from data.DatasetUtil   import *
from runner.RunnerUtil  import *
from runner.Runner      import * 
from data.Visualization import * 
from torch.nn.utils     import clip_grad_norm_

"""
Comparison for root relative is on absolute coordinates (without norm), hence  
make sure to set pck threshold as pixel distance
"""

class RootRelativeRunner(Runner):
    def __init__(self, 
            pck_dict_fn, 
            rmse_dict_fn,
            args, 
            model_checkpoint, 
        ):
        super(RootRelativeRunner, self).__init__(
            pck_dict_fn      = pck_dict_fn, 
            rmse_dict_fn     = rmse_dict_fn,  
            args             = args, 
            model_checkpoint = model_checkpoint
        )
        self.mean = torch.tensor(np.array(self.mean), device = self.args.device, dtype = torch.float32)
        self.std  = torch.tensor(np.array(self.std),  device = self.args.device, dtype = torch.float32)
        print("Input represented as Root Relative")

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

    
        self.model.train()
        for batch_idx, (observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, _, _) in enumerate(self.train_loader):
            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

            observed_poses        = observed_poses.to(self.args.device, dtype = torch.float32)
            targeted_poses        = targeted_poses.to(self.args.device, dtype = torch.float32)

            batch_size            = targeted_poses.shape[0]
            prediction_len        = targeted_poses.shape[1]

                
            predicted_act_seqs, loss = self.get_predicted_n_loss(
                train        = True,
                src_seq      = observed_act_seqs, 
                tgt_seq      = targeted_act_seqs,
                epoch        = epoch,  
            )

            train_losses.append(loss.item())
                
            self.optimizer.zero_grad()
            loss.backward()

            if self.args.max_norm > 0 : 
                clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            if not isinstance(predicted_act_seqs, list): 
                predicted_act_seqs = [predicted_act_seqs]

            predicted_poses  = reconstruct_pose_from_root(
                root_vectors           = predicted_act_seqs[-1], 
                mean                   = self.mean, 
                std                    = self.std,   
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w, 
            )

            predictions = compute_joints_metric(
                pred      = predicted_poses.contiguous().view(batch_size, prediction_len,  -1, 2),  
                gt        = targeted_poses.contiguous().view(batch_size, prediction_len, -1, 2),  
                time_step = prediction_len, 
                dataset   = self.args.dataset, 
                alpha     = self.args.pck_alpha,  
            )

            train_per_joint_rmse.append(calc_rmse(predicted_poses, targeted_poses, self.args.num_joints))

            for metric in predictions:
                for key in predictions[metric]:
                    train_total_pck[metric][key] += predictions[metric][key]

            if batch_idx > 0 and batch_idx % self.args.print_step == 0:
                acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{self.args.step_per_epoch} |'
                      f'Loss = {np.mean(train_losses):.6f} |' 
                      f'PJRMSE  = {np.mean(train_per_joint_rmse)} |' 
                      f'PCK accuracy = {acc:.4f}'
                )
                
            if batch_idx == self.args.step_per_epoch : 
                break 

        self.model.eval()
        with torch.no_grad():
           for _, (observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, _, _) in enumerate(self.val_loader):        
            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

            observed_poses        = observed_poses.to(self.args.device, dtype = torch.float32)
            targeted_poses        = targeted_poses.to(self.args.device, dtype = torch.float32)

            batch_size            = targeted_poses.shape[0]
            prediction_len        = targeted_poses.shape[1]

            predicted_act_seqs, loss = self.get_predicted_n_loss(
                train        = False,
                src_seq      = observed_act_seqs, 
                tgt_seq      = targeted_act_seqs, 
                epoch        = epoch, 
            )
            
            valid_losses.append(loss.item())
            
            if not isinstance(predicted_act_seqs, list): 
                predicted_act_seqs = [predicted_act_seqs]

            predicted_poses  = reconstruct_pose_from_root(
                root_vectors           = predicted_act_seqs[-1], 
                mean                   = self.mean, 
                std                    = self.std,   
                frame_input_h          = self.args.frame_input_h, 
                frame_input_w          = self.args.frame_input_w, 
            )


            predictions = compute_joints_metric(
                pred      = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2),  
                gt        = targeted_poses.contiguous().view(batch_size, prediction_len, -1, 2),  
                time_step = prediction_len, 
                dataset   = self.args.dataset, 
                alpha     = self.args.pck_alpha,  
            )

            valid_per_joint_rmse.append(calc_rmse(predicted_poses, targeted_poses, self.args.num_joints))
               
            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_pck[metric][key] += predictions[metric][key]
    
        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)
    

    def evaluation(self):
        test_losses         = [] 
        test_per_joint_rmse = []
        test_total_pck      = self.pck_dict_fn()
        
        self.model.eval()
        with torch.no_grad():
            for _, (observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, _, _) in enumerate(self.test_dataset): 
                observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
                targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

                observed_poses        = observed_poses.to(self.args.device, dtype = torch.float32)
                targeted_poses        = targeted_poses.to(self.args.device, dtype = torch.float32)

                batch_size            = targeted_poses.shape[0]
                prediction_len        = targeted_poses.shape[1]

                predicted_act_seqs, loss = self.get_predicted_n_loss(
                    train        = False,
                    src_seq      = observed_act_seqs, 
                    tgt_seq      = targeted_act_seqs, 
                )
                
                test_losses.append(loss.item())

                if not isinstance(predicted_act_seqs, list): 
                    predicted_act_seqs = [predicted_act_seqs]

                predicted_poses  = reconstruct_pose_from_root(
                    root_vectors           = predicted_act_seqs[-1], 
                    mean                   = self.mean, 
                    std                    = self.std,   
                    frame_input_h          = self.args.frame_input_h, 
                    frame_input_w          = self.args.frame_input_w, 
                )

                predictions = compute_joints_metric(
                    pred      = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2),  
                    gt        = targeted_poses.contiguous().view(batch_size, prediction_len, -1, 2),  
                    time_step = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha,  
                )

                test_per_joint_rmse.append(calc_rmse(predicted_poses, targeted_poses, self.args.num_joints))
                
                for metric in predictions:
                    for key in predictions[metric]:
                        test_total_pck[metric][key] += predictions[metric][key]
        
            return np.mean(test_losses), test_total_pck, np.mean(test_per_joint_rmse) 

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, start_index, name = next(iter(self.val_loader))
            
            name = name[0]

            observed_act_seqs     = observed_act_seqs.to(self.args.device, dtype = torch.float32)
            targeted_act_seqs     = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

            observed_poses        = observed_poses.to(self.args.device, dtype = torch.float32)
            targeted_poses        = targeted_poses.to(self.args.device, dtype = torch.float32)


            observed_imgs, target_imgs = self.video_getter_fn(
                name               = name, 
                start_index        = start_index, 
                num_training_frame = self.args.num_training_frame, 
                num_testing_frame  = self.args.num_testing_frame, 
                frame_input_w      = self.frame_input_w,
                frame_input_h      = self.frame_input_h, 
            )


            predicted_act_seqs, loss, self_attn, cross_attn = self.get_predicted_n_loss(
                train   = False,
                src_seq = observed_act_seqs, 
                tgt_seq = targeted_act_seqs, 
            )

            if not isinstance(predicted_act_seqs, list): 
                predicted_act_seqs = [predicted_act_seqs]

            predicted_poses  = reconstruct_pose_from_root(
                root_vectors           = predicted_act_seqs[-1],  
                mean                   = self.mean, 
                std                    = self.std,   
                frame_input_h          = self.frame_input_h, 
                frame_input_w          = self.frame_input_w, 
            )

            fig, axes = plt.subplots(2, self.args.num_testing_frame, figsize = (24, 8))  

            print(f"RMSE Loss : {math.sqrt(loss.item()):.4f}")

            videos_real = []
            videos_pred = [] 

            pck_prediction = self.pck_dict_fn()
            accuracies = [] 

            visualize_attention(self_attn)
            visualize_attention(cross_attn)


            for index in range(0, self.args.num_testing_frame):     
                curr_target_image = target_imgs[index, :, :, :]

                curr_target_pose  = targeted_poses[0, index, :]    
                curr_target_pose  = curr_target_pose.contiguous().view(self.args.num_joints, -1)
                
                curr_predict_pose = predicted_poses[0, index, :]
                curr_predict_pose = curr_predict_pose.contiguous().view(self.args.num_joints, -1) 

                pck_prediction = pck_joints(
                    curr_predict_pose, 
                    curr_target_pose, 
                    pck_prediction,
                    self.dataset_joint_name, 
                    self.args.pck_alpha, 
                )
                
                target_annotated_image = draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy())
                target_predicted_image = draw_skeleton(curr_target_image, curr_predict_pose.detach().cpu().numpy())
                
                axes[0, index].imshow(target_annotated_image)
                axes[0, index].axis('off')
                axes[1, index].imshow(target_predicted_image)
                axes[1, index].axis('off')

                videos_real.append(target_annotated_image)
                videos_pred.append(target_predicted_image)
                
                acc = get_acc(pck_prediction)
                accuracies.append(acc)
                print(acc)
            

            show_video(videos_pred, fps = 2, output_path = 'root_relative_pred.mp4')
            show_video(videos_real, fps = 2, output_path = 'root_relative_real.mp4')
            plt.tight_layout()
            plt.show() 
        

    

      
      
      