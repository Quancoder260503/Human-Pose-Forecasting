import matplotlib.pyplot as plt
import numpy as np
import torch 
import random 
from data.Visualization       import *
from data.DatasetUtil         import *
from runner.RunnerUtil        import *
from runner.Runner            import * 
from torch.nn.utils import clip_grad_norm_

class VelocityRunner(Runner):
    def __init__(self, 
            pck_dict_fn, 
            rmse_dict_fn, 
            args, 
            model_checkpoint, 
        ):
        super(VelocityRunner, self).__init__(
            pck_dict_fn     = pck_dict_fn , 
            rmse_dict_fn    = rmse_dict_fn,  
            dataset         = args,
            model_checkpoint = model_checkpoint,  
        )
        print("Input represented as Velocity")

    def process_per_epoch(self, epoch):
        train_losses = []
        valid_losses = []

        train_per_joint_rmse  = []
        valid_per_joint_rmse  = []

        train_total_pck = self.pck_dict_fn() 
        valid_total_pck = self.pck_dict_fn()

        
        self.model.train()         
        for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.train_loader):
                observed_poses = observed_poses.to(self.args.device, dtype = torch.float32)           # (B, T_train, J * 2)
                target_poses   = target_poses.to(self.args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
                observed_velocities = observed_velocities.to(self.args.device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(self.args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                
                batch_size, observed_len,   _ = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 


                predicted_velocities, loss = self.get_predicted_n_loss(
                    train   = True, 
                    src_seq = observed_velocities, 
                    tgt_seq = target_velocities, 
                    epoch   = epoch, 
                )
                
                train_losses.append(loss.item())


                #with torch.autograd.detect_anomaly(): 
                self.optimizer.zero_grad()
                loss.backward()

                if self.args.max_norm > 0 : 
                   clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                
                self.optimizer.step() 
                self.lr_scheduler.step()
                    
                target_poses = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
                
                #num_observed_velocity + 1 = num_poses 
                
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)

                if not isinstance(predicted_velocities, list):
                    predicted_velocities = [predicted_velocities]

                predicted_velocities = predicted_velocities[-1]

                predicted_poses = speed_to_pos(
                    prediction_velocities =  predicted_velocities, 
                    observed_poses        =  observed_poses, 
                    device                =  self.args.device, 
                )

                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :,    :2],  
                    time_step = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha,  # Since this is not 0-1 anymore should be bigger constant
                )

                train_per_joint_rmse.append(calc_mpjpe(predicted_poses, targeted_poses, self.args.num_joints))


                for dict in predictions:
                    for key in predictions[dict]:
                        train_total_pck[dict][key] += predictions[dict][key]    
                
                if batch_idx > 0 and batch_idx % self.args.print_step == 0:
                    acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                    print(f'Epoch {epoch + 1} | Batch {batch_idx}/{self.step_per_epoch} |'
                      f'Loss = {np.mean(train_losses):.6f} |' 
                      f'PJRMSE  = {np.mean(train_per_joint_rmse)} |' 
                      f'PCK accuracy = {acc:.4f}'
                    ) 
                

        self.model.eval()
        with torch.no_grad():
            for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.val_loader):
                observed_poses = observed_poses.to(self.args.device, dtype = torch.float32)           # (B, T_train, J * 2)
                target_poses   = target_poses.to(self.args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
                observed_velocities = observed_velocities.to(self.args.device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(self.args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                    
                batch_size, observed_len, _   = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 


                predicted_velocities, loss = self.get_predicted_n_loss(
                    train    = False, 
                    src_seq  = observed_velocities, 
                    tgt_seq  = target_velocities, 
                    epoch    = epoch, 
                )
                
                valid_losses.append(loss.item())
                
                target_poses   = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
                
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)

                if not isinstance(predicted_velocities, list):
                    predicted_velocities = [predicted_velocities]

                predicted_velocities = predicted_velocities[-1]

                predicted_poses = speed_to_pos(predicted_velocities, observed_poses) # (B, target_len, 26)
                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)


                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :, :2],  
                    time_step = prediction_len, 
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha, 
                )
                
                valid_per_joint_rmse.append(calc_mpjpe(predicted_poses, targeted_poses, self.args.num_joints))

                for dict in predictions:
                    for key in predictions[dict]:
                        valid_total_pck[dict][key] += predictions[dict][key]

        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)
    

    def evaluation(self):
        self.model.eval()
        with torch.no_grad():
            test_losses         = [] 
            test_per_joint_rmse = []
            test_total_pck      = self.pck_dict_fn()
            
            for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.test_loader):
                observed_poses = observed_poses.to(self.args.device, dtype = torch.float32)           # (B, T_train, J * 2)
                target_poses   = target_poses.to(self.args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
                observed_velocities = observed_velocities.to(self.args.device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(self.args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                        
                batch_size, observed_len, _   = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 


                predicted_velocities, loss = self.get_predicted_n_loss(
                    train = False, 
                    src_seq      = observed_velocities, 
                    tgt_seq      = target_velocities, 
                )

                if not isinstance(predicted_velocities, list):
                    predicted_velocities = [predicted_velocities]

                predicted_velocities = predicted_velocities[-1]     

                test_losses.append(loss.item())
                    
                target_poses   = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
                
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)
                
                predicted_poses = speed_to_pos(predicted_velocities, observed_poses) 
                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)


                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :, :2],  
                    time_step = prediction_len, 
                    dataset   = self.dataset, 
                    alpha     = self.args.pck_alpha
                )
                    
                test_per_joint_rmse.append(calc_mpjpe(predicted_poses, target_poses, self.args.num_joints))

                for dict in predictions:
                    for key in predictions[dict]:
                        test_total_pck[dict][key] += predictions[dict][key]

            test_per_joint_rmse = np.stack(test_per_joint_rmse, axis = 0).mean(axis = 0)  # (prediction_len,)
            test_total_pck_acc  = get_acc(test_total_pck)  # assuming get_acc gives you total scalar

            print(
                f"Test Loss: {np.mean(test_losses):.4f} | "
                f"PCK: {test_total_pck_acc:.4f} | "
                f"Per Joint RMSE: {test_per_joint_rmse.mean():.4f}"
            )

            plot_rmse_over_horizon(test_per_joint_rmse, save_path = 'velocity_rmse_horizon.png')


    def inference(self):
        self.model.eval()
        with torch.no_grad():
            observed_poses, target_poses, observed_velocity, target_velocity, start_index, name = self.val_loader[random.randint(0, len(self.val_loader))]
        
            observed_poses    = observed_poses.unsqueeze(0).to(self.args.device,    dtype = torch.float32)
            observed_velocity = observed_velocity.unsqueeze(0).to(self.args.device, dtype = torch.float32)
            target_poses      = target_poses.unsqueeze(0).to(self.args.device,      dtype = torch.float32)
            target_velocity   = target_velocity.unsqueeze(0).to(self.args.device,   dtype = torch.float32)

            observed_poses   = observed_poses.contiguous().view(1, self.args.num_training_frame, -1, 3)[:, :, :, :2]
            observed_poses   = observed_poses.contiguous().view(1, self.args.num_training_frame, -1) 

            predicted_velocity, loss = self.get_predicted_n_loss(
                train   = False, 
                src_seq = observed_velocity, 
                tgt_seq = target_velocity, 
            )


            if not isinstance(predicted_velocity, list):
                predicted_velocity = [predicted_velocity]

            predicted_velocity = predicted_velocity[-1]

            predicted_poses = speed_to_pos(observed_poses = observed_poses, prediction_velocities = predicted_velocity, device = self.args.device)

            fig, axes = plt.subplots(2, self.args.num_testing_frame, figsize = (24, 8))  

            videos_real = []
            videos_pred = [] 

            pck_prediction = self.pck_dict_fn()
            accuracies = [] 

            observed_imgs, target_imgs = self.video_getter_fn(
                name               = name, 
                start_index        = start_index, 
                num_training_frame = self.args.num_training_frame, 
                num_testing_frame  = self.args.num_testing_frame, 
                frame_input_w      = self.frame_input_w,
                frame_input_h      = self.frame_input_h, 
            )

            for index in range(0, self.args.num_testing_frame):     
                curr_target_image = target_imgs[index, :, :, :]
                curr_target_image = np.ascontiguousarray(torch2numpy(curr_target_image))

                curr_target_pose  = target_poses[:, index, :].squeeze(0)     
                curr_target_pose  = curr_target_pose.contiguous().view(13, 3)[:, :2]
                
                curr_predict_pose = predicted_poses[:, index, :].squeeze(0) 
                curr_predict_pose = curr_predict_pose.contiguous().view(13, 2) 

                pck_prediction = pck_joints(
                    curr_predict_pose, 
                    curr_target_pose, 
                    pck_prediction,
                    self.dataset_joint_name, 
                    alpha = self.args.pck_alpha, 
                )
                
                target_annotated_image = draw_skeleton(curr_target_image, curr_target_pose.detach().cpu().numpy())
                axes[0, index].imshow(target_annotated_image)
                target_predicted_image = draw_skeleton(curr_target_image, curr_predict_pose.detach().cpu().numpy())
                axes[1, index].imshow(target_predicted_image)
            
                videos_real.append(target_annotated_image)
                videos_pred.append(target_predicted_image)
                
                acc = get_acc(pck_prediction)
                accuracies.append(acc)
                print(acc)

            plot_inference(
                metric_list = accuracies,  
                dataset     = self.args.dataset,
                save_path   = 'inference_velocity.png',  
            )

            show_video(videos_pred, fps = 2, output_path = 'root_relative_pred.mp4')
            show_video(videos_real, fps = 2, output_path = 'root_relative_real.mp4')
            plt.tight_layout()
            plt.show() 