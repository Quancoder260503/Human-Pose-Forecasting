import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil           import *
from data.ImgUtil               import *
from data.Visualization         import *
from runner.RunnerUtil          import *
from runner.Runner              import * 
from torch.nn.utils import clip_grad_norm_
import torch 
import math 

class CoordinatesRunner(Runner):
    def __init__(self, 
            args, 
            pck_dict_fn, 
            rmse_dict_fn, 
            model_checkpoint, 
    ):
        super(CoordinatesRunner, self).__init__(
            pck_dict_fn      = pck_dict_fn , 
            rmse_dict_fn     = rmse_dict_fn,  
            args             = args, 
            model_checkpoint = model_checkpoint, 
        )
        print("Input represented as Coordinates")

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
        train_per_joint_rmse  = []
        valid_per_joint_rmse  = []

        train_total_pck  = self.pck_dict_fn()
        valid_total_pck  = self.pck_dict_fn()

        self.model.train()
        for batch_idx, (observed_poses, target_poses, _, _) in enumerate(self.train_loader):
            observed_poses = observed_poses.to(self.args.device, dtype = torch.float32)
            target_poses   = target_poses.to(self.args.device,   dtype = torch.float32)
            
            batch_size     = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_poses, loss = self.get_predicted_n_loss(
                train                = True, 
                src_seq              = observed_poses, 
                tgt_seq              = target_poses, 
                epoch                = epoch, 
            )

            train_losses.append(loss.item())


            self.optimizer.zero_grad()
            loss.backward()

            if self.args.max_norm > 0 : 
                clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                
            self.optimizer.step()
            self.lr_scheduler.step()

            if not isinstance(predicted_poses, list): 
                predicted_poses = [predicted_poses]
    
            predicted_poses = predicted_poses[-1].contiguous().view(batch_size, prediction_len, -1, 3) 
            target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1,        3) 


            predicted_poses[:, :, :, 0] = predicted_poses[:, :, :, 0] * self.args.frame_input_w
            predicted_poses[:, :, :, 1] = predicted_poses[:, :, :, 1] * self.args.frame_input_h

            target_poses[:, :, :, 0] = target_poses[:, :, :, 0] * self.args.frame_input_w
            target_poses[:, :, :, 1] = target_poses[:, :, :, 1] * self.args.frame_input_h             
            
            
            predictions = compute_joints_metric(
              pred      = predicted_poses, 
              gt        = target_poses,  
              time_step = prediction_len, 
              dataset   = self.args.dataset, 
              alpha     = self.args.pck_alpha, 
            )


            train_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses, self.args.num_joints))

            for metric in predictions:
                for key in predictions[metric]:
                    train_total_pck[metric][key] += predictions[metric][key]


            if batch_idx % self.args.print_step == 0 :
                acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{self.args.step_per_epoch} |'
                      f'Loss = {np.mean(train_losses):.6f} |' 
                      f'PJRMSE  = {np.mean(train_per_joint_rmse):.6f} |' 
                      f'PCK accuracy = {acc:.4f}'
                ) 
            
            if batch_idx == self.args.step_per_epoch : 
                break 

        self.model.eval()
        with torch.no_grad():
            for _, (observed_poses, target_poses, _, _) in enumerate(self.val_loader):
                observed_poses = observed_poses.to(self.args.device, dtype = torch.float32)
                target_poses   = target_poses.to(self.args.device,   dtype = torch.float32)
                
                batch_size     = target_poses.shape[0]
                prediction_len = target_poses.shape[1]
                
                predicted_poses, loss = self.get_predicted_n_loss(
                    train                = False, 
                    src_seq              = observed_poses, 
                    tgt_seq              = target_poses, 
                )
                valid_losses.append(loss.item())

                if not isinstance(predicted_poses, list): 
                        predicted_poses = [predicted_poses]

                predicted_poses = predicted_poses[-1].contiguous().view(batch_size, prediction_len, -1, 3) 
                target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1, 3)  

                predicted_poses[:, :, :, 0] = predicted_poses[:, :, :, 0] * self.args.frame_input_w
                predicted_poses[:, :, :, 1] = predicted_poses[:, :, :, 1] * self.args.frame_input_h

                target_poses[:, :, :, 0] = target_poses[:, :, :, 0] * self.args.frame_input_w
                target_poses[:, :, :, 1] = target_poses[:, :, :, 1] * self.args.frame_input_h 

                predictions = compute_joints_metric(
                    pred      = predicted_poses,  
                    gt        = target_poses,  
                    time_step = prediction_len,
                    dataset   = self.args.dataset, 
                    alpha     = self.args.pck_alpha,  
                )


                valid_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses, self.args.num_joints))
                
                for metric in predictions:
                    for key in predictions[metric]:
                        valid_total_pck[metric][key] += predictions[metric][key]

        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)
    

    def evaluation(self):
        self.model.eval()
        with torch.no_grad():
            test_losses         = [] 
            test_per_joint_rmse = []
            test_total_pck      = self.pck_dict_fn()

            for _, (observed_poses, target_poses, _, _) in enumerate(self.test_loader): 
                observed_poses = observed_poses.to(self.args.device, dtype = torch.float32) 
                target_poses   = target_poses.to(self.args.device, dtype   = torch.float32)
                
                batch_size     = target_poses.shape[0]
                prediction_len = target_poses.shape[1]
                
                predicted_poses, loss = self.get_predicted_n_loss(
                    train                = False, 
                    src_seq              = observed_poses, 
                    tgt_seq              = target_poses,  
                )
                
                test_losses.append(loss.item())

                if not isinstance(predicted_poses, list): 
                    predicted_poses = [predicted_poses]

                predicted_poses = predicted_poses[-1].contiguous().view(batch_size, prediction_len, -1, 3) 
                target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1, 3)  

                predictions = compute_joints_metric(
                    pred      = predicted_poses,  
                    gt        = target_poses,  
                    time_step = prediction_len,
                    dataset   = self.dataset, 
                    alpha     = self.args.pck_alpha,  
                )

                for metric in predictions:
                    for key in predictions[metric]:
                        test_total_pck[metric][key] += predictions[metric][key]

                predicted_poses[:, :, :, 0] = predicted_poses[:, :, :, 0] * self.frame_input_w
                predicted_poses[:, :, :, 1] = predicted_poses[:, :, :, 1] * self.frame_input_h

                target_poses[:, :, :, 0] = target_poses[:, :, :, 0] * self.frame_input_w
                target_poses[:, :, :, 1] = target_poses[:, :, :, 1] * self.frame_input_h                     
                
                test_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses, self.args.num_joints))
        
            print(
                f"Test Loss: {np.mean(test_losses):.4f} | "
                f"PCK: {test_total_pck:.4f} | "
                f"Per Joint RMSE: {np.mean(test_per_joint_rmse):.4f}"
            )


    def inference(self):
        self.model.eval()
        with torch.no_grad():
        # Sample from validation dataset and visualize model prediction 
            observed_poses, target_poses, start_index, name = self.test_dataset[random.randint(0, len(self.val_loader))]
            
            observed_poses  = observed_poses.to(self.args.device, dtype = torch.float32)[0].unsqueeze(0)
            target_poses    = target_poses.to(self.args.device,   dtype = torch.float32)[0].unsqueeze(0)
            start_index     = 0 
            name            = name
            
            predicted_poses, loss = self.get_predicted_n_loss(
                train                = False, 
                src_seq              = observed_poses, 
                tgt_seq              = target_poses, 
            )

            if not isinstance(predicted_poses, list): 
                predicted_poses = [predicted_poses]

            predicted_poses = predicted_poses[-1]
            
            print(f"RMSE Loss : {math.sqrt(loss.item()):.4f}")

            fig, axes = plt.subplots(2, self.args.num_testing_frame, figsize = (24, 8))

            videos_real = []
            videos_pred = []

            pck_prediction = self.pck_dict_fn()

            observed_imgs, target_imgs = self.video_getter_fn(
                name               = name, 
                start_index        = start_index, 
                num_training_frame = self.args.num_training_frame, 
                num_testing_frame  = self.args.num_testing_frame, 
                frame_input_w      = self.frame_input_w,
                frame_input_h      = self.frame_input_h, 
            )


            pck_acc = []
            print(name)

            for index in range(0, self.args.num_testing_frame): 
                curr_target_pose                    = target_poses[:, index, :].squeeze(0)
                curr_target_pose, curr_target_prob  = torch.split(curr_target_pose.contiguous().view(self.args.num_joints, 3), [2, 1], dim = -1)  
            
                curr_pred_pose                  = predicted_poses[:, index, :].squeeze(0)
                curr_pred_pose, curr_pred_prob  = torch.split(curr_pred_pose.contiguous().view(self.args.num_joints, 3), [2, 1], dim = -1)

                curr_pred_prob  = torch.sigmoid(curr_pred_prob)

                curr_target_image = target_imgs[index, :, :, :]

                pck_prediction = pck_joints(
                    pred        = curr_pred_pose, 
                    gt          = curr_target_pose, 
                    predictions = pck_prediction, 
                    joint_names = self.dataset_joint_name, 
                    alpha       = self.args.pck_alpha, 
                )

                target_annotated_image  = draw_skeleton(
                    image = curr_target_image, 
                    poses = curr_target_pose.detach().cpu().numpy(), 
                    prob  = curr_target_prob.detach().cpu().numpy(),
                    dataset       = self.dataset, 
                    unnormalize   = True, 
                    frame_input_w = self.frame_input_w, 
                    frame_input_h = self.frame_input_h, 
                )

                target_prediction_image = draw_skeleton(
                    image = curr_target_image, 
                    poses = curr_pred_pose.detach().cpu().numpy(), 
                    prob  = curr_pred_prob.detach().cpu().numpy(),
                    dataset       = self.dataset,  
                    unnormalize   = True, 
                    frame_input_w = self.frame_input_w, 
                    frame_input_h = self.frame_input_h, 
                )

                axes[0, index].imshow(target_annotated_image)
                axes[0, index].axis('off')
                axes[1, index].imshow(target_prediction_image)
                axes[1, index].axis('off')

                videos_real.append(target_annotated_image)
                videos_pred.append(target_prediction_image)

                acc = get_acc(pck_prediction)
                print(acc)
            show_video(videos_pred, fps = 4, output_path = 'output_pred.mp4')
            show_video(videos_real, fps = 4, output_path = 'output_real.mp4')
            plt.tight_layout()
            plt.show() 

