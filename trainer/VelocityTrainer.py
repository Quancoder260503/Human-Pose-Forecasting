import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil         import *
from trainer.TrainUtil                import *
from trainer.CoordinatesTrainer       import *
from trainer.Trainer                  import * 
import torch 

class VelocityTrainer(Trainer):
    def __init__(self, pck_dict_fn, mse_dict_fn, train_loader, val_loader, test_dataset, print_step, model_type, video_getter_fn, frame_input_h, frame_input_w):
        super(VelocityTrainer, self).__init__(
            pck_dict_fn     = pck_dict_fn , 
            mse_dict_fn     = mse_dict_fn,  
            train_loader    = train_loader, 
            val_loader      = val_loader, 
            test_dataset    = test_dataset, 
            print_step      = print_step, 
            model_type      = model_type, 
            video_getter_fn = video_getter_fn, 
            frame_input_h   = frame_input_h, 
            frame_input_w   = frame_input_w, 
        )
        self.pck_alpha     = min(self.frame_input_h, self.frame_input_w) * PCK_ALPHA 
        print("Input represented as Velocity")

    def process_per_epoch(self, args, model, optimizer, criterion, epoch):
        """
        Output poses is not normalized, may include a pixel constraint to compute this ? 
        Input represented as velocity 
        Can be trained using MLP, LSTM, MOTION_VAE, ATTENTION_MOTION_VAE, TRANSFORMER
        For now supported only VAE type model 

        :param model        : model to train 
        :param train_loader : train data loader
        :param val_loader   : valid data loader 
        :param optimizer    : set default to ADAM 
        :param criterion    : loss function
        :param epoch        : current epoch
        
        """
    
        train_losses = []
        valid_losses = []

        train_per_joint_rmse  = []
        valid_per_joint_rmse  = []

        train_total_pck = self.pck_dict_fn() 
        valid_total_pck = self.pck_dict_fn()

        
        model.train()         
        for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.train_loader):
                observed_poses = observed_poses.to(args.device, dtype = torch.float32)           # (B, T_train, J * 2)
                target_poses   = target_poses.to(args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
                observed_velocities = observed_velocities.to(args.device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                
                batch_size, observed_len,   _ = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 


                predicted_velocities, loss = self.get_loss_fn(
                    train = True, 
                    model = model, 
                    criterion = criterion, 
                    encode_input = observed_velocities, 
                    decode_input = target_velocities, 
                    teacher_forcing_rate = args.teacher_forcing_rate, 
                )
                
                train_losses.append(loss.item())


                #with torch.autograd.detect_anomaly(): 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                    
                target_poses = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
                
                #num_observed_velocity + 1 = num_poses 
                
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)

                predicted_poses = speed_to_pos(
                    prediction_velocities =  predicted_velocities, 
                    observed_poses        =  observed_poses, 
                    device                =  args.device, 
                )

                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :,    :2],  
                    time_step = prediction_len, 
                    dataset   = self.dataset, 
                    alpha     = self.pck_alpha,  # Since this is not 0-1 anymore should be bigger constant
                )

                train_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))


                for dict in predictions:
                    for key in predictions[dict]:
                        train_total_pck[dict][key] += predictions[dict][key]    
                
                if batch_idx > 0 and batch_idx % args.print_step == 0:
                    acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                    print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(self.train_loader)} |'
                      f'Loss = {np.mean(train_losses):.6f} |' 
                      f'PJRMSE  = {np.mean(train_per_joint_rmse)} |' 
                      f'PCK accuracy = {acc:.4f}'
                    ) 
                

        model.eval()
        with torch.no_grad():
            for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.val_loader):
                observed_poses = observed_poses.to(args.device, dtype = torch.float32)           # (B, T_train, J * 2)
                target_poses   = target_poses.to(args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
                observed_velocities = observed_velocities.to(args.device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                    
                batch_size, observed_len, _   = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 


                predicted_velocities, loss = self.get_loss_fn(
                    train = False, 
                    model = model, 
                    criterion = criterion, 
                    encode_input = observed_velocities, 
                    decode_input = target_velocities, 
                    teacher_forcing_rate = args.teacher_forcing_rate, 
                )
                

                valid_losses.append(loss.item())
                
                target_poses   = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
                
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
                observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)
                
                predicted_poses = speed_to_pos(predicted_velocities, observed_poses) # (B, target_len, 26)
                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)


                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :, :2],  
                    time_step = prediction_len, 
                    dataset   = self.dataset, 
                    alpha     = self.pck_alpha
                )
                
                valid_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))

                for dict in predictions:
                    for key in predictions[dict]:
                        valid_total_pck[dict][key] += predictions[dict][key]

        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)
    

    @torch.no_grad()
    def evaluation(self, args, model, criterion):
        
        test_losses         = [] 
        test_per_joint_rmse = []
        test_total_pck      = self.pck_dict_fn()
        
        for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(self.test_dataset):
            observed_poses = observed_poses.to(args.device, dtype = torch.float32)           # (B, T_train, J * 2)
            target_poses   = target_poses.to(args.device,   dtype = torch.float32)           # (B, T_test,  J * 2)
            observed_velocities = observed_velocities.to(args.device, dtype = torch.float32) # (B, T_train, J * 2)
            target_velocities   = target_velocities.to(args.device,   dtype = torch.float32) # (B, T_test,  J * 2)
                    
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 


            predicted_velocities, loss = self.get_loss_fn(
                train = False, 
                model = model, 
                criterion = criterion, 
                encode_input = observed_velocities, 
                decode_input = target_velocities, 
                teacher_forcing_rate = args.teacher_forcing_rate, 
            )
                

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
                alpha     = self.pck_alpha
            )
                
            test_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))

            for dict in predictions:
                for key in predictions[dict]:
                    test_total_pck[dict][key] += predictions[dict][key]

            return np.mean(test_losses), test_total_pck, np.mean(test_per_joint_rmse) 


    @torch.no_grad()
    def inference(self, args, model, criterion):
        observed_poses, target_poses, observed_velocity, target_velocity, start_index, name = self.val_loader[random.randint(0, len(self.val_loader))]
       
        observed_poses    = observed_poses.unsqueeze(0).to(args.device,    dtype = torch.float32)
        observed_velocity = observed_velocity.unsqueeze(0).to(args.device, dtype = torch.float32)
        target_poses      = target_poses.unsqueeze(0).to(args.device,      dtype = torch.float32)
        target_velocity   = target_velocity.unsqueeze(0).to(args.device,   dtype = torch.float32)

        observed_poses   = observed_poses.contiguous().view(1, args.num_training_frame, -1, 3)[:, :, :, :2]
        observed_poses   = observed_poses.contiguous().view(1, args.num_training_frame, -1) 

        predicted_velocity, loss = self.get_loss_fn(
            train = False, 
            model = model, 
            criterion = criterion, 
            encode_input = observed_velocity, 
            decode_input = target_velocity, 
            teacher_forcing_rate = args.teacher_forcing_rate, 
        )

        predicted_poses = speed_to_pos(observed_poses = observed_poses, prediction_velocities = predicted_velocity, device = args.device)

        fig, axes = plt.subplots(2, args.num_testing_frame, figsize = (24, 8))  

        videos_real = []
        videos_pred = [] 

        pck_prediction = self.pck_dict_fn()
        accuracies = [] 

        observed_imgs, target_imgs = self.video_getter_fn(
            name               = name, 
            start_index        = start_index, 
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame, 
            frame_input_w      = self.frame_input_w,
            frame_input_h      = self.frame_input_h, 
        )

        for index in range(0, args.num_testing_frame):     
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
                alpha = self.pck_alpha, 
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

        show_video(videos_pred, fps = 5)
        plt.tight_layout()
        plt.show() 