import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil           import *
from data.ImgUtil               import *
from data.Visualization         import *
from trainer.TrainUtil          import *
from trainer.CoordinatesTrainer import *
from trainer.Trainer            import * 
from torch.nn.utils import clip_grad_norm_
import torch 
import math 

class CoordinatesTrainer(Trainer):
    def __init__(self, 
                 pck_dict_fn, 
                 rmse_dict_fn, 
                 train_loader, 
                 val_loader, 
                 test_dataset, 
                 dataset, 
                 print_step, 
                 model_type, 
                 video_getter_fn, 
                 frame_input_w, 
                 frame_input_h, 
        ):
        super(CoordinatesTrainer, self).__init__(
            pck_dict_fn     = pck_dict_fn , 
            rmse_dict_fn    = rmse_dict_fn,  
            train_loader    = train_loader, 
            val_loader      = val_loader, 
            test_dataset    = test_dataset, 
            print_step      = print_step, 
            model_type      = model_type, 
            dataset         = dataset, 
            video_getter_fn = video_getter_fn, 
            frame_input_h   = frame_input_h, 
            frame_input_w   = frame_input_w, 
        )
        self.pck_alpha = PCK_ALPHA
        print("Input represented as Coordinates")

    def process_per_epoch(self, args, model, optimizer, criterion, epoch):

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

        model.train()
        for batch_idx, (observed_poses, target_poses, _, _) in enumerate(self.train_loader):
            observed_poses = observed_poses.to(args.device, dtype=torch.float32)
            target_poses   = target_poses.to(args.device,   dtype=torch.float32)
            
            batch_size     = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_poses, loss = self.get_loss_fn(
                args                 = args, 
                train                = True, 
                model                = model, 
                criterion            = criterion,
                encode_input         = observed_poses, 
                decode_input         = target_poses, 
                teacher_forcing_rate = args.teacher_forcing_rate,  
            )

            train_losses.append(loss.item())


            optimizer.zero_grad()
            loss.backward()

            if args.max_norm > 0 : 
                clip_grad_norm_(model.parameters(), args.max_norm)
                
            optimizer.step()

            predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 3) 
            target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1,    3) 

            
            predictions = compute_joints_metric(
              pred      = predicted_poses, 
              gt        = target_poses,  
              time_step = prediction_len, 
              dataset   = self.dataset, 
              alpha     = self.pck_alpha, 
            )
            
            train_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))


            for metric in predictions:
                for key in predictions[metric]:
                    train_total_pck[metric][key] += predictions[metric][key]


            if batch_idx % self.print_step == 0 :
                acc = train_total_pck['total']['correct'] / train_total_pck['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(self.train_loader)} |'
                      f'Loss = {np.mean(train_losses):.6f} |' 
                      f'PJRMSE  = {np.mean(train_per_joint_rmse):.6f} |' 
                      f'PCK accuracy = {acc:.4f}'
                ) 

        model.eval()
        with torch.no_grad():
            for _, (observed_poses, target_poses, _, _) in enumerate(self.val_loader):
               observed_poses = observed_poses.to(args.device, dtype=torch.float32)
               target_poses   = target_poses.to(args.device,   dtype=torch.float32)
            
               batch_size     = target_poses.shape[0]
               prediction_len = target_poses.shape[1]
            
               predicted_poses, loss = self.get_loss_fn(
                    args                 = args,
                    train                = False, 
                    model                = model, 
                    criterion            = criterion,
                    encode_input         = observed_poses, 
                    decode_input         = target_poses, 
                    teacher_forcing_rate = args.teacher_forcing_rate,  
               )
               valid_losses.append(loss.item())


               predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 3) 
               target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1, 3)  

               predictions = compute_joints_metric(
                    pred      = predicted_poses,  
                    gt        = target_poses,  
                    time_step = prediction_len,
                    dataset   = self.dataset, 
                    alpha     = self.pck_alpha,  
               )

               valid_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))
               
               for metric in predictions:
                   for key in predictions[metric]:
                       valid_total_pck[metric][key] += predictions[metric][key]

        return np.mean(train_losses), train_total_pck, np.mean(train_per_joint_rmse), np.mean(valid_losses), valid_total_pck, np.mean(valid_per_joint_rmse)
    
    @torch.no_grad()
    def evaluation(self, args, model, criterion):
        # compute the accuracy and metrics on the unseen dataset

        test_losses         = [] 
        test_per_joint_rmse = []
        test_total_pck      = self.pck_dict_fn()

        for _, (observed_poses, target_poses, _, _) in enumerate(self.test_dataset): 
            observed_poses = observed_poses.to(args.device, dtype = torch.float32) 
            target_poses   = target_poses.to(args.device, dtype   = torch.float32)
            
            batch_size     = target_poses.shape[0]
            prediction_len = target_poses.shape[1]
            
            predicted_poses, loss = self.get_loss_fn(
               args                 = args,
               train                = False, 
               model                = model, 
               criterion            = criterion,
               encode_input         = observed_poses, 
               decode_input         = target_poses, 
               teacher_forcing_rate = args.teacher_forcing_rate,  
            )
            
            test_losses.append(loss.item())

            predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 3) 
            target_poses    = target_poses.contiguous().view(batch_size, prediction_len, -1, 3)  

            predictions = compute_joints_metric(
                pred      = predicted_poses,  
                gt        = target_poses,  
                time_step = prediction_len,
                dataset   = self.dataset, 
                alpha     = self.pck_alpha,  
            )

            for metric in predictions:
                for key in predictions[metric]:
                    test_total_pck[metric][key] += predictions[metric][key]
               
            test_per_joint_rmse.append(calc_rmse(predicted_poses, target_poses))
        
        return np.mean(test_losses), test_total_pck, np.mean(test_per_joint_rmse) 


    @torch.no_grad()
    def inference(self, args, model, criterion):
        # Sample from validation dataset and visualize model prediction 
        observed_poses, target_poses, start_index, name = self.test_dataset[random.randint(0, len(self.test_dataset))]
      
        observed_poses  = observed_poses.to(args.device, dtype = torch.float32)[0].unsqueeze(0)
        target_poses    = target_poses.to(args.device,   dtype = torch.float32)[0].unsqueeze(0)
        start_index     = 0 
        name            = name
      
        predicted_poses, loss = self.get_loss_fn(
            args                 = args, 
            train                = False, 
            model                = model, 
            criterion            = criterion,
            encode_input         = observed_poses, 
            decode_input         = target_poses, 
            teacher_forcing_rate = args.teacher_forcing_rate,  
        )
       
        print(f"RMSE Loss : {math.sqrt(loss.item()):.4f}")

        fig, axes = plt.subplots(2, args.num_testing_frame, figsize = (24, 8))

        videos_real = []
        videos_pred = []

        pck_prediction = self.pck_dict_fn()

        observed_imgs, target_imgs = self.video_getter_fn(
            name               = name, 
            start_index        = start_index, 
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame, 
            frame_input_w      = self.frame_input_w,
            frame_input_h      = self.frame_input_h, 
        )


        pck_acc = []

        for index in range(0, args.num_testing_frame): 
            curr_target_pose                    = target_poses[:, index, :].squeeze(0)
            curr_target_pose, curr_target_prob  = torch.split(curr_target_pose.contiguous().view(args.num_joints, 3), [2, 1], dim = -1)  
        
            curr_pred_pose                  = predicted_poses[:, index, :].squeeze(0)
            curr_pred_pose, curr_pred_prob  = torch.split(curr_pred_pose.contiguous().view(args.num_joints, 3), [2, 1], dim = -1)

            curr_pred_prob  = torch.sigmoid(curr_pred_prob)

            curr_target_image = target_imgs[index, :, :, :]

            pck_prediction = pck_joints(
                pred        = curr_pred_pose, 
                gt          = curr_target_pose, 
                predictions = pck_prediction, 
                joint_names = self.dataset_joint_name, 
                alpha       = self.pck_alpha, 
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
        show_video(videos_pred, fps = 1)
        plt.tight_layout()
        plt.show() 

