import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.Lstm   import *
from model.Mlp    import *
from model.MotionVae import *
from TrainingUtil import *
from data.PennAction import PennAction
import os 
import argparse 

PRINT_STEP = 10 


def train_per_epoch(args, model, train_loader, optimizer, criterion, epoch, device):
    """
    :param model: model to train 
    :param train_loader: train data loader 
    :param optimizer: set default to ADAM 
    :param criterion: loss function
    :param epoch: current epoch
    :param device: 'cuda' 
    """
    local_model, global_model, criterion_local, criterion_global = None, None, None, None
    if args.model.startswith('decouple'): 
        local_model, global_model = model 
        criterion_local, criterion_global = criterion
        local_model.train() 
        global_model.train()
    else : 
      model.train()
    losses = []
    total_predictions = get_pennaction_predictions_dict()
    print_step = PRINT_STEP
    if args.parent_relative : 
        for batch_idx, (observed_bones, target_bones, observed_poses, target_poses, scale_factor) in enumerate(train_loader):
            observed_bones = observed_bones.to(device, dtype = torch.float32)
            target_bones   = target_bones.to(device,   dtype = torch.float32)
            observed_poses = observed_poses.to(device, dtype = torch.float32)
            target_poses   = target_poses.to(device,   dtype = torch.float32)
            scale_factor   = scale_factor.to(device,   dtype = torch.float32)

            batch          = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_bones = model(observed_bones, prediction_len)
            loss            = criterion(predicted_bones, target_bones, prediction_len) 

            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_poses      = reconstruct_pose(
                compressed_poses = predicted_bones, 
                scaled_factor    = scale_factor,  
                dataset          = args.dataset, 
                box_size         = args.boxsize, 
            )
            
            construct_target     = reconstruct_pose(
                compressed_poses = target_bones,    
                scaled_factor    = scale_factor, 
                dataset          = args.dataset, 
                box_size         = args.boxsize
            )
            
            predictions   = compute_joints_metric(
                pred      = predicted_poses[:, :, :,  :2],  
                gt        = construct_target[:, :, :, :2],  
                time_step = prediction_len, 
                dataset   = args.dataset, 
                alpha     = args.pck_alpha 
            )

            for dict in predictions:
                for key in predictions[dict]:
                    total_predictions[dict][key] += predictions[dict][key]

            if batch_idx > 0 and batch_idx % print_step == 0:
                acc = total_predictions['total']['correct'] / total_predictions['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.6f} | PCK accuracy = {acc:.4f}') 
   
    elif args.coordinates:
        for batch_idx, (observed_poses, target_poses) in enumerate(train_loader):
            observed_poses = observed_poses.to(device, dtype = torch.float32)
            target_poses   = target_poses.to(device, dtype = torch.float32)
            
            batch          = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_poses = model(observed_poses, prediction_len)
            loss = criterion(predicted_poses, target_poses, prediction_len) 
            losses.append(loss.item())

            #with torch.autograd.detect_anomaly():  (used for debugging only)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            predicted_poses = predicted_poses.contiguous().view(batch, prediction_len, -1, 3) 
            target_poses    = target_poses.contiguous().view(batch, prediction_len, -1, 3) 

            predictions = compute_joints_metric(
                pred      = predicted_poses, 
                gt        = target_poses,  
                time_step = prediction_len, 
                dataset   = args.dataset, 
                alpha     = args.pck_alpha, 
            )

            for dict in predictions:
                for key in predictions[dict]:
                    total_predictions[dict][key] += predictions[dict][key]

            if batch_idx % print_step == 0:
                acc = total_predictions['total']['correct'] / total_predictions['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.6f} | PCK accuracy = {acc:.4f}') 
                
    elif args.velocity : 
        for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(train_loader):
            
            observed_poses = observed_poses.to(device, dtype = torch.float32) # (B, T_train, J * 2)
            target_poses   = target_poses.to(device,   dtype = torch.float32) # (B, T_test,   J * 2)
            observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
            target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test, J * 2)
            
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 
            # Global, Local decomposition approach, from : 
            # https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Parsaeifard_Learning_Decoupled_Representations_for_Human_Pose_Forecasting_ICCVW_2021_paper.pdf
           
            if args.model.startswith('decouple'):   
                observed_globals = 0.5 * (
                   observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 7] + 
                   observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 8]
                ) # (B, observed_len, 1, 2)

                target_globals   = 0.5 * ( 
                    target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 7] + 
                    target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 8]
                ) # (B, prediction_len, 1, 2)

                observed_locals = observed_velocities.contiguous().view(batch_size, observed_len, -1, 2) - observed_globals.contiguous().view(batch_size, observed_len, 1, 2)
                observed_locals = observed_locals.contiguous().view(batch_size, observed_len, -1) # (B, observed_len, 26) 

                target_locals   = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)    - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
                target_locals   = target_locals.contiguous().view(batch_size, prediction_len, -1) # (B, target_len, 26)

                global_model.zero_grad()
                predicted_globals = global_model(observed_globals, prediction_len = prediction_len)
                loss_global       = criterion_global(predicted_globals, target_globals) # (B, target_len, 1, 2)

                local_model.zero_grad() 
                predicted_locals, mean, log_var = local_model(observed_locals, prediction_len = prediction_len) # (B, target_len, 26)
                loss_local  = criterion_local(predicted_locals, target_locals, mean, log_var)

                predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
                predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
                
                loss = loss_global + 0.1 * loss_local
                losses.append(loss.item())
                
                loss.backward() 
                optimizer.step()
                optimizer.zero_grad()

                predicted_poses = speed_to_pos(predicted_velocity, observed_poses) # (B, target_len, 26)
                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

            else : 
                # For motion use VAE to model 
                predicted_velocities, mean, log_var = model(observed_velocities, prediction_len = prediction_len)
                loss = criterion(predicted_velocities, target_velocities, mean, log_var)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                predicted_poses = speed_to_pos(predicted_velocity, observed_poses) # (B, target_len, 26)
                predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

            predictions = compute_joints_metric(
                pred      = predicted_poses[:, :, :, :2],  
                gt        = target_poses[:, :, :,    :2],  
                time_step = prediction_len, 
                dataset   = args.dataset, 
                alpha     = args.pck_alpha, 
            )

            for dict in predictions:
                for key in predictions[dict]:
                    total_predictions[dict][key] += predictions[dict][key]    
            
            if batch_idx > 0 and batch_idx % print_step == 0:
                acc = total_predictions['total']['correct'] / total_predictions['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(losses):.6f} | PCK accuracy = {acc:.4f}') 

    else : 
        raise NotImplementedError('Method not supported')
    return np.mean(losses), total_predictions


def evaluate_per_epoch(args, model, val_loader, criterion, device):
    """
    :param model: model 
    :param val_loader: validation dataloader
    :param criterion: loss function
    :param device: cuda 
    :return:
    """
    local_model, global_model, criterion_local, criterion_global = None, None, None, None
    if args.model.startswith('decouple'): 
        local_model, global_model = model 
        criterion_local, criterion_global = criterion
        local_model.eval() 
        global_model.eval()
    else : 
      model.eval()

    losses = []
    total_predictions = get_pennaction_predictions_dict()
    if args.parent_relative : 
        with torch.no_grad():
            for _, (observed_bones, target_bones, observed_poses, target_poses, scale_factor) in enumerate(val_loader):
                observed_bones = observed_bones.to(device, dtype = torch.float32)
                target_bones   = target_bones.to(device,   dtype = torch.float32)
                observed_poses = observed_poses.to(device, dtype = torch.float32)
                target_poses   = target_poses.to(device,   dtype = torch.float32)
                scale_factor   = scale_factor.to(device,   dtype = torch.float32)
                
                batch          = target_poses.shape[0]
                prediction_len = target_poses.shape[1]

                predicted_bones = model(observed_bones, prediction_len)
                loss = criterion(predicted_bones, target_bones, prediction_len) 

                losses.append(loss.item())

                predicted_poses      = reconstruct_pose(
                    compressed_poses = predicted_bones, 
                    scaled_factor    = scale_factor,  
                    dataset          = args.dataset, 
                    box_size         = args.boxsize, 
                )
                construct_target     = reconstruct_pose(
                    compressed_poses = target_bones,    
                    scaled_factor    = scale_factor, 
                    dataset          = args.dataset, 
                    box_size         = args.boxsize
                )

                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :,  :2],  
                    gt        = construct_target[:, :, :, :2],  
                    time_step = prediction_len, 
                    dataset   = args.dataset, 
                    alpha     = args.pck_alpha, 
                )

                for dict in predictions:
                    for key in predictions[dict]:
                        total_predictions[dict][key] += predictions[dict][key]

    elif args.coordinates : 
        with torch.no_grad():
            for _, (observed_poses, target_poses) in enumerate(val_loader):
                observed_poses = observed_poses.to(device, dtype = torch.float32)
                target_poses   = target_poses.to(device,   dtype = torch.float32)
                batch          = target_poses.shape[0]
                prediction_len = target_poses.shape[1]
                predicted_poses = model(observed_poses, prediction_len)
                loss = criterion(predicted_poses, target_poses, prediction_len) 
                losses.append(loss.item())

                predicted_poses = predicted_poses.contiguous().view(batch, prediction_len, -1, 3) 
                target_poses    = target_poses.contiguous().view(batch, prediction_len, -1, 3)  

                predictions = compute_joints_metric(
                    pred      = predicted_poses,  
                    gt        = target_poses,  
                    time_step = prediction_len,
                    dataset   = args.dataset, 
                    alpha     = args.pck_alpha 
                )
                for dict in predictions:
                    for key in predictions[dict]:
                        total_predictions[dict][key] += predictions[dict][key]

    elif args.velocity : 
        with torch.no_grad():
            for _, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(val_loader):
            
                observed_poses = observed_poses.to(device, dtype = torch.float32) # (B, T_train, J * 2)
                target_poses   = target_poses.to(device,   dtype = torch.float32) # (B, T_test,   J * 2)
                observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
                target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test, J * 2)
                
                batch_size, observed_len, _   = observed_velocities.shape
                batch_size, prediction_len, _ = target_velocities.shape 
                if args.model.startswith('decouple'):   
                    observed_globals = 0.5 * (
                    observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 7] + 
                    observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 8]
                    ) # (B, observed_len, 1, 2)

                    target_globals   = 0.5 * ( 
                        target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 7] + 
                        target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 8]
                    ) # (B, prediction_len, 1, 2)

                    observed_locals = observed_velocities.contiguous().view(batch_size, observed_len, -1, 2) - observed_globals.contiguous().view(batch_size, observed_len, 1, 2)
                    observed_locals = observed_locals.contiguous().view(batch_size, observed_len, -1) # (B, observed_len, 26) 

                    target_locals   = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)    - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
                    target_locals   = target_locals.contiguous().view(batch_size, prediction_len, -1) # (B, target_len, 26)

                    predicted_globals = global_model(observed_globals, prediction_len = prediction_len)
                    loss_global       = criterion_global(predicted_globals, target_globals) # (B, target_len, 1, 2)

                    predicted_locals, mean, log_var = local_model(observed_locals, prediction_len = prediction_len) # (B, target_len, 26)
                    loss_local  = criterion_local(predicted_locals, target_locals, mean, log_var)

                    predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
                    predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
                    
                    loss = loss_global + 0.1 * loss_local
                    losses.append(loss.item())

                    predicted_poses = speed_to_pos(predicted_velocity, observed_poses) # (B, target_len, 26)
                    predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)
                
                else : 
                    predicted_velocities, mean, log_var = model(observed_velocities, prediction_len = prediction_len)
                    loss = criterion(predicted_velocities, target_velocities, mean, log_var)
                    losses.append(loss.item())
                    predicted_poses = speed_to_pos(predicted_velocity, observed_poses) # (B, target_len, 26)
                    predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

                predictions = compute_joints_metric(
                    pred      = predicted_poses[:, :, :, :2],  
                    gt        = target_poses[:, :, :, :2],  
                    time_step = prediction_len, 
                    dataset   = args.dataset, 
                    alpha     = args.pck_alpha
                )

                for dict in predictions:
                    for key in predictions[dict]:
                        total_predictions[dict][key] += predictions[dict][key]

    else : 
        raise NotImplementedError('Method not supported')
    return np.mean(losses), total_predictions


def get_model(args):
    if args.model == 'mlp': 
        model = get_mlp_model(
            pose_dim      = args.num_joints, 
            coordinates   = args.coordinates, 
            train_len     = args.num_training_frame, 
            predict_len   = args.num_testing_frame, 
            hidden_layers = args.mlp_hidden_layers, 
            dropout       = args.dropout, 
            device        = args.device, 
        )
        criterion = mlp_loss_function
        return model, criterion 
    elif args.model == 'lstm': 
        model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.coordinates, 
            embedding_dim  = args.lstm_embedding_dim,
            hidden_dim     = args.lstm_hidden_dim, 
            num_layers     = args.lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
        )
        criterion = lstm_loss_function 
        return model, criterion 
    
    elif args.model == 'motion_vae': 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.motion_vae_hidden_dim, 
            latent_dim      = args.motion_vae_latent_dim,
            encoding_layers = args.motion_vae_encoding_layers, 
            decoding_layers = args.motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        return vae_model, criterion_vae

    elif args.model == 'decouple_motion_vae': 
        # Motion Vae Comprises of two models, one for the root, and one for others 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.decouple_motion_vae_hidden_dim, 
            latent_dim      = args.decouple_motion_vae_latent_dim,
            encoding_layers = args.decouple_motion_vae_encoding_layers, 
            decoding_layers = args.decouple_motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        lstm_model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.coordinates, 
            embedding_dim  = args.decouple_lstm_embedding_dim,
            hidden_dim     = args.decouple_lstm_hidden_dim, 
            num_layers     = args.decouple_lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
        )

        criterion_lstm = lstm_loss_function

        return (vae_model, lstm_model), (criterion_vae, criterion_lstm)
    else : 
        raise NotImplementedError('Model not supported for now') 

def train(args):
    train_dataset = PennAction(
        num_training_frame = args.num_training_frame,
        num_testing_frame  = args.num_testing_frame,
        train              = True, 
        coordinates        = args.coordinates,  
        parent_relative    = args.parent_relative, 
        velocity           = args.velocity,  
        on_debug_mode      = args.mode == 'debug', 
        input_h            = args.boxsize, 
        input_w            = args.boxsize, 
    )
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True) 

    val_dataset = PennAction(
        num_training_frame = args.num_training_frame, 
        num_testing_frame  = args.num_testing_frame,
        train              = False,
        coordinates        = args.coordinates, 
        parent_relative    = args.parent_relative, 
        velocity           = args.velocity,    
        on_debug_mode      = args.mode == 'debug',   
        input_h            = args.boxsize, 
        input_w            = args.boxsize,  
    )
    val_loader   = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)

    print('-' * 40)
    print('Number of train batches =',      len(train_loader))
    print('Number of validation batches =', len(val_loader))

    print('-' * 40)

    device = args.device 
    print(device, 'is available')

    model_input_type = None 
    if args.coordinates : 
        model_input_type = 'coordinates'
    elif args.parent_relative : 
        model_input_type = 'parent_relative' 
    else : 
        model_input_type = 'velocity'

    checkpoint = f"checkpoint/{args.mode}/{model_input_type}_{args.model}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"
    start_epoch = 0 
    if os.path.exists(checkpoint): 
        model_dict = torch.load(checkpoint, weights_only = False) 
        start_epoch = model_dict['epoch'] + 1 
        model, criterion = get_model(args)
        model.load_state_dict(model_dict['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)
   
    else : 
        model, criterion = get_model(args)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)

    best_accuracy = 0
    loss_list, metric_list = [], []

    print("Start Training ...\n")
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_loss, train_predictions = train_per_epoch(args, model, train_loader, optimizer, criterion, epoch, device)
        val_loss, val_predictions     = evaluate_per_epoch(args, model, val_loader, criterion, device)

        scheduler.step(train_loss)

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc, curr_val_total_acc = train_acc['total'], val_acc['total']

        print(f'Training | Loss : {train_loss:.6f} | PCK accuracy : {curr_train_total_acc:.4f}')
        print(f'Validation | Loss : {val_loss:.6f} | PCK accuracy : {curr_val_total_acc:.4f}')

        if val_acc['total'] > best_accuracy : 
            best_accuracy = val_acc['total']
            save_checkpoint(
                epoch, 
                model, 
                optimizer, 
                model_name = f'{model_input_type}_{args.model}_pennaction_{args.num_training_frame}_{args.num_testing_frame}', 
                prefix_dir_name = f'checkpoint/{args.mode}/'
            )

    plot(
        loss_list   = np.array(loss_list), 
        metric_list = np.array(metric_list), 
        title       = args.title, 
        dataset     = args.dataset
    )

    # Saved the training logs 
    npy_data = dict()
    npy_data['loss_list']   = np.array(loss_list) 
    npy_data['metric_list'] = np.array(metric_list)
    log_saved_dir           = f'log/{args.mode}/{model_input_type}_{args.model}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.npy'
    np.save(log_saved_dir, npy_data)

    return model, loss_list, metric_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',            default = 'pennaction', type = str, required = False)
    parser.add_argument('--num_joints',         default = 13,     type = int,   required = False)
    parser.add_argument('--num_training_frame', default = 7,      type = int,   required = False)
    parser.add_argument('--num_testing_frame',  default = 7,      type = int,   required = False) 
    parser.add_argument('--learning_rate',      default = 1e-4,   type = float, required = False)
    parser.add_argument('--num_epochs',         default = 100,    type = int,   required = False) 
    parser.add_argument('--batch_size',         default = 16,     type = int,   required = False) 
    parser.add_argument('--pin_memory',         default = False,  type = bool,  required = False)
    parser.add_argument('--device',             default = 'cuda', type = str,   required = False) 
    parser.add_argument('--model' ,             default = 'lstm', type = str,   required = True) 
    parser.add_argument('--coordinates',        default = False,   type = bool,  required = False)
    parser.add_argument('--velocity',           default = False,  type = bool,  required = False)
    parser.add_argument('--parent_relative',    default = False,  type = bool,  required = False) 
    parser.add_argument('--mode',               default = 'main',  type = str,  required = True)
    parser.add_argument('--scheduler_step',     default = 10000,  type = float, required = False) 
    parser.add_argument('--scheduler_gamma',    default = 0.3333, type = float, required = False)
    parser.add_argument('--betas',              default = (0.9, 0.999), type = tuple, required = False) 
    parser.add_argument('--title',              type    = str, required = True)
    parser.add_argument('--dropout',            default = 0.2,  type = float, required = False)
    parser.add_argument('--pck_alpha',          default = 0.05, type = float, required = False)
    parser.add_argument('--boxsize',            default = 480,  type = int,   required = False)
    # MLP 
    parser.add_argument('--mlp_hidden_layers',  default = [1024, 2048, 1024], type = list[int], required = False) 
    
    # LSTM 
    parser.add_argument('--lstm_embedding_dim', default = 2048, type = int, required = False) 
    parser.add_argument('--lstm_hidden_dim'   , default = 256,  type = int, required = False)
    parser.add_argument('--lstm_num_layers',    default = 2,    type = int, required = False) 

    # MOTION VAE 
    parser.add_argument('--motion_vae_hidden_dim',       default = 32, type = int, required = False) 
    parser.add_argument('--motion_vae_latent_dim',       default = 16, type = int, required = False) 
    parser.add_argument('--motion_vae_encoding_layers',  default = 2,  type = int, required = False) 
    parser.add_argument('--motion_vae_decoding_layers',  default = 1,  type = int, required = False) 

    # DECOUPLE MOTION VAE
    parser.add_argument('--decouple_motion_vae_hidden_dim',       default = 32, type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_latent_dim',       default = 16, type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_encoding_layers',  default = 2,  type = int, required = False) 
    parser.add_argument('--decouple_motion_vae_decoding_layers',  default = 1,  type = int, required = False) 
    parser.add_argument('--decouple_lstm_embedding_dim',          default = 8,   type = int, required = False)
    parser.add_argument('--decouple_lstm_hidden_dim',             default = 16,  type = int, required = False)
    parser.add_argument('--decouple_lstm_num_layers',             default = 1,   type = int, required = False)

    # TRANSFORMER 
    args = parser.parse_args()
    model, loss_list, metric_list = train(args) 


