import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Util import *
from data.PennAction import PennAction
import os 
import argparse 

PRINT_STEP = 5

'''
:TODO : PREPARE FOR MOTION VAE and Attention Motion VAE, Transformer 
:TODO : Maybe add weight decay for training transformer, in that case, remove LR scheduler. 
'''

def process_per_epoch_coordinates(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):
    """
    Input represented as absolute coordinates
    Could be trained by MLP, LSTM, TRANSFORMER

    :param model: model to train 
    :param train_loader: train data loader 
    :param val_loader  : validation data loader 
    :param optimizer: set default to ADAM 
    :param criterion: loss function
    :param epoch: current epoch
    :param device: 'cuda' 
    """

    train_losses = []
    valid_losses   = []
    train_total_predictions = get_pennaction_predictions_dict()
    valid_total_predictions = get_pennaction_predictions_dict()

    print_step = PRINT_STEP
    
    model.train()
    for batch_idx, (observed_poses, target_poses) in enumerate(train_loader):
        observed_poses = observed_poses.to(device, dtype = torch.float32)
        target_poses   = target_poses.to(device, dtype = torch.float32)
            
        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]
            
        predicted_poses = model(observed_poses, prediction_len)
        loss = criterion(predicted_poses, target_poses, prediction_len) 
        
        train_losses.append(loss.item())

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
                train_total_predictions[dict][key] += predictions[dict][key]

        if batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 
    
    model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses) in enumerate(val_loader):
            observed_poses = observed_poses.to(device, dtype = torch.float32)
            target_poses   = target_poses.to(device,   dtype = torch.float32)
            batch          = target_poses.shape[0]
            prediction_len = target_poses.shape[1]
            predicted_poses = model(observed_poses, prediction_len)
            loss = criterion(predicted_poses, target_poses, prediction_len) 
            
            valid_losses.append(loss.item())

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
                    valid_total_predictions[dict][key] += predictions[dict][key]

    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_parent_relatives(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):    
    """
    Input represented as Parent Relative 
    Could be trained using MLP, LSTM, TRANSFORMER

    :param model        : model to train 
    :param train_loader : train data loader 
    :param val_loader   : validation loader
    :param optimizer    : set default to ADAM 
    :param criterion    : loss function
    :param epoch        : current epoch
    :param device       : cuda 
    """

    train_losses = []
    valid_losses = [] 

    train_total_predictions = get_pennaction_predictions_dict()
    valid_total_predictions = get_pennaction_predictions_dict()

    print_step = PRINT_STEP
    model.train()
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

        train_losses.append(loss.item())
            
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
                train_total_predictions[dict][key] += predictions[dict][key]

        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 
    
    model.eval()
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

            valid_losses.append(loss.item())

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
                    valid_total_predictions[dict][key] += predictions[dict][key]

    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_velocity(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):
    """
    Input represented as velocity 
    Can be trained using MLP, LSTM, MOTION_VAE, ATTENTION_MOTION_VAE, TRANSFORMER
    For now supported only VAE type model 

    :param model        : model to train 
    :param train_loader : train data loader
    :param val_loader   : valid data loader 
    :param optimizer    : set default to ADAM 
    :param criterion    : loss function
    :param epoch        : current epoch
    :param device       : 'cuda' 
    
    """
   
    train_losses = []
    valid_losses = []
    
    train_total_predictions = get_pennaction_predictions_dict()
    valid_total_predictions = get_pennaction_predictions_dict()

    print_step = PRINT_STEP
     
    model.train()         
    for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(train_loader):
            observed_poses = observed_poses.to(device, dtype = torch.float32)           # (B, T_train, J * 2)
            target_poses   = target_poses.to(device,   dtype = torch.float32)           # (B, T_test,  J * 2)
            observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
            target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test,  J * 2)
            
            batch_size, observed_len,   _ = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 
            
            # For motion use VAE to model
            predicted_velocities, mean, log_var = model(observed_velocities, prediction_len = prediction_len)
            
            loss = criterion(predicted_velocities, target_velocities, mean, log_var)
            
            train_losses.append(loss.item())

            
            #with torch.autograd.detect_anomaly(): 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
                
            target_poses   = target_poses.contiguous().view(batch_size, prediction_len,   -1, 3) 
            
            #num_observed_velocity + 1 = num_poses 
            observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1, 3)[:, :, :, :2]
            observed_poses = observed_poses.contiguous().view(batch_size, observed_len + 1, -1)

            predicted_poses = speed_to_pos(
                prediction_velocities =  predicted_velocities, 
                observed_poses        =  observed_poses, 
                device                =  args.device, 
            ) # (B, target_len, 26)

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
                    train_total_predictions[dict][key] += predictions[dict][key]    
            
            if batch_idx > 0 and batch_idx % print_step == 0:
                acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 

    model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(val_loader):
            
            observed_poses = observed_poses.to(device, dtype = torch.float32)           # (B, T_train, J * 2)
            target_poses   = target_poses.to(device,   dtype = torch.float32)           # (B, T_test,  J * 2)
            observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
            target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test,  J * 2)
                
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 

            predicted_velocities, mean, log_var = model(observed_velocities, prediction_len = prediction_len)
            
            loss = criterion(predicted_velocities, target_velocities, mean, log_var)
                
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
                dataset   = args.dataset, 
                alpha     = args.pck_alpha
            )

            for dict in predictions:
                for key in predictions[dict]:
                    valid_total_predictions[dict][key] += predictions[dict][key]

    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_decouple_velocity(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):
    """
    Input represented as decoupled velocity
    For details, refer to this paper  
    https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Parsaeifard_Learning_Decoupled_Representations_for_Human_Pose_Forecasting_ICCVW_2021_paper.pdf
    
    Can be trained using MLP, LSTM, MOTION_VAE, ATTENTION_MOTION_VAE, TRANSFORMER
    For now supported only combination of VAE and LSTM

    :param model        : model to train 
    :param train_loader : train dataloader
    :param val_loader   : valid dataloader 
    :param optimizer    : set default to ADAM 
    :param criterion    : loss function
    :param epoch        : current epoch
    :param device       : 'cuda' 
    """

    train_losses = []
    valid_losses = [] 

    train_total_predictions = get_pennaction_predictions_dict()
    valid_total_predictions = get_pennaction_predictions_dict()

    print_step = PRINT_STEP
    
    local_model, global_model = model 
    criterion_local, criterion_global = criterion
    
    local_model.train() 
    global_model.train()

    for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(train_loader):
        
        observed_poses = observed_poses.to(device, dtype = torch.float32) # (B, T_train, J * 2)
        target_poses   = target_poses.to(device,   dtype = torch.float32) # (B, T_test,   J * 2)
        observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
        target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test, J * 2)
            
        batch_size, observed_len, _   = observed_velocities.shape
        batch_size, prediction_len, _ = target_velocities.shape 

        observed_globals = 0.5 * (
            observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 7] + 
            observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 8]
        ) 

        target_globals   = 0.5 * ( 
            target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 7] + 
            target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 8]
        ) 

        observed_locals = observed_velocities.contiguous().view(batch_size, observed_len, -1, 2) - observed_globals.contiguous().view(batch_size, observed_len, 1, 2)
        observed_locals = observed_locals.contiguous().view(batch_size, observed_len, -1) 

        target_locals   = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)    - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
        target_locals   = target_locals.contiguous().view(batch_size, prediction_len, -1) 

        global_model.zero_grad()
        predicted_globals = global_model(observed_globals, prediction_len = prediction_len)
        loss_global       = criterion_global(predicted_globals, target_globals) 

        local_model.zero_grad() 
        predicted_locals, mean, log_var = local_model(observed_locals, prediction_len = prediction_len) 
        loss_local  = criterion_local(predicted_locals, target_locals, mean, log_var)

        predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
        predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
        
        loss = loss_global + 0.1 * loss_local
        
        train_losses.append(loss.item())
        
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        predicted_poses = speed_to_pos(predicted_velocity, observed_poses) 
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
                train_total_predictions[dict][key] += predictions[dict][key]    
            
            if batch_idx > 0 and batch_idx % print_step == 0:
                acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
                print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}')

    local_model.eval() 
    global_model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses, observed_velocities, target_velocities) in enumerate(val_loader):
            
            observed_poses = observed_poses.to(device, dtype = torch.float32)           # (B, T_train, J * 2)
            target_poses   = target_poses.to(device,   dtype = torch.float32)           # (B, T_test,  J * 2)
            observed_velocities = observed_velocities.to(device, dtype = torch.float32) # (B, T_train, J * 2)
            target_velocities   = target_velocities.to(device,   dtype = torch.float32) # (B, T_test,  J * 2)
                
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 

            observed_globals = 0.5 * (
                observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 7] + 
                observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 8]
            ) 

            target_globals   = 0.5 * ( 
                target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 7] + 
                target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 8]
            ) 

            observed_locals = observed_velocities.contiguous().view(batch_size, observed_len, -1, 2) - observed_globals.contiguous().view(batch_size, observed_len, 1, 2)
            observed_locals = observed_locals.contiguous().view(batch_size, observed_len, -1) 

            target_locals   = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)    - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
            target_locals   = target_locals.contiguous().view(batch_size, prediction_len, -1) 

            predicted_globals = global_model(observed_globals, prediction_len = prediction_len)
            loss_global       = criterion_global(predicted_globals, target_globals) 

            predicted_locals, mean, log_var = local_model(observed_locals, prediction_len = prediction_len) 
            loss_local  = criterion_local(predicted_locals, target_locals, mean, log_var)

            predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
            predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
                
            loss = loss_global + 0.1 * loss_local
            valid_losses.append(loss.item())

            predicted_poses = speed_to_pos(predicted_velocity, observed_poses) 
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
                    valid_total_predictions[dict][key] += predictions[dict][key]
    
    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


PENNACTION_INPUT_REPRESENTATION_DICT = { 
    'coordinates'       : process_per_epoch_coordinates, 
    'parent_relatives'  : process_per_epoch_parent_relatives, 
    'velocity'          : process_per_epoch_velocity,  
    'decouple_velocity' : process_per_epoch_decouple_velocity, 
}

def train(args):
    train_dataset = PennAction(
        num_training_frame = args.num_training_frame,
        num_testing_frame  = args.num_testing_frame,
        train              = True, 
        coordinates        = args.coordinates,  
        parent_relative    = args.parent_relative, 
        velocity           = args.velocity or args.decouple_velocity,  
        on_debug_mode      = args.mode == 'debug', 
        box_size           = args.boxsize,     
    )
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = args.pin_memory) 

    val_dataset = PennAction(
        num_training_frame = args.num_training_frame, 
        num_testing_frame  = args.num_testing_frame,
        train              = False,
        coordinates        = args.coordinates, 
        parent_relative    = args.parent_relative, 
        velocity           = args.velocity or args.decouple_velocity ,    
        on_debug_mode      = args.mode == 'debug',   
        box_size           = args.boxsize, 
    )
    val_loader   = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)

    print('-' * 40)
    print('Number of train batches =',      len(train_loader))
    print('Number of validation batches =', len(val_loader))

    print('-' * 40)

    device = args.device 
    print(device, 'is available')

    model_input_type, process_per_epoch = None, None 
    if args.coordinates : 
        model_input_type  = 'coordinates'
        process_per_epoch = PENNACTION_INPUT_REPRESENTATION_DICT['coordinates']
    elif args.parent_relative : 
        model_input_type  = 'parent_relative' 
        process_per_epoch = PENNACTION_INPUT_REPRESENTATION_DICT['parent_relatives']
    elif args.velocity: 
        model_input_type  = 'velocity'
        process_per_epoch = PENNACTION_INPUT_REPRESENTATION_DICT['velocity']
    elif args.decouple_velocity : 
        model_input_type  = 'decouple_velocity'  
        process_per_epoch = PENNACTION_INPUT_REPRESENTATION_DICT['decouple_velocity']
    else : 
        raise NotImplementedError('Representation not supported')


    checkpoint = f"checkpoint/{args.mode}/{model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"

    start_epoch = 0 

    if os.path.exists(checkpoint): 
        model_dict = torch.load(checkpoint, weights_only = False) 
        start_epoch = model_dict['epoch'] + 1 
        model, criterion = get_model(args)
        model.to(device)
        model.load_state_dict(model_dict['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)
   
    else : 
        model, criterion = get_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 25, threshold = 1e-5)
        model.to(device)

    best_accuracy = 0
    loss_list, metric_list = [], []

    print("Start Training ...\n")
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_loss, train_predictions, val_loss, val_predictions = process_per_epoch(args, model, train_loader, val_loader, optimizer, criterion, epoch, device)

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
                model_name = f'{model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}', 
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
    log_saved_dir           = f'log/{args.mode}/{model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.npy'
    np.save(log_saved_dir, npy_data)

    return model, loss_list, metric_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',            default = 'pennaction', type = str, required = False)
    parser.add_argument('--model_version',                        type = str,   required = False)
    parser.add_argument('--num_joints',         default = 13,     type = int,   required = False)
    parser.add_argument('--num_training_frame', default = 7,      type = int,   required = False)
    parser.add_argument('--num_testing_frame',  default = 7,      type = int,   required = False) 
    parser.add_argument('--learning_rate',      default = 1e-4,   type = float, required = False)
    parser.add_argument('--num_epochs',         default = 100,    type = int,   required = False) 
    parser.add_argument('--batch_size',         default = 16,     type = int,   required = False) 
    parser.add_argument('--num_workers',        default = 8,      type = int,   required = False)
    parser.add_argument('--pin_memory',         default = True,   type = bool,  required = False)
    parser.add_argument('--device',             default = 'cuda', type = str,   required = False) 
    parser.add_argument('--model' ,             default = 'lstm', type = str,   required = True) 
    parser.add_argument('--coordinates',        default = False,  type = bool,  required = False)
    parser.add_argument('--velocity',           default = False,  type = bool,  required = False)
    parser.add_argument('--decouple_velocity',  default = False,  type = bool,  required = False)
    parser.add_argument('--parent_relative',    default = False,  type = bool,  required = False) 
    parser.add_argument('--mode',               default = 'baseline', type = str,  required = True)
    parser.add_argument('--scheduler_step',     default = 10000,  type = float,    required = False) 
    parser.add_argument('--scheduler_gamma',    default = 0.3333, type = float,    required = False)
    parser.add_argument('--betas',              default = (0.9, 0.999), type = tuple, required = False) 
    parser.add_argument('--title',              type    = str, required = True)
    parser.add_argument('--dropout',            default = 0.2,     type = float, required = False)
    parser.add_argument('--pck_alpha',          default = 0.05,    type = float, required = False)
    parser.add_argument('--boxsize',            default = 480,     type = int,   required = False)
    parser.add_argument('--temperature',        default = 10000,   type = float, required = False) 
    parser.add_argument('--alpha',              default = 1,       type = float, required = False)
    parser.add_argument('--weight_decay',       default = 0.00001, type = float, required = False)


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


    # ATTENTION MOTION VAE 
    parser.add_argument('--attention_motion_vae_num_heads',           default = 8,    type = int, required = False)
    parser.add_argument('--attention_motion_vae_model_dim',           default = 512,  type = int, required = False)
    parser.add_argument('--attention_motion_vae_query_dim',           default = 64,   type = int, required = False) 
    parser.add_argument('--attention_motion_vae_value_dim',           default = 64,   type = int, required = False)
    parser.add_argument('--attention_motion_vae_fc_hidden_dim',       default = 2048, type = int, required = False)
    parser.add_argument('--attention_motion_vae_num_encoding_layers', default = 6,    type = int, required = False)
    parser.add_argument('--attention_motion_vae_num_decoding_layers', default = 1,    type = int, required = False)
    parser.add_argument('--attention_motion_vae_latent_dim',          default = 128,  type = int, required = False)


    # DECOUPLE MOTION VAE
    parser.add_argument('--decouple_motion_vae_hidden_dim',       default = 32, type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_latent_dim',       default = 16, type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_encoding_layers',  default = 2,  type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_decoding_layers',  default = 1,  type  = int, required = False) 
    parser.add_argument('--decouple_lstm_embedding_dim',          default = 8,   type = int, required = False)
    parser.add_argument('--decouple_lstm_hidden_dim',             default = 16,  type = int, required = False)
    parser.add_argument('--decouple_lstm_num_layers',             default = 1,   type = int, required = False)

    # TRANSFORMER 


    
    args = parser.parse_args()
    model, loss_list, metric_list = train(args) 


