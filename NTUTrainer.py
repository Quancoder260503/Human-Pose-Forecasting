import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from model.Lstm      import *
from model.Mlp       import *
from model.RnnMotionVae import *
from Util    import *
from data.NTU        import *
import os 
import argparse 

PRINT_STEP = 100

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(LOCAL_RANK)


def process_per_epoch_coordinates(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):
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
    
    train_total_predictions = get_ntu_predictions_dict()
    valid_total_predictions = get_ntu_predictions_dict()

    print_step = PRINT_STEP
    
    model.train()
    for batch_idx, (observed_poses, target_poses, _, _) in enumerate(train_loader):
        observed_poses = observed_poses.to(device, dtype=torch.float32)
        target_poses   = target_poses.to(device,   dtype=torch.float32)
            
        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]

        predicted_poses = model(observed_poses, prediction_len)
        loss = criterion(predicted_poses, target_poses, prediction_len) 
        
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_poses = predicted_poses.contiguous().view(batch, prediction_len, -1, 3) 
        target_poses    = target_poses.contiguous().view(batch, prediction_len, -1, 3) 

        predictions = compute_joints_metric(
            pred     =predicted_poses, 
            gt       =target_poses,  
            time_step=prediction_len, 
            dataset  =args.dataset, 
            alpha    =args.pck_alpha, 
        )

        for metric in predictions:
            for key in predictions[metric]:
                    train_total_predictions[metric][key] += predictions[metric][key]

        if batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 

    model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses, _, _) in enumerate(val_loader):
            observed_poses = observed_poses.to(device, dtype=torch.float32)
            target_poses   = target_poses.to(device,   dtype=torch.float32)
            
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
                alpha     = args.pck_alpha,  
            )

            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_predictions[metric][key] += predictions[metric][key]

    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_parent_relatives(args, model, train_loader, val_loader, optimizer, criterion, epoch, device): 
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
    
    train_total_predictions = get_ntu_predictions_dict()
    valid_total_predictions = get_ntu_predictions_dict()

    print_step = PRINT_STEP
    
    model.train()
    for batch_idx, (observed_bones, target_bones, observed_poses, target_poses, scale_factor, _, _) in enumerate(train_loader):
        observed_bones = observed_bones.to(device, dtype=torch.float32)
        target_bones   = target_bones.to(device,   dtype=torch.float32)
        observed_poses = observed_poses.to(device, dtype=torch.float32)
        target_poses   = target_poses.to(device,   dtype=torch.float32)
        scale_factor   = scale_factor.to(device,   dtype=torch.float32)

        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]

        predicted_bones = model(observed_bones, prediction_len)
        loss            = criterion(predicted_bones, target_bones, prediction_len) 

        train_losses.append(loss.item())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_poses  = reconstruct_pose(
            compressed_poses = predicted_bones, 
            scaled_factor    = scale_factor,  
            dataset          = args.dataset, 
            box_size         = args.boxsize, 
        )

        construct_target = reconstruct_pose(
            compressed_poses = target_bones,    
            scaled_factor    = scale_factor, 
            dataset          = args.dataset, 
            box_size         = args.boxsize, 
        )
        
            
        predictions = compute_joints_metric(
            pred     =predicted_poses[:, :, :,  :2],  
            gt       =construct_target[:, :, :, :2],  
            time_step=prediction_len, 
            dataset  =args.dataset, 
            alpha    =args.pck_alpha 
        )

        for metric in predictions:
            for key in predictions[metric]:
                train_total_predictions[metric][key] += predictions[metric][key]

        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}')

    
    model.eval()
    with torch.no_grad():
        for _, (observed_bones, target_bones, observed_poses, target_poses, scale_factor, _, _) in enumerate(val_loader):
            
            observed_bones = observed_bones.to(device, dtype=torch.float32)
            target_bones   = target_bones.to(device,   dtype=torch.float32)
            observed_poses = observed_poses.to(device, dtype=torch.float32)
            target_poses   = target_poses.to(device,   dtype=torch.float32)
            scale_factor   = scale_factor.to(device,   dtype=torch.float32)
            
            batch          = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_bones = model(observed_bones, prediction_len)
            
            loss = criterion(predicted_bones, target_bones, prediction_len) 
            valid_losses.append(loss.item())

            predicted_poses  = reconstruct_pose(
                compressed_poses = predicted_bones, 
                scaled_factor    = scale_factor,  
                dataset          = args.dataset, 
                box_size         = args.boxsize, 
            )

            construct_target = reconstruct_pose(
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

            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_predictions[metric][key] += predictions[metric][key]
    
    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions
   

def process_per_epoch_root_relatives(args, model, train_loader, val_loader, optimizer, criterion, epoch, device): 
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
    
    train_total_predictions = get_ntu_predictions_dict()
    valid_total_predictions = get_ntu_predictions_dict()

    print_step = PRINT_STEP
    
    model.train()
    for batch_idx, (observed_root_vectors, target_root_vectors, observed_poses, target_poses, scale_factor, _, _) in enumerate(train_loader):
        
        observed_root_vectors = observed_root_vectors.to(device, dtype=torch.float32)
        target_root_vectors   = target_root_vectors.to(device,   dtype=torch.float32)
        observed_poses        = observed_poses.to(device,        dtype=torch.float32)
        target_poses          = target_poses.to(device,          dtype=torch.float32)
        scale_factor          = scale_factor.to(device,          dtype=torch.float32)

        batch          = target_poses.shape[0]
        prediction_len = target_poses.shape[1]

        predicted_root_vectors = model(observed_root_vectors, prediction_len)
        loss                   = criterion(predicted_root_vectors, target_root_vectors, prediction_len) 

        train_losses.append(loss.item())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_poses  = reconstruct_pose_from_root(
            compressed_poses=predicted_root_vectors, 
            scaled_factor   =scale_factor,  
            box_size        =args.boxsize, 
        )
        construct_target = reconstruct_pose_from_root(
            compressed_poses=target_root_vectors,    
            scaled_factor   =scale_factor, 
            box_size        =args.boxsize, 
        )
        
        predictions = compute_joints_metric(
            pred     =predicted_poses[:, :, :,  :2],  
            gt       =construct_target[:, :, :, :2],  
            time_step=prediction_len, 
            dataset  =args.dataset, 
            alpha    =args.pck_alpha 
        )

        for metric in predictions:
            for key in predictions[metric]:
                train_total_predictions[metric][key] += predictions[metric][key]

        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 

    model.eval()
    with torch.no_grad():
        for _, (observed_root_vectors, target_root_vectors, observed_poses, target_poses, scale_factor, _, _) in enumerate(val_loader):
            
            observed_root_vectors = observed_root_vectors.to(device, dtype=torch.float32)
            target_root_vectors   = target_root_vectors.to(device,   dtype=torch.float32)
            observed_poses        = observed_poses.to(device,        dtype=torch.float32)
            target_poses          = target_poses.to(device,          dtype=torch.float32)
            scale_factor          = scale_factor.to(device,          dtype=torch.float32)

            batch          = target_poses.shape[0]
            prediction_len = target_poses.shape[1]

            predicted_root_vectors = model(observed_root_vectors, prediction_len)
            loss                   = criterion(predicted_root_vectors, target_root_vectors, prediction_len) 
            
            valid_losses.append(loss.item())

            predicted_poses  = reconstruct_pose_from_root(
                compressed_poses=predicted_root_vectors, 
                scaled_factor   =scale_factor,  
                dataset         =args.dataset, 
                box_size        =args.boxsize, 
            )
            construct_target = reconstruct_pose_from_root(
                compressed_poses=target_root_vectors,    
                scaled_factor   =scale_factor, 
                dataset         =args.dataset, 
                box_size        =args.boxsize, 
            )
            
            predictions = compute_joints_metric(
                pred     =predicted_poses[:, :, :,  :2],  
                gt       =construct_target[:, :, :, :2],  
                time_step=prediction_len, 
                dataset  =args.dataset, 
                alpha    =args.pck_alpha 
            )

            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_predictions[metric][key] += predictions[metric][key]
    
    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_velocity(args, model, train_loader, val_loader, optimizer, criterion, epoch, device): 
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
    
    train_total_predictions = get_ntu_predictions_dict()
    valid_total_predictions = get_ntu_predictions_dict()

    print_step = PRINT_STEP
    
    model.train()
    for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(train_loader):
            
        observed_poses      = observed_poses.to(device,      dtype=torch.float32)
        target_poses        = target_poses.to(device,        dtype=torch.float32)
        observed_velocities = observed_velocities.to(device, dtype=torch.float32)
        target_velocities   = target_velocities.to(device,   dtype=torch.float32)
            
        batch_size, observed_len, _   = observed_velocities.shape
        batch_size, prediction_len, _ = target_velocities.shape 

        predicted_velocities, mean, log_var = model(observed_velocities, prediction_len=prediction_len)
        loss = criterion(predicted_velocities, target_velocities, mean, log_var)
        
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_poses = speed_to_pos(predicted_velocities, observed_poses)
        
        predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)
        target_poses_reshaped = target_poses.contiguous().view(batch_size, prediction_len, -1, 2)

        predictions = compute_joints_metric(
            pred      = predicted_poses[:, :, :, :2],  
            gt        = target_poses_reshaped[:, :, :, :2],  
            time_step = prediction_len, 
            dataset   = args.dataset, 
            alpha     = args.pck_alpha, 
        )

        for metric in predictions:
            for key in predictions[metric]:
                train_total_predictions[metric][key] += predictions[metric][key]
        
        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 

    model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(val_loader):
            
            observed_poses      = observed_poses.to(device,      dtype=torch.float32)
            target_poses        = target_poses.to(device,        dtype=torch.float32)
            observed_velocities = observed_velocities.to(device, dtype=torch.float32)
            target_velocities   = target_velocities.to(device,   dtype=torch.float32)
                
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 


            predicted_velocities, mean, log_var = model(observed_velocities, prediction_len = prediction_len)
            loss = criterion(predicted_velocities, target_velocities, mean, log_var)
            valid_losses.append(loss.item())
                

            predicted_poses = speed_to_pos(predicted_velocities, observed_poses)
            
            predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)
            target_poses_reshaped = target_poses.contiguous().view(batch_size, prediction_len, -1, 2)

            predictions = compute_joints_metric(
                pred     =predicted_poses[:, :, :, :2],  
                gt       =target_poses_reshaped[:, :, :, :2],  
                time_step=prediction_len, 
                dataset  =args.dataset, 
                alpha    =args.pck_alpha
            )

            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_predictions[metric][key] += predictions[metric][key]
    
    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions


def process_per_epoch_decouple_velocity(args, model, train_loader, val_loader, optimizer, criterion, epoch, device):
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
    
    train_total_predictions = get_ntu_predictions_dict()
    valid_total_predictions = get_ntu_predictions_dict()

    print_step = PRINT_STEP
    
    local_model, global_model = model 
    criterion_local, criterion_global = criterion
    
    local_model.train() 
    global_model.train()

    for batch_idx, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(train_loader):
        observed_poses      = observed_poses.to(device, dtype = torch.float32)      # (B, T_train, J * 2)
        target_poses        = target_poses.to(device,   dtype = torch.float32)      # (B, T_test,   J * 2)
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

        target_locals   = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2) - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
        target_locals   = target_locals.contiguous().view(batch_size, prediction_len, -1) 

        global_model.zero_grad()
        predicted_globals = global_model(observed_globals, prediction_len = prediction_len)
        loss_global       = criterion_global(predicted_globals, target_globals) # (B, target_len, 1, 2)

        local_model.zero_grad() 
        predicted_locals, mean, log_var = local_model(observed_locals, prediction_len = prediction_len) # (B, target_len, 26)
        loss_local  = criterion_local(predicted_locals, target_locals, mean, log_var)

        predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
        predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
                
        loss = loss_global + 0.1 * loss_local
        train_losses.append(loss.item())
        
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

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
                train_total_predictions[dict][key] += predictions[dict][key]    
        
        if batch_idx > 0 and batch_idx % print_step == 0:
            acc = train_total_predictions['total']['correct'] / train_total_predictions['total']['all']
            print(f'Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} |  Loss = {np.mean(train_losses):.6f} | PCK accuracy = {acc:.4f}') 


    local_model.eval() 
    global_model.eval()
    with torch.no_grad():
        for _, (observed_poses, target_poses, observed_velocities, target_velocities, _, _) in enumerate(val_loader):
        
            observed_poses      = observed_poses.to(device,      dtype=torch.float32)
            target_poses        = target_poses.to(device,        dtype=torch.float32)
            observed_velocities = observed_velocities.to(device, dtype=torch.float32)
            target_velocities   = target_velocities.to(device,   dtype=torch.float32)
            
            batch_size, observed_len, _   = observed_velocities.shape
            batch_size, prediction_len, _ = target_velocities.shape 

            observed_globals = 0.5 * (
                observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 7] + 
                observed_velocities.contiguous().view(batch_size, observed_len, -1, 2)[:, :, 8]
            )
            target_globals = 0.5 * ( 
                target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 7] + 
                target_velocities.contiguous().view(batch_size, prediction_len, -1, 2)[:, :, 8]
            )

            observed_locals = observed_velocities.contiguous().view(batch_size, observed_len, -1, 2) - observed_globals.contiguous().view(batch_size, observed_len, 1, 2)
            observed_locals = observed_locals.contiguous().view(batch_size, observed_len, -1)

            target_locals = target_velocities.contiguous().view(batch_size, prediction_len, -1, 2) - target_globals.contiguous().view(batch_size, prediction_len, 1, 2)
            target_locals = target_locals.contiguous().view(batch_size, prediction_len, -1)

            predicted_globals = global_model(observed_globals, prediction_len=prediction_len)
            loss_global       = criterion_global(predicted_globals, target_globals)

            predicted_locals, mean, log_var = local_model(observed_locals, prediction_len=prediction_len)
            loss_local = criterion_local(predicted_locals, target_locals, mean, log_var)

            predicted_velocity = predicted_globals.contiguous().view(batch_size, prediction_len, 1, 2) + predicted_locals.contiguous().view(batch_size, prediction_len, -1, 2)
            predicted_velocity = predicted_velocity.contiguous().view(batch_size, prediction_len, -1)
            
            loss = loss_global + 0.1 * loss_local
            valid_losses.append(loss.item())


            predicted_poses = speed_to_pos(predicted_velocity, observed_poses)
            predicted_poses = predicted_poses.contiguous().view(batch_size, prediction_len, -1, 2)

            target_poses_reshaped = target_poses.contiguous().view(batch_size, prediction_len, -1, 2)

            predictions = compute_joints_metric(
                pred     =predicted_poses[:, :, :, :2],  
                gt       =target_poses_reshaped[:, :, :, :2],  
                time_step=prediction_len, 
                dataset  =args.dataset, 
                alpha    =args.pck_alpha
            )

            for metric in predictions:
                for key in predictions[metric]:
                    valid_total_predictions[metric][key] += predictions[metric][key]
    
    return np.mean(train_losses), train_total_predictions, np.mean(valid_losses), valid_total_predictions
  

NTU_INPUT_REPRESENTATION_DICT = { 
    'coordinates'       : process_per_epoch_coordinates, 
    'parent_relatives'  : process_per_epoch_parent_relatives,
    'root_relatives'    : process_per_epoch_root_relatives, 
    'velocity'          : process_per_epoch_velocity,  
    'decouple_velocity' : process_per_epoch_decouple_velocity, 
}

def train(args):
    train_dataset = NTU(
        num_training_frame=args.num_training_frame,
        num_testing_frame =args.num_testing_frame,
        train             =True, 
        coordinates       =args.coordinates,  
        parent_relative   =args.parent_relative, 
        velocity          =args.velocity or args.decouple_velocity,  
        on_debug_mode     =args.mode == 'debug', 
        input_h           =args.boxsize, 
        input_w           =args.boxsize, 
    )

    train_sampler = DistributedSampler(train_dataset)
    train_loader  = DataLoader(
        train_dataset, 
        batch_size =args.batch_size, 
        sampler    =train_sampler,     
        num_workers=args.num_workers, 
        pin_memory =args.pin_memory
    ) 

    val_dataset = NTU(
        num_training_frame=args.num_training_frame, 
        num_testing_frame =args.num_testing_frame,
        train             =False,
        coordinates       =args.coordinates, 
        parent_relative   =args.parent_relative, 
        velocity          =args.velocity or args.decouple_velocity,    
        on_debug_mode     =args.mode == 'debug',   
        input_h           =args.boxsize, 
        input_w           =args.boxsize,  
    )
    val_sampler = DistributedSampler(val_dataset, shuffle = False)
    val_loader  = DataLoader(
        val_dataset, 
        batch_size =args.batch_size, 
        sampler    =val_sampler, 
        num_workers=args.num_workers, 
        pin_memory =args.pin_memory
    )

    print('-' * 40)
    print('Number of train batches =',      len(train_loader))
    print('Number of validation batches =', len(val_loader))
    print('-' * 40)

    device = torch.device(f'cuda:{LOCAL_RANK}')
    print(device, 'is available')

    model_input_type, process_per_epoch = None, None  
    if args.coordinates:
        model_input_type  = 'coordinates'
        process_per_epoch = NTU_INPUT_REPRESENTATION_DICT['coordinates'] 
    elif args.parent_relative:
        model_input_type = 'parent_relative' 
        process_per_epoch = NTU_INPUT_REPRESENTATION_DICT['parent_relatives']
    elif args.velocity:
        model_input_type = 'velocity'
        process_per_epoch = NTU_INPUT_REPRESENTATION_DICT['velocity']
    elif args.root_relative:
        model_input_type = 'root_relative' 
        process_per_epoch = NTU_INPUT_REPRESENTATION_DICT['root_relatives']
    elif args.decouple_velocity : 
        model_input_type = 'decouple_velocity'
        process_per_epoch = NTU_INPUT_REPRESENTATION_DICT['decouple_velocity']
    else:
        raise NotImplementedError('Method not supported')
    
    checkpoint  = f"checkpoint/{args.mode}/{model_input_type}_{args.model}_{args.model_version}_ntu_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"
    start_epoch = 0 

    if os.path.exists(checkpoint): 
        model_dict  = torch.load(checkpoint, weights_only=False) 
        start_epoch = model_dict['epoch'] + 1 
        model, criterion = get_model(args)
        model.load_state_dict(model_dict['model'])
        model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[LOCAL_RANK])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas)
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=1e-5)
    else:
        model, criterion = get_model(args)
        model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[LOCAL_RANK])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25, threshold=1e-5)

    best_accuracy = 0
    loss_list, metric_list = [], []

    print("Start Training ...\n")
    for epoch in range(start_epoch, start_epoch + args.num_epochs):

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss, train_predictions, val_loss, val_predictions = process_per_epoch(args, model, train_loader, val_loader, optimizer, criterion, epoch, device)

        scheduler.step(train_loss)

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc = train_acc['total']
        curr_val_total_acc   = val_acc['total']

        print(f'Training | Loss : {train_loss:.6f} | PCK accuracy : {curr_train_total_acc:.4f}')
        print(f'Validation | Loss : {val_loss:.6f} | PCK accuracy : {curr_val_total_acc:.4f}')

        if LOCAL_RANK == 0 and val_acc['total'] > best_accuracy:
            best_accuracy = val_acc['total']
            save_checkpoint(
                epoch, 
                model, 
                optimizer, 
                model_name     = f'{model_input_type}_{args.model}_{args.model_version}_ntu_{args.num_training_frame}_{args.num_testing_frame}', 
                prefix_dir_name= f'checkpoint/{args.mode}/'
            )

    if LOCAL_RANK == 0:
        log_saved_dir = f'log/{args.mode}/{model_input_type}_{args.model}_{args.model_version}_ntu_{args.num_training_frame}_{args.num_testing_frame}'
        plot(np.array(loss_list), metric_list, args.title, 'ntu', save_path=f'{log_saved_dir}.png')
        np.save(f'{log_saved_dir}.npy', {
            'loss_list'  : np.array(loss_list),
            'metric_list': metric_list,
        }, 
        allow_pickle=True)

    return model, loss_list, metric_list


if __name__ == '__main__':
    # Model version can be noted as v_0, v_1, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',            default='ntu',         type=str,   required=False)
    parser.add_argument('--model_version',                             type=str,   required=False)
    parser.add_argument('--num_joints',         default=25,            type=int,   required=False)
    parser.add_argument('--num_training_frame', default=14,            type=int,   required=False)
    parser.add_argument('--num_testing_frame',  default=7,             type=int,   required=False) 
    parser.add_argument('--learning_rate',      default=1e-4,          type=float, required=False)
    parser.add_argument('--num_epochs',         default=100,           type=int,   required=False) 
    parser.add_argument('--batch_size',         default=32,            type=int,   required=False)
    parser.add_argument('--num_workers',        default=16,            type=int,   required=False) 
    parser.add_argument('--pin_memory',         default=True,          type=bool,  required=False)
    parser.add_argument('--device',             default='cuda',        type=str,   required=False) 
    parser.add_argument('--model',              default='lstm',        type=str,   required=True) 
    parser.add_argument('--coordinates',        default=False,         type=bool,  required=False)
    parser.add_argument('--velocity',           default=False,         type=bool,  required=False)
    parser.add_argument('--parent_relative',    default=False,         type=bool,  required=False) 
    parser.add_argument('--root_relative',      default=False,         type=bool,  required=False)
    parser.add_argument('--mode',               default='baseline',    type=str,   required=False)
    parser.add_argument('--scheduler_step',     default=10000,         type=float, required=False) 
    parser.add_argument('--scheduler_gamma',    default=0.3333,        type=float, required=False)
    parser.add_argument('--betas',              default=(0.9, 0.999),  type=tuple, required=False) 
    parser.add_argument('--title',              type=str,                          required=True)
    parser.add_argument('--dropout',            default=0.2,           type=float, required=False)
    parser.add_argument('--pck_alpha',          default=0.05,          type=float, required=False)
    parser.add_argument('--boxsize',            default=640,           type=int,   required=False)
    
    # MLP 
    parser.add_argument('--mlp_hidden_layers',  default=[1024, 2048, 1024], type=list, required=False) 
    
    # LSTM 
    parser.add_argument('--lstm_embedding_dim', default=2048, type=int, required=False) 
    parser.add_argument('--lstm_hidden_dim',    default=256,  type=int, required=False)
    parser.add_argument('--lstm_num_layers',    default=2,    type=int, required=False) 

    # MOTION VAE 
    parser.add_argument('--motion_vae_hidden_dim',      default=32, type=int, required=False) 
    parser.add_argument('--motion_vae_latent_dim',      default=16, type=int, required=False) 
    parser.add_argument('--motion_vae_encoding_layers', default=2,  type=int, required=False) 
    parser.add_argument('--motion_vae_decoding_layers', default=1,  type=int, required=False) 

    # DECOUPLE MOTION VAE
    parser.add_argument('--decouple_motion_vae_hidden_dim',      default=32, type=int, required=False) 
    parser.add_argument('--decouple_motion_vae_latent_dim',      default=16, type=int, required=False) 
    parser.add_argument('--decouple_motion_vae_encoding_layers', default=2,  type=int, required=False) 
    parser.add_argument('--decouple_motion_vae_decoding_layers', default=1,  type=int, required=False) 
    parser.add_argument('--decouple_lstm_embedding_dim',         default=8,  type=int, required=False)
    parser.add_argument('--decouple_lstm_hidden_dim',            default=16, type=int, required=False)
    parser.add_argument('--decouple_lstm_num_layers',            default=1,  type=int, required=False)

    args = parser.parse_args()
    model, loss_list, metric_list = train(args)