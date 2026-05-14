import math
import torch 
import torch.nn as nn 


def KL_annealing(epoch, total_epochs, schedule_type = 'linear', start_beta = 0.0, end_beta = 1.0):
    progress = epoch / total_epochs
    if schedule_type == "linear":
        return start_beta + (end_beta - start_beta) * progress
    elif schedule_type == "cyclical":
        return end_beta * (1 + math.cos(math.pi * (1 - progress))) / 2
    elif schedule_type == "sigmoid":
        return start_beta + (end_beta - start_beta) / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def get_loss_fn(loss_fn):
    if  loss_fn == 'mse':
        return nn.MSELoss(reduction = 'mean')
    elif loss_fn == 'huber':
        return nn.SmoothL1Loss(reduction = 'mean')
    elif loss_fn == 'L1':
        return nn.L1Loss(reduction = 'mean')
    else : 
        raise NotImplementedError('Loss function not supported')


def penalize_deterministic_function(args, loss_fn, prediction_joints, ground_truth_joints):
    criterion = get_loss_fn(loss_fn)

    if not isinstance(prediction_joints, list):
        prediction_joints = [prediction_joints]

    B, T, _ = ground_truth_joints.shape
    ground_truth = ground_truth_joints.view(B, T, args.num_joints, -1)

    device = ground_truth.device
    all_idx = torch.arange(args.num_joints, device = device)
    normal_joints = all_idx[all_idx != args.root_position]

    loss = 0.0
    L = len(prediction_joints)

    layer_weights = torch.linspace(args.layerwise_start_lambda, args.layerwise_end_lambda, steps = L, device = args.device)
    layer_weights = layer_weights / layer_weights.sum()

    for i, prediction in enumerate(prediction_joints):
        prediction = prediction.view(B, T, args.num_joints, -1)

        prediction_root = prediction[:, :, args.root_position, :]
        ground_truth_root   = ground_truth[:, :, args.root_position, :]

        prediction_norm = prediction[:, :, normal_joints, :]
        ground_truth_norm   = ground_truth[:, :, normal_joints, :]

        loss_root_i   = criterion(prediction_root, ground_truth_root)
        loss_norm_i   = criterion(prediction_norm, ground_truth_norm)

        loss_i = args.lambda_norm * loss_norm_i + args.lambda_root * loss_root_i

        loss += layer_weights[i] * loss_i
    return loss


def penalize_vae_function(args, loss_fn, prediction_joints, ground_truth_joints, mean, log_var, beta):
    criterion = get_loss_fn(loss_fn)

    if not isinstance(prediction_joints, list):
        prediction_joints = [prediction_joints]

    B, T, _ = ground_truth_joints.shape
    ground_truth = ground_truth_joints.view(B, T, args.num_joints, -1)

    device = ground_truth.device
    all_idx = torch.arange(args.num_joints, device = device)
    normal_joints = all_idx[all_idx != args.root_position]

    reconstruction_loss = 0.0
    L = len(prediction_joints)

    layer_weights = torch.linspace(args.layerwise_start_lambda, args.layerwise_end_lambda, steps = L, device = args.device)
    layer_weights = layer_weights / layer_weights.sum()

    for i, prediction in enumerate(prediction_joints):
        prediction = prediction.view(B, T, args.num_joints, -1)

        prediction_root = prediction[:, :, args.root_position, :]
        ground_truth_root   = ground_truth[:, :, args.root_position, :]

        prediction_norm = prediction[:, :, normal_joints, :]
        ground_truth_norm   = ground_truth[:, :, normal_joints, :]

        loss_root_i   = criterion(prediction_root, ground_truth_root)
        loss_norm_i   = criterion(prediction_norm, ground_truth_norm)

        loss_i = args.lambda_norm * loss_norm_i + args.lambda_root * loss_root_i

        reconstruction_loss += layer_weights[i] * loss_i
    
    kullback_leiber_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var), dim = 1), dim = 0) 
    return reconstruction_loss + beta * kullback_leiber_loss


def deterministic_loss_function(args, loss_fn, prediction_joints, ground_truth_joints):
    criterion = get_loss_fn(loss_fn)
    if not isinstance(prediction_joints, list):
        prediction_joints = [prediction_joints]

    L = len(prediction_joints)

    layer_weights = torch.linspace(args.layerwise_start_lambda, args.layerwise_end_lambda, steps = L, device = args.device)
    layer_weights = layer_weights / layer_weights.sum()
    loss = 0.0
    for i, prediction in enumerate(prediction_joints):
        loss += layer_weights[i] * criterion(prediction, ground_truth_joints)
    return loss

def vae_loss_function(args, loss_fn, prediction_joints, ground_truth_joints, mean, log_var, beta):
    criterion = get_loss_fn(loss_fn)
    if not isinstance(prediction_joints, list):
        prediction_joints = [prediction_joints]

    L = len(prediction_joints)

    layer_weights = torch.linspace(args.layerwise_start_lambda, args.layerwise_end_lambda, steps = L, device = args.device)
    layer_weights = layer_weights / layer_weights.sum()
    reconstruction_loss = 0.0
    for i, prediction in enumerate(prediction_joints):
        reconstruction_loss += layer_weights[i] * criterion(prediction, ground_truth_joints)
    
    kullback_leiber_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var), dim = 1), dim = 0) 
    return reconstruction_loss + beta * kullback_leiber_loss
