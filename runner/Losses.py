import torch 
import torch.nn as nn 

def naive_loss_function(pred_joints, gt_joints):
    criterion = nn.SmoothL1Loss(reduction = 'mean')
    if not isinstance(pred_joints, list):
        pred_joints = [pred_joints]
    loss = 0 
    for i in range(len(pred_joints)):
        loss += criterion(pred_joints[i], gt_joints)
    loss = loss / len(pred_joints)
    return loss

def penalize_loss_function(
      pred_joints, 
      gt_joints, 
      num_joints, 
      root_position, 
      lambda_root = 2,
      lambda_norm = 1,  
    ):
    batch_size, seq_len, _ = gt_joints.shape 
    criterion = nn.SmoothL1Loss(reduction='mean')
    
    if not isinstance(pred_joints, list):
        pred_joints = [pred_joints]

    loss_normal = 0.0 
    loss_root   = 0.0 

    gt = gt_joints.view(batch_size, seq_len, num_joints, -1)
    
    for i in range(len(pred_joints)): 
        pred   = pred_joints[i].view(batch_size, seq_len, num_joints, -1)
        device = pred.device

        all_idx      = torch.arange(num_joints, device = device)
        normal_joints = all_idx[all_idx != root_position]  # all joints except root

        pred_normal = pred[:, :, normal_joints, :]
        gt_normal   =   gt[:, :, normal_joints, :]

        pred_root = pred[:, :, root_position, :]
        gt_root   =   gt[:, :, root_position, :]  

        loss_normal += criterion(pred_normal, gt_normal) 
        loss_root   += criterion(pred_root,   gt_root)

    loss = (lambda_norm * loss_normal + lambda_root * loss_root) / (lambda_norm + lambda_root) 
    loss = loss / len(pred_joints)
    return loss