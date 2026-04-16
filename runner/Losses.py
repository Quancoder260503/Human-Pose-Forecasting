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
      special_joints, 
      root_position, 
      lambda_root    = 2, 
      lambda_special = 1.5, 
      lambda_norm    = 1, 
    ):
    batch_size, seq_len, _ = gt_joints.shape 
    criterion = nn.SmoothL1Loss(reduction = 'mean')
    
    if not isinstance(pred_joints, list):
        pred_joints = [pred_joints]

    loss_special = 0.0 
    loss_normal  = 0.0 
    loss_root    = 0.0 
    gt   = gt_joints.view(batch_size, seq_len, num_joints, -1)
    
    for i in range(len(pred_joints)): 
        pred = pred_joints[i].view(batch_size, seq_len, num_joints, -1)
        device = pred.device

        special_joints = torch.tensor(special_joints, device = device)
        all_idx = torch.arange(num_joints, device = device)
        normal_joints = all_idx[~torch.isin(all_idx, special_joints)]

        pred_normal = pred[:, :, normal_joints, :]
        gt_normal   = gt[:, :, normal_joints, :]

        pred_root   = pred[:, :, root_position, :] 
        gt_root     = pred[:, :, gt_root, :]


        loss_special += criterion(pred, gt) 
        loss_normal  += criterion(pred_normal, gt_normal) 
        loss_root    += criterion(pred_root, gt_root)

    loss = lambda_norm * loss_normal + lambda_root * loss_root + lambda_special * loss_special
    loss = loss / len(pred_joints)
    loss = torch.sqrt(loss) 
    return loss