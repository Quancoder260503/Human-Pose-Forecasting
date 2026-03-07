import matplotlib.pyplot as plt
import numpy as np
from util.KeyPointUtil import get_pennaction_joint_names



def pck(pred, gt, threshold, predictions):
    """
    An estimation is considered  correct if it lies within \alpha * max(h, w) from
    the true position, H_map, W_map is the height and width of the heat map
    :param pred: tensor of shape (J, H_map, W_map)
    :param gt: tensor of shape (J, H_map, W_map)
    :param threshold: threshold to consider the boundary
    :param predictions: previous average meters of joints
    :return: predictions : next average meters of joints
    """
    alpha = 0.04
    penn_action_joint = get_pennaction_joint_names()
    for i in range(pred.shape[0] - 1): # For each joint
        predictions['total']['all'] += 1
        predictions[penn_action_joint[i]]['all'] += 1

        # If the joint is not in the image
        if np.max(gt[i, :, :]) == 0:
            if np.max(pred[i, :, :]) == 0:
                predictions['total']['correct'] += 1
                predictions[penn_action_joint[i]]['correct'] += 1
        else :
            pred_x, pred_y = np.where(pred[i, :, :] == np.max(pred[i, :, : ]))
            gt_x, gt_y     = np.where(gt[i, :, :]   == np.max(gt[i, :, :]))

            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

            if dist  < threshold * alpha :
                predictions['total']['correct'] += 1
                predictions[penn_action_joint[i]]['correct'] += 1

    return predictions

def compute_metric(pred_maps, gt_maps, threshold, time_step):
    '''
    :param pred_maps: tensor of shape (B, T, J, H_map, W_map)
    :param gt_maps: tensor of shape (B, T, J, H_map, W_map)
    :param threshold: size of the heatmap
    :param time_step: number of time steps
    :return:
      predictions : results tracker
    '''
    predictions = get_predictions_dict()
    for b in range(gt_maps.shape[0]):
        for t in range(time_step):
            gt = gt_maps[b, t, :, :, :]
            pred = np.array(pred_maps[b, t, :, :, :].cpu().data)
            predictions = pck(pred, gt, threshold, predictions)
    return predictions

def get_acc(predictions):
    acc = {}
    for key in predictions :
        acc[key] = predictions[key]['correct'] / predictions[key]['total']
    return acc


def plot_loss(ax, loss_list, title):
    ax.plot(loss_list[:, 0], label = "Train Loss")
    ax.plot(loss_list[:, 1], label = "Validation Loss")
    ax.set_title("Loss Curves - " + title, fontsize = 12)
    ax.set_xlabel("Epoch", fontsize = 10)
    ax.set_ylabel("Loss", fontsize = 10)
    ax.legend(prop = {'size' : 10})

def plot_metric(ax, metric_list, title):
    head_acc  = [x['head']  for x in metric_list]
    sho_acc   = [x['sho']   for x in metric_list]
    elb_acc   = [x['elb']   for x in metric_list]
    wri_acc   = [x['wri']   for x in metric_list]
    hip_acc   = [x['hip']   for x in metric_list]
    knee_acc  = [x['knee']  for x in metric_list]
    ank_acc   = [x['ank']   for x in metric_list]
    total_acc = [x['total'] for x in metric_list]

    ax.plot(total_acc[:], label = "Total")
    ax.plot(head_acc[:] , label = "Head")
    ax.plot(sho_acc[:]  , label = "Shoulders")
    ax.plot(elb_acc[:]  , label = "Elbows")
    ax.plot(wri_acc[:]  , label = "Wrists")
    ax.plot(hip_acc[:]  , label = "Hips")
    ax.plot(knee_acc[:] , label = "knees")
    ax.plot(ank_acc[:]  , label = "Ankles")

    ax.set_title("PCK accuracy - " + title, fontsize=12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.legend(prop={'size': 10})


def plot(loss_list, metric_list, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 8))
    plot_loss(ax1, loss_list, title)
    plot_metric(ax2, metric_list, title)
    plt.show()
