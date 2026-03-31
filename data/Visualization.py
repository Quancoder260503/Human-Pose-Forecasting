import cv2
import time
import torch
import pyrender
import numpy as np
from matplotlib import pyplot as plt
from data.ImgUtil import torch2numpy, normalize_2d_kp
from data.KeyPointUtil import *
from data.NTUmeta import *

def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors


def draw_SMPL_joints2D(joints2D, image, kintree_table=None, color='red'):
    rcolor = get_colors()['red'].tolist()
    lcolor = get_colors()['blue'].tolist()
    # color = get_colors()[color].tolist()
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]

        color = lcolor if i % 2 == 0 else rcolor

        pt1, pt2 = (joints2D[j1, 0], joints2D[j1, 1]), (joints2D[j2, 0], joints2D[j2, 1])
        cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=2)

        cv2.circle(image, pt1, 4, color, -1)
        cv2.circle(image, pt2, 4, color, -1)

    # for i in range(joints2D.shape[0]):
    #     color = lcolor if i % 2 == 0 else rcolor
    #     pt1 = (joints2D[i, 0], joints2D[i, 1])
    #     cv2.circle(image, pt1, 4, color, -1)

    return image


def draw_skeleton(image, kp_2d, prob = None, prob_draw = 0.6, dataset = 'pennaction', thickness = 1, shadow = 0.3, eps = 10):
    if prob is None : 
        prob  = np.ones(kp_2d.shape[0])
    kp_2d = np.array(kp_2d, dtype=int)

    image = image.copy()
    image = (image * shadow).astype(np.uint8)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = get_ntu_skeleton() if dataset == 'ntu' else get_pennaction_skeleton()
    for idx, pt in enumerate(kp_2d):
        if prob[idx] > prob_draw and pt[0] > eps and pt[1] > eps:
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)

    for i, (j1, j2) in enumerate(skeleton):
        if prob[j1] > prob_draw and prob[j2] > prob_draw and kp_2d[j1, 0] > eps and kp_2d[j1, 1] > eps and kp_2d[j2, 0] > eps and kp_2d[j2, 1] > eps:
            color = rcolor
            pt1 = tuple(kp_2d[j1, :2])
            pt2 = tuple(kp_2d[j2, :2])
            cv2.line(image, pt1, pt2, color, thickness= 3)

    return image

def batch_draw_skeleton(images, target, max_images=8, dataset='common'):
    if max_images is None or images.shape[0] < max_images:
        max_images = images.shape[0]


    with torch.no_grad():

        for k, v in target.items():
            if isinstance(target[k], torch.Tensor):
                target[k] = v.cpu().numpy()

    result_images = []

    for idx in range(max_images):
        single_target = {}

        for k, v in target.items():
            single_target[k] = v[idx]

        img = torch2numpy(images[idx])

        img = draw_skeleton(img.copy(), single_target['kp_2d'], dataset=dataset)
        result_images.append(img)

    result_image = np.vstack(result_images)

    return result_image

def show_video(video, fps=25, window_name = "Video"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, frame_bgr)

        # waitKey expects milliseconds
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()