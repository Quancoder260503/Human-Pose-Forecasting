import cv2
import time
import torch
import pyrender
import numpy as np
from matplotlib import pyplot as plt
from data.ImgUtil import torch2numpy, normalize_2d_kp
import data.KeyPointUtil

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear = pyrender.camera.DEFAULT_Z_NEAR,
                 zfar  = None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        P[0, 3] = self.translation[0] * self.scale
        P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1
        return P


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

def visualize_sequence(sequence):

    seqlen, size = sequence.shape
    sequence = sequence.reshape((seqlen, -1, 3))

    fig = plt.figure(figsize=(12, 7))

    for i in range(seqlen):
        ax = fig.add_subplot('111', projection='3d', aspect=1)
        show3Dpose(sequence[i], ax, radius=0.6)
        ax.view_init(-75, -90)

        plt.draw()
        plt.pause(0.01)
        plt.cla()

    plt.close()


import cv2
import numpy as np

def draw_skeleton(image, kp_2d, prob = None, dataset = 'pennaction', thickness = 1, shadow = 0.3):
    if prob is None : 
        prob  = np.ones(kp_2d.shape[0])
    kp_2d = np.array(kp_2d, dtype=int)

    image = image.copy()
    image = (image * shadow).astype(np.uint8)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    skeleton = eval(f'util.KeyPointUtil.get_{dataset}_skeleton')()
    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,1]
    for idx, pt in enumerate(kp_2d):
        if prob[idx] > 0.6 and pt[0] > 0 and pt[1] > 0:
            cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)

    for i, (j1, j2) in enumerate(skeleton):
        if prob[j1] > 0.6 and prob[j2] > 0.6 and kp_2d[j1, 0] > 0 and kp_2d[j1, 1] > 0 and kp_2d[j2, 0] > 0 and kp_2d[j2, 1] > 0:
            color = rcolor if common_lr[i] == 0 else (
                lcolor if dataset == 'common' else (lcolor if i % 2 == 0 else rcolor)
            )

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


def show_video(video, fps=25):
    for fid, frame in enumerate(video):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(f'frame {fid}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1./fps)

    cv2.destroyAllWindows()