import os
import numpy as np
import scipy.io

frame_dir = 'dataset/PennAction/frames'
label_dir = 'dataset/PennAction/labels'
train_dir = 'dataset/PennAction/train'
test_dir  = 'dataset/PennAction/test'
MIN_FRAMES = 30 


def clean_pennaction():
    numeric_infos = sorted(os.listdir(label_dir))

    for idx, num in enumerate(numeric_infos, 1):
        data = scipy.io.loadmat(os.path.join(label_dir + '/' + num))
        num = num.split('.')[0]
        npy_data = dict()
        npy_data['framepath']  = os.path.join(frame_dir, num)
        npy_data['dimensions'] = list(data['dimensions'][0][0:2])
        npy_data['pose']       = str(data['pose'][0])
        npy_data['nframes']    = data['nframes'][0][0]
        npy_data['action']     = str(data['action'][0])
        npy_data['x']          = data['x']
        npy_data['y']          = data['y']
        npy_data['bbox']       = data['bbox']
        npy_data['visibility'] = data['visibility']
        npy_data['sequence']   = int(num)

        if data['train'][0][0] == -1:
            save_dir = os.path.join(test_dir, num + '.npy')
        else :
            save_dir = os.path.join(train_dir, num + '.npy')

        np.save(save_dir, npy_data)


def filter_pennaction():

    train_dir_name = sorted(os.listdir(train_dir))
    test_dir_name  = sorted(os.listdir(test_dir))

    bad_train_videos = [] 
    bad_test_videos  = [] 
    for index, train_name in enumerate(train_dir_name):
        data   = np.load(os.path.join(train_dir, train_name), allow_pickle = True).item()
        n_frames = data['nframes']
        if n_frames < MIN_FRAMES : 
            bad_train_videos.append(train_name) 

    for index, test_name in enumerate(test_dir_name):
        data   = np.load(os.path.join(test_dir, test_name), allow_pickle = True).item()
        n_frames = data['nframes']
        if n_frames < MIN_FRAMES : 
            bad_test_videos.append(test_name)

    for bad_video in bad_train_videos:
        os.remove(os.path.join(train_dir, bad_video))
    for bad_video in bad_test_videos : 
        os.remove(os.path.join(test_dir, bad_video))    

if __name__ == '__main__':
    clean_pennaction()
    filter_pennaction()