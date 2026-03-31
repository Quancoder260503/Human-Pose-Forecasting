import os 
import numpy as np 
import scipy.io 


MIN_FRAMES = 30 
train_dir = 'dataset/PennAction/train'
test_dir  = 'dataset/PennAction/test'

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

print(len(bad_train_videos), len(train_dir_name))
print(len(bad_test_videos),  len(test_dir_name))