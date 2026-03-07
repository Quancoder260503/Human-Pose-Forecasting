import os
import numpy as np
import scipy.io

frame_dir = 'dataset/JHMDB/frames'
label_dir = 'dataset/JHMDB/labels'
train_dir = 'dataset/JHMDB/train'
test_dir  = 'dataset/JHMDB/test'

action_folders = sorted(os.listdir(label_dir))

actions = [
    'catch', 'clap', 'climb_stairs', 'golf',
    'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
    'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
    'stand', 'swing_baseball', 'throw', 'walk', 'wave'
]

num_videos = 0
for action_folder in action_folders[1 : ]:
    action_videos = sorted(os.listdir(os.path.join(label_dir, action_folder)))
    for action_video in action_videos[1 : ]:
        num_videos += 1

random_index = np.arange(num_videos)
np.random.shuffle(random_index)

video_index = 0
train_threshold = int(0.8 * num_videos)

for action_folder in action_folders[1 : ]:
    action_videos = sorted(os.listdir(os.path.join(label_dir, action_folder)))
    for action_video in action_videos[1 : ]:
      data = scipy.io.loadmat(os.path.join(label_dir, action_folder, action_video, 'joint_positions'))
      npy_data = dict()
      npy_data['framepath']      = os.path.join(frame_dir, action_folder, action_video)
      npy_data['pos_img']        = data['pos_img']
      npy_data['scale']          = data['scale']
      npy_data['action_label']   = actions.index(action_folder)
      npy_data['sequence']       = video_index
      npy_data['nframes']        = np.array(npy_data['pos_img']).shape[2]
      if random_index[video_index] <= train_threshold :
          save_dir = os.path.join(train_dir, str(video_index) + '.npy')
      else :
          save_dir = os.path.join(test_dir, str(video_index) + '.npy')
      video_index += 1
      np.save(save_dir, npy_data)
