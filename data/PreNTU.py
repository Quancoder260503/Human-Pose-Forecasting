"""
Preprocessing script for data, the code is quite messy, but only used once to extract and filter 
data from NTU-RGB dataset 
"""

import os
import numpy as np
import pickle
import logging
import json
import cv2

NTU_SKELETON_PATH = 'dataset/nturgbd_skeletons_s018_to_s032'
NTU_VIDEO_PATH    = 'dataset/NanyangTechRGB/frames/nturgbd_rgb_s001/nturgb+d_rgb'
NTU_IMAGE_RESOLUTION_H = 1920
NTU_IMAGE_RESOLUTION_W = 1080

def get_raw_body_data(skes_path, ske_name):
    """
    Get raw body data from a skeleton sequence

    Each body's data is a dictionary contains the following keys :
      - joints : raw 3D joint positions, Shape : (num_frames x 25, 3)
      - colors : raw_2D colour_locations, Shape : (num_frames, 25, 2)
      - interval : a list which stores the frame indices of this body.

    Return :
      a dict for a skeleton sequence with 3 key-value pairs :
         - name : the skeleton filename
         - data : a dict which stores raw data of each body
         - num_frames : number of frames in each body
    """

    ske_file = os.path.join(skes_path, ske_name)
    with open(ske_file, 'r') as f :
        str_data = f.readlines()
    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1
    current_line = 1

    for frame in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1
        # Nobody for this frame, throw it away
        if num_bodies == 0:
            frames_drop.append(frame)
            continue
        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype = np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype = np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n')) # 25 joints for NTU
            current_line += 1
            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[: 3], dtype = np.float32)
                colors[b, j, :] = np.array(temp_str[5 : 7], dtype = np.float32)
                current_line += 1

            if bodyID not in bodies_data :
                body_data = dict()
                body_data['joints'] = joints[b] # ndarray (25, 3)
                body_data['colors'] = colors[b, np.newaxis] # ndarray : (1, 25, 2)
                body_data['interval'] = [valid_frames]
            else :
                body_data = bodies_data[bodyID]
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_index = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_index + 1)

            bodies_data[bodyID] = body_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop <= num_frames, 'Cannot have more frame dropped than total number of frames'
    return {
        'name' : ske_name,
        'data' : bodies_data,
        'num_frames' : num_frames - num_frames_drop,
    }


def remove_video(datadir):
    data_storages = os.listdir(datadir)
    min_length = 30
    bad_videos = [] 
    for storage_id in data_storages:
        storage     = np.load(os.path.join(datadir, storage_id), allow_pickle = True).item()
        name        = storage['name']
        human_id_list = list(storage['data'].keys())
        if len(human_id_list) == 0: 
            bad_videos.append(storage_id)
            continue 
       # human_id    = human_id_list[0] # So far only support spotting the first person in the video, idea may extened later 
        max_frames = 0 
        for human_id in human_id_list : 
          data        = storage['data'][human_id]
          label       = data['colors'] 
          n_frames    = label.shape[0] 
          max_frames  = max(max_frames, n_frames)    
        if max_frames < 50: 
            bad_videos.append(storage_id) 
            continue
        #assert max_frames >= 30, "num frames must be more than total frame"
        num_joints  = label.shape[1] 
        dim         = label.shape[2]
  #  for bad_video in bad_videos: 
  #     bad_video_path = os.path.join(datadir, bad_video)
  #     os.remove(bad_video_path)
    print(len(bad_videos), len(data_storages)) 




if __name__ == '__main__':
    # saved_dir = 'dataset/NanyangTechRGB'
    # os.makedirs(saved_dir, exist_ok=True)

    # filenames = os.listdir(NTU_SKELETON_PATH)
    # directory_length = len(filenames)

    # random_index = np.arange(directory_length)
    # np.random.shuffle(random_index)

    # num_train = int(directory_length * 0.7)

    # train_index = set(random_index[:num_train])
    # test_index = set(random_index[num_train:])

    # train_dir = os.path.join(saved_dir, 'train')
    # test_dir  = os.path.join(saved_dir, 'test')

    # os.makedirs(train_dir, exist_ok = True)
    # os.makedirs(test_dir, exist_ok  = True)

    # start_index = 46140
    
    # for video_index, filename in enumerate(filenames):
    #     data_dict = get_raw_body_data(
    #         NTU_SKELETON_PATH,
    #         filename,
    #     )
    #     if video_index in train_index:
    #         save_dir = os.path.join(train_dir, f"{start_index + video_index}.npy")
    #     else:
    #         save_dir = os.path.join(test_dir, f"{start_index + video_index}.npy")

    #     np.save(save_dir, data_dict)
    #remove_video('dataset/NanyangTechRGB/test')

    # Sample script to show video 

    video_file_path = NTU_VIDEO_PATH
    cap = cv2.VideoCapture(os.path.join(video_file_path, 'S001C001P001R001A042_rgb.avi'))
    num_frames = 0 
    while True:
      ret, frame = cap.read()
      if not ret:
        break  # End of video
      num_frames += 1 
    # Display the frame
      cv2.imshow('A woman from NTU', frame)
      
      if cv2.waitKey(30) & 0xFF == 27:  # 27 = ESC key
        break
    print(num_frames)
    # Release resources
   # cap.release()
   # cv2.destroyAllWindows()