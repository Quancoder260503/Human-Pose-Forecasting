import numpy as np
import cv2 
import os 
import torch


MIN_STD = 1e-4

"""
Get Result Dict
"""
def get_pennaction_pck_dict():
    return {
        'total'       :  { 'all' : 0, 'correct' : 0 },
        'headtop'     :  { 'all' : 0, 'correct' : 0 },
        'rshoulder'   :  { 'all' : 0, 'correct' : 0 },
        'lshoulder'   :  { 'all' : 0, 'correct' : 0 },
        'lelbow'      :  { 'all' : 0, 'correct' : 0 },
        'relbow'      :  { 'all' : 0, 'correct' : 0 },
        'lwrist'      :  { 'all' : 0, 'correct' : 0 },
        'rwrist'      :  { 'all' : 0, 'correct' : 0 },
        'lhip'        :  { 'all' : 0, 'correct' : 0 },
        'rhip'        :  { 'all' : 0, 'correct' : 0 },
        'lknee'       :  { 'all' : 0, 'correct' : 0 },
        'rknee'       :  { 'all' : 0, 'correct' : 0},
        'lankle'      :  { 'all' : 0, 'correct' : 0},
        'rankle'      :  { 'all' : 0, 'correct' : 0},
    }

def get_pennaction_mse_dict():
     return {
        'total'       :  0.0,
        'headtop'     :  0.0,
        'rshoulder'   :  0.0,
        'lshoulder'   :  0.0,
        'lelbow'      :  0.0,
        'relbow'      :  0.0,
        'lwrist'      :  0.0,
        'rwrist'      :  0.0,
        'lhip'        :  0.0,
        'rhip'        :  0.0,
        'lknee'       :  0.0,
        'rknee'       :  0.0,
        'lankle'      :  0.0,
        'rankle'      :  0.0,
    }

def get_ntu_pck_dict():
    return {
     'total'         : { 'all' : 0, 'correct' : 0}, 
     'SpineBase'     : { 'all' : 0, 'correct' : 0 }, 
     'SpineMid'      : { 'all' : 0, 'correct' : 0 }, 
     'Neck'          : { 'all' : 0, 'correct' : 0 }, 
     'Head'          : { 'all' : 0, 'correct' : 0 }, 
     'ShoulderLeft'  : { 'all' : 0, 'correct' : 0 },  
     'ElbowLeft'     : { 'all' : 0, 'correct' : 0 },
     'WristLeft'     : { 'all' : 0, 'correct' : 0 },  
     'ShoulderRight' : { 'all' : 0, 'correct' : 0 }, 
     'ElbowRight'    : { 'all' : 0, 'correct' : 0 }, 
     'WristRight'    : { 'all' : 0, 'correct' : 0 },
     'HipLeft'       : { 'all' : 0, 'correct' : 0 }, 
     'KneeLeft'      : { 'all' : 0, 'correct' : 0 }, 
     'AnkleLeft'     : { 'all' : 0, 'correct' : 0 }, 
     'HipRight'      : { 'all' : 0, 'correct' : 0 },
     'KneeRight'     : { 'all' : 0, 'correct' : 0 }, 
     'AnkleRight'    : { 'all' : 0, 'correct' : 0 }, 
     'SpineShoulder' : { 'all' : 0, 'correct' : 0 }, 
    }

def get_ntu_mse_dict():
    return {
      'total'         : 0.0, 
      'SpineBase'     : 0.0, 
      'SpineMid'      : 0.0, 
      'Neck'          : 0.0, 
      'Head'          : 0.0, 
      'ShoulderLeft'  : 0.0,  
      'ElbowLeft'     : 0.0,
      'WristLeft'     : 0.0, 
      'ShoulderRight' : 0.0, 
      'ElbowRight'    : 0.0, 
      'WristRight'    : 0.0,
      'HipLeft'       : 0.0, 
      'KneeLeft'      : 0.0, 
      'AnkleLeft'     : 0.0,  
      'HipRight'      : 0.0,
      'KneeRight'     : 0.0, 
      'AnkleRight'    : 0.0, 
      'SpineShoulder' : 0.0, 
    }

DATASET_PCK_DICT = { 
    'pennaction' : get_pennaction_pck_dict, 
    'ntu'        : get_ntu_pck_dict, 
}

DATASET_MSE_DICT  = { 
    'pennaction' : get_pennaction_mse_dict, 
    'ntu'        : get_ntu_mse_dict, 
}

"""
Joint Names
"""
NTU_JOINT_NAMES = [
    'SpineBase', 
    'SpineMid', 
    'Neck', 
    'Head', 
    'ShoulderLeft', 
    'ElbowLeft',
    'WristLeft', 
    'HandLeft', 
    'ShoulderRight', 
    'ElbowRight', 
    'WristRight',
    'HandRight', 
    'HipLeft', 
    'KneeLeft', 
    'AnkleLeft', 
    'FootLeft', 
    'HipRight',
    'KneeRight', 
    'AnkleRight', 
    'FootRight', 
    'SpineShoulder', 
    'HandTipLeft',
    'ThumbLeft', 
    'HandTipRight', 
    'ThumbRight'
]

NTU_MAJOR_JOINTS = [
    'SpineBase', 
    'SpineMid', 
    'Neck', 
    'Head', 
    'ShoulderLeft', 
    'ElbowLeft',
    'WristLeft',  
    'ShoulderRight', 
    'ElbowRight', 
    'WristRight', 
    'HipLeft', 
    'KneeLeft', 
    'AnkleLeft', 
    'HipRight',
    'KneeRight', 
    'AnkleRight',  
    'SpineShoulder',
]

NTU_HIGH_SPEED_JOINTS = [
    'ShoulderLeft', 
    'ElbowLeft',
    'WristLeft',  
    'ShoulderRight', 
    'ElbowRight', 
    'WristRight', 
    'HipLeft', 
    'KneeLeft', 
    'AnkleLeft', 
    'HipRight',
    'KneeRight', 
    'AnkleRight', 
]

PENNACTION_JOINT_NAMES = [
    "headtop",    
    "lshoulder",  
    "rshoulder",  
    "lelbow",     
    "relbow",    
    "lwrist",    
    "rwrist",    
    "lhip" ,     
    "rhip" ,     
    "lknee",     
    "rknee" ,    
    "lankle",    
    "rankle"    
]

NTU_JOINT_INDS = {
    name: idx for idx, name in enumerate(NTU_JOINT_NAMES)
}

NTU_MAJOR_JOINTS_INDS = { 
    name : idx for idx, name in enumerate(NTU_MAJOR_JOINTS)
}

NTU_HIGH_SPEED_JOINT_INDS = [NTU_MAJOR_JOINTS_INDS[name] for name in NTU_HIGH_SPEED_JOINTS]

NTU_MAJOR_JOINTS_INDS_LST = [NTU_JOINT_INDS[name] for name in NTU_MAJOR_JOINTS]


PENNACTION_JOINT_INDS = { 
    name : idx for idx, name in enumerate(PENNACTION_JOINT_NAMES)
}

"""
Parent Relative
"""
PENNACTION_JOINT_PARENT_NAMES = { 
  'headtop'   : 'headtop', 
  'lshoulder' : 'headtop',
  'rshoulder' : 'headtop', 
  'lelbow'    : 'lshoulder', 
  'relbow'    : 'rshoulder', 
  'lwrist'    : 'lelbow', 
  'rwrist'    : 'relbow', 
  'lhip'      : 'lshoulder', 
  'rhip'      : 'rshoulder', 
  'lknee'     : 'lhip', 
  'rknee'     : 'rhip', 
  'lankle'    : 'lknee', 
  'rankle'    : 'rknee'
}

NTU_JOINT_PARENT_NAMES = {
    # parent root to itself for convenience
    'Head': 'Head',
    # stuff down the centre
    'Neck': 'Head',
    'SpineShoulder': 'Neck',
    'SpineMid': 'SpineShoulder',
    'SpineBase': 'SpineMid',
    # stuff on the left
    'ShoulderLeft': 'SpineShoulder',
    'ElbowLeft': 'ShoulderLeft',
    'WristLeft': 'ElbowLeft',
    'HandLeft': 'WristLeft',
    'ThumbLeft': 'HandLeft',
    'HandTipLeft': 'HandLeft',
    'HipLeft': 'SpineBase',
    'KneeLeft': 'HipLeft',
    'AnkleLeft': 'KneeLeft',
    'FootLeft': 'AnkleLeft',
    # stuff on the right
    'ShoulderRight': 'SpineShoulder',
    'ElbowRight': 'ShoulderRight',
    'WristRight': 'ElbowRight',
    'HandRight': 'WristRight',
    'ThumbRight': 'HandRight',
    'HandTipRight': 'HandRight',
    'HipRight': 'SpineBase',
    'KneeRight': 'HipRight',
    'AnkleRight': 'KneeRight',
    'FootRight': 'AnkleRight'
}


NTU_JOINT_PARENT_INDS = [
    NTU_JOINT_INDS[NTU_JOINT_PARENT_NAMES[child_name]] for child_name in NTU_JOINT_NAMES
]

PENNACTION_JOINT_PARENT_IDS = [ 
    PENNACTION_JOINT_INDS[PENNACTION_JOINT_PARENT_NAMES[child_name]] for child_name in PENNACTION_JOINT_NAMES
]

NTU_ACTION_CLASSES = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "put something inside pocket / take out something from pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
}

JOINT_NAMES = { 
    'pennaction' : PENNACTION_JOINT_NAMES, 
    'ntu'        : NTU_MAJOR_JOINTS, 
}

JOINT_INDS   = { 
    'pennaction' : PENNACTION_JOINT_INDS, 
    'ntu'        : NTU_MAJOR_JOINTS_INDS, 
}

JOINT_PARENT_NAME = { 
    'pennaction' : PENNACTION_JOINT_PARENT_NAMES, 
    'ntu'        : NTU_JOINT_PARENT_NAMES, 
}

HIGH_SPEED_JOINTS = { 
    'ntu'        : NTU_HIGH_SPEED_JOINT_INDS, 
}

NTU_VIDEO_PATH  = 'dataset/NanyangTechRGB/frames'

def get_skeleton(dataset):
    skeletons = []
    joint_names       = JOINT_NAMES[dataset]
    joint_inds        = JOINT_INDS[dataset]
    joint_parent_name = JOINT_PARENT_NAME[dataset]
    for child_index, childname in enumerate(joint_names): 
        parent_index = joint_inds[joint_parent_name[childname]]
        if parent_index == child_index : 
           continue 
        skeletons.append([parent_index, child_index])
    return skeletons


# For NTU 
def get_activity_from_file(path_file):
    # The pattern is SsssCcccPpppRrrrAaaa.skeleton
    pattern        = path_file.split('/')[-1].split('.')[0]
    setup_id       = int(pattern[1 : 4])
    camera_id      = int(pattern[5 : 8])
    subject_id     = int(pattern[9 : 12])
    replication_id = int(pattern[13 : 16])
    activity_id    = int(pattern[17 :])
    return (setup_id, camera_id, subject_id, replication_id, activity_id)

def capture_video_frame(video_file_path, video_name):
    cap = cv2.VideoCapture(os.path.join(video_file_path, video_name.split('.')[0] + '_rgb.avi'))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  

    cap.release()
    return np.array(frames) if len(frames) > 0 else None

def get_video_frames_ntu(name, start_index, num_training_frame, num_testing_frame, frame_input_w, frame_input_h):
    video_frames = capture_video_frame(os.path.join(NTU_VIDEO_PATH, 'nturgbd_rgb_s' + name[1:4], 'nturgb+d_rgb'), video_name = name)

    if video_frames is None:
        return None

    total_frames = num_training_frame + num_testing_frame
    if start_index + total_frames > len(video_frames):
        return None

    resized_frames = [cv2.resize(img, (frame_input_w, frame_input_h)) for img in video_frames]
    
    video_frames = np.array(resized_frames)
    observed_images = video_frames[start_index : start_index + num_training_frame]
    target_images = video_frames[start_index + num_training_frame : start_index + total_frames]

    return observed_images, target_images


def get_video_frames_pennaction(name, start_index, num_training_frame, num_testing_frame, frame_input_h, frame_input_w):
    total_frames = num_training_frame + num_testing_frame
    images = np.zeros((total_frames, frame_input_h, frame_input_w, 3))

    for i in range(total_frames):
        img_path = os.path.join(name, f'{start_index + i + 1:06d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue  

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (frame_input_w, frame_input_h), interpolation=cv2.INTER_CUBIC)  # ✓ (w, h)
        images[i] = img

    observed_img = images[:num_training_frame]
    target_img   = images[num_training_frame:]

    return observed_img, target_img