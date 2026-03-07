import numpy as np

def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d

def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs

def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]

def get_pennaction_skeleton():
    return [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (1, 7),
        (2, 8),
        (7, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
    ]

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

"""
The original set does not include connection between l-shoulder and head top nor r-shoulder and head top 
"""

def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [8, 13],
            [9, 13],
        ]
    )

def get_coco_joint_names():
    return [
        "headtop",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]

def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )


def get_jhmdb_joint_names():
    return [
        "neck",       # 0
        "belly",      # 1
        "face",       # 2
        "rshoulder",  # 3
        "lshoulder",  # 4
        "rhip",       # 5
        "lhip",       # 6
        "relbow",     # 7
        "lelbow",     # 8
        "rknee",      # 9
        "lknee",      # 10
        "rwrist",     # 11
        "lwrist",     # 12
        "rankle",     # 13
        "lankle",     # 14
    ]

def get_jhmdb_skeleton():
    # 0  - neck
    # 1  - belly
    # 2  - face
    # 3  - rshoulder
    # 4  - lshoulder
    # 5  - rhip
    # 6  - lhip
    # 7  - relbow
    # 8  - lelbow
    # 9  - rknee
    # 10 - lknee
    # 11 - rwrist
    # 12 - lwrist
    # 13 - rankle
    # 14 - lankle
    return np.array(
        [
            [ 0,  2],  # neck -> face
            [ 0,  3],  # neck -> rshoulder
            [ 0,  4],  # neck -> lshoulder
            [ 0,  1],  # neck -> belly
            [ 3,  7],  # rshoulder -> relbow
            [ 4,  8],  # lshoulder -> lelbow
            [ 7, 11],  # relbow -> rwrist
            [ 8, 12],  # lelbow -> lwrist
            [ 1,  5],  # belly -> rhip
            [ 1,  6],  # belly -> lhip
            [ 5,  6],  # rhip -> lhip
            [ 5,  9],  # rhip -> rknee
            [ 6, 10],  # lhip -> lknee
            [ 9, 13],  # rknee -> rankle
            [10, 14],  # lknee -> lankle
        ]
    )