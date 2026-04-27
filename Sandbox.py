# from data.DatasetUtil   import * 
# from data.PreNTU        import * 
# from data.Visualization import * 

# NAME = 'S031C002P044R002A094.skeleton'

# skeleton = get_raw_body_data(NTU_SKELETON_PATH, ske_name = NAME)
# videos_real = []

# name          = skeleton['name']
# human_id_list = list(skeleton['data'].keys())

# assert len(human_id_list), f"video {name} has no person in the video"

# human_id_list.sort(key = lambda x : skeleton['data'][x]['colors'].shape[0], reverse = True) 
# human_id    = human_id_list[0]

# data        = skeleton['data'][human_id]
# label       = data['colors'] # projected 2D locations
# frame_num   = data['interval']

# observed_images, _ = get_video_frames_ntu(
#     start_index = 0, 
#     name = NAME, 
#     num_training_frame = len(frame_num), 
#     num_testing_frame = 0, 
#     frame_input_w = 1920, 
#     frame_input_h = 1080,  
# )


# observed_images = observed_images[frame_num]

# for index in range(0, observed_images.shape[0]): 
#     curr_target_pose  = label[index]
#     curr_target_image = observed_images[index, :, :, :]


#     target_annotated_image  = draw_skeleton(
#         image = curr_target_image, 
#         poses = curr_target_pose,
#         dataset       = 'ntu', 
#         unnormalize   = False, 
#         frame_input_w = 1920, 
#         frame_input_h = 1080, 
#     )
#     videos_real.append(target_annotated_image)

# show_video(videos_real, fps = 3, output_path = 'test_output.mp4')
# plt.tight_layout()
# plt.show() 

from model.BERT import * 

if __name__ == '__main__':
    dof = 3
    output_channels = 512
    n_joints = 17
    seq_len = 14

    model = get_bert_model(num_joints = 17, coordinates = True)
    inputs = torch.rand(10, seq_len, dof * n_joints)
    X = model(inputs)
    print(X.size())