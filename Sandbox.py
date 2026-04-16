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

from model.Transformer import *

model = get_transformer_model(
    transformer_type = 'joint', 
).to('cuda')

dummy = torch.zeros((1, 14, 39)).to('cuda')
out = model(dummy)
#out = torch.cat(out, dim = 0)
print(out[0].shape)