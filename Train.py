import numpy as np
import torch
from runner.Factory import *
import os 
import argparse 
import torch.distributed as dist


def train(args):
    runner = dataset_factory(args)
    device = args.device 
    print(device, 'is available')

    checkpoint = f"checkpoint/{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"

    start_epoch = 0 

    if os.path.exists(checkpoint): 
       model_dict = torch.load(checkpoint, weights_only = False) 
       start_epoch = model_dict['epoch'] + 1 
    
       model     = get_model(args)
       criterion = get_criterions(args)
       model.load_state_dict(model_dict['model'])
       model = model.to(device)

       optimizer = torch.optim.Adam(
          model.parameters(), 
          lr           = args.learning_rate, 
          betas        = args.betas, 
          weight_decay = args.weight_decay,
       )
       
       optimizer.load_state_dict(model_dict['optimizer'])
      
       scheduler = lr_scheduler_factory(args, optimizer, start_epoch)
       scheduler.load_state_dict(model_dict['scheduler'])

   
    else: 
        model     = get_model(args)
        criterion = get_criterions(args)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(
          model.parameters(), 
          lr           = args.learning_rate, 
          betas        = args.betas, 
          weight_decay = args.weight_decay,
        )

        scheduler = lr_scheduler_factory(args, optimizer, start_epoch)
       
    best_accuracy = 0
        
    loss_list, metric_list = [], []

    trainable_params = [p for p in model.parameters() if p.requires_grad]
  
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_params:,}")


    print("Start Training ...\n")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"learning rate for epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")
        
        train_loss, train_predictions, train_rmse, val_loss, val_predictions, val_rmse = runner.process_per_epoch(
            model     = model, 
            optimizer = optimizer, 
            criterion = criterion, 
            epoch     = epoch,        
        )

        if args.lr_scheduler == 'plateau':
           scheduler.step(train_loss)
        else : 
           scheduler.step()

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc, curr_val_total_acc = train_acc['total'], val_acc['total']

        print(f'Training | Loss : {train_loss:.6f} | PCK accuracy : {curr_train_total_acc:.4f} | PJRMSE : {train_rmse:.6f}')
        print(f'Validation | Loss : {val_loss:.6f} | PCK accuracy : {curr_val_total_acc:.4f}   | PJRMSE : {val_rmse:.6f}')

        if val_acc['total'] > best_accuracy : 
            best_accuracy = val_acc['total']
            save_checkpoint(
                epoch, 
                model, 
                optimizer, 
                scheduler, 
                model_name = f'{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}', 
                prefix_dir_name = f'checkpoint/{args.mode}/'
            )

    plot(
        loss_list   = np.array(loss_list), 
        metric_list = np.array(metric_list), 
        save_path   = f'log/metrics/{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.png', 
        dataset     = args.dataset, 
    )

    # Saved the training logs 
    npy_data = dict()
    npy_data['loss_list']   = np.array(loss_list) 
    npy_data['metric_list'] = np.array(metric_list)
    log_saved_dir           = f'log/{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.npy'
    np.save(log_saved_dir, npy_data)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',              default = 'pennaction',    type = str, required = False)
    parser.add_argument('--model_version',                                   type = str, required = False)
    parser.add_argument('--model_type',           default = 'deterministic', type = str, required = True)
    parser.add_argument('--num_joints',           default = 13,       type = int,    required = False)
    parser.add_argument('--num_training_frame',   default = 7,        type = int,    required = False)
    parser.add_argument('--num_testing_frame',    default = 7,        type = int,    required = False) 
    parser.add_argument('--learning_rate',        default = 1e-4,     type = float,  required = False)
    parser.add_argument('--num_epochs',           default = 100,      type = int,    required = False) 
    parser.add_argument('--batch_size',           default = 16,       type = int,    required = False) 
    parser.add_argument('--num_workers',          default = 8,        type = int,    required = False)
    parser.add_argument('--pin_memory',           default = True,     type = bool,   required = False)
    parser.add_argument('--decouple_loss',        default = False,    type = bool,   required = False)
    parser.add_argument('--step_per_epoch',       default = 1000,      type = int,    required = True)

    parser.add_argument('--device',               default = 'cuda',     type = str, required = False) 
    
    parser.add_argument('--model' ,               default = 'lstm',   type = str,  required = True) 
    parser.add_argument('--model_input_type',     default = None,     type = str,  required = True)
    parser.add_argument('--print_step',           default = 50,       type = int,  required = False)
    
    parser.add_argument('--mode',                 default = 'baseline', type = str,   required = True)
    parser.add_argument('--dropout',              default = 0.2,     type = float, required = False)
    parser.add_argument('--pck_alpha',            default = 0.05,    type = float, required = False)

    # Dataset Config 
    parser.add_argument('--ntu_frame_input_h', default = 1080, type = int, required = False) 
    parser.add_argument('--ntu_frame_input_w', default = 1920, type = int, required = False) 
    parser.add_argument('--ntu_boxsize',       default = 640,  type = int, required = False) 

    parser.add_argument('--pennaction_frame_input_h', default = 480, type = int, required = False) 
    parser.add_argument('--pennaction_frame_input_w', default = 640, type = int, required = False) 
    parser.add_argument('--pennaction_boxsize',       default = 480,  type = int, required = False) 
    
    # Embedding and Optimizer config
    parser.add_argument('--betas',                           default = (0.9, 0.999), type = tuple, required = False) 
    parser.add_argument('--temperature',                     default = 10000,        type = float, required = False) 
    parser.add_argument('--alpha',                           default = 1,            type = float, required = False)
    parser.add_argument('--weight_decay',                    default = 1e-4,         type = float, required = False)
    parser.add_argument('--teacher_forcing_rate',            default = 0.7,          type = float, required = False)
    parser.add_argument('--positional_encoding_temperature', default = 10000,        type = float, required = False)  
    parser.add_argument('--positional_encoding_alpha',       default = 1,            type = float, required = False)
    parser.add_argument('--max_norm',                        default = 0.1,          type = float, required = False)

    # Learning rate scheduler config  

    parser.add_argument('--lr_scheduler', default = 'cosine', type = str,   required = True)
    parser.add_argument('--lr_step_size', default = 40,       type = int,   required = False)
    parser.add_argument('--lr_gamma',     default = 0.3,      type = float, required = False)
    parser.add_argument('--lr_eta_min',   default = 1e-6,     type = float, required = False)
    parser.add_argument('--lr_factor',    default = 0.5,      type = float, required = False)
    parser.add_argument('--lr_patience',  default = 3,        type = float, required = False)

    # Loss function 
    parser.add_argument('--loss_type',            default = 'penalize', type = str,    required = True)
    parser.add_argument('--lambda_norm',          default = 1.0,        type = float,  required = False)
    parser.add_argument('--lambda_specs',         default = 1.5,        type = float,  required = False)
    parser.add_argument('--lambda_root',          default = 2.0,        type = float,  required = False)
    parser.add_argument('--root_pos',             default = 0,          type = int,    required = False)
    parser.add_argument('--special_joints',       default = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], type = list, required = False) # for NTU
    
    # MODEL SPACE

    # MLP 
    parser.add_argument('--mlp_hidden_layers',  default = [1024, 2048, 1024], type = list, required = False) 
    
    # LSTM 
    parser.add_argument('--lstm_embedding_dim',       default = 2048, type = int,  required = False) 
    parser.add_argument('--lstm_hidden_dim'   ,       default = 256,  type = int,  required = False)
    parser.add_argument('--lstm_num_layers',          default = 2,    type = int,  required = False) 
    parser.add_argument('--lstm_residual_connection', default = True, type = bool, required = False)

    # MOTION VAE 
    parser.add_argument('--motion_vae_hidden_dim',       default = 32, type = int, required = False) 
    parser.add_argument('--motion_vae_latent_dim',       default = 16, type = int, required = False) 
    parser.add_argument('--motion_vae_encoding_layers',  default = 2,  type = int, required = False) 
    parser.add_argument('--motion_vae_decoding_layers',  default = 1,  type = int, required = False) 


    # DECOUPLE MOTION VAE
    parser.add_argument('--decouple_motion_vae_hidden_dim',       default = 32, type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_latent_dim',       default = 16, type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_encoding_layers',  default = 2,  type  = int, required = False) 
    parser.add_argument('--decouple_motion_vae_decoding_layers',  default = 1,  type  = int, required = False) 
    parser.add_argument('--decouple_lstm_embedding_dim',          default = 8,   type = int, required = False)
    parser.add_argument('--decouple_lstm_hidden_dim',             default = 16,  type = int, required = False)
    parser.add_argument('--decouple_lstm_num_layers',             default = 1,   type = int, required = False)

    # VELOCITY TRANSFORMER 

    # TRANSFORMER
    parser.add_argument('--transformer_num_heads',           default = 4,         type = int,  required = False)  
    parser.add_argument('--transformer_model_dim',           default = 256,       type = int,  required = False) 
    parser.add_argument('--transformer_fc_hidden_dim',       default = 2048,      type = int,  required = False) 
    parser.add_argument('--transformer_num_layers',          default = 4,         type = int,  required = False) 
    parser.add_argument('--transformer_pre_normalization',   default = False,     type = bool, required = False)
    parser.add_argument('--transformer_residual_connection', default = True,      type = bool, required = False) 
    parser.add_argument('--transformer_type',                default = 'vanilla', type = str,  required = False)
    parser.add_argument('--get_attn_weight',                 default = False,     type = bool, required = False)
    parser.add_argument('--joint_attn_embed_dim',            default = 128,       type = int,  required = False)   
    parser.add_argument('--joint_attn_num_layers',           default = 2,         type = int,  required = False)
    parser.add_argument('--joint_attn_num_heads',            default = 4,         type = int,  required = False)
    parser.add_argument('--joint_attn_fc_hidden_dim',        default = 512,       type = int,  required = False)
    parser.add_argument('--proj_model_type',                 default = 'attn',    type = str,  required = False) 



    args = parser.parse_args()
    train(args) 


