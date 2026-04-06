import torch
from trainer.Factory import *
import os 
import argparse 


def eval(args):
    trainer = dataset_factory(args)
    device = args.device 
    print(device, 'is available')

    checkpoint = f"checkpoint/{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"

    if os.path.exists(checkpoint): 
        model_dict = torch.load(checkpoint, weights_only = False) 
        model, criterion = get_model(args)
        model.to(device)
        model.load_state_dict(model_dict['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        optimizer.load_state_dict(model_dict['optimizer'])
         
    else : 
        model, criterion = get_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, betas = args.betas)
        model.to(device)
    
  #  trainer.evaluation(args, model, criterion)
    trainer.inference(args, model, criterion)



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

    parser.add_argument('--device',               default = 'cuda',   type = str,    required = False) 
    
    parser.add_argument('--model' ,               default = 'lstm',   type = str,  required = True) 
    parser.add_argument('--model_input_type',     default = None,     type = str,  required = True)
    parser.add_argument('--print_step',           default = 10,       type = int,  required = False)
    
    parser.add_argument('--mode',                 default = 'basecline', type = str,   required = True)
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
    parser.add_argument('--scheduler_step',       default = 10000,      type = float, required = False) 
    parser.add_argument('--scheduler_gamma',      default = 0.3333,     type = float, required = False)
    parser.add_argument('--betas',                default = (0.9, 0.999), type = tuple, required = False) 
    parser.add_argument('--temperature',          default = 10000,   type = float, required = False) 
    parser.add_argument('--alpha',                default = 1,       type = float, required = False)
    parser.add_argument('--weight_decay',         default = 0.00001, type = float, required = False)
    parser.add_argument('--teacher_forcing_rate', default = 0.7,   type = float, required = False)
    parser.add_argument('--positional_encoding_temperature', default = 10000, type = float, required = False)  
    parser.add_argument('--positional_encoding_alpha',       default = 1,     type = float, required = False)




    # MLP 
    parser.add_argument('--mlp_hidden_layers',  default = [1024, 2048, 1024], type = list[int], required = False) 
    
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
    parser.add_argument('--transformer_num_heads',     default = 4,    type = int, required = False)  
    parser.add_argument('--transformer_query_dim',     default = 64,   type = int, required = False) 
    parser.add_argument('--transformer_value_dim',     default = 64,   type = int, required = False) 
    parser.add_argument('--transformer_model_dim',     default = 512,  type = int, required = False) 
    parser.add_argument('--transformer_fc_hidden_dim', default = 2048, type = int, required = False) 
    parser.add_argument('--transformer_num_layers',    default = 4,    type = int, required = False) 
    

    args = parser.parse_args()
    eval(args) 


