from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler

from trainer.CoordinatesTrainer  import * 
from trainer.RootRelativeTrainer import *
from trainer.VelocityTrainer     import * 

from data.DatasetUtil           import *
from data.CoordinatesNTU        import * 
from data.CoordinatesPennaction import * 
from data.VelocityNTU           import * 
from data.VelocityPennaction    import *
from data.RootRelativeNTU       import *  
from data.Visualization         import * 
from data.ImgUtil               import *
from data.DatasetUtil           import *

from model.Lstm               import *
from model.Mlp                import *
from model.LstmMotionVae      import * 
from model.Transformer        import * 


def class_factory(args):
    if args.dataset == 'pennaction':
        if args.model_input_type == 'coordinates': 
            return PennActionCoordinates, CoordinatesTrainer
        elif args.model_input_type == 'velocity'  : 
            return PennActionVelocity, VelocityTrainer 
        else : 
            raise NotImplementedError('representation is not supported for dataset')
        
    elif args.dataset == 'ntu':
        if   args.model_input_type == 'coordinates' : 
            return NTUCoordinates,  CoordinatesTrainer
        elif args.model_input_type == 'velocity' : 
            return NTUVelocity,     VelocityTrainer 
        elif args.model_input_type == 'root_relative' :
            return NTURootRelative, RootRelativeTrainer 
        else : 
            raise NotImplementedError('representation is not supported for dataset')
    else : 
        raise NotImplementedError('dataset not supported')

def dataset_factory(args):
    dataset_class, trainer_class = class_factory(args)
    if args.dataset == 'pennaction':
        dataset = dataset_class(
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame,
            frame_input_h      = args.pennaction_frame_input_h, 
            frame_input_w      = args.pennaction_frame_input_w,
            box_size           = args.pennaction_boxsize, 
            train              = True, 
        )

        test_dataset = dataset_class(
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame,
            frame_input_h      = args.pennaction_frame_input_h, 
            frame_input_w      = args.pennaction_frame_input_w,
            box_size           = args.pennaction_boxsize, 
            train              = False, 
        )
        
        num_train = int(len(dataset) * 0.9)  # Validation set rate
        num_valid = len(dataset) - num_train
        train_dataset, val_dataset = random_split(dataset, lengths = [num_train, num_valid])


        train_sampler = DistributedSampler(train_dataset)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    

        train_loader = DataLoader(
            train_dataset,
            batch_size  = args.batch_size,
            sampler     = train_sampler,  
            num_workers = args.num_workers,
            pin_memory = args.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size  = args.batch_size,
            sampler     = val_sampler,
            num_workers = args.num_workers,
            pin_memory  = args.pin_memory
        )

        trainer       = trainer_class(
            pck_dict_fn  = get_pennaction_pck_dict, 
            rmse_dict_fn = get_pennaction_mse_dict, 
            train_loader = train_loader, 
            val_loader   = val_loader, 
            test_dataset = test_dataset, 
            dataset      = 'pennaction', 
            print_step   = args.print_step, 
            model_type   = args.model_type, 
            video_getter_fn = get_video_frames_pennaction,
            frame_input_h = args.pennaction_frame_input_h, 
            frame_input_w = args.pennaction_frame_input_w,
        )

        return trainer, train_sampler
    
    elif args.dataset == 'ntu': 
        dataset = dataset_class(
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame,
            frame_input_h      = args.ntu_frame_input_h, 
            frame_input_w      = args.ntu_frame_input_w,
            box_size           = args.ntu_boxsize, 
            train              = True, 
        )

        num_train = int(len(dataset) * 0.9)  # Validation set rate
        num_valid = len(dataset) - num_train
        train_dataset, val_dataset = random_split(dataset, lengths = [num_train, num_valid])

        test_dataset =  dataset_class(
            num_training_frame = args.num_training_frame, 
            num_testing_frame  = args.num_testing_frame,
            frame_input_h      = args.ntu_frame_input_h, 
            frame_input_w      = args.ntu_frame_input_w,
            box_size           = args.ntu_boxsize, 
            train              = False, 
        )

        
        train_sampler = DistributedSampler(train_dataset)
        val_sampler   = DistributedSampler(val_dataset, shuffle = False)
    

        train_loader = DataLoader(
            train_dataset,
            batch_size  = args.batch_size,
            sampler     = train_sampler,  
            num_workers = args.num_workers,
            pin_memory  = args.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size  = args.batch_size,
            sampler     = val_sampler,
            num_workers = args.num_workers,
            pin_memory  = args.pin_memory
        )


        trainer       = trainer_class(
            pck_dict_fn     = get_ntu_pck_dict, 
            rmse_dict_fn    = get_ntu_mse_dict, 
            train_loader    = train_loader, 
            val_loader      = val_loader, 
            test_dataset    = test_dataset, 
            dataset         = 'ntu', 
            print_step      = args.print_step, 
            model_type      = args.model_type, 
            video_getter_fn = get_video_frames_ntu, 
            frame_input_h   = args.ntu_frame_input_h, 
            frame_input_w   = args.ntu_frame_input_w,
        )

        return trainer, train_sampler 
    
    else : 
        raise NotImplementedError('dataset not supported')
    

def get_model(args):
    if args.model == 'mlp': 
        model = get_mlp_model(
            pose_dim      = args.num_joints, 
            coordinates   = args.model_input_type == 'coordinates', 
            train_len     = args.num_training_frame - (1 if args.model_input_type == 'velocity' else 0), 
            predict_len   = args.num_testing_frame, 
            hidden_layers = args.mlp_hidden_layers, 
            dropout       = args.dropout, 
            device        = args.device, 
        )
        criterion = mlp_loss_function
        return model, criterion 
    elif args.model == 'lstm': 
        model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.model_input_type == 'coordinates', 
            embedding_dim  = args.lstm_embedding_dim,
            hidden_dim     = args.lstm_hidden_dim, 
            num_layers     = args.lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
            residual_connection = args.lstm_residual_connection,
        )
        criterion = lstm_loss_function 
        return model, criterion 
    
    elif args.model == 'motion_vae': 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.motion_vae_hidden_dim, 
            latent_dim      = args.motion_vae_latent_dim,
            encoding_layers = args.motion_vae_encoding_layers, 
            decoding_layers = args.motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        return vae_model, criterion_vae
    
    elif args.model == 'transformer': 
        model = get_transformer_model(
            pose_dim          = args.num_joints, 
            coordinates       = args.model_input_type == 'coordinates',
            encoding_length   = args.num_training_frame - (1 if args.model_input_type == 'velocity' else 0), 
            decoding_length   = args.num_testing_frame, 
            num_heads         = args.transformer_num_heads, 
            model_dim         = args.transformer_model_dim, 
            fc_hidden_dim     = args.transformer_fc_hidden_dim, 
            num_layers        = args.transformer_num_layers, 
            dropout           = args.dropout, 
            pre_normalization = args.transformer_pre_normalization, 
            positional_encoding_temperature = args.positional_encoding_temperature, 
            positional_encoding_alpha       = args.positional_encoding_alpha, 
            device                          = args.device, 
        )
        criterion = transformer_loss_function 
        return model, criterion
    elif args.model == 'decouple_motion_vae': 
        # Motion Vae Comprises of two models, one for the root, and one for others 
        vae_model = get_rnn_vae_model(
            pose_dim        = args.num_joints, 
            hidden_dim      = args.decouple_motion_vae_hidden_dim, 
            latent_dim      = args.decouple_motion_vae_latent_dim,
            encoding_layers = args.decouple_motion_vae_encoding_layers, 
            decoding_layers = args.decouple_motion_vae_decoding_layers, 
            dropout         = args.dropout, 
            device          = args.device, 
        )
        criterion_vae = vae_loss_function
        lstm_model = get_lstm_model(
            pose_dim       = args.num_joints, 
            coordinates    = args.model_input_type == 'coordinates', 
            embedding_dim  = args.decouple_lstm_embedding_dim,
            hidden_dim     = args.decouple_lstm_hidden_dim, 
            num_layers     = args.decouple_lstm_num_layers, 
            dropout        = args.dropout, 
            device         = args.device, 
        )

        criterion_lstm = lstm_loss_function

        return (vae_model, lstm_model), (criterion_vae, criterion_lstm)
    else : 
        raise NotImplementedError('Model not supported for now') 


def lr_scheduler_factory(args, optimizer, start_epoch):
    if args.lr_scheduler == 'step':
        print(f"Using StepLR with step size {args.lr_step_size} and gamma {args.lr_gamma}")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer = optimizer, 
            step_size = args.lr_step_size, 
            gamma     = args.lr_gamma, 
        )
        return scheduler
    elif args.lr_scheduler == 'cosine':
        print(f"Using CosineAnnealingLR with T_max {args.num_epochs}, eta_min {args.lr_eta_min}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer  = optimizer,
            T_max      = args.num_epochs, 
            eta_min    = args.lr_eta_min, 
            last_epoch = start_epoch - 1,  
        )
        return scheduler
    elif args.lr_scheduler == 'plateau':
        print(f"Using ReduceLROnPlateau with factor {args.lr_factor} and patience {args.lr_patience}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode      = 'min', 
            factor    = args.lr_factor,
            patience  = args.lr_patience, 
        )
        return scheduler
    else : 
        raise NotImplementedError('Learning Rate Scheduler not supported for now') 
    