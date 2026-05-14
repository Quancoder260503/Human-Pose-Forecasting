import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil   import *
from runner.RunnerUtil  import *
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
from ignite.handlers import create_lr_scheduler_with_warmup
from torch.utils.data import random_split
from runner.Losses              import * 
from model.Lstm                 import *
from model.Mlp                  import *
from model.LstmMotionVae        import * 
from model.Transformer          import * 
from model.Autoencoder          import * 
from data.DatasetUtil           import *
from data.NTU                   import *  
from data.Visualization         import * 
from data.ImgUtil               import *
from data.DatasetUtil           import *
from model.GaussianDiffusion    import * 
from model.DiffusionTransformer import *
from model.EMA import * 


class Runner(ABC):
    def __init__(self, 
        args, 
        pck_dict_fn, 
        rmse_dict_fn, 
        model_checkpoint, 
        eval,
    ):
        self.pck_dict_fn           = pck_dict_fn 
        self.rmse_dict_fn          = rmse_dict_fn 
        self.args                  = args
        self.dataset_joint_name    = JOINT_NAMES[self.args.dataset]
        self.model_checkpoint      = model_checkpoint

        if self.args.pose_decoupling : 
            self.model_a, self.model_b = self.get_model()
            self.model_a = self.model_a.to(self.args.device)
            self.model_b = self.model_b.to(self.args.device)
            self.model_a_variational = False
            self.model_b_variational = False 
            self.criterion_a, self.criterion_b = self.get_criterions()
            
            self.optimizer_a = self.get_optimizer(self.model_a.parameters())
            self.optimizer_b = self.get_optimizer(self.model_b.parameters())
        else : 
            self.model_a, self.model_b = self.get_model()
            self.model_a  = self.model_a.to(self.args.device) 
            self.model_a_variational = False
            self.model_b_variational = False
            self.criterion_a, self.criterion_b = self.get_criterions()
            self.optimizer_a = self.get_optimizer(self.model_a.parameters())
            self.optimizer_b = None

        self.ema_helper = EMAHelper(mu = self.args.ema_rate)
        self.ema_helper.register(self.model_a)

        self.init_lr_scheduler()

        self.get_dataset()

        print('=' * 40)
        print('Number of train batches      =', min(self.args.step_per_epoch, len(self.train_loader)))
        print('Number of validation batches =', len(self.val_loader))
        print('Number of testing instances  =', len(self.test_loader))
        
        self.start_epoch = 0 

        if os.path.exists(self.model_checkpoint): 
            model_dict = torch.load(self.model_checkpoint, weights_only = False, map_location = "cuda:0")
            self.start_epoch = model_dict['epoch'] + 1 
            
            self.model_a.load_state_dict(model_dict['model'])

            if 'ema_helper' in model_dict:
                self.ema_helper.load_state_dict(model_dict['ema_helper'])


            if self.model_b is not None : 
                self.model_b.load_state_dict(model_dict['model_b'])

            if not eval : 
                self.optimizer_a.load_state_dict(model_dict['optimizer'])
                self.lr_scheduler_a.load_state_dict(model_dict['lr_scheduler'])

                if self.optimizer_b is not None : 
                    self.optimizer_b.load_state_dict(model_dict['optimizer_b'])
                if self.lr_scheduler_b is not None : 
                    self.lr_scheduler_b.load_state_dict(model_dict['lr_scheduler_b'])

        elif eval : 
            raise FileNotFoundError('Checkpoint of the model cannot be found')

        self._last_attn   = None 
        
    
    def _forward(self, model, model_type, src_seq, prediction_len):
        if model_type == 'transformer':
            if self.args.get_attn_weight:
                output, self_attn, cross_attn = model(src_seq, True)
                self._last_attn = (self_attn, cross_attn)
            else:
                output = model(src_seq)

        elif model_type in ('mlp', 'bert', 'lstm'):
                output = model(src_seq, prediction_len)
        else:
            raise NotImplementedError(f'{model_type} is not implemented')
        
        return output

    def _compute_loss(self, criterion_fn, loss_fn, prediction, tgt_seq, mean = None, log_var = None, beta = None, variational = False):
        if variational : 
            loss = criterion_fn(self.args, loss_fn, prediction, tgt_seq, mean, log_var, beta)
        else : 
            loss = criterion_fn(self.args, loss_fn, prediction, tgt_seq)
        return loss 


    
    @abstractmethod
    def process_per_epoch(self, epoch): 
        pass 

    @abstractmethod
    def evaluation(self): 
        pass 

    @abstractmethod
    def inference(self):
        pass 


    @abstractmethod
    def prepare_dataset(self): 
        pass

    def get_optimizer(self, model_params): 
        if self.args.optimizer == 'Adam': 
            return torch.optim.Adam(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay,
                betas        = self.args.betas, 
            )
        
        elif self.args.optimizer == 'RMSProp': 
            return torch.optim.RMSprop(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay, 
            )
        
        elif self.args.optimizer == 'SGD': 
            return torch.optim.SGD(
                params       = model_params, 
                lr           = self.args.learning_rate, 
                weight_decay = self.args.weight_decay, 
            )
        else : 
            raise NotImplementedError('Optimizer not supported')
    

    def get_lr_scheduler(self, optimizer_a, optimizer_b):
        scheduler_a, scheduler_b = None, None
        if self.args.lr_scheduler_a == 'step':
            #print(f"Using StepLR with step size {self.args.lr_step_size} and gamma {self.args.lr_gamma}")
            scheduler_a = torch.optim.lr_scheduler.StepLR(
                optimizer = optimizer_a, 
                step_size = self.args.lr_step_size, 
                gamma     = self.args.lr_gamma, 
            )
        elif self.args.lr_scheduler_a == 'cosine':
            #print(f"Using CosineAnnealingLR with T_max {args.num_epochs}, eta_min {args.lr_eta_min}")
            scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer  = optimizer_a,
                T_max      = self.args.num_epochs, 
                eta_min    = self.args.lr_eta_min, 
                last_epoch = -1, 
            )
        elif self.args.lr_scheduler_a == 'plateau':
            #print(f"Using ReduceLROnPlateau with factor {args.lr_factor} and patience {args.lr_patience}")
            scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer_a, 
                mode      = 'min', 
                factor    = self.args.lr_factor,
                patience  = self.args.lr_patience, 
            )
        else : 
            raise NotImplementedError('Learning Rate Scheduler not supported for now') 
        
        if optimizer_b is None : 
            return (scheduler_a, None)
        
        if self.args.lr_scheduler_b == 'step':
            #print(f"Using StepLR with step size {self.args.lr_step_size} and gamma {self.args.lr_gamma}")
            scheduler_b = torch.optim.lr_scheduler.StepLR(
                optimizer = optimizer_b, 
                step_size = self.args.lr_step_size, 
                gamma     = self.args.lr_gamma, 
            )
        elif self.args.lr_scheduler_b == 'cosine':
            #print(f"Using CosineAnnealingLR with T_max {args.num_epochs}, eta_min {args.lr_eta_min}")
            scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer  = optimizer_b,
                T_max      = self.args.num_epochs, 
                eta_min    = self.args.lr_eta_min, 
                last_epoch = -1, 
            )
        elif self.args.lr_scheduler_b == 'plateau':
            #print(f"Using ReduceLROnPlateau with factor {args.lr_factor} and patience {args.lr_patience}")
            scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer_b, 
                mode      = 'min', 
                factor    = self.args.lr_factor,
                patience  = self.args.lr_patience, 
            )
        else : 
            raise NotImplementedError('Learning Rate Scheduler not supported for now')

        return (scheduler_a, scheduler_b)

    def init_lr_scheduler(self):
        if self.args.use_warm_up: 
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_a,
                start_factor = self.args.warmup_learning_rate / self.args.learning_rate,
                end_factor   = 1.0,
                total_iters  = self.args.warmup_steps,
            )
            scheduler_a, scheduler_b = self.get_lr_scheduler(self.optimizer_a, self.optimizer_b)

            self.lr_scheduler_a = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer_a,
                schedulers = [warmup, scheduler_a],
                milestones = [self.args.warmup_steps],
            )
            self.lr_scheduler_b = scheduler_b

        else : 
            self.lr_scheduler_a, self.lr_scheduler_b = self.get_lr_scheduler()

    def get_model(self):
        model_a = None  
        if  self.args.model_a == 'mlp': 
            model_a = get_mlp_model(
                pose_dim      = self.args.model_a_num_joints, 
                joint_dim     = self.args.joint_dim,
                train_len     = self.args.num_training_frame, 
                predict_len   = self.args.num_testing_frame, 
                hidden_layers = self.args.mlp_hidden_layers, 
                dropout       = self.args.dropout, 
                device        = self.args.device, 
            )
            
        elif self.args.model_a == 'lstm': 
            model_a = get_lstm_model(
                pose_dim       = self.args.model_a_num_joints, 
                joint_dim      = self.args.joint_dim,
                embedding_dim  = self.args.lstm_embedding_dim,
                hidden_dim     = self.args.lstm_hidden_dim, 
                num_layers     = self.args.lstm_num_layers, 
                dropout        = self.args.dropout, 
                device         = self.args.device, 
                residual_connection = self.args.lstm_residual_connection,
            )
        
        elif self.args.model_a == 'transformer': 
            model_a = get_transformer_model(
                num_joints                      = self.args.model_a_num_joints, 
                joint_dim                       = self.args.joint_dim, 
                encoding_length                 = self.args.num_training_frame, 
                decoding_length                 = self.args.num_testing_frame, 
                num_heads                       = self.args.transformer_num_heads, 
                model_dim                       = self.args.transformer_model_dim, 
                fc_hidden_dim                   = self.args.transformer_fc_hidden_dim, 
                num_layers                      = self.args.transformer_num_layers, 
                dropout                         = self.args.dropout, 
                pre_normalization               = self.args.transformer_pre_normalization, 
                positional_encoding_temperature = self.args.positional_encoding_temperature, 
                positional_encoding_alpha       = self.args.positional_encoding_alpha, 
                device                          = self.args.device, 
                residual_connection             = self.args.transformer_residual_connection, 
                joint_attn_embed_dim            = self.args.joint_attn_embed_dim, 
                joint_attn_num_layers           = self.args.joint_attn_num_layers, 
                joint_attn_num_heads            = self.args.joint_attn_num_heads, 
                joint_attn_fc_hidden_dim        = self.args.joint_attn_fc_hidden_dim, 
                proj_model_type                 = self.args.proj_model_type,
                use_query_embedding             = self.args.use_query_embedding,  
            )
        elif self.args.model_a.starts_with('diffusion'):
            self.gaussian_diffusion = GaussianDiffusion(
                noise_schedule    = self.args.noise_scheduler, 
                model_mean_type   = self.args.model_mean_type, 
                model_var_type    = self.args.model_var_type, 
                loss_type         = self.args.loss_type, 
                rescale_timesteps = self.args.rescale_timesteps, 
                num_steps         = self.args.diffusion_num_time_steps,
            )

            model_a = get_diffusion_transformer_model(
                input_features                  = self.args.num_joints * self.args.joint_dim, 
                prediction_len                  = self.args.num_dct_channels,  # quite misleading name for DCT
                ffn_dim                         = self.args.diff_transformer_ffn_dim, 
                num_layers                      = self.args.diff_transformer_num_layers, 
                positional_encoding_temperature = self.args.positional_encoding_temperature, 
                positional_encoding_alpha       = self.args.positional_encoding_alpha, 
                use_cross_attention             = self.args.diff_transformer_use_cross_attention, 
                activation                      = self.args.diff_transformer_activation,   
            )

        else : 
            raise NotImplementedError('Model not supported for now') 
        
        if self.args.pose_decoupling is False:
            return (model_a, None)
        if self.args.model_b == 'lstm':
            model_b = get_lstm_model(
                pose_dim            = self.args.model_b_num_joints, 
                joint_dim           = self.args.joint_dim, 
                embedding_dim       = self.args.lstm_embedding_dim,
                hidden_dim          = self.args.lstm_hidden_dim, 
                num_layers          = self.args.lstm_num_layers, 
                dropout             = self.args.dropout, 
                device              = self.args.device, 
                residual_connection = self.args.lstm_residual_connection,
            )
        return (model_a, model_b) 

    def get_criterions(self):
        criterion_a, criterion_b = None, None 
        if self.args.model_a_loss_type == 'naive': 
            criterion_a = vae_loss_function if self.model_a_variational else deterministic_loss_function 
        elif self.args.model_a_loss_type == 'penalize': 
            criterion_a = penalize_vae_function if self.model_a_variational else vae_loss_function

        if self.args.model_b_loss_type   == 'naive': 
            criterion_b = vae_loss_function if self.model_b_variational else deterministic_loss_function 
        elif self.args.model_b_loss_type == 'penalize': 
            criterion_b = penalize_vae_function if self.model_b_variational else vae_loss_function 

        return criterion_a, criterion_b

    def get_dataset(self):
        dataset_class = self.prepare_dataset()

        train_dataset = dataset_class(
            num_joints         = self.args.num_joints,
            num_training_frame = self.args.num_training_frame, 
            num_testing_frame  = self.args.num_testing_frame,
            frame_input_h      = self.args.frame_input_h, 
            frame_input_w      = self.args.frame_input_w,
            box_size           = self.args.boxsize, 
            on_debug_mode      = self.args.mode == 'debug', 
            train              = True, 
        )

        test_dataset = dataset_class(
            num_joints         = self.args.num_joints,
            num_training_frame = self.args.num_training_frame, 
            num_testing_frame  = self.args.num_testing_frame,
            frame_input_h      = self.args.frame_input_h, 
            frame_input_w      = self.args.frame_input_w,
            box_size           = self.args.boxsize, 
            on_debug_mode      = self.args.mode == 'debug', 
            train              = False, 
        )

        num_valid = int(len(test_dataset) * 0.1)  # Validation set rate
        num_test = len(test_dataset) - num_valid
        test_dataset, val_dataset = random_split(test_dataset, lengths = [num_test, num_valid])


        self.train_loader = DataLoader(
            dataset     = train_dataset, 
            batch_size  = self.args.batch_size,
            num_workers = self.args.num_workers,
            pin_memory  = self.args.pin_memory, 
            shuffle     = True, 
        )

        self.val_loader = DataLoader(
            dataset     = val_dataset,
            batch_size  = self.args.batch_size,
            num_workers = self.args.num_workers,
            pin_memory  = self.args.pin_memory,  
        )

        self.test_loader = DataLoader(
            dataset     = test_dataset,
            batch_size  = self.args.batch_size,
            num_workers = self.args.num_workers,
            pin_memory  = self.args.pin_memory, 
        )

        self.mean = train_dataset.mean if hasattr(train_dataset, "mean") else None
        self.std  = train_dataset.std  if hasattr(train_dataset, "std")  else None
        self.root_mean = train_dataset.root_mean if hasattr(train_dataset, "root_mean") else None 
        self.root_std  = train_dataset.root_std  if hasattr(train_dataset, "root_std")  else None 

    def visualize_pose(self, predicted_poses, observed_poses, target_poses, start_index, name):
        if self.args.joint_dim == 2: 
            animate_ntu_pose_2d(
                observe_sequences  = observed_poses, 
                target_sequences   = target_poses,
                predict_sequences  = predicted_poses, 
                frame_input_h      = self.args.frame_input_h, 
                frame_input_w      = self.args.frame_input_w, 
                start_index        = start_index, 
                num_training_frame = self.args.num_training_frame, 
                num_testing_frame  = self.args.num_testing_frame, 
                num_joints         = self.args.num_joints, 
                fps                = 3, 
            ) 
        elif self.args.joint_dim == 3:
            (setup_id, camera_id, subject_id, replication_id, activity_id) = get_activity_from_file(name)
            print(camera_id)
            animate_ntu_pose_3d(
                observed_sequence   = observed_poses, 
                target_sequence     = target_poses,
                predicted_sequence  = predicted_poses,
                camera_id           = camera_id,  
                fps                 = 3, 
            )



    # def get_gradients(self):
    #     self.model.train()
        
    #     observed_poses, observed_act_seqs, targeted_poses, targeted_act_seqs, _, _ = next(iter(self.train_loader))
        
    #     observed_act_seqs = observed_act_seqs.to(self.args.device, dtype = torch.float32)
    #     targeted_act_seqs = targeted_act_seqs.to(self.args.device, dtype = torch.float32)

    #     predicted_act_seqs, loss = self.get_predicted_n_loss(
    #         train         = True,
    #         src_seq       = observed_act_seqs,
    #         tgt_seq       = targeted_act_seqs,
    #         epoch         = 0,
    #     )

    #     self.optimizer.zero_grad()
    #     loss.backward()

    #     print("=== ENCODER LAYER GRAD NORMS ===")
    #     for i, layer in enumerate(self.model.encoder.encoder_layers):
    #         W = layer._self_attention.in_proj_weight
    #         if W.grad is not None:
    #             print(f"Layer {i} | attn grad norm: {W.grad.norm():.8f} | max: {W.grad.abs().max():.8f}")

    #     for i, layer in enumerate(self.model.decoder.decoder_layers):
    #         W = layer._self_attention.in_proj_weight
    #         if W.grad is not None:
    #             print(f"Layer {i} | self attn grad norm: {W.grad.norm():.8f} | max: {W.grad.abs().max():.8f}")
    #         W = layer._cross_attention.in_proj_weight
    #         if W.grad is not None:
    #             print(f"Layer {i} | cross attn grad norm: {W.grad.norm():.8f} | max: {W.grad.abs().max():.8f}")

    #     self.optimizer.zero_grad()  # clean up
    #     self.model.eval()