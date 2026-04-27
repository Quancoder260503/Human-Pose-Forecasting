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
from model.BERT                 import * 
from data.DatasetUtil           import *
from data.CoordinatesNTU        import * 
from data.CoordinatesPennaction import * 
from data.VelocityNTU           import * 
from data.VelocityPennaction    import *
from data.RootRelativeNTU       import *  
from data.Visualization         import * 
from data.ImgUtil               import *
from data.DatasetUtil           import *



class Runner(ABC):
    def __init__(self, 
            args, 
            pck_dict_fn, 
            rmse_dict_fn, 
            model_checkpoint, 
        ):
        self.pck_dict_fn        = pck_dict_fn 
        self.rmse_dict_fn       = rmse_dict_fn 
        self.args               = args
        self.video_getter_fn    = VIDEO_FUNC[self.args.dataset]
        self.dataset_joint_name = JOINT_NAMES[self.args.dataset]
        self.model_checkpoint   = model_checkpoint
        self.get_predicted_n_loss  = (
            self.get_deterministic_out if self.args.model_type == 'deterministic' else self.get_vae_output
        )
        
        self.model  = self.get_model()
        self.model  = self.model.to(self.args.device) 
        self.criterion = self.get_criterions()
        self.optimizer = self.get_optimizer(self.model.parameters())
        self.init_lr_scheduler()
        self.get_dataset()

        print('-' * 40)
        print('Number of train batches      =', min(self.args.step_per_epoch, len(self.train_loader)))
        print('Number of validation batches =', len(self.val_loader))
        print('Number of testing instances  =', len(self.test_dataset))
        
        self.start_epoch = 0 

        if os.path.exists(self.model_checkpoint): 
            model_dict = torch.load(self.model_checkpoint, weights_only = False) 
            self.start_epoch = model_dict['epoch'] + 1 

            self.model.load_state_dict(model_dict['model'])
        
            
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.lr_scheduler.load_state_dict(model_dict['scheduler'])

        self._last_attn   = None 
        

    
    def _forward(self, train, src_seq, tgt_seq):
        _, prediction_len, _ = tgt_seq.shape
        model = self.args.model

        if model == 'transformer':
            if self.args.get_attn_weight:
                predicted_out, self_attn, cross_attn = self.model(src_seq, True)
                self._last_attn = (self_attn, cross_attn)
            else:
                predicted_out = self.model(src_seq)

        elif model == 'lstm':
            predicted_out = self.model(
                src_seq,
                tgt_seq if train else None,
                prediction_len,
                self.args.teacher_forcing_rate,
            )

        elif model in ('mlp', 'bert'):
            predicted_out = self.model(src_seq, prediction_len)

        else:
            raise NotImplementedError(f'{model} is not implemented')

        return predicted_out

    def _compute_loss(self, predicted_out, tgt_seq):
        """Dispatch loss computation by loss_type."""
        if self.args.loss_type == 'naive':
            return self.criterion(predicted_out, tgt_seq)

        elif self.args.loss_type == 'penalize':
            return self.criterion(
                predicted_out,
                tgt_seq,
                self.args.num_joints,
                self.args.root_pos,
                self.args.lambda_root,
                self.args.lambda_norm,
            )

        raise NotImplementedError(f'{self.args.loss_type} is not implemented')


    def get_deterministic_out(self, train, src_seq, tgt_seq, epoch = None):
        predicted_out = self._forward(train, src_seq, tgt_seq)
        loss          = self._compute_loss(predicted_out, tgt_seq)
        if self._last_attn is not None:
            self_attn, cross_attn = self._last_attn
            return predicted_out, loss, self_attn, cross_attn
        return predicted_out, loss

    def get_vae_output(self, train, src_seq, tgt_seq, epoch):
        _, prediction_len, _ = tgt_seq.shape
        predicted_out, mean, log_var = self.model(
            src_seq,
            tgt_seq if train else None,
            prediction_len,
            self.args.teacher_forcing_rate,
        )
        beta = KL_annealing(
            epoch, 
            self.args.total_epochs,
            self.kl_annealing_schedule_type,
            self.start_beta, 
            self.end_beta,
        )
        loss = self.criterion(predicted_out, tgt_seq, mean, log_var, beta)
        return predicted_out, loss



    @abstractmethod 
    def process_per_epoch(self, epoch, device): 
        pass 

    @abstractmethod 
    def evaluation(self): 
        pass 

    @abstractmethod
    def inference(self):
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
    

    def get_lr_scheduler(self):
        if self.args.lr_scheduler == 'step':
            #print(f"Using StepLR with step size {self.args.lr_step_size} and gamma {self.args.lr_gamma}")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer = self.optimizer, 
                step_size = self.args.lr_step_size, 
                gamma     = self.args.lr_gamma, 
            )
            return scheduler
        elif self.args.lr_scheduler == 'cosine':
            #print(f"Using CosineAnnealingLR with T_max {args.num_epochs}, eta_min {args.lr_eta_min}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer  = self.optimizer,
                T_max      = self.args.num_epochs, 
                eta_min    = self.args.lr_eta_min, 
                last_epoch = -1, 
            )
            return scheduler
        elif self.args.lr_scheduler == 'plateau':
            #print(f"Using ReduceLROnPlateau with factor {args.lr_factor} and patience {args.lr_patience}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = self.optimizer, 
                mode      = 'min', 
                factor    = self.args.lr_factor,
                patience  = self.args.lr_patience, 
            )
            return scheduler
        else : 
            raise NotImplementedError('Learning Rate Scheduler not supported for now') 
        
    def init_lr_scheduler(self):
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor = self.args.warmup_learning_rate / self.args.learning_rate,
            end_factor   = 1.0,
            total_iters  = self.args.warmup_steps,
        )
        after = self.get_lr_scheduler()

        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers = [warmup, after],
            milestones = [self.args.warmup_steps],
        )


    def get_model(self):
        model = None  
        if  self.args.model == 'mlp': 
            model = get_mlp_model(
                pose_dim      = self.args.num_joints, 
                coordinates   = self.args.model_input_type == 'coordinates', 
                train_len     = self.args.num_training_frame - (1 if self.args.model_input_type == 'velocity' else 0), 
                predict_len   = self.args.num_testing_frame, 
                hidden_layers = self.args.mlp_hidden_layers, 
                dropout       = self.args.dropout, 
                device        = self.args.device, 
            )
            
        elif self.args.model == 'lstm': 
            model = get_lstm_model(
                pose_dim       = self.args.num_joints, 
                coordinates    = self.args.model_input_type == 'coordinates', 
                embedding_dim  = self.args.lstm_embedding_dim,
                hidden_dim     = self.args.lstm_hidden_dim, 
                num_layers     = self.args.lstm_num_layers, 
                dropout        = self.args.dropout, 
                device         = self.args.device, 
                residual_connection = self.args.lstm_residual_connection,
            )
        
        elif self.args.model == 'transformer': 
            model = get_transformer_model(
                num_joints                      = self.args.num_joints, 
                coordinates                     = self.args.model_input_type == 'coordinates',
                encoding_length                 = self.args.num_training_frame - (1 if self.args.model_input_type == 'velocity' else 0), 
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
                transformer_type                = self.args.transformer_type, 
                residual_connection             = self.args.transformer_residual_connection, 
                joint_attn_embed_dim            = self.args.joint_attn_embed_dim, 
                joint_attn_num_layers           = self.args.joint_attn_num_layers, 
                joint_attn_num_heads            = self.args.joint_attn_num_heads, 
                joint_attn_fc_hidden_dim        = self.args.joint_attn_fc_hidden_dim, 
                proj_model_type                 = self.args.proj_model_type, 
            )

        elif self.args.model == 'bert':
            model = get_bert_model(
                num_joints                      = self.args.num_joints, 
                coordinates                     = self.args.model_input_type == 'coordinates', 
                window_size                     = self.args.num_training_frame - (1 if self.args.model_input_type == 'velocity' else 0), 
                num_heads                       = self.args.bert_num_heads, 
                model_dim                       = self.args.bert_model_dim, 
                fc_hidden_dim                   = self.args.bert_fc_hidden_dim, 
                num_layers                      = self.args.bert_num_layers, 
                dropout                         = self.args.dropout, 
                positional_encoding_temperature = self.args.positional_encoding_temperature, 
                positional_encoding_alpha       = self.args.positional_encoding_alpha, 
                device                          = self.args.device, 
                pre_normalization               = self.args.bert_pre_normalization,
                attention_mode                  = self.args.bert_attention_mode,  
                pose_encoder_type               = self.args.bert_pose_encoder_type, 
                pose_decoder_type               = self.args.bert_pose_decoder_type, 
            )
            
        else : 
            raise NotImplementedError('Model not supported for now') 
    
        return model 

    def get_criterions(self):
        criterion = None 
        if self.args.loss_type   == 'naive': 
            criterion = naive_loss_function 
        elif self.args.loss_type == 'penalize': 
            criterion = penalize_loss_function 
        else : 
            raise NotImplementedError('loss function unknown')
        return criterion
    
    def prepare_dataset(self):
        if  self.args.dataset == 'pennaction':
            if  self.args.model_input_type == 'coordinates': 
                return PennActionCoordinates
            elif self.args.model_input_type == 'velocity'  : 
                return PennActionVelocity 
            else : 
                raise NotImplementedError('representation is not supported for dataset')
            
        elif self.args.dataset == 'ntu':
            if   self.args.model_input_type == 'coordinates' : 
                return NTUCoordinates
            elif self.args.model_input_type == 'velocity' : 
                return NTUVelocity
            elif self.args.model_input_type == 'root_relative' :
                return NTURootRelative
            else : 
                raise NotImplementedError('representation is not supported for dataset')
        else : 
            raise NotImplementedError('dataset not supported')

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

