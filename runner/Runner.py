import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil   import *
from runner.RunnerUtil  import *
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import math 



class Runner(ABC):
    def __init__(self, 
            args, 
            dataset, 
            pck_dict_fn, 
            rmse_dict_fn, 
            train_loader, 
            val_loader, 
            test_dataset, 
            video_getter_fn, 
            frame_input_h, 
            frame_input_w
        ):
        self.pck_dict_fn        = pck_dict_fn 
        self.rmse_dict_fn       = rmse_dict_fn 
        self.train_loader       = train_loader 
        self.val_loader         = val_loader 
        self.test_dataset       = test_dataset
        self.dataset            = dataset
        self.video_getter_fn    = video_getter_fn
        self.frame_input_w      = frame_input_w 
        self.frame_input_h      = frame_input_h
        self.dataset_joint_name = JOINT_NAMES[dataset]
        self.args               = args 
        self.get_loss_fn        = self.get_deterministic_out if self.args.model_type == 'deterministic' else self.get_vae_output


        print('-' * 40)
        print('Number of train batches      =', min(self.args.step_per_epoch, len(train_loader)))
        print('Number of validation batches =', len(val_loader))
        print('Number of testing instances  =', len(test_dataset))
    


    def get_vae_output(self, train, model, criterion, src_seq, tgt_seq, epoch):
        """
        :param train : is the model in training mode ? 
        :param model : model to train
        :param criterion : loss fn 
        :param src_seq : encoder input, a tensor of shape (B, t_train, pose_dim) 
        :param tgt_seq : decoder input, a tensor of shape (B, t_test,  pose_dim)
        :param teacher forcing rate : hyper-parameter for rnn arch. 
        """
        _, prediction_len, _ = tgt_seq.shape
        predicted_out, mean, log_var = model(src_seq, tgt_seq if train else None, prediction_len, self.args.teacher_forcing_rate)
        beta = KL_annealing(epoch, self.args.total_epochs, self.kl_annealing_schedule_type, self.start_beta, self.end_beta)
        loss = criterion(predicted_out, tgt_seq, mean, log_var, beta)
        return predicted_out, loss 
    
    def get_deterministic_out(self, train, model, criterion, src_seq, tgt_seq):
        """
        :param train : is the model in training mode ? 
        :param model : model to train
        :param criterion : loss fn 
        :param src_seq : encoder input, a tensor of shape (B, t_train, pose_dim) 
        :param tgt_seq : decoder input, a tensor of shape (B, t_test,  pose_dim)
        :param teacher forcing rate : hyper-parameter for rnn arch. 
        """

        if self.args.loss_type == 'naive':
            if self.args.model == 'transformer':
                if self.args.get_attn_weight: 
                    predicted_out, self_attn, cross_attn = model(src_seq, True)
                    loss          = criterion(predicted_out, tgt_seq)
                    return predicted_out, loss, self_attn, cross_attn     
                else : 
                    predicted_out = model(src_seq)
                    loss          = criterion(predicted_out, tgt_seq)
                    return predicted_out, loss     
            else :
                _, prediction_len, _ = tgt_seq.shape  
                predicted_out = model(
                    src_seq, 
                    tgt_seq if train else None, 
                    prediction_len, 
                    self.args.teacher_forcing_rate
                )
                loss = criterion(predicted_out, tgt_seq) 
                return predicted_out, loss     
        
        elif self.args.loss_type == 'penalize':
            if self.args.model == 'transformer':
                if self.args.get_attn_weight: 
                    predicted_out, self_attn, cross_attn = model(src_seq, True)
                    loss          = criterion(
                        predicted_out, 
                        tgt_seq, 
                        self.args.num_joints, 
                        self.args.special_joints, 
                        self.args.root_pos, 
                        self.args.lambda_root, 
                        self.args.lambda_specs, 
                        self.args.lambda_norm, 
                    )
                    return predicted_out, loss, self_attn, cross_attn     
                else : 
                    predicted_out = model(src_seq)
                    loss          = criterion(
                        predicted_out, 
                        tgt_seq, 
                        self.args.num_joints, 
                        self.args.special_joints, 
                        self.args.root_pos, 
                        self.args.lambda_root, 
                        self.args.lambda_specs, 
                        self.args.lambda_norm, 
                    )
                    
                    return predicted_out, loss     
            else :
                _, prediction_len, _ = tgt_seq.shape  
                predicted_out = model(
                    src_seq, 
                    tgt_seq if train else None, 
                    prediction_len, 
                    self.args.teacher_forcing_rate
                )
                loss          = criterion(
                        predicted_out, 
                        tgt_seq, 
                        self.args.num_joints, 
                        self.args.special_joints, 
                        self.args.root_pos, 
                        self.args.lambda_root, 
                        self.args.lambda_specs, 
                        self.args.lambda_norm, 
                )
                return predicted_out, loss     

    @abstractmethod 
    def process_per_epoch(self, model, train_loader, val_loader, optimizer, criterion, epoch, device): 
        pass 

    @abstractmethod 
    def evaluation(self, model, criterion): 
        pass 

    @abstractmethod
    def inference(self, model, criterion):
        pass 



