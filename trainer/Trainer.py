import matplotlib.pyplot as plt
import numpy as np
from data.DatasetUtil   import *
from trainer.TrainUtil  import *
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import math 


def KL_annealing(epoch, total_epochs, schedule_type = 'linear', start_beta = 0.0, end_beta = 1.0):
    progress = epoch / total_epochs
    if schedule_type == "linear":
        return start_beta + (end_beta - start_beta) * progress
    elif schedule_type == "cyclical":
        return end_beta * (1 + math.cos(math.pi * (1 - progress))) / 2
    elif schedule_type == "sigmoid":
        return start_beta + (end_beta - start_beta) / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

"""
:TODO add some of the KL requirements to trainer 
"""

class Trainer(ABC):
    def __init__(self, dataset, pck_dict_fn, rmse_dict_fn, train_loader, val_loader, test_dataset, print_step, model_type, video_getter_fn, frame_input_h, frame_input_w):
        self.pck_dict_fn        = pck_dict_fn 
        self.rmse_dict_fn       = rmse_dict_fn 
        self.train_loader       = train_loader 
        self.val_loader         = val_loader 
        self.test_dataset       = test_dataset
        self.print_step         = print_step
        self.dataset            = dataset
        self.video_getter_fn    = video_getter_fn
        self.frame_input_w      = frame_input_w 
        self.frame_input_h      = frame_input_h
        self.dataset_joint_name = JOINT_NAMES[dataset]
        self.get_loss_fn  = self.get_deterministic_out if model_type == 'deterministic' else self.get_vae_output


        print('-' * 40)
        print('Number of train batches      =', len(train_loader))
        print('Number of validation batches =', len(val_loader))
        print('Number of testing instances  =', len(test_dataset))

    def get_vae_output(self, args, train, model, criterion, encode_input, decode_input, teacher_forcing_rate, epoch, total_epochs):
        """
        :param train : is the model in training mode ? 
        :param model : model to train
        :param criterion : loss fn 
        :param encode_input : encoder input, a tensor of shape (B, t_train, pose_dim) 
        :param decode_input : decoder input, a tensor of shape (B, t_test,  pose_dim)
        :param teacher forcing rate : hyper-parameter for rnn arch. 
        """
        _, prediction_len, _ = decode_input.shape
        predicted_out, mean, log_var = model(encode_input, decode_input if train else None, prediction_len, teacher_forcing_rate)
        beta = KL_annealing(epoch, total_epochs, self.kl_annealing_schedule_type, self.start_beta, self.end_beta)
        loss = criterion(predicted_out, decode_input, mean, log_var, beta)
        return predicted_out, loss 
    
    def get_deterministic_out(self, args, train, model, criterion, encode_input, decode_input, teacher_forcing_rate):
        """
        :param train : is the model in training mode ? 
        :param model : model to train
        :param criterion : loss fn 
        :param encode_input : encoder input, a tensor of shape (B, t_train, pose_dim) 
        :param decode_input : decoder input, a tensor of shape (B, t_test,  pose_dim)
        :param teacher forcing rate : hyper-parameter for rnn arch. 
        """
        if args.model == 'transformer':
           predicted_out = model(encode_input)
           loss          = criterion(predicted_out, decode_input)
           return predicted_out, loss     
        else :
           _, prediction_len, _ = decode_input.shape  
           predicted_out = model(encode_input, decode_input if train else None, prediction_len, teacher_forcing_rate)
           loss = criterion(predicted_out, decode_input) 
           return predicted_out, loss     
    
    @abstractmethod 
    def process_per_epoch(self, args, model, train_loader, val_loader, optimizer, criterion, epoch, device): 
        pass 

    @abstractmethod 
    def evaluation(self, args, model, criterion): 
        pass 

    @abstractmethod
    def inference(self, args, model, criterion):
        pass 


