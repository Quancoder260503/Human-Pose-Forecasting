import numpy as np
import os
import sys
import torch
import torch.nn as nn
from model.GraphConvNet import xavier_init_, normal_init_, weight_init, PoseGCN, SimpleEncoder

def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module

def pose_encoder_mlp(params):
  init_fn = normal_init_ if params['init_fn'] == 'normal_init' else xavier_init_
  pose_embedding = nn.Sequential(
    nn.Linear(params['pose_dim'], params['model_dim']),
    nn.Dropout(0.1)
  )
  weight_init(pose_embedding, init_fn_ = init_fn)
  return pose_embedding
      

def pose_decoder_mlp(params):
  pose_decoder = nn.Linear(params['model_dim'], params['pose_dim'])
  zero_module(pose_decoder)
  return pose_decoder


def pose_decoder_gcn(params):
  decoder = PoseGCN(
    input_features = params['model_dim'],
    output_features = params['joint_dim'],
    model_dim = params['model_dim'],
    output_nodes = params['num_joints'],
    p_dropout = params['dropout'],
    num_stage = 1
  )
  
  return decoder

def pose_encoder_gcn(params):
  encoder = SimpleEncoder(
    n_nodes = params['num_joints'],
    input_features = params['joint_dim'],
    model_dim = params['model_dim'], 
    p_dropout = params['dropout']
  )
  return encoder


def select_pose_encoder_decoder_fn(params):
  if params['pose_embedding_type'].lower() == 'simple':
    return pose_encoder_mlp, pose_decoder_mlp
  if params['pose_embedding_type'].lower() == 'gcn_dec':
    return pose_encoder_mlp, pose_decoder_gcn
  if params['pose_embedding_type'].lower() == 'gcn_enc':
    return pose_encoder_gcn, pose_decoder_mlp
  if params['pose_embedding_type'].lower() == 'gcn_full':
    return pose_encoder_gcn, pose_decoder_gcn
  else:
    raise ValueError('Unknown pose embedding {}'.format(params['pose_embedding_type']))