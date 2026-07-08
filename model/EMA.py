import torch.nn as nn 

class EMAHelper(object):
  def __init__(self, mu = 0.999): 
   self.mu = mu
   self.curr_step = 0

  def update_model_average(self, ema_model, model):
    for current_params, ema_params in zip(model.parameters(), ema_model.parameters()): 
      old_weight, up_weight = ema_params.data, current_params.data
      ema_params.data = self.update_average(old_weight, up_weight)

  def update_average(self, curr_model, new_model): 
    if curr_model is None :
      return new_model
    return curr_model * self.mu + (1 - self.mu) * new_model

  def update_ema(self, ema_model, model, step_start_ema = 20000): 
    if self.curr_step < step_start_ema: 
      self.curr_step += 1
      self.reset_parameters(ema_model, model) 
      return  
    self.curr_step += 1 
    self.update_model_average(ema_model, model) 
  
  def reset_parameters(self, ema_model, model): 
    ema_model.load_state_dict(model.state_dict())