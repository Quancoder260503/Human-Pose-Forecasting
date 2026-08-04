import logging
import os
import numpy as np

def create_logger(filename, file_handle = True):
  logger = logging.getLogger(filename)
  logger.propagate = False
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler() 
  ch.setLevel(logging.INFO)
  stream_formatter = logging.Formatter('%(message)s')
  ch.setFormatter(stream_formatter)
  logger.addHandler(ch)

  if file_handle:
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

  return logger


def combine_dict(new_dict, old_dict = None):
  if old_dict is None:
    return new_dict
  for k in old_dict.keys():
    if k in new_dict.keys():
      old_dict[k] = np.concatenate([old_dict[k], new_dict[k]], axis=0)
  return old_dict


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0
    self.acc = 0
    self.avg_acc = 0.  

  def update(self, val, acc = 0., n = 1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.acc += acc * n 
    self.avg_acc = self.acc / self.count