import abc
import collections 
import torch 
from dataclasses import field, dataclass 
import bisect
import math 
from model.ProbabilityPath import *
'''
====================================================================
Fixed Step Size ODE Solvers with interpolation support.
====================================================================
'''

class FixedStepSizeODESolver(object):
  def __init__(
    self, 
    velo_func, 
    step_size = None, 
    grid_constructor = None, 
    interpolation = 'linear', 
    path_refiniment_steps = 1, 
    perturb_scale = 0.02, 
  ):
    self.velo_func = velo_func
    self.step_size = step_size
    self.interpolation = interpolation
    self.path_refiniment_steps = path_refiniment_steps 
    self.perturb_scale = perturb_scale
    
    print(f">>> Sampling flow matching path using {self.__class__.__name__}")

    assert self.path_refiniment_steps >= 1, "Refinement step cannot be zero"

    if step_size is None : 
      if grid_constructor is None : 
        self.grid_constructor = lambda f, y_0, t : t 
      else : 
        self.grid_constructor = grid_constructor 
    else : 
      if grid_constructor is None : 
        self.grid_constructor = self.grid_constructor_from_step_size(step_size)
      else : 
        raise ValueError("Cannot specify both `step_size` and `grid_constructor`")  
    

  def grid_constructor_from_step_size(self, step_size):
    def grid_constructor(t):
      start_time = t[0]
      end_time   = t[-1]
      num_iters = torch.ceil((end_time - start_time) / step_size + 1).item()
      t_infer = torch.arange(0, num_iters, dtype = t.dtype, device = t.device) * step_size + start_time
      t_infer[-1] = t[-1]

      return t_infer
    return grid_constructor
  
  @abc.abstractmethod
  def step_func(self, func, t0, dt, t1, y0, cond):
    pass

  def integrate(self, t, cond, mask, dct, idct, num_dct_channels, probability_path : ProbabilityPath):
    y_0 = torch.randn(cond.shape, dtype = cond.dtype).to(cond.device)
    time_grid = self.grid_constructor(self.velo_func, y_0, t)
    results = torch.empty(len(t), * y_0.shape, dtype = y_0.dtype, device = y_0.device)
    results[0] = y_0 

    j = 1 
    # Right now can fuse a bit noise to increase the diversity if use refinement step 
    # now run without feeding raw truth and see if they can generate good poses 
    for t_0, t_1 in zip(time_grid[:-1], time_grid[1:]):
      ones = torch.ones(y_0.shape[0], device = y_0.device, dtype = y_0.dtype)
      gauss_noise = torch.randn_like(cond).to(cond.device)
      y_t = probability_path.sample(x_0 = gauss_noise, x_1 = cond, t = ones * t_0).x_t
      y_known = torch.matmul(idct[:, : num_dct_channels], y_t)
      for k in range(self.path_refiniment_steps):
       # run some refininement steps to fit the trajectory onto data manifold
        y_unknown = torch.matmul(idct[:, : num_dct_channels], y_0) 
        y_combine = y_known * mask + y_unknown * (1. - mask)
        y_0 = torch.matmul(dct[: num_dct_channels, :], y_combine)
        dy, f_0 = self.step_func(
          func = self.velo_func, 
          t_0 = ones * t_0, 
          dt  = ones * (t_1 - t_0),  
          t_1 = ones * t_1, 
          y_0 = y_0, 
          cond = cond, 
        )
        y_1 = y_0 + dy

        if t_1 < time_grid[-1] and k + 1 < self.path_refiniment_steps:
          dy, _ = self.step_func(
            func = self.velo_func, 
            t_0 = ones * t_1, 
            t_1 = ones * time_grid[-1], 
            dt = ones * (time_grid[-1] - t_1), 
            cond = cond, 
            y_0 = y_1, 
          )
          y_data = y_1 + dy 
          eps = torch.randn(cond.shape).to(cond.device) 
          tilde_y_t = probability_path.sample(x_0 = eps, x_1 = y_data, t = ones * t_0).x_t
          y_unknown_ = torch.matmul(idct[:, : num_dct_channels], tilde_y_t)
          y_combine_ = y_known * mask + y_unknown_ * (1. - mask)
    
          y_0 = torch.matmul(dct[:num_dct_channels, :], y_combine_)


      while j < len(t) and t[j] <= t_1 : 
        if self.interpolation == 'linear':
          results[j] = self.linear_interpolation(t_0, t_1, y_0, y_1, t[j])
        elif self.interpolation == 'cubic': 
          f_1 = self.velo_func(y_1, ones * t_1, cond)
          results[j] = self.cubic_interpolation(t_0, y_0, f_0, t_1, y_1, f_1, t[j])
        else:
          raise ValueError(f"Unknown interpolation method {self.interpolation}")
        j += 1

      y_0 = y_1
    return results

  def linear_interpolation(self, t_0, t_1, y_0, y_1, t):
    if t == t_0 : 
      return y_0 
    if t == t_1 : 
      return y_1 

    slope = (t - t_0) / (t_1 - t_0) 
    return y_0 + slope * (y_1 - y_0)

  def cubic_interpolation(self, t_0, y_0, f_0, t_1, y_1, f_1, t):
    h = (t - t_0) / (t_1 - t_0)
    h00 = (1 + 2 * h) * (1 - h) * (1 - h)
    h10 = h * (1 - h) * (1 - h)
    h01 = h * h * (3 - 2 * h)
    h11 = h * h * (h - 1)
    dt = (t_1 - t_0)
    return h00 * y_0 + h10 * dt * f_0 + h01 * y_1 + h11 * dt * f_1


class EulerSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0, cond):
    v_0 = func(y_0, t_0, cond)
    return dt[..., None, None] * v_0, v_0
  
class MidPointSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0, cond):
    half_dt = dt / 2
    v_0 = func(y_0, t_0, cond) 
    mid_point = y_0 + half_dt * v_0 
    return dt[..., None, None] * func(mid_point, t_0 + half_dt, cond), v_0
  
class Heun2Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0, cond):
    v_0 = func(y_0, t_0, cond)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1/2, 1/2],
    ]

    k_1 = v_0
    k_2 = func(y_0 + dt[..., None, None] * butcher_tableu[1][1] * k_1, t_0 + dt * butcher_tableu[1][0], cond)
    return dt[..., None, None] * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2), v_0

class Heun3Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0, cond):
    v_0 = func(y_0, t_0, cond)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0, 0.0],
      [1/3, 1/3, 0.0, 0.0],
      [2/3, 0.0, 2/3, 0.0],
      [0.0, 1/4, 0.0, 3/4],
    ]
    k_1 = v_0
    k_2 = func(y_0 + dt[..., None, None] * butcher_tableu[1][1] * k_1, t_0 + dt * butcher_tableu[1][0], cond)
    k_3 = func(y_0 + dt[..., None, None] * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2), t_0 + dt * butcher_tableu[2][0], cond)
    return dt[..., None, None] * (butcher_tableu[3][1] * k_1 + butcher_tableu[3][2] * k_2 + butcher_tableu[3][3] * k_3), v_0


class RungeKutta4Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0, cond):
    v_0 = func(y_0, t_0, cond)
    # sampling the trajectory field
    k_1 = v_0 
    k_2 = func(y_0 + dt[..., None, None] / 3 * k_1, t_0 + dt / 3, cond)
    k_3 = func(y_0 + dt[..., None, None] * (k_2 - k_1 / 3), t_0 + dt * 2 / 3, cond)
    k_4 = func(y_0 + dt[..., None, None] * (k_1 - k_2 + k_3), t_1, cond)
    return (k_1 + 3 * (k_2 + k_3) + k_4) * dt[..., None, None] * 0.125, v_0

