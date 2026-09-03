import enum
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal 
import tqdm 

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
  if schedule_name == "linear":
   scale = 1000 / num_diffusion_timesteps
   beta_start = scale * 0.0001
   beta_end = scale * 0.02
   return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype = np.float64)
  
  elif schedule_name == "cosine":
    return betas_for_alpha_bar(
      num_diffusion_timesteps,
      lambda t : math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
  elif schedule_name == "sqrt": 
    return betas_for_alpha_bar(
      num_diffusion_timesteps, 
      lambda t : 1. - math.sqrt(t + 0.0001)
    )
  elif schedule_name == "sigmoid":
    start, end, tau = -3.0, 3.0, 0.7  
    v_start = torch.tensor(start / tau).sigmoid() 
    v_end = torch.tensor(end / tau).sigmoid()
    return betas_for_alpha_bar(
      num_diffusion_timesteps, 
      lambda t : (v_end - ((t * (end - start) + start) / tau).sigmoid()) / (v_end - v_start)
    )
  else:
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta = 0.999):
  betas = []
  for i in range(num_diffusion_timesteps):
    t1 = i / num_diffusion_timesteps
    t2 = (i + 1) / num_diffusion_timesteps
    betas.append(1 - alpha_bar(t2) / alpha_bar(t1))
  return np.array(betas).clip(0, max_beta)

def extract_into_tensor(arr, timesteps, broadcast_shape):
  """
  Extract values from a 1-D numpy array for a batch of indices.

  :param arr             : the 1-D numpy array
  :param timesteps       : a tensor of indices into array to extract
  :param broadcast_shape : a large shape of K dimensions with the batch dimension
                             equal to the length of timesteps
  :return                : a tensor of shape [batch_size, 1, ...] where the shape has K dims
  """
  result = torch.from_numpy(arr).to(device = timesteps.device)[timesteps].float()
  while len(result.shape) < len(broadcast_shape):
    result = result[..., None]
  return result.expand(broadcast_shape)


# Convention:
# beta_{t}     = 1 - alpha_{t} = 1 - (alpha_bar_{t} ^ 2) / (alpha_bar_{t - 1} ^ 2)
# x_{t}        = N(x_{t} | sqrt(1 - beta_{t}) * x_{t-1}, beta_{t} * I)


## DDIM
class GaussianDiffusionDDIM(object):
  def __init__(
    self, 
    noise_schedule, 
    num_diffusion_steps, 
    num_sampling_steps, 
    rescale_timesteps = False,
    cfg_scale = 0.7, 
    enable_complete = True, 
    deterministic = False, 
  ):
    self.rescale_timesteps = rescale_timesteps
    self.betas = get_named_beta_schedule(noise_schedule, num_diffusion_steps)
    self.betas = np.array(self.betas, dtype = np.float64)
    self.enable_complete = enable_complete
    self.cfg_scale = cfg_scale
    self.deterministic = deterministic
    assert len(self.betas.shape) == 1, "betas must be 1-D"
    assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas must be in (0, 1]"
    print(f"Number of sampling steps: {num_sampling_steps}")

    self.num_timesteps = int(self.betas.shape[0])
    self.num_sampling_steps = num_sampling_steps
    self.sampling_indices = np.arange(0, self.num_timesteps, self.num_timesteps // self.num_sampling_steps) 

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis = 0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

    assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

    # q(x_t | x_0) = N(x_t | x_0 * sqrt(alpha_bar_t), (1 - alpha_bar_t) * I)
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.log(1.0  - self.alphas_cumprod)

    # For reconstructing x_0 from x_t and eps
    self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

    # Posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    # Clamp first step (variance is 0 at t = 0)
    self.posterior_log_variance_clipped = np.log(np.append(\
       self.posterior_variance[1], self.posterior_variance[1:]))

    self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # x_0
    self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod) # x_t 


  def q_mean_variance(self, x_start, timesteps):
    """
    Get the distribution q(x_t | x_0) = N(x_t | x_0 * sqrt(alpha_bar_t), (1 - alpha_bar_t) * I).

    :param x_start   : [N x C x ...] tensor of completely denoised inputs
    :param timesteps : 1-D tensor of diffusion timestep indices
    :return          : (mean, variance) each of x_start.shape
    """
    mean = extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
    var = extract_into_tensor(1.0 - self.alphas_cumprod, timesteps, x_start.shape)
    return mean, var

  def q_sample(self, x_start, timesteps, gaussian_noise):
    """
    Sample from q(x_t | x_0) using the reparameterization trick:
        x_t = x_0 * sqrt(alpha_bar_t) + sqrt(1 - alpha_bar_t) * eps

    :param x_start        : completely denoised input
    :param timesteps      : 1-D tensor of diffusion timestep indices
    :param gaussian_noise : optional pre-sampled noise of the same shape as x_start
    :return               : noisy x_t
    """
    assert gaussian_noise.shape == x_start.shape

    weighted_x0 = extract_into_tensor(\
      self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
    weighted_noise = extract_into_tensor(\
      self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * gaussian_noise

    return weighted_x0 + weighted_noise

  def q_posterior_mean_variance(self, x_start, x_t, timesteps):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
    :param x_start   : completely denoised input x_0
    :param x_t       : noisy input at timestep t
    :param timesteps : 1-D tensor of diffusion timestep indices
    :return          : (posterior_mean, posterior_variance)
    """
    assert x_start.shape == x_t.shape

    posterior_mean = extract_into_tensor(self.posterior_mean_coef1, timesteps, x_start.shape) * x_start \
        + extract_into_tensor(self.posterior_mean_coef2, timesteps, x_start.shape) * x_t
    
    posterior_variance = extract_into_tensor(self.posterior_variance, timesteps, x_start.shape)
    assert posterior_mean.shape[0] == posterior_variance.shape[0], "Batch sizes do not match"
    
    return posterior_mean, posterior_variance
  
  def predict_x0_from_eps(self, x_t, timesteps, gaussian_noise):
    """
    Reconstruct x_0 from x_t and the estimated noise eps:
      x_0 = x_t / sqrt(alpha_bar_t) - eps * sqrt(1 / alpha_bar_t - 1)

    :param x_t            : noisy input at timestep t
    :param timesteps      : 1-D tensor of diffusion timestep indices
    :param gaussian_noise : estimated (or sampled) Gaussian noise
    :return               : reconstructed x_0
    """
    assert x_t.shape == gaussian_noise.shape
    return (
      extract_into_tensor(self.sqrt_recip_alphas_cumprod,   timesteps, x_t.shape) * x_t -
      extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * gaussian_noise
    )

  def p_mean_variance(self, model, cond, actions, input_x, timesteps):
    """
    Apply the model to get the estimated denoising distribution parameters
    p(x_{\pi({t-1})} | x_{\pi_t}), as well as a prediction of x_0.

    :param model             : denoising model — takes (x_t, t, cond)
    :param cond : conditioning signal
    :param input_x           : noisy input tensor at timestep t
    :param timesteps         : 1-D tensor of timestep indices, shape (B,)
    :return                  : dict with keys 'mean', 'variance', 'log_variance', 'pred_x0'
    """
    B, C = input_x.shape[:2]
    assert timesteps.shape == (B,)
    model_out = model(input_x, timesteps, cond, actions) # set default to predict to white noise
    if self.cfg_scale > 1.0 : 
      uncond_model_out = model(input_x, timesteps)
      model_out = uncond_model_out + self.cfg_scale * (model_out - uncond_model_out)

    pred_x0 = self.predict_x0_from_eps(
     x_t = input_x,
     timesteps = timesteps,
     gaussian_noise = model_out,
    )
    return {"pred_x0" : pred_x0, "pred_noise" : model_out}
  

  def ddim_sample(self, model, input_x, curr_t, next_t, cond, actions, eta = 1.):
    """
    One DDIM denoising step.
    The posterior is:
        q_sigma(x_{\pi_{t-1}} | x_{\pi_{t}}, x_0) = N(x_{\pi_{t-1}} | sqrt(alpha_{t-1}) * x_0 + sqrt(1 - alpha_{t-1}) * eps, 0)
    Variance is set to 0 (fully implicit / deterministic).

    :param model : denoising model
    :param input_x : the output from the last backward diffusion steps 
    :param curr_t : 1-D tensor of timestep of the current index (i.e. x_{\eta_{i}})
    :param next_t : 1-D tensor of timestep of the next index (i.e. x_{\eta_{i - 1}})
    :param cond : conditioning signal
    :return  : dict with keys 'sample' (x_\eta_{{t-1}}) and 'pred_x0'
    """
    out = self.p_mean_variance(
      model = model,
      cond = cond,
      actions = actions, 
      input_x = input_x,
      timesteps = curr_t,
    )
    alpha_bar_prev = extract_into_tensor(self.alphas_cumprod, next_t, input_x.shape)
    alpha_bar_curr = extract_into_tensor(self.alphas_cumprod, curr_t, input_x.shape)
    eps = torch.randn_like(input_x).to(input_x.device)
    if self.deterministic :
      mean_pred = out['pred_x0'] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * out['pred_noise'] 
    else: 
      std_dev = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_curr)) * torch.sqrt(1. - alpha_bar_curr / alpha_bar_prev)
      nonzero_mask = (next_t != 0).float().view(-1, *([1] * (len(input_x.shape) - 1)))
      mean_pred = out['pred_x0'] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - std_dev ** 2) \
        * out['pred_noise'] + nonzero_mask * std_dev * eps
    return {'sample' : mean_pred}

  def inpainting_complete(
    self, 
    input_x, 
    curr_t, 
    trajectory_dct_mod,
    masks, 
    dct_num_channels, 
    dct, 
    idct
  ):
    """
    Blend the known (observed) DCT frames back into the denoised sample.
    :param input_x : current denoised DCT sample (unknown region)
    :param prev_t : current timestep tensor for forward noising the guide
    :param cond : ground-truth DCT of observed frames
    :param noise : optional pre-sampled noise; sampled internally if None
    :param masks : binary mask — 1 for observed frames, 0 for unobserved
    :param dct_num_channels  : number of DCT frequency channels used
    :param dct : DCT basis matrix
    :param idct : inverse DCT basis matrix
    :return : blended DCT tensor
    """
    noise = torch.randn_like(trajectory_dct_mod).to(trajectory_dct_mod.device)

    # Forward-noise the observed frames to the current timestep
    dct_x_curr_t_known = self.q_sample(
      x_start = trajectory_dct_mod,
      timesteps = curr_t,
      gaussian_noise = noise,
    )

    x_curr_t_known = torch.matmul(idct[:, :dct_num_channels], dct_x_curr_t_known)
    x_curr_t_unknown = torch.matmul(idct[:, :dct_num_channels], input_x)

    x_t = x_curr_t_known * masks + x_curr_t_unknown * (1 - masks)
    dct_x_t = torch.matmul(dct[:dct_num_channels], x_t)
    return dct_x_t

  def ddim_sample_loop_progressive(
    self, 
    model, 
    trajectory_dct, 
    trajectory_dct_mod, 
    actions, 
    noise = None, 
    device = None, 
    dct_num_channels = None, 
    masks = None, 
    dct = None, 
    idct = None
  ):
    """
    Run full DDIM reverse diffusion, yielding the DCT sample at every timestep.
    :param model : denoising model
    :param cond : observed DCT frames used for inpainting
    :param noise : optional starting noise; sampled internally if None
    :param device : torch device; inferred from model if None
    :param dct_num_channels : number of DCT frequency channels
    :param dct : DCT basis matrix
    :param idct : inverse DCT basis matrix
    :yields : 
      DCT tensor at each denoising step
    """
    if device is None:
      device = next(model.parameters()).device


    inp = noise if noise is not None else torch.randn_like(trajectory_dct, device = device)
    successive_indices = list(zip(self.sampling_indices[1:], self.sampling_indices[:-1])) 
  

    for curr_t, next_t in reversed(successive_indices):
      curr_t = torch.tensor([curr_t] * trajectory_dct.shape[0], device = device)
      next_t = torch.tensor([next_t] * trajectory_dct.shape[0], device = device)
      with torch.no_grad():
        out = self.ddim_sample(
          model = model,
          input_x = inp,
          curr_t = curr_t, 
          next_t = next_t,  
          cond = trajectory_dct_mod,
          actions = actions, 
        )
        dct_sample = out['sample']
        if self.enable_complete : 
          dct_sample = self.inpainting_complete(
            input_x = dct_sample,
            trajectory_dct_mod = trajectory_dct_mod, 
            curr_t = next_t,
            masks = masks, 
            dct_num_channels = dct_num_channels,
            dct = dct,
            idct = idct,
          )

        yield dct_sample
        inp = dct_sample  

  def ddim_sample_loop(
    self, 
    model, 
    trajectory_dct, 
    trajectory_dct_mod, 
    actions, 
    device = None,
    masks = None,
    dct_num_channels = None, 
    dct = None, 
    idct = None, 
  ):
    """
    Run full DDIM reverse diffusion and return the final sample and all intermediates.

    :return : (final_dct_sample, list_of_intermediate_dct_samples)
    """
    final = None

    for sample in self.ddim_sample_loop_progressive(
      model = model, 
      trajectory_dct = trajectory_dct, 
      trajectory_dct_mod = trajectory_dct_mod, 
      device = device, 
      actions = actions, 
      dct = dct, 
      idct = idct, 
      dct_num_channels = dct_num_channels, 
      masks = masks, 
    ):
      final = sample
    return final





