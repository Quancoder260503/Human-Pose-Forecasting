"""
This code is reimplemented from, 
https://github.com/BarqueroGerman/BeLFusion/blob/f83cb086a574d42ff2a39b175fd669161002f83c/models/diffusion/gaussian_diffusion.py#L3,  
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

for educational purposes, some of the comments and derivation are markdown below 
"""

import enum 
import math 
from cv2 import CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION 

import numpy as np 
import torch 
import torch.nn as nn 
from model.DiffusionUtils import normal_kl, discretized_gaussian_log_likelihood

def mean_flat(tensor): 
    """
    Take the mean over all non-batch dimensions
    """
    return tensor.mean(dim = list(range(1, len(tensor.shape))))


# alpha_bar schedulers controlling the encoding phase 
SCHEDULERS = { 
    'constant'         : lambda timesteps, num_timesteps : torch.ones(timesteps.shape[0], device = timesteps.device), 
    'linear'           : lambda timesteps, num_timesteps : 1 - timesteps / num_timesteps, 
    'linear_inverse'   : lambda timesteps, num_timesteps : timesteps / num_timesteps, 
    'sqrt'             : lambda timesteps, num_timesteps : 1 - torch.sqrt(timesteps / num_timesteps), 
    'log'              : lambda timesteps, num_timesteps : 1 - torch.log(timesteps + 1) / torch.log(num_timesteps + 1), 
    'cosine'           : lambda timesteps, num_timesteps : torch.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2,
    'cosine_inverse'   : lambda timesteps, num_timesteps : 1 - torch.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2, 
    'step'             : lambda timesteps, num_timesteps : (timesteps <= 10).type(torch.float) - (timesteps >= 90).type(torch.float), 
    'sqrt1e-4_inverse' : lambda timesteps, num_timesteps : 1 - torch.sqrt((1 - timesteps / num_timesteps) + 1e-4),  
}

def betas_for_alpha_bar(num_diffusion_steps, alpha_bar, max_beta = 0.999):
    """
    Create a beta schedules that discritizes the given alpha_t_bar function, 
    which defines the cummulative produt of (1 - beta_t) over time from t = [0, 1]
    :param num_diffusion_steps : the number of betas to produce 
    :param alpha_bar : a lambda that takes an argument t from 0 to 1 and produces the cummulative product of (1 - beta) 
                      up to that part of the diffusion process
    :param max_beta : the maximum beta, use values lower than 1 to prevent singularities
    """
    betas = []
    for curr_step in range(num_diffusion_steps): 
        t1 = curr_step       / num_diffusion_steps 
        t2 = (curr_step + 1) / num_diffusion_steps
        betas.append(min(1 - alpha_bar(t2) / max(0.00001, alpha_bar(t1)), max_beta))
    return betas

def get_beta_scheduler(schedule_name, num_diffusion_timesteps): 
    """
    Get a pre-defined beta schedule given the scheduler name 
    """

    if schedule_name == "linear": 
        scale = 1000 / num_diffusion_timesteps 
        alpha_start = scale * 0.0001 
        alpha_end   = scale * 0.02 
        return np.linspace(alpha_start, alpha_end, num_diffusion_timesteps, dtype = np.float64) 
    elif schedule_name == 'linear_v2':
        linear_start, linear_end = 1e-4, 2e-2 
        return (np.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype = np.float64) ** 2) 
    elif schedule_name == 'linear_v3': 
        linear_start, linear_end = 0.0015, 0.015 
        return (np.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype = np.float64) ** 2)
    elif schedule_name == 'linear_v4':
        linear_start, linear_end = 1e-4, 2e-3
        return (np.linspace(linear_start ** 0.5, linear_end ** 0.5, num_diffusion_timesteps, dtype = np.float64) ** 2)
    elif schedule_name == 'cosine': 
        return betas_for_alpha_bar(
            num_diffusion_timesteps, 
            lambda t : math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, 
        )
    elif schedule_name == 'sqrt': 
        return betas_for_alpha_bar(
            num_diffusion_timesteps, 
            lambda t : max(0, 1 - np.sqrt(t + 1e-4)), 
        )
    else : 
        raise NotImplementedError(f'unknown beta scheduler : {schedule_name}')


class ModelMeanType(enum.Enum):         
    PREVIOUS_X = enum.auto() # the model predicts x_{t - 1} 
    START_X    = enum.auto() # the model predicts x_0 (completely denoised version)
    EPSILON    = enum.auto() # the model predicts standard Gaussian noise 

mean_type_dict = { 
    'previous_x' : ModelMeanType.PREVIOUS_X, 
    'start_x'    : ModelMeanType.START_X, 
    'epsilon'    : ModelMeanType.EPSILON, 
}

# In DPDM, the author have the variance as a fixed constant, however some works have 
# shown that setting up learnable parameters for the variance improve the results. 

class ModelVarType(enum.Enum):
    "What is used as the model's output variance."
    LEARNED       = enum.auto() 
    FIXED_SMALL   = enum.auto() 
    FIXED_LARGE   = enum.auto()
    LEARNED_RANGE = enum.auto()


var_type_dict = {
    'learned'       : ModelVarType.LEARNED, 
    'fixed_small'   : ModelVarType.FIXED_SMALL, 
    'fixed_large'   : ModelVarType.FIXED_LARGE, 
    'learned_range' : ModelVarType.LEARNED_RANGE, 
}

class LossType(enum.Enum):
    MSE         = enum.auto()
    RESCALE_MSE = enum.auto()
    KL          = enum.auto() 
    RESCALE_KL  = enum.auto()

    def is_variational_bound(self):
        return self == LossType.KL or self == LossType.RESCALE_KL

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices 
    :param arr : the 1-D numpy array 
    :param timesteps : a tensor of indices into array to extract 
    :param broadcast_shape : a large shape of K dimensions with the batch dimension equal to the length of timesteps 
    
    :return a tensor of shape [batch_size, 1, ...] where the shape has K dims
    """
    result = torch.from_numpy(arr).to(device = timesteps.device)[timesteps].float()
    # keep expanding dimension
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None] 
    return result.expand(broadcast_shape)

# Convention : 
# beta_{t} = 1 - alpha_{t} = 1 - (alpha_bar_{t} ^ 2) / (alpha_bar_{t - 1} ^ 2) 
# x_{t} = \mathcal{N}(x_{t} | \sqrt{1 - \beta_{t}}x_{t - 1}, \beta_{t} \mathbf{I})
# 
 
class GaussianDiffusion(object):
    """
    Utils for training and sampling from diffusion models 
    :params betas : 1-D numpy array of betas for each diffusion timestep, starting at T and going up to 1, (0 is the complete denoise version) 
    :param model_mean_type  : a ModelMeanType determining what the model outputs 
    :param model_var_type   : a ModelVarType  determining how variance is output 
    :param loss_type        : loss_function
    :param rescale_timestep : if True, pass the floating point timesteps into the model so that 
    they are always scaled like in the papers (0 up to 1000 steps)
    """

    def __init__(
            self,
            *, 
            noise_schedule, 
            num_steps, 
            model_mean_type   = 'start_x', 
            model_var_type    = 'fixed_large', 
            loss_type         = LossType.KL, 
            loss_multipliers  = 1., 
            rescale_timesteps = False, 
            noise_stdev       = 1., 
            **kwargs, 
        ):
        
        assert model_mean_type in mean_type_dict.keys(), f"predict = {model_mean_type} not supported" 
        self.model_mean_type = mean_type_dict[model_mean_type] 
        assert model_var_type  in var_type_dict.keys(),  f"var_type = {model_var_type} not supported"
        self.model_var_type  = var_type_dict[model_var_type]

        self.loss_type         = loss_type
        self.loss_multipliers  = loss_multipliers 
        self.rescale_timesteps = rescale_timesteps
        self.noise_stdev       = noise_stdev # Gaussian noise standard deviation 


        self.betas = get_beta_scheduler(noise_schedule, num_steps) 
        self.betas = np.array(self.betas, dtype = np.float64)

        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas must be in the range (0, 1]"

        self.num_timesteps = int(self.betas.shape[0]) # number of perturbation steps 

        self.alphas              = 1.0 - self.betas                          # \alpha_{i} = 1 - \beta_{i}        
        self.alphas_cumprod      = np.cumprod(self.alphas, axis = 0)         # \bar{\alpha}_i = {\alpha_1}{\alpha_2}...{\alpha_{i}} for i = 1, 2, ..., N  
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # 1.0, \bar{\alpha_1}, \bar{\alpha_2}, ..., \bar{\alpha_{N - 1}},  # shift left by one
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)   # \bar{\alpha_2}, \bar{\alpha_3},      ..., \bar{\alpha_{N}}, 0.0, # shift right by one   
        
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps)            

        # distribution params for the Gaussian parameters trick 
        # q(x_{t} | x_{0}) = \mathcal{N}(x_{t} | x_{0}\sqrt{\bar{\alpha}_i}, (1 - \bar{\alpha}_i) \mathbf{I})
        self.sqrt_alphas_cumprod           = np.sqrt(self.alphas_cumprod)     
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # \log{1 - \bar{\alpha}_i}
        self.log_one_minus_alphas_cumprod  = np.log(1.0  - self.alphas_cumprod)

        # To construct eps from x_{0} and vice-versa 
        # \sqrt{\frac{1}{{\bar{\alpha}_i}}
        self.sqrt_recip_alphas_cumprod     = np.sqrt(1.0 / self.alphas_cumprod)
        # \sqrt{\frac{1}{\bar{\alpha}_i} - 1.0}
        self.sqrt_recipm1_alphas_cumprod   = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # calculate the distribution params for the ground truth Gaussian posterior q(x_{t - 1} | x_{0}, x_{t})
        # q(x_{t - 1} | x_{0}, x_{t}) = \mathcal{N}(x_{t - 1} | x_0 * \frac{\beta_{t} \sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_{t}} +
        #                                                       x_t * \frac{\sqrt{1 - \beta_{t}} (1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}, 
        #                                                       \frac{(1 - \bar{\alpha}_{t - 1})\beta_{t}}{1 - \bar{\alpha}_t} 
        
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # variance of the first step is 0, hence skip 
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])) 
       
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev)          / (1.0 - self.alphas_cumprod) # * x_0
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod) # * x_t 

    
    def process_model_output_before_loss(self, model, out, model_kwargs): 
        return out 

    
    def q_mean_variance(self, x_start, timesteps):
        """
        get the distribution q(x_t | x_0) = \mathcal{N}(x_{t} | x_{0}\sqrt{\bar{\alpha}_i}, (1 - \bar{\alpha}_i) \mathbf{I})
        :param x_start       : the [N x C x ...] tensor of completely denoise inputs 
        :param timesteps : the number of diffusion steps (minus 1). Here, 0 means one step 
        :return : A tuple (mean, variance, log_variance) all of x_0 shape
        """

        mean    = extract_into_tensor(self.sqrt_alphas_cumprod,          timesteps, x_start.shape) * x_start
        var     = extract_into_tensor(1.0 - self.alphas_cumprod,         timesteps, x_start.shape) 
        log_var = extract_into_tensor(self.log_one_minus_alphas_cumprod, timesteps, x_start.shape)

        return mean, var, log_var
    
    def q_sample(self, x_start, timesteps, gaussian_noise = None):
        """
        Diffuse the data for a given number of diffusion steps, that 
        is sample from q_(x_t | x_0) using the param tricks 
        q(x_t | x_0) = \mathcal{N}(x_{t} | x_{0}\sqrt{\bar{\alpha}_i}, (1 - \bar{\alpha}_i) \mathbf{I})
        
        => x_{t} = x_{0} \sqrt{\bar{\alpha}_i} + \sqrt{1 - \bar{\alpha}_i} * gaussian_noise
        where gaussian_noise is sampled from standard Gaussian noise.  

        :param x_start        : the initial data input (completely denoised version)
        :param timesteps      : the number of diffusion steps (minus 1). Here, 0 means one step 
        :param gaussian_noise : if specified, the split-out normal noise 
        :return a noisier version of x_0
        """
        if gaussian_noise is None : 
            gaussian_noise = torch.randn_like(x_start) * self.noise_stdev 
        
        assert gaussian_noise.shape == x_start.shape 

        weighted_x_0    = extract_into_tensor(self.sqrt_alphas_cumprod,           timesteps, x_start.shape) * x_start
        weighted_noise  = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * gaussian_noise 

        return weighted_x_0 + weighted_noise
    
    def q_posterior_mean_variance(self, x_start, x_t, timesteps):
        """ 
        Compute the mean and variance of the diffusion posterior 
        q(x_{t - 1} | x_{t}, x_0) = \mathcal{N}(x_{t - 1} | x_0 * \frac{\beta_{t} \sqrt{\bar{\alpha}_{t - 1}}}{1 - \bar{\alpha}_{t}} +
                                                            x_t * \frac{\sqrt{1 - \beta_{t}} (1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}, 
                                                            \frac{(1 - \bar{\alpha}_{t - 1})\beta_{t}}{1 - \bar{\alpha}_t})

        :param x_0 : the initial data input (completely denoised version)
        :param x_t : the noisy version of initial input at time step t 
        :param timesteps : the number of diffusion steps (minus 1). Here, 0 means one step 
        :return : 
           the distributions parameters for the posterior q(x_{t - 1} | x_{t}, x_0) whose formula stated above, 
           return a dict with the following keys: 
            `posterior_mean`, 
            `posterior_variance` 
            `posterior_log_variance` 
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, timesteps, x_start.shape) * x_start  + 
            extract_into_tensor(self.posterior_mean_coef2, timesteps, x_start.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, timesteps, x_start.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, timesteps, x_start.shape)

        assert (
            posterior_mean.shape[0] 
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def predict_x0_from_eps(self, x_t, timesteps, gaussian_noise):
        """
        Given the perturbed input at time step t, i.e. x_{t} and the estimated Gaussian noise, 
        construct x_0, recall that x_t can be written as 
           x_{t} = x_{0} \sqrt{\bar{\alpha}_i} + \sqrt{1 - \bar{\alpha}_i} * gaussian_noise 
        => x_{0} = \frac{x_{t} - \sqrt{1 - \bar{\alpha}_i} * gaussian_noise}{\sqrt{\bar{\alpha}_i}} 
                 = x_{t} * \frac{1}{\sqrt{\bar{\alpha}_i}} - gaussian_noise * \sqrt{\frac{1}{\bar{\alpha_i}} - 1}
        :param x_t            : the noisy version of initial input at time step t 
        :param timesteps      : the number of diffusion steps (minus 1). Here, 0 means one step 
        :param gaussian_noise : the sampled or estimated using neural network gaussian_noised
        :return  : 
            reconstruction of initial input x_start or (x_0)
        """
        assert x_t.shape == gaussian_noise.shape 
        return ( 
            extract_into_tensor(self.sqrt_recip_alphas_cumprod,     timesteps, x_t.shape) * x_t - 
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)   * gaussian_noise
        )
    
    def predict_x0_from_xprev(self, x_t, timesteps, x_prev):
        """
        Given the perturbed input at time step t, i.e. x_{t} and the immediate previously denoised version, i.e. $x_{t + 1}$, 
        construct x_0, recall that by inferring from the posterior Guassian q,
      
        q(x_{t} | x_{0}, x_{t + 1}) = \mathcal{N}(x_{t} | x_0       * \frac{\beta_{t + 1} \sqrt{\bar{\alpha}_{t}}}{1 - \bar{\alpha}_{t + 1}} +
                                                          x_{t + 1} * \frac{\sqrt{1 - \beta_{t + 1}} (1 - \bar{\alpha}_{t})}{1 - \bar{\alpha}_{t + 1}}, 
                                                          \frac{(1 - \bar{\alpha}_{t})\beta_{t + 1}}{1 - \bar{\alpha}_{t + 1}}  
          
        for notation simplicity, we denotes the term coef_1 which is the weighted associated with $x_0$ and 
        the term coef2 would be the weight for $x_1$, then x_{t} can be written as
            x_{t} = x_{t + 1} * coef_2 + x{0} * coef_1    # ignore the noise term  
        =>  x_{0} = x_{t} / coef_ 1 - x_{t + 1} * coef_2 / coef_1
        

        :param x_t            : the noisy version of initial input at time step t 
        :param timesteps      : the number of diffusion steps (minus 1). Here, 0 means one step 
        :param x_prev         : the noisy version of initial input at time step {t + 1}
        :return : 
            reconstruction of initial input x_start (or x_0)
        """

        assert x_t.shape == x_prev.shape
        return (
            extract_into_tensor(1. / self.posterior_mean_coef1, timesteps, x_t.shape) * x_t -  
            extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, timesteps, x_t.shape) * x_prev
        )
    
    
    def predict_eps_from_x0(self, x_t, timesteps, pred_x_start):
        """
        Given the perturbed input at time step t, i.e. x_{t} and the approximated x_start, 
        compute the noise term, recall that by reparameterization trick, 
        
           x_{t} = x_{0} \sqrt{\bar{\alpha}_i} + \sqrt{1 - \bar{\alpha}_i} * gaussian_noise 
        => gaussian noise = \frac{(x_{t} - x_{0} \sqrt{\bar{\alpha}_i})}{\sqrt{1 - \bar{\alpha}_i}} 
        => gaussian noise = x_{t} * \frac{\sqrt{\frac{1}{\bar{\alpha}_i}}}{\sqrt{\frac{1}{\bar{\alpha}_i} - 1}} - x_{0} \frac{1}{\sqrt{\frac{1}{\bar{\alpha}_i} - 1}}

        :param x_t            : the noisy version of initial input at time step t 
        :param timesteps      : the number of diffusion steps (minus 1). Here, 0 means one step 
        :param pred_x_start   : approximation of initial input 
        :return : 
            reconstruction of gaussian noise 
        """

        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t - pred_x_start
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
    

    def scale_timesteps(self, timesteps): 
        """
        Scale the timesteps if required 
        """
        if self.rescale_timesteps:
            return timesteps.float() * (1000.0 / self.num_timesteps)
        return timesteps

    def p_mean_variance(
        self, 
        model, 
        input_x, 
        timesteps, 
        clip_denoised = True, 
        denoised_fn   = None, 
        model_kwargs  = None, 
    ):
        """
        Apply the model to get the estimated denoising distribution parameters 
        p(x_{t - 1} | x_{t}), as well as a prediction of the initial x_0
       
        :param model         : the model, which takes a signal and a batch of time steps as input.
        :param input_x       : input tensor
        :param timesteps     : a 1-D tensor of timesteps.
        :param clip_denoised : if True, clip the denoised signals into [-1, 1]
        :param denoised_fn   : if not None, a function which applies to the x_0 prediction before 
        it is used to sample. Applies before clip_denoised 
        :param model_kwargs  : if not None, a dict of extra keyword arguments to pass to the models, 
        this can be used for conditioning. 
        :return a dict of the following keys : 
           - 'mean'        : the model mean output 
           - 'variance     : the model variance output 
           - 'log_variance : the log of 'variance' 
           - 'pred_x0'     : the prediction for x_0 
        """

        if model_kwargs is None : 
            model_kwargs = {} 
        
        B, C = input_x.shape[:2]
        assert timesteps.shape == (B, )
        model_out = model(input_x, self.scale_timesteps(timesteps), **model_kwargs) 

        # in the case that we do not keep the variance fixed
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_out.shape == (B, C * 2, *input_x.shape[2:])
            model_output, model_var_values = torch.split(model_out, C, dim = 1) 
            
            if self.model_var_type == ModelVarType.LEARNED:
                # Easier to learn the log variance
                model_log_variance = model_var_values  
                model_variance = torch.exp(model_log_variance) 
            else : 
                # Bounded log variance 
                min_log = extract_into_tensor(self.posterior_log_variance_clipped, timesteps, input_x.shape)
                max_log = extract_into_tensor(np.log(self.betas), timesteps, input_x.shape)                   
                # The model_var_values is [-1, 1] for [min_var, max_var]. 
                frac    = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log # Learning the fraction  
                model_variance = torch.exp(model_log_variance)
            
        else :
            model_variance, model_log_variance = { 
                ModelVarType.FIXED_LARGE : (
                  np.append(self.posterior_variance[1], self.betas[1:]), 
                  np.log(np.append(self.posterior_variance[1], self.betas[1:])), 
                ), 
                ModelVarType.FIXED_SMALL : ( 
                    self.posterior_variance, 
                    self.posterior_log_variance_clipped, 
                )
            }[self.model_var_type]

            model_variance     = extract_into_tensor(model_variance,     timesteps, input_x.shape) 
            model_log_variance = extract_into_tensor(model_log_variance, timesteps, input_x.shape)

        # post-processing 
        def process_x0(x): 
            if denoised_fn is not None : 
               x = denoised_fn(x)
            if clip_denoised : 
                return x.clamp(-1, 1)
            return x 
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X : 
            pred_x0 = process_x0(
                self.predict_x0_from_xprev(
                    x_t        = input_x, 
                    timesteps  = timesteps, 
                    x_prev     = model_out
                )
            )
            model_mean = model_out 

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X : 
                pred_x0 = process_x0(model_out)
            else : 
                pred_x0 = process_x0(
                    self.predict_x0_from_eps(
                        x_t            = input_x, 
                        timesteps      = timesteps, 
                        gaussian_noise = model_out,
                    )
                ) 
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start   = pred_x0, 
                x_t       = input_x, 
                timesteps = timesteps, 
            )
        else : 
            raise NotImplementedError(self.model_mean_type)
        
        assert (
            model_mean.shape == model_log_variance.shape == pred_x0.shape == input_x.shape 
        )
        return { 
            "mean"         : model_mean, 
            "variance"     : model_variance, 
            "log_variance" : model_log_variance,
            "pred_x0"      : pred_x0,  
        }
    



    def condition_mean(self, cond_fn, p_mean_var, x, time_index, model_kwargs = None):
        """
        Compute the mean for the previous step, given a cond_fn that computes the gradient 
        of a conditional log probability with respect to x. In particular, compute the score 
        value grad(log(p(y | x))) and we want to condition on y.
        """

        gradient = cond_fn(x, self.scale_timesteps(time_index), **model_kwargs)
        new_mean = (
            p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, p_mean_var, x, time_index, model_kwargs = None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """

        alpha_bar = extract_into_tensor(self.alphas_cumprod, time_index, x.shape)

        eps = self.predict_eps_from_x0(x, time_index, p_mean_var['pred_x0'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self.scale_timesteps(time_index), **model_kwargs)

        out = p_mean_var.copy()
        out['pred_x0'] = self.predict_x0_from_eps(x_t = x, time_index = time_index, eps = eps) 
        out['mean'], _, _ = self.q_posterior_mean_variance(
            x_0 = out['pred_x0'], x_t = x, time_index = time_index
        )
        return out 
    

    
    def p_sample(
        self, 
        model, 
        input_x,  
        timesteps, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
    ):
        """
        Sample x_{t - 1} from the model at the given timestep 
        :param model         : the model to sample from 
        :param input_x       : the current tensor at x_{t}. 
        :param clip_denoised : if True, clip the x_0 prediction to [-1, 1]
        :param denoised_fn   : if not None, a function which applies to x_0 prediction 
        before it is used to sample 
        :param cond_fn       : if not None, this is a gradient function that acts similarly to a model 
        :param model_kwargs  : if not None, a dict of extra keywords arguments to pass to the model. This can be used for conditioning. 
        :return : a dict containing the following keys : 
           - 'sample'  : a random sample from the model 
           - 'pred_x0' : a prediction of x_0 
        """

        out = self.p_mean_variance(
            model         = model, 
            input_x       = input_x, 
            timesteps     = timesteps, 
            clip_denoised = clip_denoised, 
            denoised_fn   = denoised_fn, 
            model_kwargs  = model_kwargs, 
        )

        noise = torch.randn_like(input_x) * self.noise_stdev
        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (len(input_x.shape) - 1))) # no noise when t = 0

        if cond_fn is not None : 
            out['mean'] = self.condition_mean(cond_fn, out, input_x, timesteps, model_kwargs)
        
        # The predicted mean of the posterior distribution q(x_{t - 1} | x_{t}, x_{0}), use this to sample $x_{t - 1}$. 
        noise_to_add = nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise   
        sample = out['mean'] + noise_to_add
        return { 
            'sample'       : sample, 
            'pred_x0'      : out['pred_x0'], 
            'noise_to_add' : noise_to_add, 
        }
    
    def p_sample_loop_progressive(
        self, 
        model, 
        shape, 
        noise         = None, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        device        = None,       
    ):
        """
        Generate samples from the model and yield intermediate samples from 
        each time step

        :param model         : the model to sample from 
        :param shape         : the shape of the samples, (N, C, H, W) for example
        :param input_x       : the current tensor at x_{t}. 
        :param clip_denoised : if True, clip the x_0 prediction to [-1, 1]
        :param denoised_fn   : if not None, a function which applies to x_0 prediction 
        before it is used to sample 
        :param cond_fn       : if not None, this is a gradient function that acts similarly to a model 
        :param model_kwargs  : if not None, a dict of extra keywords arguments to pass to the model. This can be used for conditioning. 
        :param device        : the device on which the model is used. 
        :return : 
           Progressively yield intermediate samples from each denoising step until reaching the fully denoised result.
        """
        if device is None : 
            device = next(model.parameters()).device 
        assert isinstance(shape, (tuple, list))
        
        if noise is not None : 
            inp = noise 
        else : 
            inp = torch.randn(*shape, device = device) * self.noise_stdev 
        
        # Reverse the timesteps 
        indices = list(range(self.num_timesteps))[::-1]

        # Denoising from a given time steps
        for i in range(indices):
            timesteps = torch.tensor([i] * shape[0], device = device)
            with torch.no_grad():
                out = self.p_sample(
                    model         = model, 
                    input_x       = inp, 
                    timesteps     = timesteps, 
                    clip_denoised = clip_denoised, 
                    denoised_fn   = denoised_fn, 
                    cond_fn       = cond_fn, 
                    model_kwargs  = model_kwargs, 
                )
                yield out 
                inp = out['sample']

        

    def p_sample_loop(
        self, 
        model, 
        shape, 
        noise         = None, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        device        = None, 
        max_step      = -1, 
    ): 
        """
        Generate samples from model. 
        :param model         : the model module. 
        :param shape         : the shape of the samples, (N, C, H, W) for example
        :param noise         : if specified, the noise from the encoder to sample.
                              Should be of the same shape as `shape`
        :param clip_denoised : if True, clip x_0 predictions to [-1, 1] 
        :param denoised_fn   : if not None, a function which applies to the x_0 prediction 
                               before it is used to sample.
        :param cond_fn       : if not None, gradient function that acts similarly to the model 
        :param model_kwargs  : if not None, a dict of extra keyword arguments to pass to the model.
                               This can be used for conditioning.
        :param device        : The device to create samples on 
        :param max_step      : Last diffusion step wants to perform. If -1, all steps are performed
        :return : 
             a non-differentiable batch of samples   
        """
        final = None 
        for ind, sample in enumerate(self.p_sample_loop_progressive(
            model         = model, 
            shape         = shape, 
            noise         = noise, 
            clip_denoised = clip_denoised, 
            denoised_fn   = denoised_fn, 
            cond_fn       = cond_fn, 
            model_kwargs  = model_kwargs, 
            device        = device,
        )):
            if max_step != -1 and ind == max_step : 
                break
            final = sample 
        return final 
    
    
    def ddim_sample(
        self, 
        model, 
        input_x, 
        timesteps, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        eta           = 0.0, 
    ):
        """
        For Denoising Diffusion Implicit Model, the posterior distribution is modeled as, 
        note that there is some notational mismatch with this to what used in the papers, 
        hence one should understand $\alpha_{t}$ as $\bar{\alpha_{t}}$. 
        q_{\sigma}(x_{t - 1} | x_{t}, x_{0}) = 
            \mathcal{N}(x_{t - 1} | \sqrt{\alpha_{t - 1}} x_0 + \sqrt{1 - \alpha_{t - 1} - \sigma_{t} ^ 2} \frac{x_t - 
                                    \sqrt{\alpha_{t}f(x_t)}{\sqrt{1 - \alpha_{t}}}, \sigma^ 2 \mathbf{I}))

        Then we can use this posterior distribution set up to sample x_{t - 1} from the model 
        The params and returns are similar to that of ddpm samples 
        """

        out = self.p_mean_variance(
            model         = model, 
            input_x       = input_x,
            timesteps     = timesteps,  
            clip_denoised = clip_denoised, 
            denoised_fn   = denoised_fn, 
            model_kwargs  = model_kwargs, 
        )


        if cond_fn is not None : 
            out = self.condition_score(cond_fn, out, input_x, timesteps, model_kwargs = model_kwargs)
        
        eps = self.predict_eps_from_x0(input_x, self.alphas_cumprod, out['pred_x0']) # construct epsilon from predicted $x_0$

        alpha_bar      = extract_into_tensor(self.alphas_cumprod,       timesteps, input_x.shape) # \bar{\alpha_{t}} 
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev,  timesteps, input_x.shape) # \bar{\alpha_{t - 1}}

        """
        In DDIM papers, the authors parameterizes the variance term as a function 
          $\sigma_{\eta}(t) = eta * \sqrt{\frac{1 - \alpha_{t - 1}}{1 - \alpha_{t}}} \sqrt{1 - \frac{\alpha_{t}}{\alpha_{t - 1}}}$ 
        """
        sigma = (
            eta 
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev) 
        ) # the standard deviation term for q_{\sigma}(x_{t - 1} | x_{t}, x_{0})

        noise = torch.randn_like(input_x) * self.noise_stdev
        mean_pred = out['pred_x0'] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps 
        
        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (len(timesteps.shape) - 1)))

        sample = mean_pred + nonzero_mask * sigma * noise 

        return { 
            'sample' : sample, 
            'pred_x0' : out['pred_x0'], 
        }                
    
    
    def ddim_reverse_sample(
        self, 
        model, 
        input_x,
        timesteps,
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        eta           = 0.0, 
    ):
        """
        Sample x_{t + 1} from the model using DDIM reverse ODE 
        From the sampling equation of x_{t - 1}, which is of the form 
        x_{t - 1} = \sqrt{\alpha_{t - 1}} * predicted_x_{0} + \sqrt{1 - \alpha_{t - 1}} * predicted_noise, 
        when eta = 0.0 and the predicted noise is constructed from the predicted x_0, 
        from this we can shift the index by 1 to get, 
        
        x_{t} = \sqrt{\alpha_{t}} * predicted_x_{0} + \sqrt{1 - \alpha_{t}} * predicted_noise, 
        which is the same as ddpm forward diffusion step,  
        """

        assert eta == 0.0, "Reverse ODE only for deterministic path"

        out = self.p_mean_variance(
            model         = model, 
            input_x       = input_x,
            timesteps     = timesteps,           
            clip_denoised = clip_denoised, 
            denoised_fn   = denoised_fn, 
            model_kwargs  = model_kwargs,   
        )

        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, input_x.shape) * input_x - out['pred_x0']
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, input_x.shape)

        alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, timesteps, input_x.shape)

        mean_pred = (out['pred_x0'] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps)
        
        return {
            'sample'  : mean_pred, 
            'pred_x0' : out['pred_x0'], 
        }
    
    
    def ddim_sample_loop_progressive(
        self, 
        model, 
        shape, 
        noise         = None, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        device        = None, 
        eta           = 0.0,        
    ):
        """
        Use DDIM to Generate samples from the model and yield intermediate samples from 
        each time step of diffusion
        """
        if device is None : 
            device = next(model.parameters()).device 

        assert isinstance(shape, (tuple, list))
        
        if noise is not None : 
            inp = noise 
        else : 
            inp = torch.randn(*shape, device = device) * self.noise_stdev 
        
        indices = list(range(self.num_timesteps))[::-1]

        # Denoising from a given time steps
        for i in indices:
            timesteps = torch.tensor([i] * shape[0], device = device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model         = model, 
                    input_x       = inp, 
                    timesteps     = timesteps, 
                    clip_denoised = clip_denoised, 
                    denoised_fn   = denoised_fn, 
                    cond_fn       = cond_fn, 
                    model_kwargs  = model_kwargs, 
                    eta           = eta,
                )
                yield out 
                inp = out['sample']

    
    def ddim_sample_loop(
        self, 
        model, 
        shape, 
        noise         = None, 
        clip_denoised = True, 
        denoised_fn   = None, 
        cond_fn       = None, 
        model_kwargs  = None, 
        device        = None, 
        eta           = 0.0, 
    ):
        """
        Generate samples from model using DDIM
        """
        final = None 
        for sample in self.ddim_sample_loop_progressive(
            model          = model, 
            shape          = shape,
            noise          = noise, 
            clip_denoised  = clip_denoised, 
            denoised_fn    = denoised_fn, 
            cond_fn        = cond_fn, 
            model_kwargs   = model_kwargs,
            device         = device, 
            eta            = eta,  
        ):
            final = sample 
        return final 
    


    def variational_bound_terms_bits_per_dim(
        self, 
        model, 
        x_start, 
        x_t, 
        timesteps, 
        clip_denoised = True, 
        model_kwargs  = None, 
    ):
        """
        Get a term for variational lower-bound. 
        The resulting units are bits (rather than nats)
        :return : a dict with the following keys : 
          - 'output'      : a shape [B] tensor of NLLs or KLs 
          - 'pred_x0'     : the x_0 predictions

        where B is the batch size  
        """

        # Ground truth posterior params  
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start    = x_start, 
            x_t        = x_t, 
            timesteps  = timesteps, 
        )

        # Approximated posterior param using Neural Networks
        out = self.p_mean_variance(
            model         = model, 
            x_t           = x_t, 
            timesteps     = timesteps,
            clip_denoised = clip_denoised,
            model_kwargs  = model_kwargs,   
        )
        

        # KL divergence of two Gaussians
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out['mean'], out['log_variance']
        )

        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, 
            out['mean'], 
            0.5 * out['log_variance'], 
        )

        
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return D_KL(q(x_{t - 1}| x_t, x_0) || p_{\theta}(x_{t - 1}| x_t))
        output = torch.where((timesteps == 0), decoder_nll, kl)
    
        return { 
            'output'  : output, 
            'pred_x0' : out['pred_x0'], 
        }
    

    def prior_bits_per_dim(self, x_start):
        """
        Get the prior KL term for the variational lower bound, measured in bits per dim
        This term cannot be optimized, as it only depends on the encoder.

        :param x_start : the [B x C x ...] tensor of inputs 
        :return : a batch of [B] KL values (in bits), one per batch element
        """
        batch_size = x_start.shape[0]
        timesteps  = torch.tensor([self.num_timesteps - 1] * batch_size, device = x_start.device)
        q_true_mean, _, q_true_log_var = self.q_mean_variance(x_start = x_start, timesteps = timesteps)
        kl_prior = normal_kl(
            q_true_mean, q_true_log_var, 0.0, 0.0 
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    
    def calc_bpd_loop(self, model, x_start, clip_denoised = True, model_kwargs = None):
        """
        Compute the entire variational lower bound, measure in bits-per-dim  
        :param model         : the model to evaluate the loss on. 
        :param x_start       : the [B x C x ...] tensor of inputs 
        :param clip_denoised : if True, clip denoised samples.
        :param model_kwargs  : if not None, a dict of extra keyword arguments to pass to the model. This 
        can be used for conditioning. 

        :return    : a dict containing the following keys : 
          - total bpd : the total variational lower-bound, per batch element 
          - prior bpd : the prior term in the lower bound  
        """
        device     = x_start.device 
        batch_size = x_start.shape[0]

        variational_bound = [] 
        x_0_mse   = []
        noise_mse = []

        # in reverse order
        for t in list(range(self.num_timesteps))[::-1]:
            batch_timesteps = torch.tensor([t] * batch_size, device = device) 
            noise      = torch.randn_like(x_start)

            # sample x_{t} for each items in the batch using reparameterization trick 
            x_t        = self.q_sample(x_start = x_start, timesteps = batch_timesteps, noise = noise)

            with torch.no_grad():
                # compute the term $\mathbb{E}[D_KL(q(x_{t - 1}| x_t, x_0) || p_{\theta}(x_{t - 1}| x_t))], the 
                # term is evaluated empirically by Monte Carlo Approximation

                out = self.variational_bound_terms_bits_per_dim(
                    model         = model, 
                    x_start       = x_start, 
                    x_t           = x_t, 
                    timesteps     = batch_timesteps,
                    clip_denoised = clip_denoised, 
                    model_kwargs  = model_kwargs,  
                )
                variational_bound.append(out['output'])
                x_0_mse.append(mean_flat((out['pred_x0'] - x_start) ** 2))
                eps  = self.predict_eps_from_x0(x_t = x_t, timesteps = batch_timesteps, x_start = out['pred_x0'])
                noise_mse.append(mean_flat((eps - noise) ** 2))
        
        variational_bound = torch.stack(variational_bound, dim = 1) 
        x_0_mse           = torch.stack(x_0_mse,           dim = 1) 
        noise_mse         = torch.stack(noise_mse,         dim = 1)

        prior_bit_per_dim = self.prior_bits_per_dim(x_start = x_start)
        # Compute the ELBO term 
        total_bit_per_dim = variational_bound.sum(dim = 1) + prior_bit_per_dim
    
        return { 
            'total_bpd'         : total_bit_per_dim, 
            'prior_bpd'         : prior_bit_per_dim, 
            'variational_bound' : variational_bound,
            'x_0_mse'           : x_0_mse,
            'noise_mse'         : noise_mse, 
        }
    
    def training_losses(self, model, x_start, timesteps, model_kwargs = None, noise = None):
        """
        Compute training losses for a single time step. 
        :param model        : the model to evaluate loss on
        :param x_start      : the [B x C x ...] tensor of inputs. 
        :param timesteps    : a batch of timestep indices
        :param model_kwargs : if not None, a dict of extra keyword arguments to pass to 
        the model. This can be used for conditioning. 
        :param noise : if specified, the specific Gaussian noise to try to remove 
        :return a dict with the key loss containing a tensor of shape [B]. 
        where B is the batch size. 
        """

        if model_kwargs is None: 
            model_kwargs = {}
        if noise is None : 
            noise = torch.rand_like(x_start)
        
        # sample $x_{t}$ using reparameterization trick 
        x_t = self.q_sample(x_start = x_start, time_index = timesteps, noise = noise) 

        results = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALE_KL:
            results['loss'] = self.variational_bound_terms_bits_per_dim(
                model         = model, 
                x_start       = x_start, 
                x_t           = x_t, 
                timesteps     = timesteps, 
                clip_denoised = False, 
                model_kwargs  = model_kwargs, 
            )['output']

        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALE_MSE:
            model_out = model(x_t, timesteps, **model_kwargs)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_out.shape == (B, C * 2, *x_t.shape[2:])

                model_out, model_var_values = torch.split(model_out, C, dim = 1) 

                # Learn the variance using the variational bound, but do not let it 
                # affect the mean prediction

                frozen_out = torch.cat([model_out.detach(), model_var_values], dim = 1)
                results['variational_bound'] = self.variational_bound_terms_bits_per_dim(
                    model = lambda *args, r = frozen_out : r, 
                    x_start       = x_start, 
                    x_t           = x_t, 
                    timesteps     = timesteps,
                    clip_denoised = False,  
                    model_kwargs  = model_kwargs, 
                )['output']

                if self.loss_type == LossType.RESCALE_MSE : 
                    # Divide by 1000 for equivalence with initial implementation 
                    # Without a factor of 1 / 1000, the VB terms hit the MSE terms bad
                    results['variational_bound'] *= self.num_timesteps / 1000.0
            

            target = { 
                ModelMeanType.PREVIOUS_X : self.q_posterior_mean_variance(x_start = x_start, x_t = x_t, timesteps = timesteps)[0], 
                ModelMeanType.START_X    : x_start, 
                ModelMeanType.EPSILON    : noise, 
            }[self.model_mean_type]

            results['mse'] = mean_flat((target - model_out) ** 2)

            if 'variational_bound' in results : 
                results['loss'] = results['mse'] + results['variational_bound']
            else : 
                results['loss'] = results['mse']
        else : 
            raise NotImplementedError(f'{self.loss_type} not supported')
        
        return results





        
    



        


