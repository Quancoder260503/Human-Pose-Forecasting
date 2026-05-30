"""
This code is implemented based on,
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
for educational purposes, some of the comments and derivation are markdown below
"""

import enum
import math
import numpy as np
import torch
import torch.nn as nn


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# alpha_bar schedulers controlling the encoding phase
SCHEDULERS = {
    'constant'       : lambda timesteps, num_timesteps: torch.ones(timesteps.shape[0], device=timesteps.device),
    'linear'         : lambda timesteps, num_timesteps: 1 - timesteps / num_timesteps,
    'linear_inverse' : lambda timesteps, num_timesteps: timesteps / num_timesteps,
    'sqrt'           : lambda timesteps, num_timesteps: 1 - torch.sqrt(timesteps / num_timesteps),
    'log'            : lambda timesteps, num_timesteps: 1 - torch.log(timesteps + 1) / torch.log(num_timesteps + 1),
    'cosine'         : lambda timesteps, num_timesteps: torch.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2,
    'cosine_inverse' : lambda timesteps, num_timesteps: 1 - torch.cos((timesteps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2,
    'step'           : lambda timesteps, num_timesteps: (timesteps <= 10).type(torch.float) - (timesteps >= 90).type(torch.float),
}


def betas_for_alpha_bar(num_diffusion_steps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1 - beta_t) over time from t = [0, 1].

    :param num_diffusion_steps : the number of betas to produce
    :param alpha_bar           : a lambda that takes an argument t from 0 to 1 and produces
                                 the cumulative product of (1 - beta) up to that part of the diffusion process
    :param max_beta            : the maximum beta, use values lower than 1 to prevent singularities
    """
    betas = []
    for curr_step in range(num_diffusion_steps):
        t1 = curr_step / num_diffusion_steps
        t2 = (curr_step + 1) / num_diffusion_steps
        betas.append(min(1 - alpha_bar(t2) / max(0.00001, alpha_bar(t1)), max_beta))
    return betas


def get_beta_scheduler(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule given the scheduler name.
    """
    if schedule_name == "linear":
        scale       = 1000 / num_diffusion_timesteps
        alpha_start = scale * 0.0001
        alpha_end   = scale * 0.02
        return np.linspace(alpha_start, alpha_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f'unknown beta scheduler: {schedule_name}')


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0 (completely denoised version)
    EPSILON = enum.auto()  # the model predicts standard Gaussian noise


mean_type_dict = {
    'start_x' : ModelMeanType.START_X,
    'epsilon'  : ModelMeanType.EPSILON,
}


class ModelVarType(enum.Enum):
    "What is used as the model's output variance."
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()


var_type_dict = {
    'fixed_small' : ModelVarType.FIXED_SMALL,
    'fixed_large' : ModelVarType.FIXED_LARGE,
}


class LossType(enum.Enum):
    MSE         = enum.auto()
    RESCALE_MSE = enum.auto()


loss_type_dict = {
    'mse'         : LossType.MSE,
    'rescale_mse' : LossType.RESCALE_MSE,
}


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr             : the 1-D numpy array
    :param timesteps       : a tensor of indices into array to extract
    :param broadcast_shape : a large shape of K dimensions with the batch dimension
                             equal to the length of timesteps
    :return                : a tensor of shape [batch_size, 1, ...] where the shape has K dims
    """
    result = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


# Convention:
# beta_{t}     = 1 - alpha_{t} = 1 - (alpha_bar_{t} ^ 2) / (alpha_bar_{t - 1} ^ 2)
# x_{t}        = N(x_{t} | sqrt(1 - beta_{t}) * x_{t-1}, beta_{t} * I)


class GaussianDiffusion(object):
    """
    Utils for training and sampling from diffusion models.

    :param noise_schedule    : name of the beta schedule ('linear' or 'cosine')
    :param num_steps         : number of diffusion timesteps
    :param model_mean_type   : a ModelMeanType determining what the model outputs
    :param model_var_type    : a ModelVarType determining how variance is output
    :param loss_type         : loss function
    :param loss_multipliers  : scalar multiplier on the loss
    :param rescale_timesteps : if True, pass floating point timesteps scaled to [0, 1000]
    :param noise_stdev       : standard deviation of the Gaussian noise
    """

    def __init__(
        self,
        *,
        noise_schedule,
        num_steps,
        model_mean_type  = 'start_x',
        model_var_type   = 'fixed_small',
        loss_type        = LossType.MSE,
        loss_multipliers = 1.,
        rescale_timesteps = False,
        noise_stdev      = 1.,
        **kwargs,
    ):
        
        if isinstance(model_mean_type, ModelMeanType):
            self.model_mean_type = model_mean_type
        elif isinstance(model_mean_type, str):
            model_mean_type = model_mean_type.lower()
            assert model_mean_type in mean_type_dict, f"predict = {model_mean_type} not supported"
            self.model_mean_type = mean_type_dict[model_mean_type]
        elif isinstance(model_mean_type, int):
            mean_type_keys = ['previous_x', 'start_x', 'epsilon']
            assert 0 <= model_mean_type < len(mean_type_keys), f"predict = {model_mean_type} not supported"
            self.model_mean_type = mean_type_dict[mean_type_keys[model_mean_type]]
        else:
            raise TypeError(f"Unsupported model_mean_type type: {type(model_mean_type)}")

        if isinstance(model_var_type, ModelVarType):
            self.model_var_type = model_var_type
        elif isinstance(model_var_type, str):
            model_var_type = model_var_type.lower()
            assert model_var_type in var_type_dict, f"var_type = {model_var_type} not supported"
            self.model_var_type = var_type_dict[model_var_type]
        elif isinstance(model_var_type, int):
            var_type_keys = ['fixed_small', 'fixed_large']
            assert 0 <= model_var_type < len(var_type_keys), f"var_type = {model_var_type} not supported"
            self.model_var_type = var_type_dict[var_type_keys[model_var_type]]
        else:
            raise TypeError(f"Unsupported model_var_type type: {type(model_var_type)}")


        if isinstance(loss_type, LossType):
            self.loss_type = loss_type
        elif isinstance(loss_type, str):
            loss_type = loss_type.lower()
            assert loss_type in loss_type_dict, f"loss_type = {loss_type} not supported"
            self.loss_type = loss_type_dict[loss_type]
        elif isinstance(loss_type, int):
            loss_type_keys = ['mse', 'rescale_mse']
            assert 0 <= loss_type < len(loss_type_keys), f"loss_type = {loss_type} not supported"
            self.loss_type = loss_type_dict[loss_type_keys[loss_type]]
        else:
            raise TypeError(f"Unsupported loss_type type: {type(loss_type)}")

        self.loss_multipliers  = loss_multipliers
        self.rescale_timesteps = rescale_timesteps
        self.noise_stdev       = noise_stdev

        # --- beta schedule ---
        self.betas = get_beta_scheduler(noise_schedule, num_steps)
        self.betas = np.array(self.betas, dtype = np.float64)

        assert len(self.betas.shape) == 1,                                "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all(),        "betas must be in (0, 1]"

        self.num_timesteps = int(self.betas.shape[0])

        self.alphas              = 1.0 - self.betas
        self.alphas_cumprod      = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # q(x_t | x_0) = N(x_t | x_0 * sqrt(alpha_bar_t), (1 - alpha_bar_t) * I)
        self.sqrt_alphas_cumprod           = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod  = np.log(1.0  - self.alphas_cumprod)

        # For reconstructing x_0 from x_t and eps
        self.sqrt_recip_alphas_cumprod   = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Clamp first step (variance is 0 at t = 0)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod)
        )


    def q_mean_variance(self, x_start, timesteps):
        """
        Get the distribution q(x_t | x_0) = N(x_t | x_0 * sqrt(alpha_bar_t), (1 - alpha_bar_t) * I).

        :param x_start   : [N x C x ...] tensor of completely denoised inputs
        :param timesteps : 1-D tensor of diffusion timestep indices
        :return          : (mean, variance, log_variance) each of x_start.shape
        """
        mean    = extract_into_tensor(self.sqrt_alphas_cumprod,          timesteps, x_start.shape) * x_start
        var     = extract_into_tensor(1.0 - self.alphas_cumprod,         timesteps, x_start.shape)
        log_var = extract_into_tensor(self.log_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return mean, var, log_var

    def q_sample(self, x_start, timesteps, gaussian_noise = None):
        """
        Sample from q(x_t | x_0) using the reparameterization trick:
            x_t = x_0 * sqrt(alpha_bar_t) + sqrt(1 - alpha_bar_t) * eps

        :param x_start        : completely denoised input
        :param timesteps      : 1-D tensor of diffusion timestep indices
        :param gaussian_noise : optional pre-sampled noise of the same shape as x_start
        :return               : noisy x_t
        """
        if gaussian_noise is None:
            gaussian_noise = torch.randn_like(x_start) * self.noise_stdev

        assert gaussian_noise.shape == x_start.shape

        weighted_x0    = extract_into_tensor(self.sqrt_alphas_cumprod,           timesteps, x_start.shape) * x_start
        weighted_noise = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * gaussian_noise

        return weighted_x0 + weighted_noise

    def q_posterior_mean_variance(self, x_start, x_t, timesteps):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).

        :param x_start   : completely denoised input x_0
        :param x_t       : noisy input at timestep t
        :param timesteps : 1-D tensor of diffusion timestep indices
        :return          : (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, timesteps, x_start.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, timesteps, x_start.shape) * x_t
        )
        posterior_variance             = extract_into_tensor(self.posterior_variance,             timesteps, x_start.shape)
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
        Reconstruct x_0 from x_t and the estimated noise eps:
            x_0 = x_t / sqrt(alpha_bar_t) - eps * sqrt(1/alpha_bar_t - 1)

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

    def predict_x0_from_xprev(self, x_t, timesteps, x_prev):
        """
        Reconstruct x_0 from x_t and the immediately previous denoised sample x_{t-1}.

        :param x_t       : noisy input at timestep t
        :param timesteps : 1-D tensor of diffusion timestep indices
        :param x_prev    : denoised sample at timestep t-1
        :return          : reconstructed x_0
        """
        assert x_t.shape == x_prev.shape
        return (
            extract_into_tensor(1.0 / self.posterior_mean_coef1, timesteps, x_t.shape) * x_t -
            extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, timesteps, x_t.shape) * x_prev
        )

    def predict_eps_from_x0(self, x_t, timesteps, pred_x_start):
        """
        Reconstruct noise eps from x_t and the predicted x_0:
            eps = (x_t / sqrt(alpha_bar_t) - x_0) / sqrt(1/alpha_bar_t - 1)

        :param x_t          : noisy input at timestep t
        :param timesteps    : 1-D tensor of diffusion timestep indices
        :param pred_x_start : predicted x_0
        :return             : reconstructed noise eps
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t - pred_x_start
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)


    def scale_timesteps(self, timesteps):
        """Scale timesteps to [0, 1000] range if rescale_timesteps is True."""
        if self.rescale_timesteps:
            return timesteps.float() * (1000.0 / self.num_timesteps)
        return timesteps

    def p_mean_variance(self, model, conditional_guide, input_x, timesteps):
        """
        Apply the model to get the estimated denoising distribution parameters
        p(x_{t-1} | x_t), as well as a prediction of x_0.

        :param model             : denoising model — takes (x_t, t, conditional_guide)
        :param conditional_guide : conditioning signal
        :param input_x           : noisy input tensor at timestep t
        :param timesteps         : 1-D tensor of timestep indices, shape (B,)
        :return                  : dict with keys 'mean', 'variance', 'log_variance', 'pred_x0'
        """
        B, C = input_x.shape[:2]
        assert timesteps.shape == (B,)

        model_out = model(input_x, self.scale_timesteps(timesteps), conditional_guide)

        model_variance, model_log_variance = {
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance     = extract_into_tensor(model_variance,     timesteps, input_x.shape)
        model_log_variance = extract_into_tensor(model_log_variance, timesteps, input_x.shape)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_x0 = model_out
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_x0 = self.predict_x0_from_eps(
                x_t            = input_x,
                timesteps      = timesteps,
                gaussian_noise = model_out,
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start   = pred_x0,
            x_t       = input_x,
            timesteps = timesteps,
        )

        assert model_mean.shape == model_log_variance.shape == pred_x0.shape == input_x.shape

        return {
            "mean"         : model_mean,
            "variance"     : model_variance,
            "log_variance" : model_log_variance,
            "pred_x0"      : pred_x0,
        }

    def ddim_sample(self, model, input_x, timesteps, conditional_guide):
        """
        One DDIM denoising step.
        The posterior is:
            q_sigma(x_{t-1} | x_t, x_0) = N(x_{t-1} | sqrt(alpha_{t-1}) * x_0 + sqrt(1 - alpha_{t-1}) * eps, 0)
        Variance is set to 0 (fully implicit / deterministic).

        :param model             : denoising model
        :param input_x           : noisy input at timestep t
        :param timesteps         : 1-D tensor of timestep indices
        :param conditional_guide : conditioning signal
        :return                  : dict with keys 'sample' (x_{t-1}) and 'pred_x0'
        """
        out = self.p_mean_variance(
            model             = model,
            conditional_guide = conditional_guide,
            input_x           = input_x,
            timesteps         = timesteps,
        )

        # Reconstruct eps from predicted x_0
        eps = self.predict_eps_from_x0(input_x, timesteps, out['pred_x0'])

        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, timesteps, input_x.shape)

        # Deterministic DDIM update (sigma = 0)
        mean_pred = out['pred_x0'] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps

        return {
            'sample'  : mean_pred,
            'pred_x0' : out['pred_x0'],
        }

    def inpainting_complete(
        self,
        input_x,
        prev_t,
        conditional_guide,
        noise,
        masks,
        dct_num_channels,
        dct,
        idct,
    ):
        """
        Blend the known (observed) DCT frames back into the denoised sample.

        :param input_x           : current denoised DCT sample (unknown region)
        :param prev_t            : current timestep tensor for forward noising the guide
        :param conditional_guide : ground-truth DCT of observed frames
        :param noise             : optional pre-sampled noise; sampled internally if None
        :param masks             : binary mask — 1 for observed frames, 0 for unobserved
        :param dct_num_channels  : number of DCT frequency channels used
        :param dct               : DCT basis matrix
        :param idct              : inverse DCT basis matrix
        :return                  : blended DCT tensor
        """
        if noise is None:
            noise = torch.randn_like(conditional_guide)

        # Forward-noise the observed frames to the current timestep
        dct_x_prev_t_known = self.q_sample(
            x_start       = conditional_guide,
            timesteps     = prev_t,
            gaussian_noise = noise,
        )

        x_prev_t_known   = torch.matmul(idct[:, :dct_num_channels], dct_x_prev_t_known)
        x_prev_t_unknown = torch.matmul(idct[:, :dct_num_channels], input_x)

        # Ensure masks broadcasts correctly over joint/feature dims
        x_t = x_prev_t_known * masks + x_prev_t_unknown * (1 - masks)

        dct_x_t = torch.matmul(dct[:dct_num_channels], x_t)
        return dct_x_t


    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        conditional_guide = None,
        noise             = None,
        device            = None,
        dct_num_channels  = None,
        masks             = None, 
        dct               = None,
        idct              = None,
    ):
        """
        Run full DDIM reverse diffusion, yielding the DCT sample at every timestep.

        :param model             : denoising model
        :param shape             : output shape tuple, e.g. (B, C, T)
        :param conditional_guide : observed DCT frames used for inpainting
        :param noise             : optional starting noise; sampled internally if None
        :param device            : torch device; inferred from model if None
        :param dct_num_channels  : number of DCT frequency channels
        :param dct               : DCT basis matrix
        :param idct              : inverse DCT basis matrix
        :yields                  : DCT tensor at each denoising step
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        inp = noise if noise is not None else torch.randn(*shape, device=device) * self.noise_stdev

        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            timesteps = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():
                # FIX: keep the full dict from ddim_sample, not just ['sample']
                out = self.ddim_sample(
                    model             = model,
                    input_x           = inp,
                    timesteps         = timesteps,
                    conditional_guide = conditional_guide,
                )

                # Blend observed frames back in (inpainting replacement step)
                dct_sample = self.inpainting_complete(
                    input_x           = out['sample'],
                    conditional_guide = conditional_guide,
                    prev_t            = timesteps,
                    noise             = None,
                    masks             = masks, 
                    dct_num_channels  = dct_num_channels,
                    dct               = dct,
                    idct              = idct,
                )

            yield dct_sample
            inp = dct_sample  # FIX: inp is a tensor, not a dict

    def ddim_sample_loop(
        self,
        model,
        shape,
        conditional_guide = None,
        noise             = None,
        device            = None,
        dct               = None,
        masks             = None, 
        idct              = None,
        dct_num_channels  = None,
    ):
        """
        Run full DDIM reverse diffusion and return the final sample and all intermediates.

        :return : (final_dct_sample, list_of_intermediate_dct_samples)
        """
        final = None
        intermediate_samples = []

        for sample in self.ddim_sample_loop_progressive(
            model             = model,
            shape             = shape,
            conditional_guide = conditional_guide,
            noise             = noise,
            device            = device,
            dct               = dct,
            idct              = idct,
            dct_num_channels  = dct_num_channels,
            masks             = masks, 
        ):
            final = sample
            intermediate_samples.append(sample)

        return final, intermediate_samples

    def training_losses(self, model, x_start, timesteps, condition_guide, noise = None):
        """
        Compute training losses for a single timestep.

        :param model           : denoising model
        :param x_start         : [B x T x D] tensor of clean inputs
        :param timesteps       : 1-D tensor of timestep indices, shape (B,)
        :param condition_guide : conditioning signal (e.g. padded past poses)
        :param noise           : optional pre-sampled Gaussian noise
        :return                : dict with keys 'mse' and 'loss', each of shape (B,)
        """
        if noise is None:
            noise = torch.randn_like(x_start)


        x_t = self.q_sample(x_start = x_start, timesteps = timesteps, gaussian_noise = noise)

        model_out = model(x_t, self.scale_timesteps(timesteps), condition_guide)

        target = {
            ModelMeanType.START_X : x_start,
            ModelMeanType.EPSILON  : noise,
        }[self.model_mean_type]

        results         = {}
        results['mse']  = mean_flat((target - model_out) ** 2)
        results['loss'] = results['mse'] * self.loss_multipliers
        return results