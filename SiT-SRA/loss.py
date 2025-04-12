
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import smooth_l1_loss
import random

# simple loss function
class Simpleloss(nn.Module):
    def __init__(self):
        super(Simpleloss, self).__init__()

    def forward(self, a, b, loss_type="sml1"):
        if loss_type == "sml1":
            align_loss = smooth_l1_loss(a, b, beta=0.05)
        elif loss_type == "l2":
            align_loss = F.mse_loss(a, b)
        elif loss_type == "l1":
            align_loss = F.l1_loss(a, b)
        else:
            raise NotImplementedError()
        return align_loss



def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SRALoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            latents_scale=None, 
            latents_bias=None,
            block_out_s=4,
            block_out_t=8,
            t_max=0.2,
            loss_type="sml1"

            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.criterion = Simpleloss()
        self.block_out_s = block_out_s
        self.block_out_t = block_out_t
        self.t_max = t_max
        self.loss_type = loss_type

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images,teacher,labels):

        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))

        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)

            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        time_input_teacher = time_input - (self.t_max * torch.rand_like(time_input))
        time_input = time_input.to(device=images.device, dtype=images.dtype)


        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError()

        model_output_gen, xr, labels_train = model(model_input, time_input.flatten(), y=labels,ad=self.block_out_s)

        # teacher
        time_input_teacher = torch.clamp(time_input_teacher, 0, 1)
        time_input_teacher = time_input_teacher.to(device=images.device, dtype=images.dtype)
        labels_teacher = labels_train
        noises_t = noises
        images_t = images
        alpha_teacher, sigma_teacher, d_alpha_teacher, d_sigma_teacher = self.interpolant(time_input_teacher)
        teacher_input = alpha_teacher * images_t + sigma_teacher * noises_t

        xr_t = teacher(teacher_input, time_input_teacher.flatten(), y=labels_teacher, ad=self.block_out_t)[1]

        # loss
        denoising_loss = mean_flat((model_output_gen - model_target) ** 2)
        align_loss = self.criterion(xr, xr_t, loss_type=self.loss_type)

        return denoising_loss, align_loss
