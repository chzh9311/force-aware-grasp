import numpy as np
import torch
import torch.nn as nn


class DiffScheduler:
    def __init__(self, beta_schedule, beta_start, beta_end, num_timesteps):
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps
        self._get_beta_schedule()
        self._get_alpha()

    def _get_beta_schedule(self):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if self.beta_schedule == "quad":
            betas = (
                    torch.linspace(
                        self.beta_start ** 0.5,
                        self.beta_end ** 0.5,
                        self.num_timesteps,
                        dtype=torch.float32,
                        )
                    ** 2
            )
        elif self.beta_schedule == "linear":
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32
            )
        elif self.beta_schedule == "const":
            betas = self.beta_end * torch.ones(self.num_timesteps, dtype=torch.float32)
        elif self.beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / torch.linspace(
                self.num_timesteps, 1, self.num_timesteps, dtype=torch.float32
            )
        elif self.beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.num_timesteps)
            betas = sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise NotImplementedError(self.beta_schedule)
        assert betas.shape == (self.num_timesteps,)
        self.betas = betas

    def _get_alpha(self):
        self.alphas = 1 - self.betas
        log_alphas = torch.log(self.alphas)
        alphabar = torch.cumsum(log_alphas, dim=0).exp()
        self.sqrtab = torch.sqrt(alphabar)
        self.sqrtmab = torch.sqrt(1 - alphabar)
        self.sqrtbeta = torch.sqrt(self.betas)
        self.oneoversqrta = 1 / self.alphas
        self.betaoversqrtmab = self.betas / self.sqrtmab

    def add_noise(self, x, noise, steps):
        """
        x, noise, steps: batch_size x ...
        returns the x_t as in DDPM
        """
        shape = x.shape
        broad_sab = self.sqrtab.to(x.device)[steps].view(-1, *((1,)*(len(shape)-1)))
        broad_msab = self.sqrtmab.to(x.device)[steps].view(-1, *((1,)*(len(shape)-1)))
        return broad_sab * x + broad_msab * noise

    def denoise_step(self, x_i, eps, z, t):
        return self.oneoversqrta[t].to(x_i.device) * (x_i - self.betaoversqrtmab[t] * eps) \
                + self.sqrtbeta[t].to(x_i.device) * z

