import math

import torch
from torch import nn


class FISTA(nn.Module):

    def __init__(
        self,
        n_iters=100,
        learnable=False,
        tau=2e-4,
        n_inner=5,
        pad_factor=2,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.n_inner = n_inner
        self.pad_factor = pad_factor

        log_scale = torch.zeros(n_iters)
        log_tau = torch.full((n_iters,), math.log(tau))
        if learnable:
            self.log_scale = nn.Parameter(log_scale)
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_scale", log_scale)
            self.register_buffer("log_tau", log_tau)

    def _params(self, k):
        return torch.exp(self.log_scale[k]), torch.exp(self.log_tau[k])

    def _pad(self, x, full_shape):
        h, w = x.shape[-2:]
        full_h, full_w = full_shape
        top = (full_h - h) // 2
        bottom = full_h - h - top
        left = (full_w - w) // 2
        right = full_w - w - left
        return nn.functional.pad(x, (left, right, top, bottom))

    def _crop(self, x, shape):
        h, w = shape
        full_h, full_w = x.shape[-2:]
        top = (full_h - h) // 2
        left = (full_w - w) // 2
        return x[..., top : top + h, left : left + w]

    @staticmethod
    def _grad(x):
        dh = torch.roll(x, shifts=1, dims=-2) - x
        dv = torch.roll(x, shifts=1, dims=-1) - x
        return dh, dv

    @staticmethod
    def _grad_transpose(dh, dv):
        return (torch.roll(dh, shifts=-1, dims=-2) - dh) + (
            torch.roll(dv, shifts=-1, dims=-1) - dv
        )

    @staticmethod
    def _conv(x, psf_fft):
        return torch.fft.ifft2(psf_fft * torch.fft.fft2(x)).real

    @staticmethod
    def _conv_transpose(x, psf_fft):
        return torch.fft.ifft2(torch.conj(psf_fft) * torch.fft.fft2(x)).real

    def _prox_tv(self, z, weight):
        p_h = torch.zeros_like(z)
        p_v = torch.zeros_like(z)
        for _ in range(self.n_inner):
            x = z - self._grad_transpose(p_h, p_v)
            dh, dv = self._grad(x)
            p_h = torch.clamp(p_h + dh / 8.0, min=-weight, max=weight)
            p_v = torch.clamp(p_v + dv / 8.0, min=-weight, max=weight)
        return z - self._grad_transpose(p_h, p_v)

    def forward(self, measurement, psf):
        shape = measurement.shape[-2:]
        full_shape = (shape[0] * self.pad_factor, shape[1] * self.pad_factor)

        psf_full = self._pad(psf, full_shape)
        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_full, dim=(-2, -1)))
        lipschitz = (torch.abs(psf_fft) ** 2).amax(dim=(-2, -1), keepdim=True) + 1e-6

        crop_mask = self._pad(torch.ones_like(measurement), full_shape)
        padded_measurement = self._pad(measurement, full_shape)

        x = torch.zeros_like(crop_mask)
        y = torch.zeros_like(crop_mask)
        t = 1.0

        for k in range(self.n_iters):
            scale, tau = self._params(k)
            step = scale / lipschitz

            residual = crop_mask * self._conv(y, psf_fft) - padded_measurement
            grad = self._conv_transpose(residual, psf_fft)
            z = y - step * grad

            x_next = self._prox_tv(z, step * tau)
            x_next = torch.clamp(x_next, min=0.0)

            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next

        x = self._crop(x, shape)
        return torch.clamp(x, min=0.0)
