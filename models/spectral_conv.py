"""Spectral convolution layers, pure PyTorch.

The layer transforms to Fourier space, keeps the lowest k_max modes, applies a
learned complex linear map per mode (mixing channels), zero-pads the truncated
high frequencies, and transforms back. The complex channel-mixing multiply is
factored into compl_mul1d / compl_mul2d so the custom CUDA kernel can be dropped
in later via the mul_fn argument without changing this module.
"""

import torch
import torch.nn as nn


def compl_mul1d(inp, weight):
    """inp [B, C_in, K] x weight [C_in, C_out, K] -> [B, C_out, K], complex."""
    return torch.einsum("bik,iok->bok", inp, weight)


def compl_mul2d(inp, weight):
    """inp [B, C_in, K1, K2] x weight [C_in, C_out, K1, K2] -> [B, C_out, K1, K2]."""
    return torch.einsum("bikl,iokl->bokl", inp, weight)


class SpectralConv1d(nn.Module):
    """1D spectral convolution keeping k_max Fourier modes."""

    def __init__(self, in_channels, out_channels, k_max, mul_fn=compl_mul1d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max
        self.mul_fn = mul_fn
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, k_max, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: [B, C_in, N]. Run the FFT path in fp32 under autocast (so bf16/half
        # inputs are promoted), but leave float64 alone so a double-precision
        # gradcheck of the full layer is exact.
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        b, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                 # [B, C_in, n//2+1]
        k = min(self.k_max, x_ft.shape[-1])
        out_ft = torch.zeros(b, self.out_channels, x_ft.shape[-1],
                             dtype=x_ft.dtype, device=x.device)
        # .to() matches the weight dtype to the FFT (no-op in fp32; promotes the
        # cfloat weight to cdouble when x is float64, since Module.double() does
        # not cast complex parameters).
        w = self.weight[:, :, :k].to(x_ft.dtype)
        out_ft[:, :, :k] = self.mul_fn(x_ft[:, :, :k], w)
        return torch.fft.irfft(out_ft, n=n, dim=-1)      # [B, C_out, N]


class SpectralConv2d(nn.Module):
    """2D spectral convolution keeping (k_max, k_max) modes.

    rfft2 returns only the non-redundant half-plane (last axis length W//2+1),
    so the four conceptual quadrants of the 2D spectrum collapse to two corners
    along the first transformed axis: the low positive frequencies and the low
    negative frequencies. Hence two weight tensors, not four. This matches the
    canonical Li et al. FNO2d implementation.
    """

    def __init__(self, in_channels, out_channels, k_max, mul_fn=compl_mul2d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max
        self.mul_fn = mul_fn
        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, k_max, k_max,
                               dtype=torch.cfloat))
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, k_max, k_max,
                               dtype=torch.cfloat))

    def forward(self, x):
        # x: [B, C_in, H, W]. Promote bf16/half to fp32 for the FFT; keep
        # float64 so a double-precision gradcheck of the full layer is exact.
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))          # [B, C_in, H, W//2+1]
        k1 = min(self.k_max, h // 2)
        k2 = min(self.k_max, x_ft.shape[-1])
        out_ft = torch.zeros(b, self.out_channels, h, x_ft.shape[-1],
                             dtype=x_ft.dtype, device=x.device)
        # Match the weight dtype to the FFT (no-op in fp32; promotes cfloat to
        # cdouble for float64 input, which Module.double() leaves untouched).
        w1 = self.weight1[:, :, :k1, :k2].to(x_ft.dtype)
        w2 = self.weight2[:, :, :k1, :k2].to(x_ft.dtype)
        out_ft[:, :, :k1, :k2] = self.mul_fn(x_ft[:, :, :k1, :k2], w1)
        out_ft[:, :, -k1:, :k2] = self.mul_fn(x_ft[:, :, -k1:, :k2], w2)
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))   # [B, C_out, H, W]
