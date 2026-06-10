"""Full FNO models for 1D and 2D.

    lifting (Linear)  ->  N x FNOBlock  ->  projection (Linear -> GELU -> Linear)

Lifting and projection mix only the channel dimension, so they are applied with
the channel axis moved last, which keeps the model code identical for 1D and 2D.
Input and output are channels-first: [B, in_channels, *spatial].
"""

import torch.nn as nn
import torch.nn.functional as F

from models.fno_block import FNOBlock


def _channel_linear(lin, x):
    """Apply a Linear over the channel axis (dim 1) of [B, C, *spatial]."""
    x = x.movedim(1, -1)
    x = lin(x)
    return x.movedim(-1, 1)


class _FNO(nn.Module):
    """Shared FNO body, parameterized by spatial dimension."""

    def __init__(self, dim, in_channels, out_channels, width, k_max,
                 n_blocks, proj_hidden):
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        # Activation on every block except the last (standard FNO; the final
        # spectral layer feeds the regression head linearly).
        self.blocks = nn.ModuleList([
            FNOBlock(width, k_max, dim=dim,
                     activation=F.gelu if i < n_blocks - 1 else nn.Identity())
            for i in range(n_blocks)
        ])
        self.proj1 = nn.Linear(width, proj_hidden)
        self.proj2 = nn.Linear(proj_hidden, out_channels)

    def forward(self, x):
        x = _channel_linear(self.lift, x)
        for blk in self.blocks:
            x = blk(x)
        x = _channel_linear(self.proj1, x)
        x = F.gelu(x)
        x = _channel_linear(self.proj2, x)
        return x


class FNO1d(_FNO):
    def __init__(self, in_channels, out_channels, width, k_max,
                 n_blocks=4, proj_hidden=128):
        super().__init__(1, in_channels, out_channels, width, k_max,
                         n_blocks, proj_hidden)


class FNO2d(_FNO):
    def __init__(self, in_channels, out_channels, width, k_max,
                 n_blocks=4, proj_hidden=128):
        super().__init__(2, in_channels, out_channels, width, k_max,
                         n_blocks, proj_hidden)


def count_params(model):
    """Real-valued parameter count (a complex parameter counts as two)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.is_complex() else 1)
    return total
