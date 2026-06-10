"""FNO block: spectral convolution in parallel with a pointwise bypass.

    x --> SpectralConv --------> (+) --> activation
      --> Conv1x1 (pointwise) -->

The spectral path carries global dependencies; the pointwise 1x1 conv carries
local information that the truncated spectrum drops.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.spectral_conv import SpectralConv1d, SpectralConv2d


class FNOBlock(nn.Module):
    def __init__(self, width, k_max, dim=1, activation=F.gelu):
        super().__init__()
        if dim == 1:
            self.spectral = SpectralConv1d(width, width, k_max)
            self.pointwise = nn.Conv1d(width, width, 1)
        elif dim == 2:
            self.spectral = SpectralConv2d(width, width, k_max)
            self.pointwise = nn.Conv2d(width, width, 1)
        else:
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        self.act = activation

    def forward(self, x):
        return self.act(self.spectral(x) + self.pointwise(x))
