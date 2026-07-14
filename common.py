"""Shared helpers: config loading, seeding, the relative-L2 loss, normalizers."""

import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make data generation and training deterministic given a seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relative_l2_per_sample(pred, true, eps=1e-8):
    """Per-sample ||pred - true||_2 / ||true||_2, norms over all non-batch dims.

    The denominator is clamped so a zero-norm target can never divide by zero.
    """
    b = pred.shape[0]
    num = torch.linalg.vector_norm((pred - true).reshape(b, -1), dim=1)
    den = torch.linalg.vector_norm(true.reshape(b, -1), dim=1).clamp_min(eps)
    return num / den


def relative_l2(pred, true, eps=1e-8):
    """Mean over batch of the per-sample relative-L2 error (the FNO metric)."""
    return relative_l2_per_sample(pred, true, eps).mean()


def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianNormalizer(torch.nn.Module):
    """Per-channel zero-mean unit-std normalizer, fit on training tensors.

    Statistics are stored as buffers so they travel with the model in
    checkpoints. Shapes are [1, C, 1, ...] so encode/decode broadcast over the
    batch and spatial axes.
    """

    def __init__(self, x, eps=1e-6):
        super().__init__()
        dims = [0] + list(range(2, x.dim()))      # all but the channel axis
        self.register_buffer("mean", x.mean(dims, keepdim=True))
        self.register_buffer("std", x.std(dims, keepdim=True) + eps)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


class Normalized(torch.nn.Module):
    """Wrap a model so it consumes and emits physical-scale tensors.

    Input is encoded before the model and the output decoded after, so the
    training loss and reported metric are always computed in physical units.
    """

    def __init__(self, model, x_norm, y_norm):
        super().__init__()
        self.model = model
        self.x_norm = x_norm
        self.y_norm = y_norm

    def forward(self, x):
        return self.y_norm.decode(self.model(self.x_norm.encode(x)))

    def __getattr__(self, name):
        # Forward attribute lookups (e.g. .blocks) to the wrapped model so the
        # model stays introspectable after wrapping. nn.Module's own
        # __getattr__ (params/buffers/submodules) is tried first.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
