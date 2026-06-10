"""Validate the spectral convolution against an independent naive reference."""

import torch

from models.spectral_conv import (SpectralConv1d, SpectralConv2d,
                                   compl_mul1d, compl_mul2d)


def _naive_mul1d(inp, weight):
    """Reference complex channel mix, looping modes explicitly."""
    b, ci, k = inp.shape
    co = weight.shape[1]
    out = torch.zeros(b, co, k, dtype=inp.dtype)
    for m in range(k):
        out[:, :, m] = inp[:, :, m] @ weight[:, :, m]
    return out


def test_compl_mul1d_matches_naive():
    inp = torch.randn(3, 4, 5, dtype=torch.cfloat)
    w = torch.randn(4, 6, 5, dtype=torch.cfloat)
    assert torch.allclose(compl_mul1d(inp, w), _naive_mul1d(inp, w), atol=1e-5)


def test_spectralconv1d_matches_manual_forward():
    torch.manual_seed(0)
    b, ci, co, n, k = 2, 3, 5, 64, 16
    layer = SpectralConv1d(ci, co, k)
    x = torch.randn(b, ci, n)

    y = layer(x)

    # Manual forward: rfft, truncate, per-mode complex matmul, zero-pad, irfft.
    x_ft = torch.fft.rfft(x, dim=-1)
    out_ft = torch.zeros(b, co, x_ft.shape[-1], dtype=torch.cfloat)
    out_ft[:, :, :k] = _naive_mul1d(x_ft[:, :, :k], layer.weight[:, :, :k])
    y_ref = torch.fft.irfft(out_ft, n=n, dim=-1)

    assert y.shape == (b, co, n)
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_spectralconv1d_shape_and_grad():
    layer = SpectralConv1d(2, 4, 8)
    x = torch.randn(3, 2, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 4, 32)
    y.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_compl_mul2d_matches_einsum_reference():
    inp = torch.randn(2, 3, 4, 4, dtype=torch.cfloat)
    w = torch.randn(3, 5, 4, 4, dtype=torch.cfloat)
    ref = torch.zeros(2, 5, 4, 4, dtype=torch.cfloat)
    for a in range(4):
        for bb in range(4):
            ref[:, :, a, bb] = inp[:, :, a, bb] @ w[:, :, a, bb]
    assert torch.allclose(compl_mul2d(inp, w), ref, atol=1e-5)


def test_spectralconv2d_shape_and_grad():
    layer = SpectralConv2d(3, 4, 6)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 4, 32, 32)
    y.pow(2).mean().backward()
    assert torch.isfinite(layer.weight1.grad).all()
    assert torch.isfinite(layer.weight2.grad).all()
