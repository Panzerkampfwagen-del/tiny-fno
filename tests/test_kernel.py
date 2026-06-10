"""Validate the custom spectral_mm CUDA kernel against the einsum reference.

Skipped automatically when no CUDA device is available. The kernel and its
toolchain are only meaningful on GPU.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _einsum1d(u, w):
    return torch.einsum("bik,iok->bok", u, w)


@cuda
def test_forward_matches_einsum_1d():
    from kernels.spectral_mm import spectral_mm
    torch.manual_seed(0)
    u = torch.randn(64, 64, 16, dtype=torch.cfloat, device="cuda")
    w = torch.randn(64, 64, 16, dtype=torch.cfloat, device="cuda")
    err = (spectral_mm(u, w) - _einsum1d(u, w)).abs().max().item()
    assert err < 1e-4, f"forward error {err}"


@cuda
def test_backward_matches_einsum_1d():
    from kernels.spectral_mm import spectral_mm
    torch.manual_seed(1)
    u = torch.randn(32, 64, 16, dtype=torch.cfloat, device="cuda")
    w = torch.randn(64, 32, 16, dtype=torch.cfloat, device="cuda")
    g = torch.randn(32, 32, 16, dtype=torch.cfloat, device="cuda")

    uk, wk = u.clone().requires_grad_(True), w.clone().requires_grad_(True)
    ue, we = u.clone().requires_grad_(True), w.clone().requires_grad_(True)
    spectral_mm(uk, wk).backward(g)
    _einsum1d(ue, we).backward(g)
    assert (uk.grad - ue.grad).abs().max().item() < 1e-4
    assert (wk.grad - we.grad).abs().max().item() < 1e-4


@cuda
def test_gradcheck_double():
    from kernels.spectral_mm import spectral_mm
    torch.manual_seed(2)
    # Small double-precision problem so gradcheck's numerical jacobian is tight.
    u = torch.randn(3, 4, 5, dtype=torch.cdouble, device="cuda", requires_grad=True)
    w = torch.randn(4, 6, 5, dtype=torch.cdouble, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(spectral_mm, (u, w), atol=1e-6, rtol=1e-4)


@cuda
def test_matches_einsum_2d_flattened():
    from kernels.spectral_mm import spectral_mm
    torch.manual_seed(3)
    u = torch.randn(8, 32, 12, 12, dtype=torch.cfloat, device="cuda")
    w = torch.randn(32, 32, 12, 12, dtype=torch.cfloat, device="cuda")
    ref = torch.einsum("bikl,iokl->bokl", u, w)
    assert (spectral_mm(u, w) - ref).abs().max().item() < 1e-4


@cuda
def test_kernel_drops_into_full_fno():
    """The kernel mul_fn gives the same FNO2d forward and weight grads as einsum,
    proving the autograd Function integrates cleanly in the real model."""
    from models.fno import FNO2d
    from models.spectral_conv import compl_mul2d
    from kernels.spectral_mm import spectral_mm

    def build(mul_fn):
        torch.manual_seed(0)
        m = FNO2d(3, 1, 16, 8, n_blocks=2, proj_hidden=32).cuda()
        for blk in m.blocks:
            blk.spectral.mul_fn = mul_fn
        return m

    me, mk = build(compl_mul2d), build(spectral_mm)
    x = torch.randn(4, 3, 32, 32, device="cuda")
    ye, yk = me(x), mk(x)
    assert (ye - yk).abs().max().item() < 1e-3
    ye.pow(2).mean().backward()
    yk.pow(2).mean().backward()
    ge = me.blocks[0].spectral.weight1.grad
    gk = mk.blocks[0].spectral.weight1.grad
    assert (ge - gk).abs().max().item() < 1e-3
