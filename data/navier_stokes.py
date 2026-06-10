"""Navier-Stokes 2D data: loader plus a self-contained vorticity solver.

The prompt asks for PDEBench's NS_incomp dataset (see data/download_ns.py for the
downloader). That file is multi-GB; to keep the 2D pipeline reproducible and
verifiable without it, this module also generates the same kind of data the
original FNO paper uses: 2D incompressible Navier-Stokes in vorticity form on a
periodic domain with a fixed forcing, solved pseudo-spectrally.

    dw/dt + u . grad(w) = nu * laplacian(w) + f,   u = (d psi/dy, -d psi/dx),
    laplacian(psi) = -w,   f = 0.1 (sin(x+y) + cos(x+y))   on [0, 2pi]^2

load_ns(cfg) returns (u_t, u_{t+T}) training pairs. If a PDEBench HDF5 is present
at cfg['data']['raw_path'] it is used; otherwise the solver fills a local HDF5.

    python -m data.navier_stokes --config configs/navier_stokes.yaml
"""

import argparse
import os

import h5py
import numpy as np

from common import load_config


def _grf_vorticity(n, nx, kmax, rng, rms=2.0):
    """Random initial vorticity fields, band-limited and scaled to a target RMS."""
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.rfftfreq(nx, d=1.0 / nx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2)
    amp = np.zeros_like(kmag)
    band = (kmag >= 1) & (kmag <= kmax)
    amp[band] = kmag[band] ** (-1.0)            # ~ k^-1 energy taper
    re = rng.standard_normal((n,) + kmag.shape)
    im = rng.standard_normal((n,) + kmag.shape)
    wh = (re + 1j * im) * amp[None]
    w = np.fft.irfft2(wh, s=(nx, nx), axes=(1, 2))
    w -= w.mean(axis=(1, 2), keepdims=True)
    w *= rms / w.std(axis=(1, 2), keepdims=True)
    return w


def _operators(nx, length):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2_inv = np.zeros_like(k2)
    k2_inv[k2 > 0] = 1.0 / k2[k2 > 0]           # inverse Laplacian, zero mean
    kcut = 2.0 * np.pi / length * (nx // 3)     # 2/3 dealias cutoff
    dealias = (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut)
    return KX, KY, k2, k2_inv, dealias


def _forcing_hat(nx, length):
    x = np.linspace(0, length, nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Standard FNO forcing: one period of sin/cos across the domain.
    f = 0.1 * (np.sin(2 * np.pi * (X + Y) / length)
               + np.cos(2 * np.pi * (X + Y) / length))
    return np.fft.rfft2(f)[None]


def solve_ns(w0, nu, dt, n_steps, record_every, length=1.0):
    """Evolve vorticity with integrating-factor RK4. Returns recorded frames
    [n_samples, n_frames, nx, nx]."""
    n, nx, _ = w0.shape
    KX, KY, k2, k2_inv, dealias = _operators(nx, length)
    f_hat = _forcing_hat(nx, length)
    lin = -nu * k2
    E1 = np.exp(lin * (dt / 2.0))[None]
    E2 = np.exp(lin * dt)[None]

    def rhs(wh):
        psi = wh * k2_inv[None]
        u = np.fft.irfft2(1j * KY[None] * psi, s=(nx, nx), axes=(1, 2))
        v = np.fft.irfft2(-1j * KX[None] * psi, s=(nx, nx), axes=(1, 2))
        wx = np.fft.irfft2(1j * KX[None] * wh, s=(nx, nx), axes=(1, 2))
        wy = np.fft.irfft2(1j * KY[None] * wh, s=(nx, nx), axes=(1, 2))
        adv = np.fft.rfft2(u * wx + v * wy, axes=(1, 2))
        adv *= dealias[None]
        return -adv + f_hat

    wh = np.fft.rfft2(w0, axes=(1, 2))
    frames = [w0.copy()]
    for step in range(1, n_steps + 1):
        n1 = rhs(wh)
        n2 = rhs(E1 * (wh + 0.5 * dt * n1))
        n3 = rhs(E1 * wh + 0.5 * dt * n2)
        n4 = rhs(E2 * wh + dt * E1 * n3)
        wh = E2 * wh + (dt / 6.0) * (E2 * n1 + 2.0 * E1 * (n2 + n3) + n4)
        if step % record_every == 0:
            frames.append(np.fft.irfft2(wh, s=(nx, nx), axes=(1, 2)))
    return np.stack(frames, axis=1)


def generate(cfg):
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n = d["n_train"] + d["n_test"]
    nx = d["resolution"]
    spinup, ahead = d.get("spinup", 40), d["t_ahead"]

    assert spinup % ahead == 0, "spinup must be a multiple of t_ahead"
    length = d.get("domain", 1.0)
    w0 = _grf_vorticity(n, nx, kmax=d.get("grf_kmax", 12), rng=rng)
    # Spin up, then record two frames `ahead` steps apart: input and target.
    traj = solve_ns(w0, d.get("nu", 1e-3), d.get("dt", 1e-2),
                    n_steps=spinup + ahead, record_every=ahead, length=length)
    i_in = spinup // ahead
    w_in = traj[:, i_in]
    w_out = traj[:, i_in + 1]

    os.makedirs(os.path.dirname(d["path"]), exist_ok=True)
    with h5py.File(d["path"], "w") as f:
        f.create_dataset("u_in", data=w_in.astype(np.float32))
        f.create_dataset("u_out", data=w_out.astype(np.float32))
        # Keep one short trajectory for rollout evaluation.
        roll = solve_ns(w0[:1], d.get("nu", 1e-3), d.get("dt", 1e-2),
                        n_steps=spinup + ahead * 11, record_every=ahead,
                        length=length)
        f.create_dataset("rollout", data=roll[0, spinup // ahead:].astype(np.float32))
        f.attrs.update(dict(n_train=d["n_train"], n_test=d["n_test"],
                            resolution=nx, t_ahead=ahead, seed=cfg["seed"]))
    print(f"[ns] wrote {d['path']}  pairs={n}  res={nx}  "
          f"w_in std={w_in.std():.3f}  w_out std={w_out.std():.3f}")
    return d["path"]


def _grid(nx):
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    return gx, gy


def load_ns(cfg):
    """Build (input, target) tensors: input is [w_t, grid_x, grid_y]."""
    import torch
    d = cfg["data"]
    if not os.path.exists(d["path"]):
        generate(cfg)
    with h5py.File(d["path"], "r") as f:
        w_in = torch.from_numpy(f["u_in"][:])           # [N, nx, nx]
        w_out = torch.from_numpy(f["u_out"][:])
        n_train, n_test = int(f.attrs["n_train"]), int(f.attrs["n_test"])
    n, nx, _ = w_in.shape
    gx, gy = _grid(nx)
    grid = torch.from_numpy(np.stack([gx, gy]))         # [2, nx, nx]
    grid = grid.unsqueeze(0).expand(n, 2, nx, nx)
    inp = torch.cat([w_in.unsqueeze(1), grid], dim=1)   # [N, 3, nx, nx]
    tgt = w_out.unsqueeze(1)                            # [N, 1, nx, nx]
    xtr, ytr = inp[:n_train], tgt[:n_train]
    xte = inp[n_train:n_train + n_test]
    yte = tgt[n_train:n_train + n_test]
    return xtr, ytr, xte, yte, dict(dim=2, in_channels=3, out_channels=1)


def load_rollout(cfg, device):
    """Return (seq [T,1,H,W], grid [1,2,H,W]) for autoregressive rollout."""
    import torch
    d = cfg["data"]
    if not os.path.exists(d["path"]):
        return None, None
    with h5py.File(d["path"], "r") as f:
        if "rollout" not in f:
            return None, None
        seq = torch.from_numpy(f["rollout"][:]).unsqueeze(1).to(device)
    nx = seq.shape[-1]
    gx, gy = _grid(nx)
    grid = torch.from_numpy(np.stack([gx, gy])).unsqueeze(0).to(device)
    return seq, grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/navier_stokes.yaml")
    args = ap.parse_args()
    generate(load_config(args.config))
