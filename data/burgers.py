"""Burgers' 1D data generation with a pseudo-spectral solver.

Equation on the periodic domain [0, 1):

    du/dt + u du/dx = nu d^2u/dx^2

Initial conditions are drawn from a Gaussian random field with power spectrum
P(k) ~ k^-power, truncated at k = kmax. Time integration uses integrating-factor
RK4: the stiff linear viscous term is solved exactly via the integrating factor
exp(-nu k^2 t), and classical RK4 advances only the nonlinear term. This is what
lets us take nt_full steps of dt = T/nt_full without the explicit-RK4 viscous
stability limit blowing up at nx = 1024.

Run:
    python -m data.burgers --config configs/burgers.yaml
    python -m data.burgers --config configs/burgers.yaml --plot
"""

import argparse
import os

import h5py
import numpy as np

from common import load_config


def grf_1d(n_samples, nx, power, kmax, rng):
    """Sample real GRF initial conditions, each normalized to unit std.

    Returns an array of shape [n_samples, nx]. Built in rfft space so the
    result is exactly real, with mode-0 (mean) zeroed and Nyquist made real.
    """
    nk = nx // 2 + 1
    k = np.arange(nk)
    amp = np.zeros(nk)
    band = (k >= 1) & (k <= kmax)
    amp[band] = k[band].astype(np.float64) ** (-power / 2.0)  # sqrt(P(k))

    re = rng.standard_normal((n_samples, nk))
    im = rng.standard_normal((n_samples, nk))
    coeff = (re + 1j * im) * amp[None, :]
    coeff[:, 0] = 0.0                       # zero mean
    if nx % 2 == 0:
        coeff[:, -1] = coeff[:, -1].real    # Nyquist mode must be real

    u0 = np.fft.irfft(coeff, n=nx, axis=1)
    u0 /= u0.std(axis=1, keepdims=True)     # unit std per sample
    return u0


# Physical domain length. The standard pseudo-spectral convention [0, 2pi]
# makes angular wavenumbers equal the integer mode index, so nu*k^2 stays
# moderate and the flow exhibits genuine nonlinear steepening rather than
# being dominated by viscous decay (as it would be on [0, 1]).
DOMAIN = 2.0 * np.pi


def _wavenumbers(nx, length=DOMAIN):
    """Angular wavenumbers for rfft on a domain of given length."""
    return 2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx)


def _dealias_mask(nx):
    """2/3-rule mask over rfft modes (True = keep)."""
    nk = nx // 2 + 1
    k = np.arange(nk)
    return k <= (nx // 3)


def spectral_downsample(u, nx_target):
    """Anti-aliased downsampling along the last axis via spectral truncation.

    Strided slicing aliases the thin shocks (thickness ~ nu) into the coarse
    grid, injecting sample-dependent noise into the targets. Truncating to the
    coarse grid's resolved modes instead keeps a clean band-limited field.
    """
    nx_full = u.shape[-1]
    if nx_target == nx_full:
        return u
    ft = np.fft.rfft(u, axis=-1)
    nk = nx_target // 2 + 1
    ft_lp = ft[..., :nk].copy() * (nx_target / nx_full)
    if nx_target % 2 == 0:
        ft_lp[..., -1] = ft_lp[..., -1].real     # Nyquist mode must be real
    return np.fft.irfft(ft_lp, n=nx_target, axis=-1)


def solve_burgers(u0, nu, T, nt_full, n_frames):
    """Integrate Burgers' forward from u0 over [0, T].

    u0: [n_samples, nx]. Returns traj [n_samples, n_frames, nx] sampled at
    n_frames time points (including t=0 and t=T) and uT [n_samples, nx] at t=T.
    """
    n, nx = u0.shape
    h = T / nt_full
    k = _wavenumbers(nx)                    # [nk]
    ik = 1j * k
    keep = _dealias_mask(nx)

    lin = -nu * k * k                       # linear operator (diagonal)
    E1 = np.exp(lin * (h / 2.0))[None, :]   # half-step integrating factor
    E2 = np.exp(lin * h)[None, :]           # full-step integrating factor

    def nonlinear(vhat):
        """N(vhat) = -dealias( rfft( u * u_x ) ), the advection term."""
        u = np.fft.irfft(vhat, n=nx, axis=1)
        ux = np.fft.irfft(ik[None, :] * vhat, n=nx, axis=1)
        phat = np.fft.rfft(u * ux, axis=1)
        phat[:, ~keep] = 0.0
        return -phat

    frame_steps = np.round(np.linspace(0, nt_full, n_frames)).astype(int)
    frame_steps[0], frame_steps[-1] = 0, nt_full
    step_to_frame = {s: i for i, s in enumerate(frame_steps)}

    traj = np.empty((n, n_frames, nx), dtype=np.float64)
    v = np.fft.rfft(u0, axis=1)
    traj[:, 0, :] = u0

    for step in range(1, nt_full + 1):
        # Integrating-factor RK4. Only E2*K_i combinations appear in the
        # update, all bounded, so the stiff high modes never overflow.
        n1 = nonlinear(v)
        n2 = nonlinear(E1 * (v + 0.5 * h * n1))
        n3 = nonlinear(E1 * v + 0.5 * h * n2)
        n4 = nonlinear(E2 * v + h * E1 * n3)
        v = E2 * v + (h / 6.0) * (E2 * n1 + 2.0 * E1 * (n2 + n3) + n4)

        if step in step_to_frame:
            traj[:, step_to_frame[step], :] = np.fft.irfft(v, n=nx, axis=1)

    uT = traj[:, -1, :].copy()
    return traj, uT


def generate(cfg):
    d = cfg["data"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    n_total = d["n_train"] + d["n_test"]
    nx_full, nx = d["nx_full"], d["nx"]
    nt, nt_full = d["nt"], d["nt_full"]

    u0 = grf_1d(n_total, nx_full, d["grf_power"], d["grf_kmax"], rng)
    traj_full, _ = solve_burgers(u0, d["nu"], d["T"], nt_full, nt + 1)

    # Anti-aliased spatial downsampling to the training resolution.
    traj = spectral_downsample(traj_full, nx)      # [n, nt+1, nx]
    x = np.linspace(0.0, 1.0, nx, endpoint=False)

    os.makedirs(os.path.dirname(d["path"]), exist_ok=True)
    with h5py.File(d["path"], "w") as f:
        f.create_dataset("u0", data=traj[:, 0, :].astype(np.float32))
        f.create_dataset("uT", data=traj[:, -1, :].astype(np.float32))
        f.create_dataset("traj", data=traj.astype(np.float32))
        f.create_dataset("x", data=x.astype(np.float32))
        f.attrs.update(dict(nu=d["nu"], T=d["T"], nx=nx, nt=nt,
                            n_train=d["n_train"], n_test=d["n_test"], seed=seed))
    print(f"[burgers] wrote {d['path']}  traj={traj.shape}  "
          f"u_range=[{traj.min():.3f}, {traj.max():.3f}]")
    return d["path"]


def plot_samples(cfg, n_show=10):
    import matplotlib.pyplot as plt
    d = cfg["data"]
    with h5py.File(d["path"], "r") as f:
        x = f["x"][:]
        u0 = f["u0"][:n_show]
        uT = f["uT"][:n_show]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(x, u0[i], label="u(t=0)", lw=1.2)
        ax.plot(x, uT[i], label="u(t=T)", lw=1.2)
        ax.set_title(f"sample {i}")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Burgers' GRF initial vs final states")
    fig.tight_layout()
    out = "results/burgers_samples.png"
    fig.savefig(out, dpi=110)
    print(f"[burgers] saved {out}")


def load_burgers(cfg):
    """Build train/test tensors from the generated HDF5.

    Input is two channels [u(t=0), grid x]; target is u(t=T), one channel.
    Returns (x_train, y_train, x_test, y_test, meta) with channels-first tensors
    shaped [N, C, nx].
    """
    import torch
    d = cfg["data"]
    with h5py.File(d["path"], "r") as f:
        u0 = torch.from_numpy(f["u0"][:])          # [N, nx]
        uT = torch.from_numpy(f["uT"][:])          # [N, nx]
        x = torch.from_numpy(f["x"][:])            # [nx]
        n_train = int(f.attrs["n_train"])
        n_test = int(f.attrs["n_test"])

    n, nx = u0.shape
    grid = x.reshape(1, nx).expand(n, nx)
    inp = torch.stack([u0, grid], dim=1)           # [N, 2, nx]
    tgt = uT.unsqueeze(1)                          # [N, 1, nx]

    xtr, ytr = inp[:n_train], tgt[:n_train]
    xte, yte = inp[n_train:n_train + n_test], tgt[n_train:n_train + n_test]
    meta = dict(dim=1, in_channels=2, out_channels=1)
    return xtr, ytr, xte, yte, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/burgers.yaml")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    generate(cfg)
    if args.plot:
        plot_samples(cfg)
