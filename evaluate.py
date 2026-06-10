"""Evaluate a trained FNO: relative L2, error fields, and rollout drift.

    python evaluate.py --config configs/burgers.yaml

Rebuilds the exact training-time model (including the Gaussian normalizers,
which are deterministic given the seed) and loads the checkpoint. Plots go to
results/.
"""

import argparse
import os

import torch

from common import (GaussianNormalizer, Normalized, device_auto, load_config,
                    relative_l2, relative_l2_per_sample, set_seed)
from train import build_data, build_model


def load_trained(cfg, device):
    set_seed(cfg["seed"])
    xtr, ytr, xte, yte, meta = build_data(cfg)
    model = build_model(cfg, meta)
    if cfg["train"].get("normalize", True):
        model = Normalized(model, GaussianNormalizer(xtr), GaussianNormalizer(ytr))
    ckpt = torch.load(cfg["train"]["checkpoint"], map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, (xte, yte), meta, ckpt


@torch.no_grad()
def relative_l2_dataset(model, x, y, device, bs=64):
    errs = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        yb = y[i:i + bs].to(device)
        pred = model(xb)
        errs.append(relative_l2_per_sample(pred, yb).cpu())
    return torch.cat(errs)


def plot_burgers(model, xte, yte, device, out):
    import matplotlib.pyplot as plt
    n_show = 4
    xb = xte[:n_show].to(device)
    with torch.no_grad():
        pred = model(xb).cpu()
    u0 = xte[:n_show, 0]
    true = yte[:n_show, 0]
    pred = pred[:, 0]

    fig, axes = plt.subplots(2, n_show, figsize=(16, 6))
    for i in range(n_show):
        ax = axes[0, i]
        ax.plot(u0[i], "--", color="gray", lw=1, label="u(t=0)")
        ax.plot(true[i], color="C0", lw=1.4, label="true u(t=1)")
        ax.plot(pred[i], color="C1", lw=1.2, label="pred u(t=1)")
        ax.set_title(f"test sample {i}")
        if i == 0:
            ax.legend(fontsize=8)
        axes[1, i].plot((pred[i] - true[i]).abs(), color="C3", lw=1)
        axes[1, i].set_title("|pred - true|")
    fig.suptitle("FNO1d Burgers': prediction vs ground truth")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"[evaluate] saved {out}")

    # Error heatmap across the whole test set.
    with torch.no_grad():
        allpred = model(xte.to(device)).cpu()[:, 0]
    err = (allpred - yte[:, 0]).abs()
    fig2, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(err, aspect="auto", cmap="magma")
    ax.set_xlabel("x"); ax.set_ylabel("test sample")
    ax.set_title("|u_pred - u_true| over test set")
    fig2.colorbar(im, ax=ax)
    heat = out.replace(".png", "_heatmap.png")
    fig2.tight_layout(); fig2.savefig(heat, dpi=110)
    print(f"[evaluate] saved {heat}")


def plot_ns(model, xte, yte, device, out):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        pred = model(xte[:1].to(device)).cpu()[0, 0]
    true = yte[0, 0]
    err = (pred - true).abs()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, title in zip(axes, [true, pred, err],
                                ["ground truth", "prediction", "|error|"]):
        im = ax.imshow(field, cmap="twilight" if title != "|error|" else "magma")
        ax.set_title(title); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("FNO2d Navier-Stokes: one test sample")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"[evaluate] saved {out}")


@torch.no_grad()
def rollout_drift(model, cfg, device, steps=10):
    """Autoregressive rollout error vs the ground-truth NS trajectory."""
    from data.navier_stokes import load_rollout
    seq, grid = load_rollout(cfg, device)        # seq: [T, 1, H, W]
    if seq is None:
        return None
    x = torch.cat([seq[0:1], grid], dim=1)       # [1, C, H, W]
    errs = []
    # seq[0] is the input frame; valid targets are seq[1 .. T-1].
    for t in range(1, min(steps, seq.shape[0] - 1) + 1):
        pred = model(x)
        true = seq[t:t + 1]
        errs.append(relative_l2(pred, true).item())
        x = torch.cat([pred, grid], dim=1)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["train"]["checkpoint"] = args.checkpoint
    device = device_auto()
    os.makedirs("results", exist_ok=True)

    model, (xte, yte), meta, ckpt = load_trained(cfg, device)
    errs = relative_l2_dataset(model, xte, yte, device)
    print(f"[evaluate] {cfg['problem']}  checkpoint epoch {ckpt['epoch']}")
    print(f"  test relative L2: mean {errs.mean():.4e}  median {errs.median():.4e}"
          f"  max {errs.max():.4e}")

    if meta["dim"] == 1:
        plot_burgers(model, xte, yte, device, "results/eval_burgers.png")
    else:
        plot_ns(model, xte, yte, device, "results/eval_ns.png")
        drift = rollout_drift(model, cfg, device, steps=10)
        if drift:
            print("  rollout relative L2 per step:",
                  " ".join(f"{d:.3f}" for d in drift))


if __name__ == "__main__":
    main()
