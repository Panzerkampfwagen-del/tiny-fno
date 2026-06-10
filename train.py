"""Train an FNO on Burgers' (1D) or Navier-Stokes (2D).

    python train.py --config configs/burgers.yaml
    python train.py --config configs/navier_stokes.yaml

Loss and the reported metric are both the relative-L2 error. Mixed precision
uses BF16 autocast (no GradScaler needed for BF16). The spectral FFTs run in
fp32 internally, so autocast is safe.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (GaussianNormalizer, Normalized, device_auto, load_config,
                    relative_l2, set_seed)
from models.fno import FNO1d, FNO2d, count_params


def build_data(cfg):
    problem = cfg["problem"]
    if problem == "burgers":
        from data.burgers import load_burgers
        return load_burgers(cfg)
    if problem == "navier_stokes":
        from data.navier_stokes import load_ns
        return load_ns(cfg)
    raise ValueError(f"unknown problem: {problem}")


def build_model(cfg, meta):
    m = cfg["model"]
    ctor = FNO1d if meta["dim"] == 1 else FNO2d
    return ctor(in_channels=m["in_channels"], out_channels=m["out_channels"],
                width=m["width"], k_max=m["k_max"], n_blocks=m["n_blocks"],
                proj_hidden=m["proj_hidden"])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total += relative_l2(pred, yb) * xb.shape[0]   # keep on GPU
        n += xb.shape[0]
    return (total / n).item()                          # single sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = device_auto()
    tr = cfg["train"]
    epochs = args.epochs if args.epochs is not None else tr["epochs"]

    xtr, ytr, xte, yte, meta = build_data(cfg)
    train_loader = DataLoader(TensorDataset(xtr, ytr),
                              batch_size=tr["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte),
                             batch_size=tr["batch_size"], shuffle=False)

    model = build_model(cfg, meta)
    if tr.get("normalize", True):
        # Standard FNO Gaussian normalization, fit on the training split.
        model = Normalized(model, GaussianNormalizer(xtr), GaussianNormalizer(ytr))
    model = model.to(device)
    print(f"[train] problem={cfg['problem']} device={device} "
          f"params={count_params(model):,} normalize={tr.get('normalize', True)} "
          f"train={len(xtr)} test={len(xte)}")

    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                           weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tr["sched_t_max"], eta_min=tr["sched_eta_min"])
    use_amp = tr["amp"] and device.type == "cuda"

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run = torch.zeros((), device=device)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                pred = model(xb)
                loss = relative_l2(pred, yb)
            loss.backward()
            opt.step()
            run += loss.detach() * xb.shape[0]          # accumulate on GPU
        sched.step()
        train_loss = (run / len(xtr)).item()            # single sync per epoch

        if epoch % tr["log_every"] == 0 or epoch == epochs - 1:
            test_err = evaluate(model, test_loader, device)
            if test_err < best:
                best = test_err
                torch.save({"model": model.state_dict(), "config": cfg,
                            "epoch": epoch, "test_rel_l2": best},
                           tr["checkpoint"])
            dt = time.time() - t0
            print(f"epoch {epoch:4d}  train {train_loss:.4e}  "
                  f"test relL2 {test_err:.4e}  best {best:.4e}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {dt*1e3:.0f} ms/ep")

    print(f"[train] best test relative L2: {best:.4e}  "
          f"checkpoint: {tr['checkpoint']}")


if __name__ == "__main__":
    main()
