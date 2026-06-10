"""Profile one FNO training step and locate the bottleneck.

    python profiling.py --config configs/burgers.yaml --batch 64

torch.profiler's CUDA timing needs CUPTI, which is restricted under WSL2 here
(CUDA kernel times come back empty), so the per-region breakdown is measured with
CUDA events instead, which do work. We time the spectral path, the pointwise
bypass, and within the spectral path the FFT vs the complex multiply, then report
which dominates. A Chrome trace is still exported (CPU-side timeline plus memory;
CUDA kernels may be absent under WSL2).
"""

import argparse
from collections import defaultdict

import torch

from common import device_auto, load_config, relative_l2
from train import build_model


class Regions:
    """Accumulate GPU time per named region using CUDA event pairs."""

    def __init__(self):
        self.pairs = defaultdict(list)

    def _timed(self, label, fn, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*a, **k)
        e.record()
        self.pairs[label].append((s, e))
        return out

    def wrap_module(self, module, label):
        orig = module.forward
        module.forward = lambda *a, **k: self._timed(label, orig, *a, **k)

    def wrap_mul(self, spectral, label):
        orig = spectral.mul_fn
        spectral.mul_fn = lambda *a, **k: self._timed(label, orig, *a, **k)

    def totals_ms(self):
        torch.cuda.synchronize()
        return {label: sum(s.elapsed_time(e) for s, e in pairs)
                for label, pairs in self.pairs.items()}


def make_batch(cfg, batch, device):
    m = cfg["model"]
    if cfg["problem"] == "burgers":
        n = cfg["data"]["nx"]
        x = torch.randn(batch, m["in_channels"], n, device=device)
        y = torch.randn(batch, m["out_channels"], n, device=device)
    else:
        r = cfg["data"]["resolution"]
        x = torch.randn(batch, m["in_channels"], r, r, device=device)
        y = torch.randn(batch, m["out_channels"], r, r, device=device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/burgers.yaml")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trace", default="results/profile_trace.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    if device.type != "cuda":
        raise SystemExit("profiling requires a CUDA device")

    model = build_model(cfg, {"dim": 1 if cfg["problem"] == "burgers" else 2})
    model = model.to(device)
    reg = Regions()
    for blk in model.blocks:
        reg.wrap_module(blk.spectral, "SpectralConv (FFT+multiply)")
        reg.wrap_module(blk.pointwise, "Pointwise bypass")
        reg.wrap_mul(blk.spectral, "  - complex multiply")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = make_batch(cfg, args.batch, device)

    def step():
        opt.zero_grad(set_to_none=True)
        relative_l2(model(x), y).backward()
        opt.step()

    for _ in range(10):                      # warm up plans/allocator
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    reg.pairs.clear()

    # Whole-step time via events, plus a profiler pass for the trace.
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters):
        step()
    e.record()
    torch.cuda.synchronize()
    step_ms = s.elapsed_time(e) / args.iters

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True) as prof:
        step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(args.trace)

    t = reg.totals_ms()
    n = args.iters
    spec = t.get("SpectralConv (FFT+multiply)", 0) / n
    pw = t.get("Pointwise bypass", 0) / n
    mul = t.get("  - complex multiply", 0) / n
    fft = spec - mul

    print(f"\n[profile] {cfg['problem']}  batch={args.batch}  "
          f"width={cfg['model']['width']}  k_max={cfg['model']['k_max']}")
    print(f"  full training step      : {step_ms:7.3f} ms")
    print(f"  SpectralConv (forward)  : {spec:7.3f} ms   (4 blocks summed)")
    print(f"    - FFT (rfft+irfft)    : {fft:7.3f} ms   ({100*fft/spec:.0f}% of spectral)")
    print(f"    - complex multiply    : {mul:7.3f} ms   ({100*mul/spec:.0f}% of spectral)")
    print(f"  Pointwise bypass        : {pw:7.3f} ms")
    print(f"  peak CUDA memory        : {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    dom = "SpectralConv" if spec > pw else "Pointwise"
    print(f"  dominant block path     : {dom}")
    print(f"  bottleneck within spectral: {'FFT' if fft > mul else 'complex multiply'}")
    print(f"\n[profile] chrome trace -> {args.trace}  "
          f"(CUDA kernels may be absent under WSL2/CUPTI)")


if __name__ == "__main__":
    main()
