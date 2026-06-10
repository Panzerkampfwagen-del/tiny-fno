"""Benchmark the spectral channel-mix and the SpectralConv2d layer:
einsum baseline vs custom CUDA kernel vs the neuraloperator reference.

    CUDA_HOME=/home/aryan/anaconda3/envs/tinyinfer \
        python benchmark.py --cin 32 --cout 32 --kmax 12

Three views:
  1. the isolated complex multiply the kernel replaces (latency, bandwidth,
     roofline),
  2. the full SpectralConv2d layer, einsum vs kernel vs neuraloperator, and
  3. the full FNO2d training-step speedup the kernel buys.
"""

import argparse

import torch

from common import device_auto


def time_us(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters       # ms -> us


def mul_bytes(B, Ci, Co, K):
    return (B * Ci * K + Ci * Co * K + B * Co * K) * 8     # read u, w; write v


def mul_flops(B, Ci, Co, K):
    return 8 * B * Ci * Co * K                             # complex madd


def bench_multiply(cin, cout, kmax, batches, device):
    """Isolated multiply on the 2D mode grid (K = kmax^2), einsum vs kernel."""
    from kernels.spectral_mm import spectral_mm, _load_ext
    _load_ext()
    K = kmax * kmax
    einsum = lambda u, w: torch.einsum("bik,iok->bok", u, w)

    print(f"\n[tinyfno] isolated complex multiply  C_in={cin} C_out={cout} "
          f"K={K} (= {kmax}^2)")
    print("  batch | einsum us | kernel us | speedup | kernel GB/s")
    print("  " + "-" * 52)
    for B in batches:
        u = torch.randn(B, cin, K, dtype=torch.cfloat, device=device)
        w = torch.randn(cin, cout, K, dtype=torch.cfloat, device=device)
        assert (spectral_mm(u, w) - einsum(u, w)).abs().max().item() < 1e-3
        te = time_us(lambda: einsum(u, w))
        tk = time_us(lambda: spectral_mm(u, w))
        gb = mul_bytes(B, cin, cout, K) / (tk * 1e-6) / 1e9
        print(f"  {B:5d} | {te:9.1f} | {tk:9.1f} | {te/tk:6.2f}x | {gb:8.1f}")

    B = batches[-1]
    ai = mul_flops(B, cin, cout, K) / mul_bytes(B, cin, cout, K)
    ridge = 9000.0 / 200.0                  # RTX 3050: ~9 TFLOP/s, ~200 GB/s
    print(f"  arithmetic intensity {ai:.1f} FLOP/byte (ridge ~{ridge:.0f}) -> "
          f"{'memory-bound' if ai < ridge else 'compute-bound'}")


def bench_layer(cin, cout, kmax, batches, device):
    """Full SpectralConv2d forward: einsum vs custom kernel vs neuraloperator."""
    from models.spectral_conv import SpectralConv2d, compl_mul2d
    from kernels.spectral_mm import spectral_mm

    base = SpectralConv2d(cin, cout, kmax, mul_fn=compl_mul2d).to(device)
    kern = SpectralConv2d(cin, cout, kmax, mul_fn=spectral_mm).to(device)
    kern.load_state_dict(base.state_dict())

    impls = [("torch.einsum (base)", lambda x: base(x)),
             ("custom CUDA kernel", lambda x: kern(x))]
    try:
        from neuralop.layers.spectral_convolution import SpectralConv
        ref = SpectralConv(cin, cout, (kmax, kmax)).to(device)
        impls.append(("neuraloperator ref", lambda x: ref(x)))
    except Exception as e:
        print(f"  (neuraloperator unavailable: {e})")

    for B in batches:
        x = torch.randn(B, cin, 64, 64, device=device)
        bytes_ = 2 * mul_bytes(B, cin, cout, kmax * kmax)    # two spectral corners
        print(f"\n[tinyfno] SpectralConv2d benchmark  C_in={cin} C_out={cout} "
              f"k_max={kmax}  batch={B}")
        print("  Implementation       | Latency (us) | BW (GB/s) | vs baseline")
        print("  ---------------------|--------------|-----------|------------")
        t_base = None
        for name, fn in impls:
            t = time_us(lambda: fn(x))
            if t_base is None:
                t_base = t
            bw = bytes_ / (t * 1e-6) / 1e9
            print(f"  {name:20s} | {t:12.1f} | {bw:9.1f} | {t_base/t:9.2f}x")


def bench_training_step(cin, cout, kmax, device):
    """Full FNO2d training-step speedup, kernel mul_fn vs einsum mul_fn."""
    from models.fno import FNO2d
    from models.spectral_conv import compl_mul2d
    from kernels.spectral_mm import spectral_mm
    from common import relative_l2

    x = torch.randn(16, 3, 64, 64, device=device)
    y = torch.randn(16, 1, 64, 64, device=device)
    res = {}
    for name, mul_fn in [("einsum", compl_mul2d), ("kernel", spectral_mm)]:
        torch.manual_seed(0)
        m = FNO2d(3, 1, cout, kmax, 4, 128).to(device)
        for blk in m.blocks:
            blk.spectral.mul_fn = mul_fn
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        def run():
            opt.zero_grad(set_to_none=True)
            relative_l2(m(x), y).backward()
            opt.step()
        res[name] = time_us(run, iters=50, warmup=20)
    print(f"\n  full FNO2d train step:  einsum {res['einsum']/1e3:.2f} ms  "
          f"kernel {res['kernel']/1e3:.2f} ms  "
          f"({res['einsum']/res['kernel']:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cin", type=int, default=32)
    ap.add_argument("--cout", type=int, default=32)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64])
    args = ap.parse_args()

    device = device_auto()
    if device.type != "cuda":
        raise SystemExit("benchmark requires a CUDA device")

    bench_multiply(args.cin, args.cout, args.kmax, args.batches, device)
    bench_layer(args.cin, args.cout, args.kmax, args.batches, device)
    bench_training_step(args.cin, args.cout, args.kmax, device)


if __name__ == "__main__":
    main()
