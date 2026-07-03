"""Python wrapper for the custom spectral_mm CUDA kernel.

Exposes `spectral_mm(u, w)`, a drop-in replacement for the einsum complex
channel-mix used by SpectralConv1d / SpectralConv2d. It is a torch.autograd
Function, so it composes with autograd and the rest of the FNO. The extension is
JIT-compiled on first use via torch.utils.cpp_extension.load.

The kernel works on a single flattened mode axis. For 2D the (k1, k2) mode grid
is flattened to one axis before the call and restored after; each mode is an
independent complex matmul, so this is exact.
"""

import os
import shutil
import sys

import torch

def _is_toolkit(path):
    """True if `path` looks like a CUDA toolkit root (has bin/nvcc)."""
    return bool(path) and os.path.isfile(os.path.join(path, "bin", "nvcc"))


def _discover_cuda_home():
    """Find a CUDA toolkit without hardcoding a machine.

    Order: nvcc already on PATH, the active conda env, then sibling conda envs
    (some setups keep nvcc in a separate env from torch). Returns None if no
    toolkit is found; the caller then requires CUDA_HOME to be set manually.
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    cands = [os.environ.get("CONDA_PREFIX"), sys.prefix]
    envs_dir = os.path.dirname(sys.prefix)               # .../envs/<name> -> .../envs
    if os.path.basename(envs_dir) == "envs" and os.path.isdir(envs_dir):
        cands += [os.path.join(envs_dir, e) for e in sorted(os.listdir(envs_dir))]
    return next((c for c in cands if _is_toolkit(c)), None)


def _prepare_build_env():
    """Make the JIT compile work no matter how python was launched.

    cpp_extension.load shells out to the `ninja` and `nvcc` executables, which
    must be on PATH. When python is invoked by full path (no `conda activate`),
    the env's bin dir is not on PATH; pip-installed ninja then looks missing
    ("Ninja is required to load C++ extensions"). We put both on PATH and set
    CUDA_HOME here. This MUST run before importing torch.utils.cpp_extension,
    which snapshots CUDA_HOME at its own import time.
    """
    if shutil.which("ninja") is None:
        try:
            import ninja
            bindir = ninja.BIN_DIR                       # pip ninja's executable
        except Exception:
            bindir = os.path.dirname(sys.executable)     # the env's bin dir
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")

    if not _is_toolkit(os.environ.get("CUDA_HOME")):
        found = _discover_cuda_home()
        if found:
            os.environ["CUDA_HOME"] = found
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and shutil.which("nvcc") is None:
        nvcc_bin = os.path.join(cuda_home, "bin")
        if os.path.isdir(nvcc_bin):
            os.environ["PATH"] = nvcc_bin + os.pathsep + os.environ.get("PATH", "")

    # If the system GCC is newer than what CUDA supports (e.g. GCC 15 vs CUDA
    # 12.9 which tops out at GCC 14), point CC/CXX at the conda-bundled GCC
    # inside the CUDA toolkit env so both the .cpp and .cu host paths compile.
    if cuda_home:
        conda_gxx = os.path.join(cuda_home, "bin", "g++")
        if os.path.isfile(conda_gxx):
            os.environ.setdefault("CC",  os.path.join(cuda_home, "bin", "gcc"))
            os.environ.setdefault("CXX", conda_gxx)


_prepare_build_env()
from torch.utils.cpp_extension import load  # noqa: E402  (after env prep above)

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAME = "spectral_mm_ext"
_SOURCES = [os.path.join(_HERE, "spectral_mm.cu"),
            os.path.join(_HERE, "spectral_mm_binding.cpp")]
_ext = None


def _needs_build():
    """True if the .so is missing or any source is newer than it.

    Mirrors cpp_extension.load's own staleness check, so we can warn before the
    slow nvcc compile rather than appear hung on a fresh checkout or after a
    kernel edit.
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory
        so = os.path.join(_get_build_directory(_NAME, False), _NAME + ".so")
    except Exception:
        return True
    if not os.path.exists(so):
        return True
    so_mtime = os.path.getmtime(so)
    return any(os.path.getmtime(s) > so_mtime for s in _SOURCES)


def _cuda_cflags():
    flags = ["-O2", "-arch=sm_86", "-Xcicc", "-O1"]
    cuda_home = os.environ.get("CUDA_HOME", "")
    conda_gxx = os.path.join(cuda_home, "bin", "g++") if cuda_home else ""
    if conda_gxx and os.path.isfile(conda_gxx):
        flags += ["-ccbin", conda_gxx]
    return flags


def _load_ext():
    global _ext
    if _ext is None:
        building = _needs_build()
        if building:
            print(f"[spectral_mm] building the CUDA kernel with nvcc; the first "
                  f"compile takes a few minutes on this GPU (cicc is slow on the "
                  f"complex-math kernel) and is cached afterward. Streaming "
                  f"build output below...", file=sys.stderr, flush=True)
        _ext = load(
            name=_NAME,
            sources=_SOURCES,
            # Lower the cicc (device NVVM) optimization level: it is the slow
            # pass on this complex-arithmetic kernel and the op is memory-bound,
            # so -O1 device code costs little runtime but cuts compile time a lot.
            extra_cuda_cflags=_cuda_cflags(),
            extra_cflags=["-O3"],
            verbose=building,
        )
    return _ext


class _SpectralMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, w):
        # u: [B, Ci, *modes], w: [Ci, Co, *modes], complex.
        ext = _load_ext()
        b, ci = u.shape[0], u.shape[1]
        co = w.shape[1]
        modes = u.shape[2:]
        uf = u.reshape(b, ci, -1).contiguous()
        wf = w.reshape(ci, co, -1).contiguous()
        v = ext.forward(uf, wf)
        ctx.save_for_backward(uf, wf)
        ctx.modes = modes
        return v.reshape(b, co, *modes)

    @staticmethod
    def backward(ctx, gv):
        ext = _load_ext()
        uf, wf = ctx.saved_tensors
        b, ci = uf.shape[0], uf.shape[1]
        co = wf.shape[1]
        modes = ctx.modes
        gvf = gv.reshape(b, co, -1).contiguous()
        gu, gw = ext.backward(gvf, uf, wf)
        return gu.reshape(b, ci, *modes), gw.reshape(ci, co, *modes)


def spectral_mm(u, w):
    """Complex channel mix v[b,o,...] = sum_i w[i,o,...] u[b,i,...]."""
    return _SpectralMM.apply(u, w)
