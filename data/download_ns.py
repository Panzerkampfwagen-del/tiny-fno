"""Download PDEBench's 2D incompressible Navier-Stokes dataset and convert it
to the (u_in, u_out) HDF5 this project trains on.

PDEBench is hosted on DaRUS (the Zenodo-equivalent institutional repository),
dataset DOI 10.18419/darus-2986. We query its file listing through the DaRUS
dataverse API, pick a 2D incompressible NS file, stream it down, and verify the
MD5 the API reports (this is the checksum source). The files are large (a few
GB), so this is opt-in and separate from training, which by default uses the
self-generated solver in navier_stokes.py.

    python -m data.download_ns --list
    python -m data.download_ns --config configs/navier_stokes.yaml --convert

If you already have a PDEBench file, skip the download and just convert it:
    python -m data.download_ns --raw path/to/file.h5 --convert
"""

import argparse
import hashlib
import json
import os
import urllib.request

import h5py
import numpy as np

from common import load_config

DARUS = "https://darus.uni-stuttgart.de"
DOI = "doi:10.18419/darus-2986"          # PDEBench on DaRUS


def _api(path):
    with urllib.request.urlopen(f"{DARUS}{path}", timeout=60) as r:
        return json.load(r)


def list_files():
    """Return PDEBench files whose names look like 2D incompressible NS."""
    meta = _api(f"/api/datasets/:persistentId/?persistentId={DOI}")
    files = meta["data"]["latestVersion"]["files"]
    out = []
    for f in files:
        name = f["dataFile"]["filename"]
        low = name.lower()
        if "ns_incom" in low or ("incom" in low and "2d" in low):
            out.append((name, f["dataFile"]["id"],
                        f["dataFile"].get("md5", ""),
                        f["dataFile"].get("filesize", 0)))
    return out


def download(file_id, expected_md5, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = f"{DARUS}/api/access/datafile/{file_id}"
    h = hashlib.md5()
    got = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            got += len(chunk)
            print(f"\r  {got/1e6:8.1f} MB", end="", flush=True)
    print()
    digest = h.hexdigest()
    if expected_md5 and digest != expected_md5:
        raise RuntimeError(f"checksum mismatch: got {digest}, want {expected_md5}")
    print(f"[download] saved {dest}  md5={digest}  "
          f"({'verified' if expected_md5 else 'no reference md5'})")


def _vorticity(vx, vy):
    """Vorticity dvy/dx - dvx/dy via spectral derivatives on a periodic grid.

    Fields are laid out [X, Y] (axis -2 is x, axis -1 is y), so the x-derivative
    multiplies the first-axis wavenumber and the y-derivative the last-axis one.
    """
    nx, ny = vx.shape[-2], vx.shape[-1]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=1.0/nx)[:, None]   # d/dx along axis -2
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=1.0/ny)[None, :]   # d/dy along axis -1
    dvy_dx = np.fft.ifft2(1j * kx * np.fft.fft2(vy)).real
    dvx_dy = np.fft.ifft2(1j * ky * np.fft.fft2(vx)).real
    return dvy_dx - dvx_dy


def convert(raw_path, cfg):
    """Convert a PDEBench incompressible NS HDF5 into (u_in, u_out) pairs.

    PDEBench stores velocity with shape [sample, time, x, y, 2]. We derive the
    scalar vorticity and pair frame t_in with frame t_in + t_ahead.
    """
    d = cfg["data"]
    ahead = d["t_ahead"]
    n_train, n_test = d["n_train"], d["n_test"]
    with h5py.File(raw_path, "r") as f:
        key = "velocity" if "velocity" in f else list(f.keys())[0]
        vel = f[key]                                   # [N, T, X, Y, 2]
        n = min(vel.shape[0], n_train + n_test)
        T = vel.shape[1]
        t0 = T // 2
        if t0 + ahead >= T:
            raise ValueError(f"t_ahead={ahead} too large for trajectory length {T}")
        vx = vel[:n, t0, ..., 0]
        vy = vel[:n, t0, ..., 1]
        vx2 = vel[:n, t0 + ahead, ..., 0]
        vy2 = vel[:n, t0 + ahead, ..., 1]
        # One trajectory (sample 0) for rollout: frames `ahead` apart from t0.
        roll_idx = list(range(t0, T, ahead))[:12]
        roll = np.stack([_vorticity(vel[0, ti, ..., 0], vel[0, ti, ..., 1])
                         for ti in roll_idx])
    w_in = np.stack([_vorticity(vx[i], vy[i]) for i in range(len(vx))])
    w_out = np.stack([_vorticity(vx2[i], vy2[i]) for i in range(len(vx2))])

    # The file may hold fewer samples than the requested split; clamp so the
    # stored attrs match what is actually written (load_ns slices on these).
    n_train = min(n_train, len(w_in))
    n_test = min(n_test, len(w_in) - n_train)

    os.makedirs(os.path.dirname(d["path"]) or ".", exist_ok=True)
    with h5py.File(d["path"], "w") as f:
        f.create_dataset("u_in", data=w_in.astype(np.float32))
        f.create_dataset("u_out", data=w_out.astype(np.float32))
        f.create_dataset("rollout", data=roll.astype(np.float32))
        f.attrs.update(dict(n_train=n_train, n_test=n_test,
                            resolution=w_in.shape[-1], t_ahead=ahead, seed=cfg["seed"]))
    print(f"[convert] wrote {d['path']}  pairs={len(w_in)}  "
          f"train={n_train} test={n_test}  rollout={len(roll)}  res={w_in.shape[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/navier_stokes.yaml")
    ap.add_argument("--list", action="store_true", help="list candidate files")
    ap.add_argument("--raw", default=None, help="existing PDEBench HDF5")
    ap.add_argument("--convert", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.list:
        for name, fid, md5, size in list_files():
            print(f"  id={fid}  {size/1e9:6.2f} GB  md5={md5[:12]}...  {name}")
        return

    raw = args.raw or cfg["data"]["raw_path"]
    if not args.raw and not os.path.exists(raw):
        cands = list_files()
        if not cands:
            raise SystemExit("no NS incompressible files found in the dataset")
        cands.sort(key=lambda c: c[3])             # smallest first
        name, fid, md5, size = cands[0]
        print(f"[download] {name}  ({size/1e9:.2f} GB)")
        download(fid, md5, raw)

    if args.convert:
        convert(raw, cfg)


if __name__ == "__main__":
    main()
