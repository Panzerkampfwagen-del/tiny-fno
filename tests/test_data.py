"""Check Burgers data shapes, statistics, and solver determinism."""

import os

import h5py
import numpy as np
import pytest

from common import load_config
from data.burgers import grf_1d, solve_burgers

CFG = load_config("configs/burgers.yaml")
H5 = CFG["data"]["path"]
pytestmark = pytest.mark.filterwarnings("ignore")


def test_grf_is_deterministic_and_unit_std():
    a = grf_1d(16, 256, power=3.5, kmax=50, rng=np.random.default_rng(7))
    b = grf_1d(16, 256, power=3.5, kmax=50, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)                      # same seed -> identical
    assert np.allclose(a.std(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(a).all()


def test_solver_deterministic_and_dissipative():
    u0 = grf_1d(8, 256, 3.5, 50, np.random.default_rng(0))
    traj1, uT1 = solve_burgers(u0, nu=0.01, T=1.0, nt_full=50, n_frames=5)
    traj2, uT2 = solve_burgers(u0, nu=0.01, T=1.0, nt_full=50, n_frames=5)
    assert np.array_equal(uT1, uT2)
    assert np.isfinite(traj1).all()
    # Viscosity removes energy: final energy below initial.
    assert (uT1 ** 2).mean() < (u0 ** 2).mean()


@pytest.mark.skipif(not os.path.exists(H5), reason="run data/burgers.py first")
def test_h5_shapes_and_split():
    d = CFG["data"]
    with h5py.File(H5, "r") as f:
        n = d["n_train"] + d["n_test"]
        assert f["u0"].shape == (n, d["nx"])
        assert f["uT"].shape == (n, d["nx"])
        assert f["traj"].shape == (n, d["nt"] + 1, d["nx"])
        assert f["x"].shape == (d["nx"],)
        assert np.isfinite(f["uT"][:]).all()
        assert int(f.attrs["n_train"]) == d["n_train"]
