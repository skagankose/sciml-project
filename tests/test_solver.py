"""Numerical-solver baseline: grids and resampling (NumPy) and scoring parity with `evaluate`."""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from numpy.polynomial import chebyshev as C

from rbbench.config import load_config
from rbbench.data import FIELDS, full_trajectory_dataset, list_trajectories, load_trajectory
from rbbench.evaluate import _physical, evaluate, make_cohort
from rbbench.manifests import atomic_write_json
from rbbench.metrics import physics
from rbbench.rollout import Persistence
from rbbench.solver_grid import (
    FIELDS as SOLVER_FIELDS,
)
from rbbench.solver_grid import (
    chebyshev_coefficients,
    gauss_nodes,
    resample,
    resample_x,
    resample_y,
    rollout_frames,
    rollout_name,
    solver_cohort,
    x_grid,
)

GRID = "task.spatial_resolution=[64,32]"


def test_field_order_matches_the_models():
    assert tuple(FIELDS) == SOLVER_FIELDS


def test_gauss_nodes_are_the_evaluation_y_grid():
    """The solver's y nodes are the ones the metrics use for rayleigh_benard (D18)."""
    for n in (16, 128):
        cfg = {"y_grid": "gauss", "Ly": 1.0}
        assert np.allclose(gauss_nodes(n), physics.y_nodes(np.linspace(0, 1, n), cfg), atol=1e-14)
        assert np.all(np.diff(gauss_nodes(n)) > 0)


def test_fourier_resampling_is_exact_for_band_limited_fields():
    x_f, x_c = x_grid(64), x_grid(16)
    f = lambda x: 1.3 + np.cos(2 * np.pi * x / 4) - 0.4 * np.sin(2 * np.pi * 5 * x / 4)
    assert np.allclose(resample_x(f(x_f)[:, None], 16, axis=0)[:, 0], f(x_c), atol=1e-12)
    assert np.allclose(resample_x(f(x_c)[:, None], 64, axis=0)[:, 0], f(x_f), atol=1e-12)
    # the coarse grid's Nyquist mode is dropped (Dedalus's RealFourier has none)
    nyq = np.cos(np.pi * np.arange(16))
    assert np.allclose(resample_x(nyq[:, None], 64, axis=0), 0.0, atol=1e-12)
    g = np.random.default_rng(0).standard_normal((8, 5))
    assert np.array_equal(resample_x(g, 8, axis=0), g)


def test_chebyshev_resampling_is_exact_for_polynomials():
    coef = np.array([0.5, -1.0, 0.25, 0.1, -0.05, 0.02])
    s = lambda y: 2 * y - 1  # [0, 1] -> [-1, 1]
    fine, coarse = gauss_nodes(32), gauss_nodes(8)
    vals = C.chebval(s(fine), coef)
    assert np.allclose(resample_y(vals, 8), C.chebval(s(coarse), coef), atol=1e-12)
    assert np.allclose(resample_y(resample_y(vals, 8), 32), vals, atol=1e-12)
    a = chebyshev_coefficients(vals)
    assert np.allclose(a[:6], coef, atol=1e-12) and np.allclose(a[6:], 0.0, atol=1e-12)


def test_resample_round_trip_on_both_axes():
    rng = np.random.default_rng(1)
    coarse = rng.standard_normal((4, 16, 8))
    coarse = resample(resample(coarse, (8, 4)), (16, 8))       # band-limit to what (8, 4) holds
    up = resample(coarse, (64, 32))
    assert np.allclose(resample(up, (16, 8)), coarse, atol=1e-12)


def test_rollout_frames_match_the_models_targets():
    init, targets = rollout_frames(20, 4, 30)
    assert init == 23 and targets[0] == 24 and targets[-1] == 53 and len(targets) == 30


def test_solver_cohort_and_names():
    cohort = {"trajectories": [
        {"id": "data/test/rb_Rayleigh_1e6_Prandtl_1.hdf5#0", "rayleigh": 1e6, "prandtl": 1.0, "starts": [20]},
        {"id": "data/test/rb_Rayleigh_1e6_Prandtl_1.hdf5#1", "rayleigh": 1e6, "prandtl": 1.0, "starts": [20]},
        {"id": "data/test/rb_Rayleigh_1e7_Prandtl_1.hdf5#0", "rayleigh": 1e7, "prandtl": 1.0, "starts": [20]},
    ]}
    assert [e["id"][-1] for e in solver_cohort(cohort)] == ["0", "0"]
    assert len(solver_cohort(cohort, per_regime=2)) == 3
    assert rollout_name("data/test/rb_Rayleigh_1e6_Prandtl_1.hdf5#3", 80) == "rb_Rayleigh_1e6_Prandtl_1__3__s080"


def _write_rollout(path, traj_id, start, pred, status="ok"):
    with h5py.File(path, "w") as h:
        h.create_dataset("pred", data=pred.astype(np.float32))
        attrs = {"traj_id": traj_id, "start": start, "status": status, "complete": True,
                 "steps_done": int(np.isfinite(pred).all(axis=(1, 2, 3)).sum()), "build_seconds": 1.0,
                 "step_seconds": 2.0, "iterations": 10, "dt_min": 0.01, "dt_max": 0.1, "initial_dt": 0.01,
                 "cores": 1, "cpu": "test", "hostname": "test", "slurm_job": "_", "dedalus_version": "none",
                 "nx": pred.shape[1], "ny": pred.shape[2]}
        for k, v in attrs.items():
            h.attrs[k] = v


@pytest.fixture
def persistence_as_solver(dataset_dir, tmp_path):
    """Persistence frames written in the solver's file format, plus `evaluate`'s own persistence scores."""
    from rbbench.solver_grid import rollout_name

    cfg = load_config("cnext_unet", "smoke", "tiny", [GRID])
    fd, total = full_trajectory_dataset(dataset_dir, "test")
    cohort = make_cohort(dataset_dir, cfg["eval"], 4, total)
    audit = tmp_path / "audit.json"
    atomic_write_json(audit, {"physics_checks": {"passed": True, "failures": [], "y_grid_used": "gauss"}})
    evaluate({"na": Persistence()}, cfg, dataset_dir, cohort, tmp_path / "ref", torch.device("cpu"), audit_path=audit)
    refs = {r.id: r for r in list_trajectories(dataset_dir, "test")}
    horizon = cohort["rollout_steps"]
    rollouts = tmp_path / "solver" / "64x32"
    rollouts.mkdir(parents=True)
    for entry in cohort["trajectories"]:
        traj = _physical(fd.norm, load_trajectory(fd, refs[entry["id"]], dataset_dir)).double().numpy()
        for t0 in entry["starts"]:
            pred = np.repeat(traj[t0 + 3][None], horizon, axis=0)
            _write_rollout(rollouts / f"{rollout_name(entry['id'], t0)}.h5", entry["id"], t0, pred)
    return cfg, cohort, audit, tmp_path


def test_solver_scoring_equals_the_models_scoring(dataset_dir, persistence_as_solver):
    """Scored through rbbench.solver_eval, persistence gets the numbers `evaluate` gives it."""
    from rbbench.solver_eval import score_all

    cfg, cohort, audit, root = persistence_as_solver
    summary = score_all(root / "solver", cfg, dataset_dir, cohort, root / "out", audit)["64x32"]
    out = root / "out" / "baselines" / "solver_64x32" / "evaluations"
    keys = ["traj_id", "start", "step"]
    for table, cols in (("rollout", [f"vrmse_{f}" for f in FIELDS] + ["vrmse_mean"]),
                        ("physics", ["nu_pred", "nu_true", "nu_rel_err", "log_spectral_error"])):
        mine = pd.read_parquet(out / f"{table}_na.parquet").sort_values(keys).reset_index(drop=True)
        ref = pd.read_parquet(root / "ref" / f"{table}_na.parquet")
        ref = ref[ref.sigma == 0].sort_values(keys).reset_index(drop=True)   # the smoke cohort also adds noise
        assert len(mine) == len(ref) > 0
        pd.testing.assert_frame_equal(mine[keys], ref[keys])
        for c in cols:
            assert np.allclose(mine[c], ref[c], rtol=1e-4, atol=1e-6, equal_nan=True), (table, c)
    spec_mine = pd.read_parquet(out / "spectra_na.parquet")
    spec_ref = pd.read_parquet(root / "ref" / "spectra_na.parquet").query("sigma == 0")
    assert len(spec_mine) == len(spec_ref)
    spec_keys = ["rayleigh", "prandtl", "step", "source", "k_index"]
    a = spec_mine.sort_values(spec_keys).reset_index(drop=True)
    b = spec_ref.sort_values(spec_keys).reset_index(drop=True)
    assert np.allclose(a.energy, b.energy, rtol=1e-4, atol=1e-12)
    assert summary["n_rollouts"] == sum(len(e["starts"]) for e in cohort["trajectories"])
    assert summary["status_counts"] == {"ok": summary["n_rollouts"]}


def test_diverged_rollouts_count_as_failures(dataset_dir, persistence_as_solver):
    from rbbench.solver_eval import score_all

    cfg, cohort, audit, root = persistence_as_solver
    path = min((root / "solver" / "64x32").glob("*.h5"))
    with h5py.File(path, "r") as h:
        pred = h["pred"][...].astype(np.float64)
        tid, start = h.attrs["traj_id"], int(h.attrs["start"])
    pred[2:] = np.nan
    _write_rollout(path, tid, start, pred, status="diverged")
    score_all(root / "solver", cfg, dataset_dir, cohort, root / "out", audit)
    out = root / "out" / "baselines" / "solver_64x32" / "evaluations"
    roll = pd.read_parquet(out / "rollout_na.parquet")
    mine = roll[(roll.traj_id == tid) & (roll.start == start)].sort_values("step")
    assert np.isfinite(mine.vrmse_mean.iloc[:2]).all() and np.isinf(mine.vrmse_mean.iloc[2:]).all()
    phys = pd.read_parquet(out / "physics_na.parquet")
    assert phys[(phys.traj_id == tid) & (phys.start == start)].step.max() == 2


# The Dedalus side runs only in the rb-dedalus env, which has no torch: `scripts/solver_rollouts.py --selftest`
# (run by cluster/setup_dedalus.sh) checks its grids, restart reproducibility and a resolution change.
