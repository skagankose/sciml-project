"""Score the numerical-solver rollouts (scripts/solver_rollouts.py) with the models' metric code.

Each solver resolution becomes a baseline next to persistence, under
    outputs/baselines/solver_<resolution>/evaluations/
with the same tables the models write (`rollout_na.parquet`, `physics_na.parquet`, `spectra_na.parquet`) plus
`runs_na.parquet` (one row per rollout: status, wall time, iterations, CFL steps) and `summary_na.json`.
Predictions are brought back to 512 x 128 by spectral interpolation (rbbench.solver_grid) and scored against the
same frames, in physical units, with the upstream VRMSE, Nusselt number and kinetic-energy spectra. A rollout that
diverged scores inf from its first missing step (so it counts as a failure); one that timed out scores NaN there.
`tests/test_solver.py` checks that persistence frames scored here equal persistence scored by `evaluate`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from rbbench import METRIC_VERSION
from rbbench.data import FIELDS, full_trajectory_dataset, list_trajectories, load_trajectory
from rbbench.evaluate import _physical, physics_status, read_grid
from rbbench.manifests import atomic_write_json, now_utc
from rbbench.metrics import physics
from rbbench.metrics.well_metric import vrmse
from rbbench.solver_grid import FIELDS as SOLVER_FIELDS
from rbbench.solver_grid import resample

RUN_ATTRS = ("status", "steps_done", "build_seconds", "step_seconds", "iterations", "dt_min", "dt_max",
             "initial_dt", "cores", "cpu", "hostname", "slurm_job", "dedalus_version", "nx", "ny")


def read_rollout(path: Path, native: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Predicted frames at native resolution [H, X, Y, 4] (NaN where none) and the rollout's attributes."""
    with h5py.File(path, "r") as h:
        pred = np.asarray(h["pred"][...], np.float64)
        attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in h.attrs.items()}
    if pred.shape[1:3] != native:
        pred = np.moveaxis(resample(np.moveaxis(pred, -1, 1), native), 1, -1)
    return pred, attrs


def score_frames(pred_phys: np.ndarray, true_phys: torch.Tensor, base: dict, t0: int, status: str,
                 ops, kappa: float, delta_b: float, spectra_steps: list[int], lx: float,
                 spec_acc: dict) -> tuple[list[dict], list[dict]]:
    """Rollout and physics rows for one rollout, built as `evaluate` builds them (sigma = 0)."""
    horizon = true_phys.shape[0]
    done = np.isfinite(pred_phys).all(axis=(1, 2, 3))
    fill = np.inf if status == "diverged" else np.nan
    safe = np.where(done[:, None, None, None], pred_phys, 0.0)
    v = vrmse(torch.from_numpy(safe).float(), true_phys).numpy()                  # [H, F]
    v[~done] = fill
    rows = [{**base, "start": t0, "sigma": 0.0, "step": k + 1,
             **{f"vrmse_{f}": float(a) for f, a in zip(FIELDS, v[k])}, "vrmse_mean": float(v[k].mean())}
            for k in range(horizon)]
    phys_rows = []
    if ops is None:
        return rows, phys_rows
    tp = true_phys.double().numpy()
    nu_ref = physics.nusselt(tp[..., 0], tp[..., 3], ops, kappa, delta_b)
    nu = physics.nusselt(safe[..., 0], safe[..., 3], ops, kappa, delta_b)
    for k in range(horizon):
        if not done[k]:
            continue
        row = {**base, "start": t0, "sigma": 0.0, "step": k + 1,
               "nu_pred": float(nu["nu_volume"][k]), "nu_true": float(nu_ref["nu_volume"][k]),
               "nu_wall_bottom_true": float(nu_ref["nu_wall_bottom"][k]),
               "nu_wall_top_true": float(nu_ref["nu_wall_top"][k]),
               "nu_wall_bottom_pred": float(nu["nu_wall_bottom"][k])}
        row["nu_abs_err"] = abs(row["nu_pred"] - row["nu_true"])
        row["nu_rel_err"] = row["nu_abs_err"] / max(abs(row["nu_true"]), 1e-12)
        if k + 1 in spectra_steps:
            e_pred = physics.kinetic_energy_spectrum(safe[k, ..., 2], safe[k, ..., 3], ops, lx)[1]
            e_true = physics.kinetic_energy_spectrum(tp[k, ..., 2], tp[k, ..., 3], ops, lx)[1]
            row["log_spectral_error"] = float(physics.log_spectral_error(e_pred, e_true)["log_spectral_error"])
            for src, e in (("pred", e_pred), ("true", e_true)):
                acc = spec_acc[(base["rayleigh"], base["prandtl"], 0.0, k + 1, src)]
                if isinstance(acc[0], float):
                    acc[0] = np.zeros_like(e)
                acc[0] += e
                acc[1] += 1
        phys_rows.append(row)
    return rows, phys_rows


def spectra_rows(spec_acc: dict, n_x: int, lx: float) -> list[dict]:
    k_axis = 2 * np.pi * np.arange(n_x // 2 + 1) / lx
    out = []
    for (ra, pr, s, step, src), (total, n) in spec_acc.items():
        mean = total / n
        out += [{"rayleigh": ra, "prandtl": pr, "sigma": s, "step": step, "source": src, "k_index": i,
                 "k": float(k_axis[i]), "energy": float(e), "n": n} for i, e in enumerate(mean)]
    return out


def score_resolution(rollout_dir: Path, cfg: dict, dataset_dir: Path, cohort: dict, out_dir: Path,
                     audit_path: Path | None = None) -> dict:
    """Score every rollout file of one solver resolution; returns the summary written to summary_na.json."""
    if SOLVER_FIELDS != tuple(FIELDS):
        raise RuntimeError("solver field order differs from the models'")
    files = sorted(Path(rollout_dir).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no solver rollouts in {rollout_dir}")
    n_in, pcfg, ecfg = cfg["task"]["n_steps_input"], cfg["physics"], cfg["eval"]
    horizon = cohort["rollout_steps"]
    fd, _ = full_trajectory_dataset(dataset_dir, "test", n_in)
    refs = {r.id: r for r in list_trajectories(dataset_dir, "test")}
    in_cohort = {e["id"]: set(e["starts"]) for e in cohort["trajectories"]}
    do_physics, physics_note = physics_status(audit_path, pcfg.get("y_grid", "auto"))
    x_grid, y_stored = read_grid(dataset_dir)
    ops = physics.ChebyshevY(physics.y_nodes(y_stored, pcfg), 0.0, pcfg["Ly"]) if do_physics else None
    delta_b = pcfg["b_bottom"] - pcfg["b_top"]
    spectra_steps = [s for s in ecfg["spectra_steps"] if s <= horizon]
    native = (len(x_grid), len(y_stored))

    by_traj: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for f in files:
        with h5py.File(f, "r") as h:
            if not h.attrs.get("complete", False):
                continue
            tid, start = str(h.attrs["traj_id"]), int(h.attrs["start"])
        if start not in in_cohort.get(tid, ()):
            raise ValueError(f"{f}: trajectory {tid} start {start} is not in the cohort")
        by_traj[tid].append((f, start))

    rows, phys_rows, runs = [], [], []
    spec_acc: dict = defaultdict(lambda: [0.0, 0])
    for tid, items in sorted(by_traj.items()):
        ref = refs[tid]
        traj_phys = _physical(fd.norm, load_trajectory(fd, ref, dataset_dir))            # [T, X, Y, F]
        kappa = float(physics.kappa_from(ref.rayleigh, ref.prandtl))
        base = {"traj_id": ref.id, "rayleigh": ref.rayleigh, "prandtl": ref.prandtl}
        for path, t0 in sorted(items, key=lambda it: it[1]):
            pred, attrs = read_rollout(path, native)
            true_phys = traj_phys[t0 + n_in:t0 + n_in + horizon]
            r, p = score_frames(pred, true_phys, base, t0, str(attrs["status"]), ops, kappa, delta_b,
                                spectra_steps, pcfg["Lx"], spec_acc)
            rows += r
            phys_rows += p
            runs.append({**base, "start": t0, **{k: attrs.get(k) for k in RUN_ATTRS},
                         "seconds_per_rollout": float(attrs["build_seconds"]) + float(attrs["step_seconds"])})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roll = pd.DataFrame(rows)
    roll.to_parquet(out_dir / "rollout_na.parquet", index=False)
    pd.DataFrame(runs).to_parquet(out_dir / "runs_na.parquet", index=False)
    if phys_rows:
        pd.DataFrame(phys_rows).to_parquet(out_dir / "physics_na.parquet", index=False)
        pd.DataFrame(spectra_rows(spec_acc, native[0], pcfg["Lx"])).to_parquet(out_dir / "spectra_na.parquet",
                                                                               index=False)
    run_df = pd.DataFrame(runs)
    ok = roll.groupby(["traj_id", "start"]).vrmse_mean.apply(lambda s: bool(np.isfinite(s).all()))
    summary = {
        "metric_version": METRIC_VERSION, "label": "na", "resolution": f"{run_df.nx.iloc[0]}x{run_df.ny.iloc[0]}",
        "cohort_sha256": cohort["sha256"], "n_rollouts": len(run_df), "n_trajectories": int(run_df.traj_id.nunique()),
        "status_counts": run_df.status.value_counts().to_dict(), "all_steps_finite": int(ok.sum()),
        "rollout_vrmse_mean_at": {str(k): float(roll[roll.step == k].vrmse_mean.mean())
                                  for k in ecfg["report_steps"] if k <= horizon},
        "seconds_per_rollout": {"median": float(run_df.seconds_per_rollout.median()),
                                "mean": float(run_df.seconds_per_rollout.mean()),
                                "max": float(run_df.seconds_per_rollout.max())},
        "build_seconds_median": float(run_df.build_seconds.median()),
        "core_hours_total": float((run_df.seconds_per_rollout * run_df.cores.astype(float)).sum() / 3600),
        "physics": physics_note, "scored_utc": now_utc(),
    }
    atomic_write_json(out_dir / "summary_na.json", summary)
    return summary


def score_all(solver_root: Path, cfg: dict, dataset_dir: Path, cohort: dict, output_root: Path,
              audit_path: Path | None = None) -> dict:
    out = {}
    for res_dir in sorted(p for p in Path(solver_root).iterdir() if p.is_dir()):
        out[res_dir.name] = score_resolution(res_dir, cfg, dataset_dir, cohort,
                                             output_root / "baselines" / f"solver_{res_dir.name}" / "evaluations",
                                             audit_path)
    return out
