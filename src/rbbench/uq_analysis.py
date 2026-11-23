"""Uncertainty analysis of the `eval-uq` outputs (D30, docs/final_plan.md WP8/WP9; course Lecture 17).

Everything here works on the stored samples, never on the data: `cells.npz` holds each member's error y - mu_m (and
predicted std) at 128 fixed random grid points per window (32 under input noise), in normalised units; the exact
per-window means are in `summary.parquet`. Conventions:

* Split conformal (validation -> test): score s = |y - mu(x)| / sigma(x), quantile at level ceil((n+1)(1-a))/n of
  the calibration scores, coverage = share of test scores <= q, width = 2 q sigma(x). Scales sigma(x):
  `constant` (a point forecast: the first member alone), `spread` (the members' standard deviation, floored at the
  calibration set's 5th percentile so a near-zero spread cannot dominate), `gaussian` (the ensemble's predictive
  std, sqrt(var_ep + mean_m sigma_m^2)).
* Rank histogram of M point members: rank = number of members below y (0..M); uniform for a calibrated ensemble.
* PIT of a Gaussian mixture: mean_m Phi((y - mu_m) / sigma_m); uniform for a calibrated forecast.
* Spread-skill ratio: sqrt(mean(c var_ep + var_al)) / RMSE(ensemble mean), c = (M+1)/(M-1) for the population
  variance of M members (a perfect finite ensemble then scores 1).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

FIELDS = ("buoyancy", "pressure", "velocity_x", "velocity_y")  # rbbench.data.FIELDS order
SPREAD_FLOOR_QUANTILE = 0.05


def rayleigh_of(traj_id: str) -> float:
    m = re.search(r"Rayleigh_([0-9.e+-]+)_Prandtl", traj_id)
    return float(m.group(1)) if m else np.nan


def load_cells(path: Path, kind: str, sigma: float = 0.0) -> dict | None:
    """err [M, N, K, F] and std (or None) as float32, index [N, 3] (trajectory row, start, step), traj_ids, members."""
    with np.load(path) as c:
        tag = f"{kind}__s{sigma:g}"
        if f"err__{tag}" not in c.files:
            return None
        return {"err": c[f"err__{tag}"].astype(np.float32),
                "std": c[f"std__{tag}"].astype(np.float32) if f"std__{tag}" in c.files else None,
                "index": c[f"index__{tag}"], "traj_ids": c["traj_ids"].astype(str),
                "members": c["members"].astype(str)}


def window_table(cells: dict, onset: dict[str, int | None] | None = None) -> pd.DataFrame:
    """One row per stored window: trajectory, Rayleigh number, start, step and phase.

    A window is onset if its first input frame precedes the trajectory's t_on (rbbench.onset); for a rollout step k
    from start s, the phase is that of its start (rollouts from 20 are onset, from 80 and 140 developed on test)."""
    idx = cells["index"]
    traj = cells["traj_ids"][idx[:, 0]]
    w = pd.DataFrame({"traj_id": traj, "rayleigh": [rayleigh_of(t) for t in traj], "start": idx[:, 1],
                      "step": idx[:, 2]})
    if onset is not None:
        w["onset"] = [onset.get(t) is None or s < onset[t] for t, s in zip(w.traj_id, w.start)]
    return w


# --- scores ---------------------------------------------------------------------------------------------

def scores(cells: dict, scale: str, floor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Conformal scores and scales [N, K, F] for one of the three choices of sigma(x) (see the module docstring)."""
    err, std = cells["err"], cells["std"]
    if scale == "constant":
        resid, sig = np.abs(err[0]), np.ones_like(err[0])
    elif scale == "spread":
        if err.shape[0] < 2:
            raise ValueError("a spread needs at least two members")
        resid = np.abs(err.mean(0))
        sig = err.std(0)
        if floor is not None:
            sig = np.maximum(sig, floor)
    elif scale == "gaussian":
        if std is None:
            raise ValueError("a Gaussian scale needs predicted stds")
        resid = np.abs(err.mean(0))
        sig = np.sqrt(err.var(0) + (std.astype(np.float64) ** 2).mean(0)).astype(np.float32)
    else:
        raise ValueError(scale)
    return resid / sig, sig


def spread_floor(cells: dict) -> np.ndarray:
    """Per field, the calibration set's 5th percentile of the members' spread (the `spread` scale's floor)."""
    sig = cells["err"].std(0)
    return np.quantile(sig.reshape(-1, sig.shape[-1]), SPREAD_FLOOR_QUANTILE, axis=0)


def conformal_quantile(s: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (inf if that exceeds n)."""
    s = np.asarray(s, np.float64).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.inf) if k > n or n == 0 else float(np.partition(s, k - 1)[k - 1])


def rank_histogram(err: np.ndarray) -> np.ndarray:
    """Share of each rank 0..M: the number of members whose forecast lies below y (err_m = y - mu_m > 0)."""
    m = err.shape[0]
    ranks = (err > 0).sum(0).ravel()
    return np.bincount(ranks, minlength=m + 1)[: m + 1] / ranks.size


def pit_values(err: np.ndarray, std: np.ndarray) -> np.ndarray:
    """PIT of the uniform mixture of the members' Gaussians at y."""
    return ndtr(err.astype(np.float64) / np.maximum(std.astype(np.float64), 1e-12)).mean(0)


def spread_skill(summary: pd.DataFrame, n_members: int, fields=FIELDS) -> float:
    """sqrt(mean(c var_ep + var_al)) / sqrt(mean(mse)) over the rows and fields given (module docstring)."""
    if summary.empty:
        return float("nan")
    c = (n_members + 1) / (n_members - 1) if n_members > 1 else 0.0
    var = sum(c * summary[f"var_ep_{f}"] + summary[f"var_al_{f}"] for f in fields).mean() / len(fields)
    mse = sum(summary[f"mse_{f}"] for f in fields).mean() / len(fields)
    return float(np.sqrt(var / mse)) if mse > 0 else float("nan")


# --- conformal calibration on validation, evaluated on test ---------------------------------------------

def conformal_by_group(cal: dict, test: dict, scale: str, alphas=(0.5, 0.1), cal_mask=None, test_mask=None,
                       group: str = "step", test_cells: dict | None = None) -> pd.DataFrame:
    """Per group (rollout step, or a single pooled group) and field: q-hat on validation, coverage and mean width
    on test. `test_cells` overrides the test samples (e.g. noisy inputs) while `test` gives their windows."""
    floor = spread_floor(cal) if scale == "spread" else None
    s_cal, _ = scores(cal, scale, floor)
    s_te, sig_te = scores(test_cells or test, scale, floor)
    w_cal, w_te = window_table(cal), window_table(test_cells or test)
    mc = np.ones(len(w_cal), bool) if cal_mask is None else np.asarray(cal_mask)
    mt = np.ones(len(w_te), bool) if test_mask is None else np.asarray(test_mask)
    groups = sorted(set(w_te.step[mt])) if group == "step" else [None]
    rows = []
    for g in groups:
        gc = mc & ((w_cal.step == g).to_numpy() if g is not None else True)
        gt = mt & ((w_te.step == g).to_numpy() if g is not None else True)
        for fi, f in enumerate(FIELDS):
            for a in alphas:
                q = conformal_quantile(s_cal[gc][..., fi], a)
                st, sg = s_te[gt][..., fi], sig_te[gt][..., fi]
                rows.append({"step": g, "field": f, "nominal": 1 - a, "q": q, "coverage": float((st <= q).mean()),
                             "width": float(2 * q * sg.mean()), "n_cal": int(gc.sum()), "n_test": int(gt.sum())})
    return pd.DataFrame(rows)


def coverage_with(q_table: pd.DataFrame, test: dict, scale: str, floor=None, test_mask=None) -> pd.DataFrame:
    """Coverage on `test` of the q-hats of another calibration (shifts): q_table rows (step, field, nominal, q);
    a q_table with step None applies its pooled q-hat to every step."""
    s_te, sig_te = scores(test, scale, floor)
    w = window_table(test)
    mt = np.ones(len(w), bool) if test_mask is None else np.asarray(test_mask)
    rows = []
    for step in sorted(set(w.step[mt])):
        g = mt & (w.step == step).to_numpy()
        for fi, f in enumerate(FIELDS):
            for nominal in sorted(set(q_table.nominal)):
                qt = q_table[(q_table.field == f) & (q_table.nominal == nominal)]
                qq = qt[qt.step == step] if qt.step.notna().any() else qt
                if qq.empty:
                    continue
                q = float(qq.q.iloc[0])
                rows.append({"step": step, "field": f, "nominal": nominal, "q": q,
                             "coverage": float((s_te[g][..., fi] <= q).mean()),
                             "width": float(2 * q * sig_te[g][..., fi].mean())})
    return pd.DataFrame(rows)
