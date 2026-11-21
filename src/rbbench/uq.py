"""Uncertainty quantification of the forecasts (D30; docs/final_plan.md WP9; course Lecture 17).

An *ensemble* is a list of members that forecast the same windows: the two seeds of a core configuration (point
forecasts, UQ-B), five CNextU-Nets with a variance output trained with the beta-NLL (UQ-A), or a single model or
persistence (for conformal intervals of a point forecast). `evaluate_uq` runs an ensemble over one split, reading
each trajectory once, and writes:

* `summary.parquet`: one row per one-step window and per rollout step, with exact means over all grid points, per
  field (normalised units): the ensemble mean's squared and absolute error (`mse_*`, `mae_*`), the members' mean
  absolute error (`mae_member_*`), the epistemic variance (spread of the members' means, `var_ep_*`), the
  aleatoric variance (mean of their predicted variances, `var_al_*`; 0 for point members), the CRPS of the
  ensemble's predictive distribution (`crps_*`), the coverage of its Gaussian central 50% and 90% intervals
  (`cov50_*`, `cov90_*`), and the ensemble mean's physical-field VRMSE (`vrmse_ens_*`);
* `members.parquet`: every member's physical-field VRMSE (and correlation) per window and step, for every noise
  level: the noise-robustness results (WP8);
* `cells.npz`: each member's error y - mu_m (and predicted std) at a fixed random sample of grid points per window,
  the same points for every ensemble. Split conformal calibration (validation) and coverage (test), PIT and rank
  histograms, binned spread against error, and any grouping (phase, Rayleigh number) are computed from it later;
* `yprofile.npz`: squared error of the ensemble mean and predicted total variance per y row (boundary layers);
* `UQ_COMPLETE.json`: members, checkpoints, split, cohort, noise levels and wall time.

Rollouts: every member feeds back its own mean, as in `rbbench.rollout`, so the spread of the members includes how
their rollouts diverge. A member's variance at a step is its one-step variance given its own inputs; it is not
propagated by sampling (a stated limitation).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rbbench.data import FIELDS, full_trajectory_dataset, list_trajectories, load_trajectory
from rbbench.manifests import atomic_write_json, stable_int
from rbbench.metrics import physics
from rbbench.metrics.well_metric import correlation, vrmse
from rbbench.noise import perturb_window, wall_rows

ONE_STEP_STRIDE = 4            # every 4th window: 49 of 196 per trajectory
CELLS_CLEAN, CELLS_NOISY = 128, 32
YPROFILE_STEPS = (1, 10, 30)
Z50, Z90 = 0.6744897501960817, 1.6448536269514722  # Gaussian central 50% and 90% half-widths
METRICS = ("mse", "mae", "mae_member", "var_ep", "var_al", "crps", "cov50", "cov90", "vrmse_ens")

DistPredictor = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]  # window -> (mean, std | None)


@dataclass
class Member:
    name: str
    predict: DistPredictor
    gaussian: bool


def model_member(name: str, model: torch.nn.Module) -> Member:
    """A trained model; with `forward_dist` (beta-NLL, D30) it also returns a predicted std."""
    model.eval()
    gaussian = hasattr(model, "forward_dist")

    @torch.no_grad()
    def predict(window):
        if gaussian:
            mean, log_var = model.forward_dist(window)
            return mean.float(), (0.5 * log_var.float()).exp()
        return model(window).float(), None

    return Member(name, predict, gaussian)


def persistence_member() -> Member:
    return Member("persistence", lambda window: (window[:, -1].float(), None), False)


@torch.no_grad()
def member_rollout(member: Member, context: torch.Tensor, n_steps: int):
    """Like `rbbench.rollout.rollout`, feeding back the mean: context [B, T, X, Y, F] -> means [B, H, X, Y, F] and
    predicted stds (or None)."""
    window, means, stds = context, [], []
    for _ in range(n_steps):
        mean, std = member.predict(window)
        means.append(mean)
        if std is not None:
            stds.append(std)
        window = torch.cat([window[:, 1:], mean.unsqueeze(1)], dim=1)
    return torch.stack(means, 1), (torch.stack(stds, 1) if stds else None)


# --- scores -------------------------------------------------------------------------------

def _abs_gauss(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """E|Z| for Z ~ N(mu, var), elementwise; |mu| where var = 0."""
    sd = var.clamp_min(0).sqrt()
    z = mu / sd.clamp_min(1e-30)
    val = 2 * sd * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) + mu * (2 * torch.special.ndtr(z) - 1)
    return torch.where(sd > 0, val, mu.abs())


def crps_mixture(y: torch.Tensor, means: torch.Tensor, stds: torch.Tensor | None) -> torch.Tensor:
    """CRPS of the uniform mixture of N(mu_m, sigma_m^2) at y, elementwise (closed form):
    mean_m E|X_m - y| - 1/2 mean_{m,n} E|X_m - X'_n|. With point members (stds None) it is the ensemble CRPS
    mean_m |mu_m - y| - 1/2 mean_{m,n} |mu_m - mu_n|, and with one point member the absolute error (Lecture 17)."""
    m = means.shape[0]
    var = stds ** 2 if stds is not None else torch.zeros_like(means)
    first = sum(_abs_gauss(y - means[i], var[i]) for i in range(m)) / m
    second = sum(_abs_gauss(means[i] - means[j], var[i] + var[j]) for i in range(m) for j in range(m)) / (m * m)
    return first - 0.5 * second


def ensemble_stats(y: torch.Tensor, means: torch.Tensor, stds: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """y [B, X, Y, F] (normalised), member means and stds [M, B, X, Y, F] -> {metric: [B, F]} over all grid points."""
    mu = means.mean(0)
    err = y - mu
    var_ep = means.var(0, unbiased=False)
    var_al = (stds ** 2).mean(0) if stds is not None else torch.zeros_like(mu)
    sd_tot = (var_ep + var_al).sqrt()
    grid = (1, 2)
    return {
        "mse": err.pow(2).mean(grid),
        "mae": err.abs().mean(grid),
        "mae_member": (y.unsqueeze(0) - means).abs().mean((2, 3)).mean(0),
        "var_ep": var_ep.mean(grid),
        "var_al": var_al.mean(grid),
        "crps": crps_mixture(y, means, stds).mean(grid),
        "cov50": (err.abs() <= Z50 * sd_tot).float().mean(grid),
        "cov90": (err.abs() <= Z90 * sd_tot).float().mean(grid),
    }


def sample_cells(traj_id: str, start: int, n_cells: int, n_points: int, k: int) -> np.ndarray:
    """`k` fixed flat grid indices for one window (the same for every ensemble, so comparisons are paired)."""
    rng = np.random.default_rng(stable_int("uq-cells", traj_id, start, k))
    return np.sort(rng.choice(n_points, size=min(k, n_points), replace=False)).astype(np.int64)


# --- the pass -----------------------------------------------------------------------------

def split_entries(dataset_dir: Path, split: str, cohort: dict) -> list[dict]:
    """Test: the frozen cohort (D15). Validation (calibration): every trajectory with the cohort's starts."""
    if split == "test":
        return cohort["trajectories"]
    return [{"id": r.id, "rayleigh": r.rayleigh, "prandtl": r.prandtl, "starts": list(cohort["starts"])}
            for r in list_trajectories(dataset_dir, split)]


class _Collector:
    def __init__(self, members: list[Member], gaussian: bool):
        self.names = [m.name for m in members]
        self.gaussian = gaussian
        self.summary, self.member_rows = [], []
        self.cells = defaultdict(lambda: {"err": [], "std": [], "index": []})
        self.yprof = defaultdict(lambda: {"err2": [], "var": [], "index": []})

    def add(self, kind, base, starts, steps, sigma, y, means, stds, phys, n_cells, traj_row, keep_yprof,
            with_correlation):
        """One batch: y [B, X, Y, F]; means/stds [M, B, X, Y, F]; starts/steps per row of the batch."""
        stats = {k: v.cpu().numpy() for k, v in ensemble_stats(y, means, stds).items()}
        y_phys = phys(y)
        stats["vrmse_ens"] = vrmse(phys(means.mean(0)), y_phys).cpu().numpy()
        member_v = [vrmse(phys(means[i]), y_phys).cpu().numpy() for i in range(means.shape[0])]
        member_c = ([correlation(phys(means[i]), y_phys).cpu().numpy() for i in range(means.shape[0])]
                    if with_correlation and kind == "rollout" else None)
        b, nx, ny, nf = y.shape
        flat_err = (y.unsqueeze(0) - means).reshape(means.shape[0], b, nx * ny, nf)
        flat_std = stds.reshape(stds.shape[0], b, nx * ny, nf) if stds is not None else None
        for j in range(b):
            row = {"kind": kind, **base, "start": int(starts[j]), "step": int(steps[j]), "sigma": float(sigma)}
            for metric, v in stats.items():
                row |= {f"{metric}_{f}": float(a) for f, a in zip(FIELDS, v[j])}
            row["vrmse_ens_mean"] = float(stats["vrmse_ens"][j].mean())
            self.summary.append(row)
            for i, name in enumerate(self.names):
                mrow = {"kind": kind, "member": name, **base, "start": int(starts[j]), "step": int(steps[j]),
                        "sigma": float(sigma)}
                mrow |= {f"vrmse_{f}": float(a) for f, a in zip(FIELDS, member_v[i][j])}
                mrow["vrmse_mean"] = float(member_v[i][j].mean())
                if member_c is not None:
                    mrow |= {f"corr_{f}": float(a) for f, a in zip(FIELDS, member_c[i][j])}
                    mrow["corr_mean"] = float(np.nanmean(member_c[i][j]))
                self.member_rows.append(mrow)
            idx = torch.as_tensor(sample_cells(base["traj_id"], int(starts[j]), n_cells, nx * ny, n_cells),
                                  device=y.device)
            key = (kind, float(sigma))
            self.cells[key]["err"].append(flat_err[:, j, idx].to(torch.float16).cpu().numpy())
            if flat_std is not None:
                self.cells[key]["std"].append(flat_std[:, j, idx].to(torch.float16).cpu().numpy())
            self.cells[key]["index"].append((traj_row, int(starts[j]), int(steps[j])))
        if keep_yprof:
            err2 = (y - means.mean(0)).pow(2).mean(1).cpu().numpy()  # [B, Y, F]
            var = (means.var(0, unbiased=False) + ((stds ** 2).mean(0) if stds is not None else 0)).mean(1)
            var = var.cpu().numpy() if torch.is_tensor(var) else var
            for j in range(b):
                self.yprof[kind]["err2"].append(err2[j])
                self.yprof[kind]["var"].append(var[j])
                self.yprof[kind]["index"].append((traj_row, int(starts[j]), int(steps[j])))

    def write(self, out_dir: Path, traj_ids: list[str]):
        pd.DataFrame(self.summary).to_parquet(out_dir / "summary.parquet", index=False)
        pd.DataFrame(self.member_rows).to_parquet(out_dir / "members.parquet", index=False)
        arrays = {"traj_ids": np.array(traj_ids), "members": np.array(self.names), "fields": np.array(FIELDS)}
        for (kind, sigma), d in self.cells.items():
            tag = f"{kind}__s{sigma:g}"
            arrays[f"err__{tag}"] = np.stack(d["err"], 1)  # [M, N, K, F]
            if d["std"]:
                arrays[f"std__{tag}"] = np.stack(d["std"], 1)
            arrays[f"index__{tag}"] = np.asarray(d["index"], np.int32)  # [N, 3]: trajectory row, start, step
        np.savez_compressed(out_dir / "cells.npz", **arrays)
        prof = {"traj_ids": np.array(traj_ids)}
        for kind, d in self.yprof.items():
            prof[f"err2__{kind}"] = np.stack(d["err2"]).astype(np.float32)  # [N, Y, F]
            prof[f"var__{kind}"] = np.stack(d["var"]).astype(np.float32)
            prof[f"index__{kind}"] = np.asarray(d["index"], np.int32)
        np.savez_compressed(out_dir / "yprofile.npz", **prof)


def evaluate_uq(members: list[Member], cfg: dict, dataset_dir: Path, split: str, cohort: dict, sigmas: list[float],
                out_dir: Path, device: torch.device, meta: dict | None = None, max_trajectories: int | None = None,
                one_step_stride: int = ONE_STEP_STRIDE, cells: tuple[int, int] = (CELLS_CLEAN, CELLS_NOISY),
                with_correlation: bool = True) -> dict:
    """Run one ensemble over `split` (see the module docstring) and write its outputs to `out_dir`."""
    t_pass = time.time()
    gaussian = {m.gaussian for m in members}
    if len(gaussian) != 1:
        raise ValueError("an ensemble mixes members with and without a variance output")
    gaussian = gaussian.pop()
    sigmas = [float(s) for s in sigmas]
    if not sigmas or sigmas[0] != 0.0:
        raise ValueError("the first noise level must be 0 (clean)")
    n_in, horizon = cfg["task"]["n_steps_input"], int(cohort["rollout_steps"])
    pcfg = cfg["physics"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fd, n_total = full_trajectory_dataset(dataset_dir, split, n_in)
    norm = fd.norm
    refs = {r.id: r for r in list_trajectories(dataset_dir, split)}
    entries = split_entries(dataset_dir, split, cohort)
    entries = entries[:max_trajectories] if max_trajectories else entries
    from rbbench.evaluate import read_grid

    _, y_stored = read_grid(dataset_dir, split)
    keep_y = wall_rows(physics.y_nodes(y_stored, pcfg), 0.0, pcfg["Ly"])
    noise_seed = int(cfg["eval"]["noise_seed"])
    bs = int(cfg["eval"]["batch_size"])

    def phys(t):  # physical units, on the CPU (as `evaluate`)
        return norm.denormalize_flattened(t.float().cpu(), "variable")

    col = _Collector(members, gaussian)
    traj_ids = [e["id"] for e in entries]
    for row_i, entry in enumerate(entries):
        ref = refs[entry["id"]]
        traj = load_trajectory(fd, ref, dataset_dir)  # [T, X, Y, F], normalised, CPU
        base = {"traj_id": ref.id, "rayleigh": ref.rayleigh, "prandtl": ref.prandtl}

        # one-step windows (every `one_step_stride`-th), at every noise level
        starts = list(range(0, traj.shape[0] - n_in, one_step_stride))
        for sigma in sigmas:
            n_cells = cells[0] if sigma == 0 else cells[1]
            for i in range(0, len(starts), bs):
                chunk = starts[i:i + bs]
                x = torch.stack([perturb_window(traj[s:s + n_in], ref.id, s, sigma, noise_seed, keep_y)
                                 for s in chunk]).to(device)
                outs = [m.predict(x) for m in members]
                means = torch.stack([o[0] for o in outs])
                stds = torch.stack([o[1] for o in outs]) if gaussian else None
                y = torch.stack([traj[s + n_in] for s in chunk]).to(device).float()
                col.add("one_step", base, chunk, [1] * len(chunk), sigma, y, means, stds, phys, n_cells, row_i,
                        keep_yprof=sigma == 0, with_correlation=False)

        # rollouts from the cohort's starts, all noise levels batched (identical noise for every member)
        for t0 in entry["starts"]:
            ctx = torch.stack([perturb_window(traj[t0:t0 + n_in], ref.id, t0, s, noise_seed, keep_y)
                               for s in sigmas]).to(device)
            outs = [member_rollout(m, ctx, horizon) for m in members]  # [S, H, X, Y, F] each
            means = torch.stack([o[0] for o in outs])  # [M, S, H, X, Y, F]
            stds = torch.stack([o[1] for o in outs]) if gaussian else None
            truth = traj[t0 + n_in:t0 + n_in + horizon].to(device).float()  # [H, X, Y, F]
            for si, sigma in enumerate(sigmas):
                n_cells = cells[0] if sigma == 0 else cells[1]
                mk = means[:, si]  # [M, H, X, Y, F]: the steps play the role of the batch
                sk = stds[:, si] if stds is not None else None
                col.add("rollout", base, [t0] * horizon, list(range(1, horizon + 1)), sigma, truth, mk, sk, phys,
                        n_cells, row_i, keep_yprof=False, with_correlation=with_correlation)
                if sigma == 0:  # y profiles at a few steps only
                    steps = [k for k in YPROFILE_STEPS if k <= horizon]
                    sel = [k - 1 for k in steps]
                    col_kind = col.yprof["rollout"]
                    err2 = (truth[sel] - mk[:, sel].mean(0)).pow(2).mean(1).cpu().numpy()
                    var = mk[:, sel].var(0, unbiased=False)
                    if sk is not None:
                        var = var + (sk[:, sel] ** 2).mean(0)
                    var = var.mean(1).cpu().numpy()
                    for j, k in enumerate(steps):
                        col_kind["err2"].append(err2[j])
                        col_kind["var"].append(var[j])
                        col_kind["index"].append((row_i, t0, k))
        print(f"  {split} {row_i + 1}/{len(entries)} {ref.id}", flush=True)

    col.write(out_dir, traj_ids)
    complete = {"split": split, "members": [m.name for m in members], "gaussian": gaussian, "sigmas": sigmas,
                "noise_seed": noise_seed, "one_step_stride": one_step_stride, "cells": list(cells),
                "rollout_steps": horizon, "starts": list(cohort["starts"]), "cohort_sha256": cohort.get("sha256"),
                "n_trajectories": len(entries), "n_steps_total": int(n_total),
                "wall_seconds": round(time.time() - t_pass, 1), **(meta or {})}
    atomic_write_json(out_dir / "UQ_COMPLETE.json", complete)
    return complete


# --- ensembles ------------------------------------------------------------------------------

NOISE_SIGMAS = (0.0, 0.01, 0.05, 0.10, 0.20)  # configs/experiments/noise.yaml (D17, WP8)


def default_ensembles() -> dict:
    """The ensembles of WP8 and WP9 (`tasks/uq_ensembles.json`, made by `rbbench make-uq-ensembles`).

    * UQ-B: the two seeds of each core configuration at 4 h (tasks/train.jsonl), and persistence; validation clean
      (calibration), test at every noise level (WP8 and the input-shift test).
    * UQ-A: the five beta-NLL CNextU-Nets at 10% and 100% (tasks/uq.jsonl, 1 h), and their MSE point baselines with
      the same 1 h schedule (tasks/train_1h.jsonl at 10%, tasks/train_1h_f100.jsonl at 100%); clean.
    """
    clean = {"valid": [0.0], "test": [0.0]}
    noisy = {"valid": [0.0], "test": list(NOISE_SIGMAS)}
    pairs = [("cnext_unet", 0.1, 0, 1), ("cnext_unet", 1.0, 2, 3), ("tfno", 0.1, 4, 5), ("tfno", 1.0, 6, 7),
             ("avit", 0.1, 8, 9), ("avit", 1.0, 10, 11)]
    ens = [{"name": f"{m}_f{int(round(f * 100)):03d}_4h_seeds", "members": [
        {"manifest": "tasks/train.jsonl", "row": a}, {"manifest": "tasks/train.jsonl", "row": b}],
        "label": "4h", "splits": noisy} for m, f, a, b in pairs]
    ens.append({"name": "persistence", "members": [{"baseline": "persistence"}], "label": None, "splits": noisy})
    for f, rows in ((0.1, range(0, 5)), (1.0, range(5, 10))):
        ens.append({"name": f"cnext_unet_f{int(round(f * 100)):03d}_1h_betanll", "label": "1h", "splits": clean,
                    "members": [{"manifest": "tasks/uq.jsonl", "row": r} for r in rows]})
    for f, manifest in ((0.1, "tasks/train_1h.jsonl"), (1.0, "tasks/train_1h_f100.jsonl")):
        ens.append({"name": f"cnext_unet_f{int(round(f * 100)):03d}_1h_mse", "label": "1h", "splits": clean,
                    "members": [{"manifest": manifest, "row": r} for r in (0, 1)]})
    return {"description": "Ensembles for uncertainty quantification (D30) and noise robustness (D17); "
                           "rbbench eval-uq --index i runs entry i", "ensembles": ens}
