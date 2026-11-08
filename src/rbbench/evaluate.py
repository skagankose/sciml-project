"""Identical evaluation for every model, checkpoint and baseline.

Each test trajectory is read once (upstream loader, normalised) and used for:
  * one-step VRMSE on every window (or every `one_step_window_stride`-th),
  * autoregressive rollouts from the frozen cohort starts, clean and with initial-window
    noise at each sigma (identical draws for all models),
  * Nusselt numbers at every rollout step and horizontal KE spectra at selected steps,
    in physical units, only if the dataset audit passed its physics checks.
Several checkpoints of one run can share the pass, so the data is read once.
`evaluate_one_step` runs the same one-step code on any split without a rollout cohort; the
released-checkpoint pipeline check uses it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from rbbench import METRIC_VERSION
from rbbench.config import config_hash
from rbbench.data import (
    FIELDS,
    full_trajectory_dataset,
    list_trajectories,
    load_trajectory,
    split_files,
)
from rbbench.device import autocast_context, synchronize
from rbbench.manifests import atomic_write_json, now_utc, read_json, sha256_json
from rbbench.metrics import physics
from rbbench.metrics.well_metric import vrmse
from rbbench.noise import perturb_window, sigma_key, wall_rows
from rbbench.rollout import rollout

Predictor = Callable[[torch.Tensor], torch.Tensor]


# --- cohort -------------------------------------------------------------------------------

def make_cohort(dataset_dir: Path, eval_cfg: dict, n_steps_input: int, n_steps_total: int) -> dict:
    """Frozen rollout cohort: every test trajectory x configured start indices."""
    horizon = eval_cfg["rollout_steps"]
    starts = [s for s in eval_cfg["rollout_starts"] if s + n_steps_input + horizon <= n_steps_total]
    dropped = sorted(set(eval_cfg["rollout_starts"]) - set(starts))
    refs = list_trajectories(dataset_dir, "test")
    entries = [{"id": r.id, "rayleigh": r.rayleigh, "prandtl": r.prandtl, "starts": starts} for r in refs]
    body = {
        "split": "test", "rollout_steps": horizon, "n_steps_input": n_steps_input,
        "n_steps_total": n_steps_total, "starts": starts, "dropped_starts": dropped,
        "noise_sigmas": eval_cfg["noise_sigmas"], "noise_seed": eval_cfg["noise_seed"],
        "trajectories": entries,
    }
    return {**body, "sha256": sha256_json(body), "created_utc": now_utc()}


def check_cohort(cohort: dict) -> None:
    body = {k: v for k, v in cohort.items() if k not in ("sha256", "created_utc")}
    if sha256_json(body) != cohort["sha256"]:
        raise ValueError("cohort file was modified after it was frozen")


# --- helpers ------------------------------------------------------------------------------

def read_grid(dataset_dir: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(split_files(dataset_dir, split)[0], "r") as h:
        return np.asarray(h["dimensions/x"][...], np.float64), np.asarray(h["dimensions/y"][...], np.float64)


def physics_status(audit_path: Path | None, y_grid: str = "auto") -> tuple[bool, str]:
    if audit_path is None or not Path(audit_path).exists():
        return False, f"no dataset audit at {audit_path}; physics diagnostics skipped"
    audit = read_json(audit_path)
    checks = audit.get("physics_checks", {})
    if not checks.get("passed"):
        return False, f"dataset audit physics checks did not pass: {checks.get('failures')}"
    if y_grid != "auto" and checks.get("y_grid_used") != y_grid:
        return False, f"dataset audit checked y grid {checks.get('y_grid_used')!r}, evaluation uses {y_grid!r}"
    return True, "audit passed"


def model_predictor(model: torch.nn.Module, device: torch.device, precision: str) -> Predictor:
    model.eval()

    @torch.no_grad()
    def predict(window: torch.Tensor) -> torch.Tensor:
        with autocast_context(device, precision):
            return model(window).float()

    return predict


@torch.no_grad()
def measure_latency(predict: Predictor, window: torch.Tensor, device: torch.device,
                    warmup: int, iters: int, rollout_steps: int) -> dict:
    """Batch-1 latency per predicted frame and for one full rollout, plus peak memory."""
    window = window[:1].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        predict(window)
    synchronize(device)
    t = time.perf_counter()
    for _ in range(iters):
        predict(window)
    synchronize(device)
    per_frame = (time.perf_counter() - t) / iters
    t = time.perf_counter()
    rollout(predict, window, rollout_steps)
    synchronize(device)
    out = {
        "batch_size": 1, "ms_per_frame": 1e3 * per_frame,
        "ms_per_rollout": 1e3 * (time.perf_counter() - t), "rollout_steps": rollout_steps,
        "warmup": warmup, "iters": iters, "device": str(device),
    }
    if device.type == "cuda":
        out["peak_mem_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        out["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
        out["gpu_name"] = torch.cuda.get_device_name(device)
    return out


def _physical(norm, t: torch.Tensor) -> torch.Tensor:
    return norm.denormalize_flattened(t.float(), "variable")


def one_step_rows(predict: Predictor, traj: torch.Tensor, traj_phys: torch.Tensor, base: dict, n_in: int,
                  stride: int, batch_size: int, device: torch.device, norm,
                  paper_protocols: bool = False) -> list[dict]:
    """VRMSE (physical units) of every `stride`-th one-step window of one normalised trajectory.

    `paper_protocols` adds the paper-era scores used for the released checkpoints (decisions D20):
    `paper_valid_*` scores the prediction in normalised units against the normalised target;
    `paper_test_*` feeds the raw (physical) window, as the_well v1.0.x's unnormalised test set did, and
    scores the raw output against the raw target.
    """
    rows = []
    starts = list(range(0, traj.shape[0] - n_in, stride))
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        x = torch.stack([traj[s:s + n_in] for s in chunk]).to(device)
        out = predict(x).cpu()
        pred = _physical(norm, out)
        true = torch.stack([traj_phys[s + n_in] for s in chunk])
        scores = {"vrmse": vrmse(pred, true).numpy()}
        if paper_protocols:
            scores["paper_valid_vrmse"] = vrmse(out, torch.stack([traj[s + n_in] for s in chunk]).float()).numpy()
            raw = torch.stack([traj_phys[s:s + n_in] for s in chunk]).to(device)
            scores["paper_test_vrmse"] = vrmse(predict(raw).cpu(), true).numpy()
        for j, s in enumerate(chunk):
            row = {**base, "start": s}
            for key, v in scores.items():
                row |= {f"{key}_{f}": float(a) for f, a in zip(FIELDS, v[j])} | {f"{key}_mean": float(v[j].mean())}
            rows.append(row)
    return rows


def failure_step(curve: np.ndarray, threshold: float) -> int | None:
    """First 1-based step whose VRMSE exceeds `threshold`; None if never (right-censored)."""
    above = np.nonzero(np.asarray(curve) > threshold)[0]
    return int(above[0]) + 1 if above.size else None


def kaplan_meier(events: list[int | None], horizon: int) -> np.ndarray:
    """S(k) for k = 1..horizon; None means censored at `horizon` (no failure observed)."""
    surv = np.ones(horizon)
    at_risk = len(events)
    s = 1.0
    times = sorted(e for e in events if e is not None)
    for k in range(1, horizon + 1):
        d = sum(1 for e in times if e == k)
        if at_risk > 0 and d:
            s *= 1 - d / at_risk
        surv[k - 1] = s
        at_risk -= d
    return surv


def km_median(surv: np.ndarray) -> int | None:
    idx = np.nonzero(surv <= 0.5)[0]
    return int(idx[0]) + 1 if idx.size else None


# --- main pass ----------------------------------------------------------------------------

def evaluate(
    predictors: dict[str, Predictor],
    cfg: dict,
    dataset_dir: Path,
    cohort: dict,
    out_dir: Path,
    device: torch.device,
    audit_path: Path | None = None,
    meta: dict | None = None,
    max_trajectories: int | None = None,
) -> dict[str, dict]:
    """Evaluate each labelled predictor on the same data and write per-label outputs."""
    t_pass = time.time()
    check_cohort(cohort)
    ecfg, pcfg = cfg["eval"], cfg["physics"]
    n_in = cfg["task"]["n_steps_input"]
    horizon = cohort["rollout_steps"]
    sigmas = [float(s) for s in cohort["noise_sigmas"]]
    if sigmas[0] != 0.0:
        raise ValueError("first noise level must be 0 (clean)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fd, n_total = full_trajectory_dataset(dataset_dir, "test", n_in)
    if n_total != cohort["n_steps_total"]:
        raise RuntimeError(f"test trajectories have {n_total} steps, cohort expects {cohort['n_steps_total']}")
    norm = fd.norm
    refs = {r.id: r for r in list_trajectories(dataset_dir, "test")}
    entries = cohort["trajectories"][:max_trajectories] if max_trajectories else cohort["trajectories"]

    do_physics, physics_note = physics_status(audit_path, pcfg.get("y_grid", "auto"))
    x_grid, y_stored = read_grid(dataset_dir)
    y_grid = physics.y_nodes(y_stored, pcfg)  # the stored y is a placeholder for rayleigh_benard (D18)
    ops = physics.ChebyshevY(y_grid, 0.0, pcfg["Ly"]) if do_physics else None
    keep_y = wall_rows(y_grid, 0.0, pcfg["Ly"])
    delta_b = pcfg["b_bottom"] - pcfg["b_top"]
    spectra_steps = [s for s in ecfg["spectra_steps"] if s <= horizon]

    rows = {lab: defaultdict(list) for lab in predictors}
    spec_acc = {lab: defaultdict(lambda: [0.0, 0]) for lab in predictors}
    stride = ecfg["one_step_window_stride"]
    bs = ecfg["batch_size"]
    latency_window = None

    def phys(t: torch.Tensor) -> torch.Tensor:
        return _physical(norm, t)

    for entry in entries:
        ref = refs[entry["id"]]
        traj = load_trajectory(fd, ref, dataset_dir)  # [T, X, Y, F], normalised, CPU
        traj_phys = phys(traj)
        kappa = float(physics.kappa_from(ref.rayleigh, ref.prandtl))
        base = {"traj_id": ref.id, "rayleigh": ref.rayleigh, "prandtl": ref.prandtl}
        if latency_window is None:
            latency_window = traj[None, :n_in]

        # one-step windows
        for lab, predict in predictors.items():
            rows[lab]["one_step"] += one_step_rows(predict, traj, traj_phys, base, n_in, stride, bs, device, norm)

        # rollouts (all sigmas batched together; identical noise for every predictor)
        reference_physics: dict[int, dict] = {}
        for t0 in entry["starts"]:
            ctx = torch.stack(
                [perturb_window(traj[t0:t0 + n_in], ref.id, t0, s, cohort["noise_seed"], keep_y) for s in sigmas]
            )
            true_phys = traj_phys[t0 + n_in:t0 + n_in + horizon]  # [H, X, Y, F]
            if do_physics:
                tp = true_phys.double().numpy()
                reference_physics[t0] = physics.nusselt(tp[..., 0], tp[..., 3], ops, kappa, delta_b)
                ref_spec = {k: physics.kinetic_energy_spectrum(tp[k - 1, ..., 2], tp[k - 1, ..., 3], ops, pcfg["Lx"])[1]
                            for k in spectra_steps}
            for lab, predict in predictors.items():
                preds = rollout(predict, ctx.to(device), horizon).cpu()  # [S, H, X, Y, F]
                pred_phys = phys(preds)
                v = vrmse(pred_phys, true_phys.unsqueeze(0).expand_as(pred_phys)).numpy()  # [S, H, F]
                for si, s in enumerate(sigmas):
                    for k in range(horizon):
                        rows[lab]["rollout"].append(
                            {**base, "start": t0, "sigma": s, "step": k + 1,
                             **{f"vrmse_{f}": float(a) for f, a in zip(FIELDS, v[si, k])},
                             "vrmse_mean": float(v[si, k].mean())}
                        )
                if not do_physics:
                    continue
                pp = pred_phys.double().numpy()
                nu = physics.nusselt(pp[..., 0], pp[..., 3], ops, kappa, delta_b)  # arrays [S, H]
                nu_ref = reference_physics[t0]
                for si, s in enumerate(sigmas):
                    for k in range(horizon):
                        row = {**base, "start": t0, "sigma": s, "step": k + 1,
                               "nu_pred": float(nu["nu_volume"][si, k]), "nu_true": float(nu_ref["nu_volume"][k]),
                               "nu_wall_bottom_true": float(nu_ref["nu_wall_bottom"][k]),
                               "nu_wall_top_true": float(nu_ref["nu_wall_top"][k]),
                               "nu_wall_bottom_pred": float(nu["nu_wall_bottom"][si, k])}
                        row["nu_abs_err"] = abs(row["nu_pred"] - row["nu_true"])
                        row["nu_rel_err"] = row["nu_abs_err"] / max(abs(row["nu_true"]), 1e-12)
                        if k + 1 in spectra_steps:
                            e_pred = physics.kinetic_energy_spectrum(pp[si, k, ..., 2], pp[si, k, ..., 3], ops, pcfg["Lx"])[1]
                            e_true = ref_spec[k + 1]
                            row["log_spectral_error"] = float(physics.log_spectral_error(e_pred, e_true)["log_spectral_error"])
                            for src, e in (("pred", e_pred), ("true", e_true)):
                                acc = spec_acc[lab][(ref.rayleigh, ref.prandtl, s, k + 1, src)]
                                if isinstance(acc[0], float):
                                    acc[0] = np.zeros_like(e)
                                acc[0] += e
                                acc[1] += 1
                        rows[lab]["physics"].append(row)

    # latency/memory, once per predictor, after the data pass
    summaries = {}
    for lab, predict in predictors.items():
        lat = measure_latency(predict, latency_window, device, ecfg["latency_warmup"], ecfg["latency_iters"], horizon)
        tables = {name: pd.DataFrame(r) for name, r in rows[lab].items()}
        for name, df in tables.items():
            df.to_parquet(out_dir / f"{name}_{lab}.parquet", index=False)
        if spec_acc[lab]:
            spec_rows = []
            k_axis = 2 * np.pi * np.arange(len(x_grid) // 2 + 1) / pcfg["Lx"]
            for (ra, pr, s, step, src), (total, n) in spec_acc[lab].items():
                mean = total / n
                spec_rows += [{"rayleigh": ra, "prandtl": pr, "sigma": s, "step": step, "source": src,
                               "k_index": i, "k": float(k_axis[i]), "energy": float(e), "n": n}
                              for i, e in enumerate(mean)]
            pd.DataFrame(spec_rows).to_parquet(out_dir / f"spectra_{lab}.parquet", index=False)
        summaries[lab] = summarize(lab, tables, ecfg, sigmas, horizon, lat, cohort, cfg, physics_note, meta or {})
        for s in sigmas:
            name = "clean" if s == 0 else f"noise_{sigma_key(s)}"
            atomic_write_json(out_dir / f"{name}_{lab}.json", summaries[lab]["by_sigma"][sigma_key(s)])
    # Markers last, each with the wall time of the whole pass it shared with the other labels
    # (aggregation counts it once per run as evaluation GPU-hours).
    pass_info = {"wall_seconds": round(time.time() - t_pass, 3), "labels": list(predictors), "device": str(device)}
    for lab in predictors:
        summaries[lab]["pass"] = pass_info
        atomic_write_json(out_dir / f"EVAL_COMPLETE_{lab}.json", {k: v for k, v in summaries[lab].items() if k != "by_sigma"})
    return summaries


def evaluate_one_step(
    predictors: dict[str, Predictor],
    dataset_dir: Path,
    split: str,
    n_steps_input: int,
    eval_cfg: dict,
    device: torch.device,
    max_trajectories: int | None = None,
    paper_protocols: bool = False,
) -> dict[str, pd.DataFrame]:
    """One-step VRMSE of every window of one split for each predictor, reading the data once."""
    fd, _ = full_trajectory_dataset(dataset_dir, split, n_steps_input)
    refs = list_trajectories(dataset_dir, split)
    refs = refs[:max_trajectories] if max_trajectories else refs
    rows: dict[str, list[dict]] = {lab: [] for lab in predictors}
    for ref in refs:
        traj = load_trajectory(fd, ref, dataset_dir)
        traj_phys = _physical(fd.norm, traj)
        base = {"traj_id": ref.id, "rayleigh": ref.rayleigh, "prandtl": ref.prandtl}
        for lab, predict in predictors.items():
            rows[lab] += one_step_rows(predict, traj, traj_phys, base, n_steps_input,
                                       eval_cfg["one_step_window_stride"], eval_cfg["batch_size"], device, fd.norm,
                                       paper_protocols)
    return {lab: pd.DataFrame(r) for lab, r in rows.items()}


def summarize(lab, tables, ecfg, sigmas, horizon, latency, cohort, cfg, physics_note, meta) -> dict:
    one = tables["one_step"]
    roll = tables["rollout"]
    phys_df = tables.get("physics")
    by_sigma = {}
    for s in sigmas:
        r = roll[roll.sigma == s]
        curve = r.groupby("step")["vrmse_mean"].mean()
        per_rollout = r.pivot_table(index=["traj_id", "start"], columns="step", values="vrmse_mean")
        failures = {}
        for thr in ecfg["thresholds"]:
            events = [failure_step(row.values, thr) for _, row in per_rollout.iterrows()]
            surv = kaplan_meier(events, horizon)
            failures[f"{thr:g}"] = {
                "km_median_step": km_median(surv),
                "censored_fraction": float(np.mean([e is None for e in events])),
                "survival_at": {str(k): float(surv[k - 1]) for k in ecfg["report_steps"] if k <= horizon},
                "n_rollouts": len(events),
            }
        entry = {
            "label": lab, "sigma": s, "metric_version": METRIC_VERSION,
            "rollout_vrmse_mean_at": {str(k): float(curve.loc[k]) for k in ecfg["report_steps"] if k in curve.index},
            "rollout_vrmse_per_field_at": {
                str(k): {f: float(r[r.step == k][f"vrmse_{f}"].mean()) for f in FIELDS}
                for k in ecfg["report_steps"] if k in curve.index
            },
            "failure": failures, "meta": meta,
        }
        if s == 0:
            entry["one_step"] = {
                "vrmse_mean": float(one["vrmse_mean"].mean()),
                **{f"vrmse_{f}": float(one[f"vrmse_{f}"].mean()) for f in FIELDS},
                "n_windows": len(one),
            }
            entry["latency"] = latency
        if phys_df is not None and len(phys_df):
            p = phys_df[phys_df.sigma == s]
            entry["nusselt"] = {
                "abs_err_mean_at": {str(k): float(p[p.step == k]["nu_abs_err"].mean()) for k in ecfg["report_steps"]
                                    if k in set(p.step)},
                "rel_err_mean_at": {str(k): float(p[p.step == k]["nu_rel_err"].mean()) for k in ecfg["report_steps"]
                                    if k in set(p.step)},
            }
            if "log_spectral_error" in p:
                lse = p.dropna(subset=["log_spectral_error"]).groupby("step")["log_spectral_error"].mean()
                entry["log_spectral_error_at"] = {str(k): float(v) for k, v in lse.items()}
        else:
            entry["physics"] = physics_note
        by_sigma[sigma_key(s)] = entry
    return {
        "label": lab, "evaluation_completed": True, "metric_version": METRIC_VERSION,
        "cohort_sha256": cohort["sha256"], "eval_config_sha256": sha256_json(cfg["eval"]),
        "config_sha256": config_hash(cfg), "physics": physics_note, "n_one_step_windows": len(one),
        "n_rollouts": int(len(roll) / max(len(sigmas) * horizon, 1)), "meta": meta,
        "completed_utc": now_utc(), "by_sigma": by_sigma,
    }
