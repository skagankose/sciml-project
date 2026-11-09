"""Equal-budget learning-rate search: results, each model's rate, and the pilot timings (decisions D7, D24).

Reads only the search runs' training artifacts (COMPLETE.json, FAILED.json, provenance.json): the
validation loss at the search budget. Nothing here reads test data.

The same runs are the real-data pilots: GPU seconds per optimizer step and the time spent waiting for data,
which give the steps and epochs each model gets in the study's budgets (abstract milestone "budgets fixed
from pilot timings").
"""

from __future__ import annotations

import math
from pathlib import Path

from rbbench.config import load_config
from rbbench.manifests import read_json
from rbbench.subsets import load_subset_ids, subset_filename


def run_results(rows: list[dict], output_root: Path) -> list[dict]:
    """One record per search row: status, validation at the budget, timing."""
    out = []
    for row in rows:
        run = Path(output_root) / "runs" / row["run_id"]
        rec = {"task_index": row["task_index"], "model": row["model"], "lr": row["lr"], "run_id": row["run_id"]}
        prov = read_json(run / "provenance.json") if (run / "provenance.json").exists() else {}
        rec["wall_hours"] = sum(a.get("wall_seconds") or 0.0 for a in prov.get("attempts", [])) / 3600
        if (run / "COMPLETE.json").exists():
            done = read_json(run / "COMPLETE.json")
            ms = max(done["milestones"].values(), key=lambda m: m["target_seconds"])
            steps = ms["step"]
            rec.update(
                status="complete", val_loss=ms["validation"]["val_loss"], val_vrmse=ms["validation"]["val_vrmse"],
                steps=steps, batch_size=ms["samples_seen"] // steps, gpu_s_per_step=ms["train_seconds"] / steps,
                wait_s_per_step=(ms.get("data_wait_seconds") or 0.0) / steps,
                n_train_windows=prov.get("n_train_windows"), n_train_trajectories=prov.get("n_train_trajectories"),
            )
        elif (run / "FAILED.json").exists():
            rec.update(status="diverged", val_loss=math.inf, val_vrmse=math.inf)
        else:
            rec.update(status="running or preempted" if run.exists() else "not started")
        out.append(rec)
    return out


def choose(results: list[dict]) -> dict:
    """Per model: the rate with the lowest validation loss, once every rate of that model has finished.

    A diverged rate counts as infinitely bad. `at_grid_edge` flags a choice at the smallest or largest rate
    tried, where a wider grid might have done better.
    """
    chosen = {}
    for model in dict.fromkeys(r["model"] for r in results):
        mine = [r for r in results if r["model"] == model]
        pending = [r["lr"] for r in mine if r["status"] not in ("complete", "diverged")]
        if pending:
            chosen[model] = {"lr": None, "pending_lrs": pending}
            continue
        finite = [r for r in mine if math.isfinite(r["val_loss"])]
        if not finite:
            chosen[model] = {"lr": None, "pending_lrs": [], "note": "every rate diverged"}
            continue
        best = min(finite, key=lambda r: r["val_loss"])
        grid = sorted(r["lr"] for r in mine)
        chosen[model] = {"lr": best["lr"], "val_loss": best["val_loss"], "val_vrmse": best["val_vrmse"],
                         "at_grid_edge": best["lr"] in (grid[0], grid[-1]), "pending_lrs": []}
    return chosen


def pilot_timings(results: list[dict], tasks_dir: Path, experiment: str = "main") -> list[dict]:
    """Per model: measured step times and what the study's budgets buy, at 10% and 100% data.

    GPU time per step does not depend on the rate, so every completed rate of a model is averaged. The data
    wait is measured on the 10% subset; the 100% estimate assumes the same wait per step.
    """
    out = []
    for model in dict.fromkeys(r["model"] for r in results):
        done = [r for r in results if r["model"] == model and r["status"] == "complete"]
        if not done:
            continue
        gpu = sum(r["gpu_s_per_step"] for r in done) / len(done)
        wait = sum(r["wait_s_per_step"] for r in done) / len(done)
        batch = done[0]["batch_size"]
        per_traj = done[0]["n_train_windows"] / done[0]["n_train_trajectories"]
        rec = {"model": model, "runs": len(done), "batch_size": batch, "gpu_s_per_step": gpu,
               "wait_s_per_step": wait, "gpu_samples_per_s": batch / gpu,
               "wait_fraction_of_wall": wait / (gpu + wait)}
        cfg = load_config(model, experiment)
        for label, seconds in cfg["train"]["budgets_seconds"].items():
            steps = float(seconds) / gpu
            rec[f"steps_{label}"] = steps
            rec[f"est_wall_hours_{label}"] = float(seconds) * (gpu + wait) / gpu / 3600
            for frac in cfg["sweep"]["fractions"]:
                path = Path(tasks_dir) / subset_filename(frac)
                if path.exists():
                    windows = len(load_subset_ids(read_json(path))) * per_traj
                    rec[f"epochs_{label}_f{int(round(frac * 100)):03d}"] = steps * batch / windows
        out.append(rec)
    return out


def overrides_yaml(chosen: dict) -> str:
    """The model_overrides block for configs/experiments/main.yaml."""
    lines = ["model_overrides:"]
    for model, c in chosen.items():
        if c["lr"] is not None:
            lines += [f"  {model}:", "    optim:", f"      lr: {c['lr']:.1e}"]
    return "\n".join(lines)
