#!/usr/bin/env python
"""Findings ledger for the final report: what every training run cost, how long it took and how its loss
evolved, rebuilt from the pulled outputs so the notes in docs/report_notes.md never drift from the data.

    HPC3_USER=<UCInetID> bash cluster/pull.sh && .venv/bin/python scripts/findings.py

Reads outputs/runs/ (pulled without checkpoints), the task manifests and outputs/baselines/released/.
Writes outputs/findings/:
  attempts.csv        one row per Slurm attempt: job, partition, host, commit, end status, wall time, SU estimate
  runs.csv            one row per run: wall time split into training / data wait / other, steps, epochs,
                      training loss, validation VRMSE at each milestone, attempts, SU estimate
  evaluated.csv       the (run, checkpoint) pairs with a test evaluation on disk
and figures, each as .pdf and .png:
  training (milestone 3)
    training_curves_<manifest>_f<fraction>       training loss, validation VRMSE and learning rate vs training hours
    time_breakdown_<manifest>_f<fraction>        wall time per run: training, waiting for data, other
    data_wait_timeline_<manifest>_f<fraction>    cumulative data wait over UTC time (stalls, crashes)
  pipeline check (milestone 1)
    released_paper_rollout_<split>               released checkpoints rolled out from frame 0 (Table 3)
    released_paper_rollout_by_ra_<split>         the same, per trajectory, by Rayleigh number
    released_one_step_by_frame_<split>           released checkpoints' one-step VRMSE by window
  budgets from pilot timings (milestone 2)
    profile_batch_size                           compute throughput and memory vs batch size, fp32/bf16
    data_loader_benchmark                        samples/s from /pub vs workers, against each model's demand
    pilot_vs_actual_wall_time                    pilot wall-time predictions against the 10% runs
  evaluation (test cohort, train manifest)
    eval_one_step_by_frame_test                  one-step VRMSE by window: ours 1 h / 4 h, released, persistence
    eval_rollout_by_start_<label>_test           rollouts from starts 20 / 80 / 140 against persistence
    eval_paper_rollout_4h_test                   the paper's frame-0 rollouts: ours, released, persistence
    eval_spectra_ratio_4h_test                   predicted / true kinetic-energy spectrum by wavenumber
The learning-rate search has its own figures from `rbbench select-lr` (outputs/lr_search/); tables and the
paper-protocol table come from `rbbench aggregate`. Prints the run table as Markdown.

SU estimate: RCIC charges paid partitions 32 SU per GPU-hour plus 1 SU per core-hour of wall time; free
partitions cost nothing. Attempts before 2026-11-10 did not record their core count, so the 8 cores of
train_a100.sbatch are assumed. An attempt that never closed (crashed, or still running) is timed up to
its last log line, a lower bound. `sacct` gives the billed values.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import findings_figures as figs  # scripts/, next to this file

from rbbench import MODEL_NAMES
from rbbench.aggregate import GRID, INK, MUTED, STYLE, _save, _style_axes
from rbbench.data import FIELDS
from rbbench.manifests import read_json
from rbbench.sweep import read_jsonl

PAID_PARTITIONS = {"gpu"}
SU_PER_GPU_HOUR, SU_PER_CORE_HOUR = 32, 1
ASSUMED_CPUS = 8  # train_a100.sbatch
MANIFESTS = ("train", "lr_search", "replication")
CURVE_MANIFESTS = ("train", "replication")  # select-lr already plots the learning-rate search


def manifest_rows(tasks_dir: Path) -> dict[str, dict]:
    rows = {}
    for name in MANIFESTS:
        path = tasks_dir / f"{name}.jsonl"
        if path.exists():
            for r in read_jsonl(path):
                rows[r["run_id"]] = {**r, "manifest": name}
    return rows


def load_metrics(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw rows, and rows with the steps a resumed attempt replayed taken from that attempt only."""
    raw = pd.read_csv(run_dir / "train_metrics.csv")
    clean = raw
    for _, r in raw[raw.event == "resume"].iterrows():
        clean = clean[~((clean.attempt < r.attempt) & (clean.step > r.step))]
    return raw, clean


def attempt_rows(run_id: str, run_dir: Path, raw: pd.DataFrame) -> list[dict]:
    prov = read_json(run_dir / "provenance.json") if (run_dir / "provenance.json").exists() else {}
    out = []
    for a in prov.get("attempts", []):
        s = a.get("slurm") or {}
        wall = a.get("wall_seconds")
        closed = wall is not None
        if not closed:
            mine = raw[raw.attempt == a.get("attempt")]
            wall = float(mine.wall_seconds.max()) if len(mine) else 0.0
        part = s.get("SLURM_JOB_PARTITION")
        cpus = int(s["SLURM_CPUS_PER_TASK"]) if s.get("SLURM_CPUS_PER_TASK") else ASSUMED_CPUS
        gpus = len(s["SLURM_JOB_GPUS"].split(",")) if s.get("SLURM_JOB_GPUS") else 1
        rate = SU_PER_GPU_HOUR * gpus + SU_PER_CORE_HOUR * cpus if part in PAID_PARTITIONS else 0
        array = s.get("SLURM_ARRAY_JOB_ID")
        out.append({
            "run_id": run_id, "attempt": a.get("attempt"), "job_id": s.get("SLURM_JOB_ID"),
            "array_task": f"{array}_{s.get('SLURM_ARRAY_TASK_ID')}" if array else "",
            "partition": part, "account": s.get("SLURM_JOB_ACCOUNT"), "host": a.get("hostname"),
            "gpu": (a.get("gpu") or {}).get("name"), "commit": ((a.get("git") or {}).get("commit") or "")[:7],
            "started_utc": a.get("time_utc"), "ended_utc": a.get("ended_utc"),
            "end_status": a.get("end_status") or "unclosed (crashed or running)",
            "resumed_from_step": a.get("resumed_from_step"), "wall_hours": wall / 3600,
            "wall_is_lower_bound": not closed, "cpus": cpus, "cpus_assumed": not s.get("SLURM_CPUS_PER_TASK"),
            "su_estimate": wall / 3600 * rate,
        })
    return out


def run_row(run_id: str, run_dir: Path, meta: dict, clean: pd.DataFrame, attempts: list[dict]) -> dict:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    complete = read_json(run_dir / "COMPLETE.json") if (run_dir / "COMPLETE.json").exists() else None
    train_s = complete["milestones"][max(complete["milestones"], key=lambda k: complete["milestones"][k]
                                         ["train_seconds"])]["train_seconds"] if complete else clean.train_seconds.max()
    wait_s = complete["data_wait_seconds"] if complete else clean.data_wait_seconds.max()
    wall_h = sum(a["wall_hours"] for a in attempts)
    train_rows = clean[clean.event == "train"]
    val = clean[clean.val_vrmse.notna()].drop_duplicates("step", keep="last")
    milestones = clean[clean.event.str.startswith("milestone_")]
    status = ("complete" if complete else "failed (diverged)" if (run_dir / "FAILED.json").exists()
              else "partial")
    row = {
        "manifest": meta.get("manifest"), "task_index": meta.get("task_index"), "run_id": run_id,
        "model": cfg["model"]["name"], "fraction": meta.get("fraction"), "seed": meta.get("seed"),
        "lr": cfg["optim"]["lr"], "batch_size": cfg["train"]["batch_size"], "status": status,
        "budget_hours": max(cfg["train"]["budgets_seconds"].values()) / 3600,
        "attempts": len(attempts), "partitions": "+".join(sorted({str(a["partition"]) for a in attempts})),
        "jobs": " ".join(str(a["job_id"]) for a in attempts),
        "wall_hours": wall_h, "train_hours": (train_s or 0) / 3600, "data_wait_hours": (wait_s or 0) / 3600,
        "other_hours": wall_h - ((train_s or 0) + (wait_s or 0)) / 3600,
        "gpu_busy_share": (train_s or 0) / 3600 / wall_h if wall_h else np.nan,
        "steps": int(clean.step.max()), "epochs": int(clean.epoch.max()),
        "samples_seen": int(clean.samples_seen.max()) if clean.samples_seen.notna().any() else None,
        "steps_per_train_second": clean.step.max() / train_s if train_s else np.nan,
        "loss_last": float(train_rows.loss.tail(10).mean()) if len(train_rows) else np.nan,
        "val_vrmse_last": float(val.val_vrmse.iloc[-1]) if len(val) else np.nan,
        "val_vrmse_best": float(val.val_vrmse.min()) if len(val) else np.nan,
        "peak_mem_reserved_gb": float(clean.gpu_mem_reserved_gb.max()),
        "su_estimate": sum(a["su_estimate"] for a in attempts),
        "wall_is_lower_bound": any(a["wall_is_lower_bound"] for a in attempts),
    }
    for _, m in milestones.iterrows():
        row[f"val_vrmse_{m.event.removeprefix('milestone_')}"] = m.val_vrmse
    return row


def _hours_label(h: float) -> str:
    return f"{h:.1f} h"


def curves_figure(runs: pd.DataFrame, metrics: dict[str, pd.DataFrame], title: str, path_stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = [m for m in MODEL_NAMES if m in set(runs.model)]
    fig, axes = plt.subplots(3, len(models), figsize=(3.4 * len(models), 6.6), sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.55]})
    seed_styles = {0: "-", 1: "--", 2: ":"}
    for j, model in enumerate(models):
        st = STYLE[model]
        mine = runs[runs.model == model].sort_values("seed")
        for _, r in mine.iterrows():
            m = metrics[r.run_id]
            ls = seed_styles.get(int(r.seed), "-.")
            tr = m[m.event == "train"]
            smooth = tr.loss.rolling(5, min_periods=1).mean()
            axes[0][j].plot(tr.train_seconds / 3600, smooth, color=st["color"], lw=1.6, ls=ls)
            val = m[m.val_vrmse.notna()].drop_duplicates("step", keep="last")
            axes[1][j].plot(val.train_seconds / 3600, val.val_vrmse, color=st["color"], lw=1.6, ls=ls,
                            marker=st["marker"], ms=4)
            axes[2][j].plot(tr.train_seconds / 3600, tr.lr * 1e3, color=st["color"], lw=1.4, ls=ls)
            if r.status != "complete":
                axes[1][j].annotate(f"s{int(r.seed)} {r.status}", (val.train_seconds.iloc[-1] / 3600,
                                    val.val_vrmse.iloc[-1]), xytext=(4, 4), textcoords="offset points",
                                    fontsize=7, color=MUTED)
        for ms in sorted({float(x) for x in metrics[mine.run_id.iloc[0]].query("event.str.startswith('milestone_')",
                                                                               engine="python").train_seconds}):
            for ax in axes[:, j]:
                ax.axvline(ms / 3600, color=MUTED, lw=0.8, ls=":")
        axes[0][j].set_title(st["label"], fontsize=9, color=INK)
        axes[0][j].set_yscale("log")
        axes[2][j].set_xlabel("training time (h, budget clock)")
        for ax in axes[:, j]:
            _style_axes(ax)
    axes[0][0].set_ylabel("training loss")
    axes[1][0].set_ylabel("validation VRMSE")
    axes[2][0].set_ylabel("learning rate (×1e-3)")
    budget = max(runs.budget_hours)
    for ax in axes.flat:
        ax.set_xlim(0, budget * 1.03)
    handles = [Line2D([], [], color=MUTED, lw=1.6, ls=seed_styles[s], label=f"seed {s}")
               for s in sorted(int(x) for x in set(runs.seed))]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=":", label="milestone checkpoint"))
    axes[0][-1].legend(handles=handles, fontsize=7, frameon=False)
    fig.suptitle(f"{title}. Loss: rolling mean of 5 logs; validation: 64 fixed windows", fontsize=10, color=INK)
    fig.tight_layout()
    written = _save(fig, path_stem.parent, path_stem.name)
    plt.close(fig)
    return written


def time_figure(runs: pd.DataFrame, title: str, path_stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    runs = runs.sort_values(["model", "seed"], key=lambda c: c.map(order) if c.name == "model" else c)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(runs) + 1.2))
    for i, (_, r) in enumerate(runs.iterrows()):
        color = STYLE[r.model]["color"]
        y = len(runs) - 1 - i
        ax.barh(y, r.train_hours, color=color, height=0.62, edgecolor="white", linewidth=2)
        ax.barh(y, r.data_wait_hours, left=r.train_hours, color="white", edgecolor=color, hatch="////",
                height=0.62, linewidth=0)
        ax.barh(y, max(r.other_hours, 0), left=r.train_hours + r.data_wait_hours, color=MUTED, alpha=0.3,
                height=0.62, edgecolor="white", linewidth=2)
        cost = f"{r.su_estimate:,.0f} SU" if r.su_estimate else "free"
        note = "" if r.status == "complete" else f", {r.status}"
        ax.text(r.wall_hours + 0.1, y, f"{r.wall_hours:.1f} h, {cost}{note}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(runs)), [f"{STYLE[r.model]['label']} s{int(r.seed)}" for _, r in runs[::-1].iterrows()],
                  fontsize=8)
    ax.set_xlabel("wall time (h)")
    ax.set_xlim(0, runs.wall_hours.max() * 1.35)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.legend(handles=[Patch(color=MUTED, label="training (GPU busy)"),
                       Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="waiting for data (/pub)"),
                       Patch(color=MUTED, alpha=0.3, label="other: startup, validation, checkpoints, replays")],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    written = _save(fig, path_stem.parent, path_stem.name)
    plt.close(fig)
    return written


def released_rollout_figure(released_dir: Path, split: str, path_stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = {m: pd.read_parquet(released_dir / m / f"paper_rollout_{split}.parquet")
              for m in ("cnext_unet", "tfno") if (released_dir / m / f"paper_rollout_{split}.parquet").exists()}
    if not frames:
        return []
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    top = [fig.add_subplot(gs[0, 0:2])]
    top.append(fig.add_subplot(gs[0, 2:4], sharey=top[0]))
    bottom = [fig.add_subplot(gs[1, 0])]
    bottom += [fig.add_subplot(gs[1, k], sharey=bottom[0]) for k in range(1, 4)]

    def windows(ax):
        for x in (5.5, 12.5):
            ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.axhline(10, color=INK, lw=0.9, ls="--")

    for ax, col, title in ((top[0], "normalised_vrmse_mean", "Field mean, normalised units (the paper's protocol)"),
                           (top[1], "vrmse_mean", "Field mean, physical units (our protocol)")):
        for model, df in frames.items():
            st = STYLE[model]
            g = df.groupby("step")[col]
            ax.fill_between(g.mean().index, g.quantile(0.25), g.quantile(0.75), color=st["color"], alpha=0.15, lw=0)
            ax.plot(g.mean().index, g.mean(), color=st["color"], lw=2, marker=st["marker"], ms=3.5,
                    label=st["label"] if ax is top[0] else None)
        windows(ax)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("rollout step (step 1 = frame 4)")
        _style_axes(ax)
    top[0].set_ylabel("VRMSE")
    ymax = top[0].get_ylim()[1]
    for x, text in ((9, "steps 6–12"), (21.5, "steps 13–30")):
        top[0].text(x, ymax * 0.7, text, ha="center", fontsize=7, color=MUTED)
    top[0].text(30, 10 * 1.15, "paper: > 10", ha="right", fontsize=7, color=INK)
    for ax, field in zip(bottom, FIELDS):
        for model, df in frames.items():
            st = STYLE[model]
            g = df.groupby("step")[f"normalised_vrmse_{field}"].mean()
            ax.plot(g.index, g.values, color=st["color"], lw=1.6, marker=st["marker"], ms=3)
        windows(ax)
        ax.set_yscale("log")
        ax.set_title(f"{field}, normalised units", fontsize=9, color=INK)
        ax.set_xlabel("rollout step")
        _style_axes(ax)
    bottom[0].set_ylabel("VRMSE")
    n = len(next(iter(frames.values())).traj_id.unique())
    fig.suptitle(f"The Well's released checkpoints rolled out from frame 0, {split} split ({n} trajectories). "
                 "Lines: mean over trajectories; bands: middle 50%", fontsize=10, color=INK)
    fig.legend(*top[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    written = _save(fig, path_stem.parent, path_stem.name)
    plt.close(fig)
    return written


def markdown(df: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str]) -> str:
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt.get(c, "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--tasks", default=str(REPO / "tasks"))
    args = ap.parse_args()
    outputs, out = Path(args.outputs), Path(args.outputs) / "findings"
    out.mkdir(parents=True, exist_ok=True)

    meta = manifest_rows(Path(args.tasks))
    attempts, runs, metrics, raws = [], [], {}, {}
    for run_id, row in meta.items():
        run_dir = outputs / "runs" / run_id
        if not (run_dir / "train_metrics.csv").exists():
            continue
        raw, clean = load_metrics(run_dir)
        mine = attempt_rows(run_id, run_dir, raw)
        attempts += mine
        runs.append(run_row(run_id, run_dir, row, clean, mine))
        metrics[run_id], raws[run_id] = clean, raw
    attempts_df, runs_df = pd.DataFrame(attempts), pd.DataFrame(runs)
    attempts_df.to_csv(out / "attempts.csv", index=False)
    runs_df.to_csv(out / "runs.csv", index=False)
    written = [str(out / "attempts.csv"), str(out / "runs.csv")]

    for (manifest, fraction), group in runs_df.groupby(["manifest", "fraction"]):
        if manifest not in CURVE_MANIFESTS:
            continue
        name = f"{manifest}_f{round(fraction * 100):03d}"
        label = f"{round(fraction * 100)}% data ({manifest} manifest)"
        written += curves_figure(group, metrics, f"Training curves, {label}", out / f"training_curves_{name}")
        written += time_figure(group, f"Where the wall time went, {label}", out / f"time_breakdown_{name}")
    for (manifest, fraction), group in runs_df.groupby(["manifest", "fraction"]):
        label = f"{round(fraction * 100)}% data ({manifest} manifest)"
        written += figs.wait_timeline_figure(group, raws, attempts_df, f"Waiting for data, {label}",
                                             out / f"data_wait_timeline_{manifest}_f{round(fraction * 100):03d}")

    # milestone 1: the pipeline check
    for split in ("test", "valid"):
        written += released_rollout_figure(outputs / "baselines" / "released", split,
                                           out / f"released_paper_rollout_{split}")
        written += figs.released_one_step_by_frame(outputs, split, out / f"released_one_step_by_frame_{split}")
        written += figs.released_rollout_by_rayleigh(outputs, split, out / f"released_paper_rollout_by_ra_{split}")

    # milestone 2: budgets from pilot timings
    pilot_timings = outputs / "lr_search" / "pilot_timings.csv"
    written += figs.profile_figure(outputs / "pilots" / "feasibility.csv", out / "profile_batch_size")
    written += figs.data_loader_figure(outputs / "pilots" / "bench_data.json", pilot_timings,
                                       out / "data_loader_benchmark")
    if not runs_df.empty:
        written += figs.pilot_vs_actual_figure(pilot_timings, runs_df, out / "pilot_vs_actual_wall_time")

    # evaluation of the core runs on the frozen test cohort
    evals = figs.evaluated_runs(runs_df, outputs) if not runs_df.empty else pd.DataFrame()
    if not evals.empty:
        evals.drop(columns="dir").to_csv(out / "evaluated.csv", index=False)
        written.append(str(out / "evaluated.csv"))
        written += figs.eval_one_step_by_frame(evals, outputs, out / "eval_one_step_by_frame_test")
        written += figs.eval_paper_rollout(evals, outputs, "4h", out / "eval_paper_rollout_4h_test")
        for lab in sorted(set(evals.label)):
            written += figs.eval_rollout_by_start(evals, outputs, lab, out / f"eval_rollout_by_start_{lab}_test")
        written += figs.eval_spectra_ratio(evals, outputs, "4h", out / "eval_spectra_ratio_4h_test")

    cols = {"manifest": "manifest", "task_index": "row", "model": "model", "fraction": "data", "seed": "seed",
            "lr": "lr", "status": "status", "attempts": "attempts", "partitions": "partition",
            "train_hours": "training h", "data_wait_hours": "data wait h", "other_hours": "other h",
            "wall_hours": "wall h", "gpu_busy_share": "GPU busy", "steps": "steps", "epochs": "epochs",
            "val_vrmse_1h": "val 1 h", "val_vrmse_4h": "val 4 h", "val_vrmse_last": "val last",
            "su_estimate": "SU est."}
    fmt = {"fraction": ".0%", "lr": ".0e", "train_hours": ".2f", "data_wait_hours": ".2f", "other_hours": ".2f",
           "wall_hours": ".2f", "gpu_busy_share": ".0%", "val_vrmse_1h": ".3f", "val_vrmse_4h": ".3f",
           "val_vrmse_last": ".3f", "su_estimate": ",.0f"}
    if not runs_df.empty:
        print(markdown(runs_df.sort_values(["manifest", "task_index"]), cols, fmt))
        paid = attempts_df[attempts_df.su_estimate > 0]
        print(f"\npaid attempts: {len(paid)}, wall {paid.wall_hours.sum():.1f} h, ~{paid.su_estimate.sum():,.0f} SU "
              f"(lower bound where an attempt never closed)")
    print("\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
