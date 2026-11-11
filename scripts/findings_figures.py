"""Figures for the findings ledger (scripts/findings.py), grouped by the report section they illustrate.

Pipeline check (milestone 1): released checkpoints' one-step error by start frame, frame-0 rollouts by
Rayleigh number. Budgets (milestone 2): batch-size profile, data-loader benchmark, pilot predictions against
the 10% runs. Training (milestone 3): cumulative data wait over wall-clock time. Evaluation: one-step error by
start frame, rollouts by start, the paper's frame-0 rollouts and kinetic-energy spectrum ratios, for every
evaluated run beside the released checkpoints and persistence.

Each function returns the written paths (pdf and png) or [] when its inputs are missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rbbench import MODEL_NAMES
from rbbench.aggregate import INK, MUTED, ONSET_FRAMES, STYLE, _save, _style_axes

PERSIST = STYLE["persistence"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, stem: Path) -> list[str]:
    written = _save(fig, stem.parent, stem.name)
    _plt().close(fig)
    return written


def _models(names) -> list[str]:
    present = set(names)
    return [m for m in MODEL_NAMES if m in present]


# --- evaluated runs ---------------------------------------------------------------------------

def evaluated_runs(runs: pd.DataFrame, outputs: Path, manifest: str = "train") -> pd.DataFrame:
    """One row per (run, checkpoint label) with an evaluation on disk."""
    rows = []
    for _, r in runs[runs.manifest == manifest].iterrows():
        ev = outputs / "runs" / r.run_id / "evaluations"
        for done in sorted(ev.glob("EVAL_COMPLETE_*.json")):
            label = done.stem.removeprefix("EVAL_COMPLETE_")
            rows.append({"run_id": r.run_id, "model": r.model, "seed": int(r.seed), "fraction": r.fraction,
                         "label": label, "dir": ev})
    return pd.DataFrame(rows)


def _load(evals: pd.DataFrame, kind: str, label: str) -> dict[str, list[pd.DataFrame]]:
    """Per model, the `kind` tables of every seed evaluated at `label`."""
    out: dict[str, list[pd.DataFrame]] = {}
    for _, e in evals[evals.label == label].iterrows():
        p = e.dir / f"{kind}_{label}.parquet"
        if p.exists():
            out.setdefault(e.model, []).append(pd.read_parquet(p))
    return out


def _seed_mean(tables: list[pd.DataFrame], by: str, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mean over trajectories per `by` for each seed; returns the seed mean, min and max."""
    per_seed = pd.concat([t.groupby(by)[col].mean() for t in tables], axis=1)
    return per_seed.mean(axis=1), per_seed.min(axis=1), per_seed.max(axis=1)


def _persistence(outputs: Path, kind: str) -> pd.DataFrame | None:
    p = outputs / "baselines" / "persistence" / "evaluations" / f"{kind}_na.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _released(outputs: Path, model: str, kind: str, split: str = "test") -> pd.DataFrame | None:
    p = outputs / "baselines" / "released" / model / f"{kind}_{split}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _onset_marker(ax, y_text: float | None = None):
    ax.axvline(ONSET_FRAMES, color=MUTED, lw=0.8, ls=":")
    if y_text is not None:
        ax.text(ONSET_FRAMES - 3, y_text, "onset", ha="right", va="top", fontsize=7, color=MUTED)
        ax.text(ONSET_FRAMES + 3, y_text, "developed convection", ha="left", va="top", fontsize=7, color=MUTED)


# --- milestone 1: pipeline check ----------------------------------------------------------------

def released_one_step_by_frame(outputs: Path, split: str, stem: Path) -> list[str]:
    plt = _plt()
    from matplotlib.lines import Line2D

    tables = {m: t for m in ("cnext_unet", "tfno") if (t := _released(outputs, m, "one_step", split)) is not None}
    if not tables:
        return []
    persist = _persistence(outputs, "one_step") if split == "test" else None
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    for ax, col, title in ((axes[0], "normalised_vrmse_mean", "Normalised units (the paper's protocol)"),
                           (axes[1], "vrmse_mean", "Physical units (our protocol)")):
        for model, t in tables.items():
            st = STYLE[model]
            g = t.groupby("start")[col]
            ax.fill_between(g.mean().index, g.quantile(0.25), g.quantile(0.75), color=st["color"], alpha=0.15, lw=0)
            ax.plot(g.mean().index, g.mean(), color=st["color"], lw=1.8)
        if persist is not None:
            g = persist.groupby("start")[col].mean()
            ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.4, ls=":")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("first input frame of the window")
        _style_axes(ax)
        _onset_marker(ax, y_text=ax.get_ylim()[1] * 0.8)
    axes[0].set_ylabel("one-step VRMSE (field mean)")
    pub = json.loads((outputs / "baselines" / "released" / "released_check.json").read_text())["results"]
    handles = []
    for model, t in tables.items():
        res = pub.get(model, {}).get(split, {})
        note = f"all-window mean {t.normalised_vrmse_mean.mean():.3f} normalised, {t.vrmse_mean.mean():.3f} physical"
        if res.get("published_vrmse_mean") is not None and split == "test":
            note += f"; paper {res['published_vrmse_mean']:.3f}"
        handles.append(Line2D([], [], color=STYLE[model]["color"], lw=1.8,
                              label=f"released {STYLE[model]['label']} ({note})"))
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], lw=1.4, ls=":",
                              label=f"persistence ({persist.normalised_vrmse_mean.mean():.3f} / "
                                    f"{persist.vrmse_mean.mean():.3f})"))
    n = next(iter(tables.values())).traj_id.nunique()
    fig.suptitle(f"The Well's released checkpoints, one-step error by window, {split} split ({n} trajectories). "
                 "Lines: mean over trajectories; bands: middle 50%", fontsize=10, color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _finish(fig, stem)


def released_rollout_by_rayleigh(outputs: Path, split: str, stem: Path) -> list[str]:
    plt = _plt()
    from matplotlib.patches import Patch

    tables = {m: t for m in ("cnext_unet", "tfno") if (t := _released(outputs, m, "paper_rollout", split)) is not None}
    if not tables:
        return []
    persist = _persistence(outputs, "paper_rollout") if split == "test" else None
    if persist is not None:
        tables["persistence"] = persist
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    for ax, (lo, hi) in zip(axes, ((6, 12), (13, 30))):
        ras = sorted(next(iter(tables.values())).rayleigh.unique())
        width = 0.8 / len(tables)
        for j, (model, t) in enumerate(tables.items()):
            per_traj = t[t.step.between(lo, hi)].groupby(["traj_id", "rayleigh"]).normalised_vrmse_mean.mean()
            data = [per_traj.xs(ra, level="rayleigh").values for ra in ras]
            pos = np.arange(len(ras)) + (j - (len(tables) - 1) / 2) * width
            color = STYLE[model]["color"]
            ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "lw": 1.2},
                       boxprops={"facecolor": color, "alpha": 0.55, "edgecolor": color},
                       whiskerprops={"color": color}, capprops={"color": color})
        ax.axhline(10, color=INK, lw=0.9, ls="--")
        ax.text(-0.45, 11.5, "paper: > 10", ha="left", fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(ras)), [f"{ra:.0e}".replace("+", "") for ra in ras])
        ax.set_xlabel("Rayleigh number")
        ax.set_title(f"Mean over steps {lo}–{hi}, per trajectory", fontsize=9, color=INK)
        _style_axes(ax)
    axes[0].set_ylabel("rollout VRMSE, normalised units")
    handles = [Patch(color=STYLE[m]["color"], alpha=0.55,
                     label=("released " if m != "persistence" else "") + STYLE[m]["label"]) for m in tables]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Frame-0 rollouts (the paper's Table 3) by Rayleigh number, {split} split. "
                 "Boxes: middle 50% of trajectories; whiskers 1.5 IQR", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _finish(fig, stem)


# --- milestone 2: budgets from pilot timings --------------------------------------------------

def profile_figure(feasibility: Path, stem: Path) -> list[str]:
    if not feasibility.exists():
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    f = pd.read_csv(feasibility)
    f = f[(f.status == "ok") & (f.device == "cuda")].copy()
    f["compute_s"] = f.forward_s + f.backward_s + f.optimizer_s
    f["gpu_samples_per_s"] = f.batch_size / f.compute_s
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for model in _models(f.model):
        st = STYLE[model]
        for prec, ls in (("fp32", "-"), ("bf16", "--")):
            d = f[(f.model == model) & (f.precision == prec)].sort_values("batch_size")
            if d.empty:
                continue
            axes[0].plot(d.batch_size, d.gpu_samples_per_s, color=st["color"], marker=st["marker"], ms=5, lw=1.6, ls=ls)
            axes[1].plot(d.batch_size, d.peak_reserved_gb, color=st["color"], marker=st["marker"], ms=5, lw=1.6, ls=ls)
    axes[1].axhline(80, color=INK, lw=0.9, ls="--")
    axes[1].text(8, 76, "A100 memory (80 GB)", fontsize=7, color=INK, va="top")
    panels = ((axes[0], "training samples per GPU-second", "Compute throughput (forward + backward + optimizer)"),
              (axes[1], "peak GPU memory reserved (GB)", "Memory"))
    for ax, ylabel, title in panels:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, color=INK)
        _style_axes(ax)
    axes[0].set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ticks = [t for t in (20, 40, 60, 100, 200, 400, 700) if f.gpu_samples_per_s.min() * 0.8 <= t
             <= f.gpu_samples_per_s.max() * 1.2]
    axes[0].set_yticks(ticks, [str(t) for t in ticks])
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], lw=1.6, label=STYLE[m]["label"])
               for m in _models(f.model)]
    handles += [Line2D([], [], color=MUTED, lw=1.6, ls="-", label="fp32 (used)"),
                Line2D([], [], color=MUTED, lw=1.6, ls="--", label="bf16 (TFNO: unsupported, D8)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    gpu = f.gpu_name.dropna().iloc[0] if f.gpu_name.notna().any() else "GPU"
    fig.suptitle(f"Batch-size profile at 512 × 128 on one {gpu}; the study uses batch 64, fp32 (D7, D8)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _finish(fig, stem)


def data_loader_figure(bench: Path, pilot_timings: Path, stem: Path) -> list[str]:
    if not bench.exists():
        return []
    plt = _plt()

    b = pd.DataFrame(json.loads(bench.read_text())["rows"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for loader, color, label in (("upstream", MUTED, "upstream WellDataset (reopens files per sample)"),
                                 ("persistent", INK, "PersistentWellDataset (ours, D23)")):
        d = b[b.loader == loader].sort_values("workers")
        ax.plot(d.workers, d.samples_per_s, color=color, marker="o", ms=5, lw=1.8, label=label)
    top = b.samples_per_s.max()
    if pilot_timings.exists():
        pt = pd.read_csv(pilot_timings)
        for _, r in pt.iterrows():
            st = STYLE[r.model]
            ax.axhline(r.gpu_samples_per_s, color=st["color"], lw=1.2, ls="--")
            ax.text(b.workers.max() + 5.3, r.gpu_samples_per_s + 4, f"{st['label']} needs {r.gpu_samples_per_s:.0f}/s",
                    ha="right", va="bottom", fontsize=7, color=INK)
            top = max(top, r.gpu_samples_per_s)
    ax.set_ylim(0, top * 1.12)
    ax.set_xlim(b.workers.min() - 0.5, b.workers.max() + 5.5)
    ax.set_xticks(sorted(b.workers.unique()))
    ax.set_xlabel("data-loader worker processes (one node)")
    ax.set_ylabel("training samples per second")
    ax.set_title(f"Reading the {json.loads(bench.read_text()).get('split', 'train')} split from /pub, batch 64. "
                 "Dashed: each model's GPU demand", fontsize=9, color=INK)
    _style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    return _finish(fig, stem)


def pilot_vs_actual_figure(pilot_timings: Path, runs: pd.DataFrame, stem: Path) -> list[str]:
    actual = runs[(runs.manifest == "train") & (runs.status == "complete")]
    if not pilot_timings.exists() or actual.empty:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    pt = pd.read_csv(pilot_timings).set_index("model")
    models = [m for m in _models(actual.model) if m in pt.index]
    fig, ax = plt.subplots(figsize=(7.0, 0.8 * len(models) + 1.3))
    for i, model in enumerate(models):
        y = len(models) - 1 - i
        st = STYLE[model]
        ax.plot(pt.loc[model, "est_wall_hours_4h_no_stalls"], y + 0.15, marker="o", mfc="white", mec=st["color"],
                ms=8, ls="none")
        ax.plot(pt.loc[model, "est_wall_hours_4h"], y + 0.15, marker="s", mfc="white", mec=st["color"], ms=8,
                ls="none")
        for k, (_, r) in enumerate(actual[actual.model == model].sort_values("seed").iterrows()):
            yy = y - 0.1 - 0.2 * k
            ax.plot(r.wall_hours, yy, marker=st["marker"], color=st["color"], ms=7, ls="none")
            ax.text(r.wall_hours + 0.18, yy, f"seed {int(r.seed)}: {r.wall_hours:.1f} h", va="center", fontsize=7,
                    color=MUTED)
    ax.axvline(4, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(models)), [STYLE[m]["label"] for m in models[::-1]], fontsize=8)
    ax.set_ylim(-0.7, len(models) - 0.4)
    ax.set_xlim(0, max(actual.wall_hours.max(), pt.est_wall_hours_4h.max()) * 1.2)
    ax.set_xlabel("wall time for 4 h of GPU training (h)")
    _style_axes(ax)
    ax.grid(False, axis="y")
    handles = [Line2D([], [], marker="o", mfc="white", mec=MUTED, ls="none", ms=7, label="pilot, without stalls"),
               Line2D([], [], marker="s", mfc="white", mec=MUTED, ls="none", ms=7,
                      label="pilot, at the stall rate of 2026-11-08"),
               Line2D([], [], marker="o", color=MUTED, ls="none", ms=7, label="10% run (all attempts)")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("Pilot predictions (learning-rate search) against the 10% runs", fontsize=9, color=INK)
    fig.tight_layout()
    return _finish(fig, stem)


# --- milestone 3: training --------------------------------------------------------------------

def wait_timeline_figure(runs: pd.DataFrame, raw: dict[str, pd.DataFrame], attempts: pd.DataFrame, title: str,
                         stem: Path, gap_hours: float = 2.0) -> list[str]:
    """Cumulative data wait over UTC time; idle stretches longer than `gap_hours` are cut out."""
    plt = _plt()
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    if runs.empty or attempts.empty:
        return []
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    runs = runs.sort_values(["model", "seed", "lr"], key=lambda c: c.map(order) if c.name == "model" else c)
    multi_lr = runs.groupby("model").lr.nunique().max() > 1
    styles = ["-", "--", ":"]
    lr_styles = {lr: styles[i % 3] for i, lr in enumerate(sorted(runs.lr.unique()))}
    lines = []  # (x, y, color, ls, ended_early)
    for _, r in runs.iterrows():
        m = raw[r.run_id]
        ls = lr_styles[r.lr] if multi_lr else styles[int(r.seed) % 3]
        for _, a in attempts[attempts.run_id == r.run_id].iterrows():
            rows = m[(m.attempt == a.attempt) & m.data_wait_seconds.notna() & m.wall_seconds.notna()]
            if rows.empty or not isinstance(a.started_utc, str):
                continue
            x = pd.Timestamp(a.started_utc).tz_convert(None) + pd.to_timedelta(rows.wall_seconds.values, unit="s")
            lines.append((x, rows.data_wait_seconds.values / 3600, STYLE[r.model]["color"], ls,
                          a.end_status not in ("trained", "complete")))
    if not lines:
        return []
    spans = sorted((x.min(), x.max()) for x, *_ in lines)
    segments = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - segments[-1][1] > pd.Timedelta(hours=gap_hours):
            segments.append([lo, hi])
        else:
            segments[-1][1] = max(segments[-1][1], hi)
    widths = [max((hi - lo).total_seconds(), 3 * 3600) for lo, hi in segments]
    fig, axes = plt.subplots(1, len(segments), figsize=(10.0, 3.9), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": widths, "wspace": 0.06})
    for ax, (lo, hi) in zip(axes[0], segments):
        pad = (hi - lo) * 0.03 + pd.Timedelta(minutes=5)
        for x, y, color, ls, early in lines:
            if x.max() < lo or x.min() > hi:
                continue
            ax.plot(x, y, color=color, lw=1.5, ls=ls)
            if early:
                ax.plot(x[-1], y[-1], marker="x", color=INK, ms=6, mew=1.5)
        ax.set_xlim(lo - pad, hi + pad)
        hours = (hi - lo).total_seconds() / 3600
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(np.ceil(hours / 4)))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(f"{lo:%m-%d %H:%M} to {hi:%m-%d %H:%M}", fontsize=8, color=MUTED)
        _style_axes(ax)
    for ax in axes[0][1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0][0].set_ylabel("cumulative wait for data (h)")
    fig.supxlabel("time (UTC); stretches with no training cut out", fontsize=9, color=INK)
    handles = [Line2D([], [], color=STYLE[m]["color"], lw=1.5, label=STYLE[m]["label"]) for m in _models(runs.model)]
    if multi_lr:
        handles += [Line2D([], [], color=MUTED, lw=1.5, ls=s, label=f"lr {lr:.0e}") for lr, s in lr_styles.items()]
    else:
        handles += [Line2D([], [], color=MUTED, lw=1.5, ls=styles[s % 3], label=f"seed {s}")
                    for s in sorted(int(x) for x in set(runs.seed))]
    handles.append(Line2D([], [], marker="x", color=INK, ls="none", ms=6, mew=1.5,
                          label="attempt ended early (crash or preemption)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{title}. Each line is one attempt; steep rises are waits for /pub,\nand rises at the same moment "
                 "on different runs are stalls that hit several nodes", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.subplots_adjust(top=0.8)
    return _finish(fig, stem)


# --- evaluation ---------------------------------------------------------------------------------

def eval_one_step_by_frame(evals: pd.DataFrame, outputs: Path, stem: Path) -> list[str]:
    if evals.empty:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    models = _models(evals.model)
    by_label = {lab: _load(evals, "one_step", lab) for lab in sorted(set(evals.label))}
    persist = _persistence(outputs, "one_step")
    fig, axes = plt.subplots(2, len(models), figsize=(max(3.9 * len(models), 6.4) + 0.6, 6.0), sharex=True,
                             sharey="row", squeeze=False)
    for j, model in enumerate(models):
        st = STYLE[model]
        released = _released(outputs, model, "one_step")
        for i, (col, unit) in enumerate((("vrmse_mean", "physical"), ("normalised_vrmse_mean", "normalised (paper)"))):
            ax = axes[i][j]
            for lab, lw, alpha in (("1h", 1.1, 0.5), ("4h", 1.9, 1.0)):
                tables = by_label.get(lab, {}).get(model)
                if not tables:
                    continue
                mean, lo, hi = _seed_mean(tables, "start", col)
                if len(tables) > 1:
                    ax.fill_between(mean.index, lo, hi, color=st["color"], alpha=0.15 * alpha, lw=0)
                ax.plot(mean.index, mean.values, color=st["color"], lw=lw, alpha=alpha)
            if released is not None:
                g = released.groupby("start")[col].mean()
                ax.plot(g.index, g.values, color=INK, lw=1.2, ls="--")
            if persist is not None:
                g = persist.groupby("start")[col].mean()
                ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.3, ls=":")
            ax.set_yscale("log")
            _style_axes(ax)
            if i == 0:
                n_seeds = len(by_label.get("4h", {}).get(model, []))
                ax.set_title(f"{st['label']} ({n_seeds} seed{'s' if n_seeds != 1 else ''} at 4 h)", fontsize=9,
                             color=INK)
            else:
                ax.set_xlabel("first input frame of the window")
            if j == 0:
                ax.set_ylabel(f"one-step VRMSE, {unit}", fontsize=9)
    for ax in axes.flat:
        _onset_marker(ax, y_text=ax.get_ylim()[1] * 0.8)
    handles = [Line2D([], [], color=MUTED, lw=1.9, label="ours, 4 h checkpoint (10% data)"),
               Line2D([], [], color=MUTED, lw=1.1, alpha=0.5, label="ours, 1 h checkpoint"),
               Line2D([], [], color=INK, lw=1.2, ls="--", label="released checkpoint (100% data, the paper's)"),
               Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("One-step test error by window: mean over 175 trajectories (band: range over seeds)", fontsize=10,
                 color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return _finish(fig, stem)


def eval_rollout_by_start(evals: pd.DataFrame, outputs: Path, label: str, stem: Path) -> list[str]:
    tables = _load(evals, "rollout", label)
    if not tables:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    persist = _persistence(outputs, "rollout")
    starts = sorted(next(iter(tables.values()))[0].start.unique())
    names = {20: "convection onset", 80: "developed", 140: "late"}
    fig, axes = plt.subplots(1, len(starts), figsize=(3.6 * len(starts) + 0.6, 3.4), sharey=True, squeeze=False)
    for ax, start in zip(axes[0], starts):
        for model in _models(tables):
            st = STYLE[model]
            mine = [t[(t.start == start) & (t.sigma == 0)] for t in tables[model]]
            mean, lo, hi = _seed_mean(mine, "step", "vrmse_mean")
            if len(mine) > 1:
                ax.fill_between(mean.index, lo, hi, color=st["color"], alpha=0.15, lw=0)
            ax.plot(mean.index, mean.values, color=st["color"], lw=1.9, marker=st["marker"], ms=3, markevery=5)
        if persist is not None:
            g = persist[(persist.start == start) & (persist.sigma == 0)].groupby("step").vrmse_mean.mean()
            ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.3, ls=":")
        ax.set_title(f"start frame {start} ({names.get(int(start), '')})", fontsize=9, color=INK)
        ax.set_xlabel("rollout step")
        _style_axes(ax)
    axes[0][0].set_ylabel("test VRMSE, physical units (field mean)")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], lw=1.9,
                      label=f"{STYLE[m]['label']} ({len(tables[m])} seed{'s' if len(tables[m]) != 1 else ''})")
               for m in _models(tables)]
    handles.append(Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Rollouts from the study's start frames (D15), {label} checkpoints, 10% data: mean over 175 "
                 "trajectories (band: range over seeds)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _finish(fig, stem)


def eval_paper_rollout(evals: pd.DataFrame, outputs: Path, label: str, stem: Path) -> list[str]:
    tables = _load(evals, "paper_rollout", label)
    if not tables:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    persist = _persistence(outputs, "paper_rollout")
    released = {m: t for m in ("cnext_unet", "tfno") if (t := _released(outputs, m, "paper_rollout")) is not None}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=True)
    for ax, col, title in ((axes[0], "normalised_vrmse_mean", "Normalised units (the paper's protocol)"),
                           (axes[1], "vrmse_mean", "Physical units")):
        for model in _models(list(tables) + list(released)):
            st = STYLE[model]
            if model in tables:
                mean, lo, hi = _seed_mean(tables[model], "step", col)
                if len(tables[model]) > 1:
                    ax.fill_between(mean.index, lo, hi, color=st["color"], alpha=0.15, lw=0)
                ax.plot(mean.index, mean.values, color=st["color"], lw=1.9, marker=st["marker"], ms=3, markevery=3)
            if model in released:
                g = released[model].groupby("step")[col].mean()
                ax.plot(g.index, g.values, color=st["color"], lw=1.3, ls="--")
        if persist is not None:
            g = persist.groupby("step")[col].mean()
            ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.3, ls=":")
        for x in (5.5, 12.5):
            ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.axhline(10, color=INK, lw=0.9, ls="--")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("rollout step from frame 0 (step 1 = frame 4)")
        _style_axes(ax)
    axes[0].set_ylabel("rollout VRMSE (field mean)")
    axes[0].text(30, 11.5, "paper: > 10", ha="right", fontsize=7, color=INK)
    handles = [Line2D([], [], color=STYLE[m]["color"], lw=1.9, label=STYLE[m]["label"])
               for m in _models(list(tables) + list(released))]
    handles += [Line2D([], [], color=MUTED, lw=1.9, label=f"ours, {label} checkpoint (10% data)"),
                Line2D([], [], color=MUTED, lw=1.3, ls="--", label="released checkpoint (100% data)"),
                Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("The paper's rollout protocol (Table 3): 30 steps from frame 0 of every test trajectory; windows "
                 "6–12 and 13–30 marked", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _finish(fig, stem)


SMALL_SCALES_K = 100.0  # k_x above which the 10% models' spectra depart from the truth (eval_spectra_*)


def eval_spectra_ratio(evals: pd.DataFrame, outputs: Path, label: str, stem: Path,
                       regime: tuple[float, float] = (1e8, 1.0)) -> list[str]:
    """Top: absolute spectra for one regime (true, models, persistence). Bottom: predicted / true over all regimes.

    The shaded band marks the smallest scales (k_x > SMALL_SCALES_K); left of it are the large and middle scales.
    """
    tables = _load(evals, "spectra", label)
    if not tables:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    def clean(t: pd.DataFrame) -> pd.DataFrame:
        return t[(t.sigma == 0) & (t.k_index > 0) & (t.k_index < t.k_index.max())]   # no mean, no Nyquist mode

    def ratio(t: pd.DataFrame) -> pd.DataFrame:
        t = clean(t)
        key = ["rayleigh", "prandtl", "step", "k_index", "k"]
        m = t[t.source == "pred"].merge(t[t.source == "true"], on=key, suffixes=("_p", "_t"))
        m["log_ratio"] = np.log10(m.energy_p / m.energy_t)
        return m

    persist = _persistence(outputs, "spectra")
    sources = {m: ts for m, ts in tables.items()}
    if persist is not None:
        sources["persistence"] = [persist]
    first = clean(next(iter(tables.values()))[0])
    steps = sorted(first.step.unique())
    regimes = sorted({(r, p) for r, p in zip(first.rayleigh, first.prandtl)})
    ra, pr = regime if regime in regimes else regimes[len(regimes) // 2]
    fig, axes = plt.subplots(2, len(steps), figsize=(3.6 * len(steps) + 0.6, 6.4), sharex=True, sharey="row",
                             squeeze=False, gridspec_kw={"height_ratios": [1.1, 1]})
    for j, step in enumerate(steps):
        top, bottom = axes[0][j], axes[1][j]
        truth = first[(first.source == "true") & (first.step == step) & (first.rayleigh == ra)
                      & (first.prandtl == pr)].sort_values("k")
        top.loglog(truth.k, truth.energy, color=INK, lw=2.2)
        for model, ts in sources.items():
            st = STYLE[model]
            ls = ":" if model == "persistence" else "-"
            spec = pd.concat([clean(t)[(clean(t).source == "pred") & (clean(t).step == step)
                                       & (clean(t).rayleigh == ra) & (clean(t).prandtl == pr)]
                              .set_index("k").energy for t in ts], axis=1).mean(axis=1)
            top.loglog(spec.index, spec.values, color=st["color"], lw=1.4, ls=ls)
            rs = [ratio(t) for t in ts]
            med = pd.concat([r[r.step == step].groupby("k").log_ratio.median() for r in rs], axis=1).mean(axis=1)
            q_lo = pd.concat([r[r.step == step].groupby("k").log_ratio.quantile(0.25) for r in rs], axis=1).mean(axis=1)
            q_hi = pd.concat([r[r.step == step].groupby("k").log_ratio.quantile(0.75) for r in rs], axis=1).mean(axis=1)
            bottom.fill_between(med.index, 10 ** q_lo, 10 ** q_hi, color=st["color"], alpha=0.12, lw=0)
            bottom.plot(med.index, 10 ** med, color=st["color"], lw=1.6, ls=ls)
        bottom.axhline(1, color=INK, lw=0.9)
        for ax in (top, bottom):
            ax.axvspan(SMALL_SCALES_K, truth.k.max() * 1.05, color=MUTED, alpha=0.08, lw=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _style_axes(ax)
        top.set_title(f"rollout step {int(step)}", fontsize=9, color=INK)
        bottom.set_xlabel("horizontal wavenumber k_x")
    top_ylim = axes[0][0].get_ylim()
    axes[0][0].text(SMALL_SCALES_K * 1.1, top_ylim[1] / 5, "smallest\nscales", fontsize=7, color=MUTED, va="top")
    axes[0][0].text(2.0, top_ylim[0] * 5, "large and middle scales", fontsize=7, color=MUTED, va="bottom")
    axes[0][0].set_ylabel(f"E(k_x), Ra = {ra:.0e}, Pr = {pr:g}")
    axes[1][0].set_ylabel("E_pred / E_true, all 35 regimes")
    names = _models(tables) + (["persistence"] if persist is not None else [])
    handles = [Line2D([], [], color=INK, lw=2.2, label="true flow")]
    handles += [Line2D([], [], color=STYLE[m]["color"], lw=1.6, ls=":" if m == "persistence" else "-",
                       label=STYLE[m]["label"]) for m in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Horizontal kinetic-energy spectra ({label} checkpoints, 10% data). Top: one regime, mean over "
                 "its rollouts. Bottom: predicted / true, median over regimes (band: middle 50%); 1 = exact",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _finish(fig, stem)


# --- numerical solver (Dedalus) against the models -------------------------------------------------------

SOLVER_STYLE = {"512x128": {"color": INK, "marker": "P"}, "256x64": {"color": "#6f6e69", "marker": "X"},
                "128x32": {"color": "#a8a7a2", "marker": "D"}}


def _solvers(outputs: Path) -> dict[str, Path]:
    """Scored solver resolutions (outputs/baselines/solver_<res>/evaluations), finest first."""
    base = outputs / "baselines"
    found = {d.parent.name.removeprefix("solver_"): d for d in sorted(base.glob("solver_*/evaluations"))
             if (d / "rollout_na.parquet").exists()} if base.exists() else {}
    return dict(sorted(found.items(), key=lambda kv: -int(kv[0].split("x")[0])))


def _subset(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    idx = pd.MultiIndex.from_frame(df[["traj_id", "start"]])
    return df[idx.isin(list(keys))]


def solver_rollout_comparison(evals: pd.DataFrame, outputs: Path, label: str, stem: Path) -> list[str]:
    solvers = _solvers(outputs)
    if not solvers:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    rolls = {res: pd.read_parquet(d / "rollout_na.parquet") for res, d in solvers.items()}
    keys = set(map(tuple, next(iter(rolls.values()))[["traj_id", "start"]].drop_duplicates().values))
    models = _load(evals, "rollout", label) if not evals.empty else {}
    persist = _persistence(outputs, "rollout")
    starts = sorted({k[1] for k in keys})
    names = {20: "convection onset", 80: "developed", 140: "late"}
    fig, axes = plt.subplots(1, len(starts), figsize=(3.6 * len(starts) + 0.6, 3.6), sharey=True, squeeze=False)
    handles, diverged = [], {}
    for ax, start in zip(axes[0], starts):
        for model in _models(models):
            st = STYLE[model]
            mine = [_subset(t[(t.start == start) & (t.sigma == 0)], keys) for t in models[model]]
            mean, _, _ = _seed_mean(mine, "step", "vrmse_mean")
            ax.plot(mean.index, mean.values, color=st["color"], lw=1.8)
        if persist is not None:
            g = _subset(persist[(persist.start == start) & (persist.sigma == 0)], keys).groupby("step").vrmse_mean.mean()
            ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.3, ls=":")
        for res, roll in rolls.items():
            st = SOLVER_STYLE.get(res, {"color": MUTED, "marker": "o"})
            r = roll[(roll.start == start) & (roll.sigma == 0)]
            bad = r.groupby("traj_id").vrmse_mean.apply(lambda s: not np.isfinite(s).all())
            diverged[res] = diverged.get(res, 0) + int(bad.sum())
            good = r[~r.traj_id.isin(bad[bad].index)]
            g = good.groupby("step").vrmse_mean.mean()
            ax.plot(g.index, g.values, color=st["color"], lw=1.6, marker=st["marker"], ms=3.5, markevery=5)
        ax.set_yscale("log")
        ax.set_title(f"start frame {start} ({names.get(int(start), '')})", fontsize=9, color=INK)
        ax.set_xlabel("rollout step")
        _style_axes(ax)
    axes[0][0].set_ylabel("test VRMSE, physical units (field mean)")
    n = len({k[0] for k in keys})
    handles += [Line2D([], [], color=STYLE[m]["color"], lw=1.8, label=f"{STYLE[m]['label']} ({label}, 10% data)")
                for m in _models(models)]
    handles += [Line2D([], [], color=SOLVER_STYLE.get(r, {}).get("color", MUTED), lw=1.6,
                       marker=SOLVER_STYLE.get(r, {}).get("marker", "o"),
                       label=f"Dedalus {r}" + (f" ({diverged[r]} diverged, left out)" if diverged.get(r) else ""))
                for r in rolls]
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Numerical solver against the models on the same {n} test trajectories (one per regime): mean "
                 "over trajectories", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    return _finish(fig, stem)


def solver_cost_accuracy(evals: pd.DataFrame, outputs: Path, label: str, stem: Path) -> list[str]:
    solvers = _solvers(outputs)
    if not solvers:
        return []
    plt = _plt()

    rolls = {res: pd.read_parquet(d / "rollout_na.parquet") for res, d in solvers.items()}
    runs = {res: pd.read_parquet(d / "runs_na.parquet") for res, d in solvers.items()
            if (d / "runs_na.parquet").exists()}
    keys = set(map(tuple, next(iter(rolls.values()))[["traj_id", "start"]].drop_duplicates().values))
    starts = sorted({k[1] for k in keys})
    points = []  # (start, name, seconds, error, color, marker)
    models = _load(evals, "rollout", label) if not evals.empty else {}
    for model in _models(models):
        lat = []
        for _, e in evals[(evals.model == model) & (evals.label == label)].iterrows():
            p = e.dir / f"clean_{label}.json"
            if p.exists():
                lat.append(json.loads(p.read_text())["latency"]["ms_per_rollout"] / 1000)
        for start in starts:
            mine = [_subset(t[(t.start == start) & (t.sigma == 0)], keys) for t in models[model]]
            err = float(np.mean([m.vrmse_mean.mean() for m in mine]))
            points.append((start, f"{STYLE[model]['label']} {label}", float(np.mean(lat)) if lat else np.nan, err,
                           STYLE[model]["color"], STYLE[model]["marker"]))
    pj = outputs / "baselines" / "persistence" / "evaluations" / "clean_na.json"
    persist = _persistence(outputs, "rollout")
    if persist is not None and pj.exists():
        sec = json.loads(pj.read_text())["latency"]["ms_per_rollout"] / 1000
        for start in starts:
            err = float(_subset(persist[(persist.start == start) & (persist.sigma == 0)], keys).vrmse_mean.mean())
            points.append((start, "persistence", sec, err, PERSIST["color"], "o"))
    for res, roll in rolls.items():
        st = SOLVER_STYLE.get(res, {"color": MUTED, "marker": "o"})
        for start in starts:
            r = roll[(roll.start == start) & (roll.sigma == 0)]
            fin = r.groupby("traj_id").vrmse_mean.apply(lambda s: np.isfinite(s).all())
            err = float(r[r.traj_id.isin(fin[fin].index)].vrmse_mean.mean()) if fin.any() else np.nan
            sec = float(runs[res][runs[res].start == start].seconds_per_rollout.median()) if res in runs else np.nan
            points.append((start, f"Dedalus {res}", sec, err, st["color"], st["marker"]))
    fig, axes = plt.subplots(1, len(starts), figsize=(3.8 * len(starts) + 0.6, 3.7), sharex=True, sharey=True,
                             squeeze=False)
    for ax, start in zip(axes[0], starts):
        for s, name, sec, err, color, marker in points:
            if s != start or not np.isfinite(sec) or not np.isfinite(err):
                continue
            ax.plot(sec, err, marker=marker, color=color, ms=7, ls="none")
            ax.annotate(name, (sec, err), xytext=(5, 3), textcoords="offset points", fontsize=6.5, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"start frame {start}", fontsize=9, color=INK)
        ax.set_xlabel("seconds per 30-step rollout")
        _style_axes(ax)
    axes[0][0].set_ylabel("mean VRMSE over the 30 steps")
    fig.suptitle("Cost against accuracy. Models and persistence: one A100, batch 1 (training cost not included); "
                 "Dedalus: one CPU core, including setup", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, stem)


# --- Slurm accounting (sacct) -----------------------------------------------------------------------------

# Arrays none of whose tasks wrote a run record, identified from the AGENTS.md progress log.
KNOWN_ARRAYS = {
    "59269977": "first replication try (stopped, I/O-bound; D23)",
    "59369074": "learning-rate search",   # first submission, cancelled after 9 min
    "59395480": "10% runs",               # refused a dirty tree during a push (2026-11-09)
    "59403878": "10% runs",               # free-gpu resubmission, cancelled while pending
    "59427099": "10% runs",               # AViT resubmission, cancelled while pending for the push
}


def _hours(elapsed: str) -> float:
    """sacct Elapsed ([D-]HH:MM:SS) in hours."""
    days, _, rest = elapsed.rpartition("-")
    h, m, s = (int(x) for x in rest.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def _tres(alloc: str) -> dict:
    out = {}
    for item in str(alloc).split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def sacct_table(path: Path, attempts: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """One row per Slurm job (array task) with billed SU, GPU-hours, queue wait and what the job was for.

    RCIC bills `billing` x elapsed hours on paid partitions (32 per GPU + 1 per core); free partitions cost 0.
    Jobs cancelled before they started have no allocation and cost nothing.
    """
    df = pd.read_csv(path, sep="|", dtype=str)
    rows = []
    manifest_of = dict(zip(runs.run_id, runs.manifest)) if not runs.empty else {}
    task_of = {}
    if not attempts.empty:
        for _, a in attempts.iterrows():
            if isinstance(a.array_task, str) and a.array_task:
                task_of[a.array_task] = manifest_of.get(a.run_id, "")
            if pd.notna(a.job_id):
                task_of.setdefault(str(a.job_id), manifest_of.get(a.run_id, ""))
    for _, r in df.iterrows():
        tres = _tres(r.AllocTRES)
        started = isinstance(r.Start, str) and r.Start not in ("None", "Unknown")
        ended = isinstance(r.End, str) and r.End not in ("None", "Unknown")
        elapsed = _hours(r.Elapsed) if started else 0.0
        billing = float(tres.get("billing", 0) or 0)
        gpus = int(tres.get("gres/gpu", 0) or 0)
        name = str(r.JobName)
        what = {"rb4-setup": "environment", "rb4-check": "environment", "bash": "environment",
                "rb4-download": "data", "rb4-audit": "data", "rb4-probe": "data", "rb4-subsets": "data",
                "rb4-conv": "data", "rb4-bench": "data", "rb4-released": "pipeline check",
                "rb4-profile": "profiling", "rb4-eval": "evaluation"}.get(name, name)
        if name == "rb4-train":
            m = task_of.get(str(r.JobID), "")
            what = {"lr_search": "learning-rate search", "train": "10% runs"}.get(m, "training (other)")
        queue = ((pd.Timestamp(r.Start) - pd.Timestamp(r.Submit)).total_seconds() / 3600 if started else np.nan)
        rows.append({"job": r.JobID, "array": str(r.JobID).split("_")[0], "name": name, "what": what,
                     "partition": r.Partition, "account": r.Account,
                     "gpus": gpus, "cpus": int(tres.get("cpu", 0) or 0), "billing": billing,
                     "submit": r.Submit, "start": r.Start if started else None, "end": r.End if ended else None,
                     "elapsed_hours": elapsed, "queue_hours": queue, "state": r.State,
                     "running": not ended and started,
                     "su": billing * elapsed if r.Partition in ("gpu", "standard") else 0.0,
                     "gpu_hours": gpus * elapsed})
    out = pd.DataFrame(rows)
    # tasks that never wrote a run record (failed at start, cancelled, preempted early) take their array's purpose
    known = out[out.what.isin(["learning-rate search", "10% runs"])].groupby("array").what.first()
    other = out.what == "training (other)"
    out.loc[other, "what"] = (out.loc[other, "array"].map(known).fillna(out.loc[other, "array"].map(KNOWN_ARRAYS))
                              .fillna("other training"))
    return out.drop(columns="array")


def compute_timeline_figure(jobs: pd.DataFrame, stem: Path) -> list[str]:
    """Every GPU job: queue wait (light) and run (solid), coloured by purpose, paid jobs outlined."""
    gpu = jobs[(jobs.gpus > 0) & jobs.start.notna()].copy()
    if gpu.empty:
        return []
    plt = _plt()
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    colors = {"learning-rate search": "#1baf7a", "10% runs": "#2a78d6", "evaluation": "#eda100",
              "pipeline check": "#eb6834", "profiling": "#8a6fd1", "environment": MUTED,
              "first replication try (stopped, I/O-bound; D23)": "#a8a7a2", "other training": "#a8a7a2"}
    gpu["t_submit"] = pd.to_datetime(gpu.submit)
    gpu["t_start"] = pd.to_datetime(gpu.start)
    gpu["t_end"] = gpu.t_start + pd.to_timedelta(gpu.elapsed_hours, unit="h")
    gpu = gpu.sort_values("t_start").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.0, 0.16 * len(gpu) + 1.8))
    for i, r in gpu.iterrows():
        y = len(gpu) - 1 - i
        c = colors.get(r.what, MUTED)
        ax.barh(y, (r.t_start - r.t_submit).total_seconds() / 86400, left=mdates.date2num(r.t_submit),
                height=0.7, color=c, alpha=0.18, lw=0)
        ax.barh(y, (r.t_end - r.t_start).total_seconds() / 86400, left=mdates.date2num(r.t_start), height=0.7,
                color=c, edgecolor=INK if r.partition == "gpu" else "none", lw=0.8)
        if r.state.startswith(("PREEMPTED", "FAILED", "TIMEOUT")):
            ax.text(mdates.date2num(r.t_end), y, " " + r.state.split()[0].lower(), va="center", fontsize=5.5,
                    color=MUTED)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.set_yticks([])
    ax.set_xlabel("time (HPC3 local)")
    _style_axes(ax)
    ax.grid(False, axis="y")
    handles = [Patch(color=c, label=k) for k, c in colors.items() if k in set(gpu.what)]
    handles += [Patch(facecolor="white", edgecolor=INK, label="paid `gpu` partition (class account)"),
                Patch(color=MUTED, alpha=0.18, label="waiting in the queue")]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    paid = gpu[gpu.partition == "gpu"]
    ax.set_title(f"All {len(gpu)} GPU jobs (from sacct): {gpu.gpu_hours.sum():.0f} GPU-hours, "
                 f"{paid.su.sum():,.0f} SU billed on the paid partition", fontsize=9, color=INK)
    fig.tight_layout()
    return _finish(fig, stem)


# --- flow snapshots: truth against our models (scripts/visualize_rollouts.py) --------------------------------

FIELD_LABELS = {"buoyancy": "buoyancy b", "pressure": "pressure p", "velocity_x": "horizontal velocity u",
                "velocity_y": "vertical velocity w"}


def rollout_snapshots_figure(npz: Path, field: str, stem: Path, steps=(1, 6, 12, 30), errors: bool = False
                             ) -> list[str]:
    """Rows: rollout steps; columns: the true flow and each model (or each model's absolute error)."""
    plt = _plt()
    from rbbench.solver_grid import LX, LY, gauss_nodes, x_grid

    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    fields = meta["fields"]
    fi = fields.index(field)
    have = [int(s) for s in d["steps"]]
    rows = [s for s in steps if s in have] or have
    models = [m for m in MODEL_NAMES if m in d.files]
    cols = models if errors else ["truth", *models]
    truth = d["truth"][..., fi]
    nx, ny = truth.shape[1:3]
    x, y = x_grid(nx, LX), gauss_nodes(ny, LY)
    if errors:
        vmax = float(max(np.abs(d[m][..., fi] - truth).max() for m in models)) or 1.0
        norm_kw, cmap = {"vmin": 0.0, "vmax": vmax}, "Blues"
    elif field == "buoyancy":
        norm_kw, cmap = {"vmin": 0.0, "vmax": 1.0}, "RdBu_r"
    else:
        lim = float(np.percentile(np.abs(truth - truth.mean()), 99.5)) or 1.0
        c = 0.0 if field.startswith("velocity") else float(truth.mean())
        norm_kw, cmap = {"vmin": c - lim, "vmax": c + lim}, "RdBu_r"
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.55 * len(cols) + 0.9, 0.78 * len(rows) + 1.0),
                             squeeze=False)
    mesh = None
    for i, step in enumerate(rows):
        k = have.index(step)
        for j, col in enumerate(cols):
            ax = axes[i][j]
            img = (np.abs(d[col][k, ..., fi] - truth[k]) if errors else
                   (truth[k] if col == "truth" else d[col][k, ..., fi]))
            mesh = ax.pcolormesh(x, y, img.T, shading="nearest", cmap=cmap, rasterized=True, **norm_kw)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                name = "true flow" if col == "truth" else STYLE[col]["label"]
                ax.set_title(name + (" |error|" if errors else ""), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"step {step}", fontsize=8, color=INK)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    what = f"absolute error in {FIELD_LABELS.get(field, field)}" if errors else FIELD_LABELS.get(field, field)
    fig.suptitle(f"{what}: Ra = {meta['rayleigh']:.0e}, Pr = {meta['prandtl']:g}, rollout from frame "
                 f"{meta['start']} (test trajectory), our {meta['label']} checkpoints, "
                 f"{round(meta['fraction'] * 100)}% data, seed {meta['seed']}", fontsize=9, color=INK)
    return _finish(fig, stem)
