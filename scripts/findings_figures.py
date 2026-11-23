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

def fraction_text(evals: pd.DataFrame) -> str:
    """'10% data', '100% data' (or both) for the runs a figure is drawn from."""
    fr = sorted({round(float(f), 3) for f in evals.fraction}) if not evals.empty else []
    return (" / ".join(f"{f:.0%}" for f in fr) + " data") if fr else "data"


def evaluated_runs(runs: pd.DataFrame, outputs: Path, manifest: str = "train",
                   fraction: float | None = None) -> pd.DataFrame:
    """One row per (run, checkpoint label) with an evaluation on disk; only runs at `fraction` when given.

    `_load` pools every run of a model it is handed as seeds, so a figure for one data fraction must pass `fraction`:
    with the 100% runs evaluated, an unfiltered call mixes 10% and 100% runs into one seed mean.
    """
    rows = []
    sel = runs[runs.manifest == manifest]
    if fraction is not None:
        sel = sel[sel.fraction.round(3) == round(fraction, 3)]
    for _, r in sel.iterrows():
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
    # the pilots predicted the 10% runs, which read /pub; the 100% runs read a node-local copy (D29)
    actual = runs[(runs.manifest == "train") & (runs.status == "complete") & (runs.fraction.round(3) == 0.1)]
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
    import warnings

    with warnings.catch_warnings():  # supxlabel is not a tight_layout axes; the layout is fixed just below
        warnings.simplefilter("ignore", UserWarning)
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
    handles = [Line2D([], [], color=MUTED, lw=1.9, label=f"ours, 4 h checkpoint ({fraction_text(evals)})"),
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
    fig.suptitle(f"Rollouts from the study's start frames (D15), {label} checkpoints, {fraction_text(evals)}: mean over 175 "
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
    handles += [Line2D([], [], color=MUTED, lw=1.9, label=f"ours, {label} checkpoint ({fraction_text(evals)})"),
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
    fig.suptitle(f"Horizontal kinetic-energy spectra ({label} checkpoints, {fraction_text(evals)}). Top: one regime, mean over "
                 "its rollouts. Bottom: predicted / true, median over regimes (band: middle 50%); 1 = exact",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _finish(fig, stem)


# --- 10% against 100% data (WP3): one row per run and checkpoint, plus the two references ---------------------

COMPARISON_FIELDS = ("buoyancy", "velocity_x", "velocity_y", "pressure")


def _comparison_row(one: pd.DataFrame, roll: pd.DataFrame, t_on: dict, persist_dev: pd.Series) -> dict:
    """One-step error over all windows and split at each trajectory's onset frame; rollout error and stability.

    Onset: a window whose first input frame precedes the trajectory's t_on (rbbench.onset; a trajectory without a
    crossing is all onset). Rollouts from frames 80 and 140 count as developed flow (t_on <= 78 on test).
    """
    on = np.array([s < t_on[t] if t_on.get(t) is not None else True for t, s in zip(one.traj_id, one.start)])
    r = {"one_step_all": one.vrmse_mean.mean(), "one_step_onset": one.vrmse_mean[on].mean(),
         "one_step_developed": one.vrmse_mean[~on].mean()}
    for f in COMPARISON_FIELDS:
        r[f"developed_{f}"] = one[f"vrmse_{f}"][~on].mean()
        r[f"onset_{f}"] = one[f"vrmse_{f}"][on].mean()
    roll = roll[roll.sigma == 0] if "sigma" in roll else roll
    for s in (20, 80, 140):
        r[f"rollout_step30_from{s}"] = roll[(roll.start == s) & (roll.step == 30)].vrmse_mean.mean()
    worst = roll.groupby(["traj_id", "start"]).vrmse_mean.max()
    r["share_below1_30steps"] = float((worst < 1).mean())
    dev = roll[roll.start >= 80].groupby("step").vrmse_mean.mean()
    ahead = (dev < persist_dev.reindex(dev.index)).to_numpy()
    r["steps_ahead_of_persistence_developed"] = int(len(ahead) if ahead.all() else np.argmin(ahead))
    return r


def fraction_comparison(runs: pd.DataFrame, outputs: Path, onset_table: Path) -> pd.DataFrame:
    """Per run and checkpoint (and persistence, the released CNextU-Net) the numbers of the 10%-vs-100% comparison."""
    from rbbench.onset import load_onset_frames

    evals = evaluated_runs(runs, outputs)
    persist_dir = outputs / "baselines" / "persistence" / "evaluations"
    if evals.empty or not onset_table.exists() or not (persist_dir / "rollout_na.parquet").exists():
        return pd.DataFrame()
    t_on = load_onset_frames(onset_table)
    p_roll = pd.read_parquet(persist_dir / "rollout_na.parquet")
    p_roll = p_roll[p_roll.sigma == 0]
    persist_dev = p_roll[p_roll.start >= 80].groupby("step").vrmse_mean.mean()
    rows = []
    for _, e in evals.iterrows():
        one, roll = (pd.read_parquet(e.dir / f"{k}_{e.label}.parquet") for k in ("one_step", "rollout"))
        rows.append({"model": e.model, "fraction": e.fraction, "seed": e.seed, "budget": e.label,
                     "run_id": e.run_id, **_comparison_row(one, roll, t_on, persist_dev)})
    for name, d in (("persistence", persist_dir),
                    ("released_cnext_unet", outputs / "baselines" / "released" / "cnext_unet" / "evaluations")):
        if (d / "one_step_na.parquet").exists():
            one, roll = (pd.read_parquet(d / f"{k}_na.parquet") for k in ("one_step", "rollout"))
            rows.append({"model": name, "fraction": np.nan, "seed": np.nan, "budget": "-", "run_id": "",
                         **_comparison_row(one, roll, t_on, persist_dev)})
    return pd.DataFrame(rows).sort_values(["budget", "model", "fraction", "seed"], na_position="first")


# --- heat transport per Rayleigh number (WP5; D28) --------------------------------------------------------------

NU_STARTS = (80, 140)  # rollouts from developed flow


def _nu_rollout_means(phys: pd.DataFrame) -> pd.DataFrame:
    """Predicted and true volume Nusselt number averaged over each 30-step rollout from developed flow."""
    phys = phys[phys.sigma == 0] if "sigma" in phys else phys
    phys = phys[phys.start.isin(NU_STARTS)]
    g = phys.groupby(["rayleigh", "prandtl", "traj_id", "start"]).agg(
        nu_true=("nu_true", "mean"), nu_pred=("nu_pred", "mean"), steps=("step", "size")).reset_index()
    g["rel_err"] = (g.nu_pred - g.nu_true) / g.nu_true
    return g


def nusselt_rollouts(runs: pd.DataFrame, outputs: Path) -> pd.DataFrame:
    """One row per (run or reference, checkpoint, rollout): the rollout-averaged predicted and true Nu.

    The volume Nusselt number Nu = 1 + <w b>/kappa of every predicted and true frame comes from the evaluation
    (`physics_<label>.parquet`, rbbench.metrics.physics.nusselt). Averaging over the rollout, as the professors asked
    (D28), damps the instantaneous value's wide swings.
    """
    rows = []
    evals = evaluated_runs(runs, outputs)
    for _, e in evals.iterrows():
        p = e.dir / f"physics_{e.label}.parquet"
        if p.exists():
            g = _nu_rollout_means(pd.read_parquet(p))
            rows.append(g.assign(model=e.model, fraction=e.fraction, seed=e.seed, budget=e.label))
    for name, d in (("persistence", outputs / "baselines" / "persistence" / "evaluations"),
                    ("released_cnext_unet", outputs / "baselines" / "released" / "cnext_unet" / "evaluations")):
        if (d / "physics_na.parquet").exists():
            g = _nu_rollout_means(pd.read_parquet(d / "physics_na.parquet"))
            rows.append(g.assign(model=name, fraction=np.nan, seed=0, budget="-"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def nusselt_by_rayleigh(roll: pd.DataFrame) -> pd.DataFrame:
    """Per model, fraction, budget and Ra (and over all Ra): medians over rollouts, then the mean over seeds.

    nu_true / nu_pred: median rollout-averaged Nu with the quartiles of the prediction; rel_err_signed /
    rel_err_abs: median of (pred - true) / true and of its absolute value (negative: too little heat transport).
    """
    def stats(g):
        return pd.Series({"n_rollouts": len(g), "nu_true": g.nu_true.median(), "nu_pred": g.nu_pred.median(),
                          "nu_pred_q25": g.nu_pred.quantile(0.25), "nu_pred_q75": g.nu_pred.quantile(0.75),
                          "nu_true_q25": g.nu_true.quantile(0.25), "nu_true_q75": g.nu_true.quantile(0.75),
                          "rel_err_signed": g.rel_err.median(), "rel_err_abs": g.rel_err.abs().median()})
    keys = ["model", "fraction", "budget", "seed"]
    per_ra = roll.groupby(keys + ["rayleigh"], dropna=False).apply(stats, include_groups=False).reset_index()
    overall = roll.groupby(keys, dropna=False).apply(stats, include_groups=False).reset_index().assign(rayleigh=np.nan)
    both = pd.concat([per_ra, overall], ignore_index=True)
    out = (both.groupby(["model", "fraction", "budget", "rayleigh"], dropna=False)
           .agg(n_seeds=("seed", "size"), **{c: (c, "mean") for c in both.columns
                                             if c not in keys + ["rayleigh"]}).reset_index())
    return out


def nusselt_by_rayleigh_figure(summary: pd.DataFrame, label: str, stem: Path) -> list[str]:
    """(a) median rollout-averaged Nu against Ra, true and predicted; (b) median signed relative error;
    (c) median |relative error|. Our models at 10% (open, dashed) and 100% (filled, solid) at one checkpoint."""
    s = summary[summary.rayleigh.notna()]
    ours = s[(s.budget == label) & s.model.isin(MODEL_NAMES)]
    if ours.empty:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12.0, 3.6))
    truth = s[s.model == "persistence"].sort_values("rayleigh")
    a.fill_between(truth.rayleigh, truth.nu_true_q25, truth.nu_true_q75, color=INK, alpha=0.12, lw=0)
    a.plot(truth.rayleigh, truth.nu_true, color=INK, lw=2.0)
    for m in _models(ours.model):
        st = STYLE[m]
        for frac, ls, fill in ((0.1, "--", "white"), (1.0, "-", st["color"])):
            g = ours[(ours.model == m) & (ours.fraction.round(3) == frac)].sort_values("rayleigh")
            if g.empty:
                continue
            kw = dict(color=st["color"], ls=ls, marker=st["marker"], ms=5, mfc=fill, lw=1.4)
            a.plot(g.rayleigh, g.nu_pred, **kw)
            b.plot(g.rayleigh, 100 * g.rel_err_signed, **kw)
            c.plot(g.rayleigh, 100 * g.rel_err_abs, **kw)
    for name, color, ls in (("persistence", PERSIST["color"], ":"), ("released_cnext_unet", INK, "-.")):
        g = s[s.model == name].sort_values("rayleigh")
        if not g.empty:
            if name != "persistence":
                a.plot(g.rayleigh, g.nu_pred, color=color, ls=ls, lw=1.2)
            b.plot(g.rayleigh, 100 * g.rel_err_signed, color=color, ls=ls, lw=1.3)
            c.plot(g.rayleigh, 100 * g.rel_err_abs, color=color, ls=ls, lw=1.3)
    a.set_yscale("log")
    a.set_ylabel("Nu averaged over the rollout (median)")
    a.set_title("(a) Heat transport, true (black, band: quartiles) and predicted", fontsize=9, color=INK)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_ylabel("median signed error (%)")
    b.set_title("(b) Bias: below 0 = too little heat transport", fontsize=9, color=INK)
    c.set_ylabel("median |error| (%)")
    c.set_title("(c) Size of the error", fontsize=9, color=INK)
    for ax in (a, b, c):
        ax.set_xscale("log")
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
        ax.set_xlabel("Rayleigh number")
        _style_axes(ax)
    handles = [Line2D([], [], color=INK, lw=2.0, label="true")]
    handles += [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=5, label=STYLE[m]["label"])
                for m in _models(ours.model)]
    handles += [Line2D([], [], color=MUTED, ls="--", marker="o", mfc="white", ms=5, label="10% data"),
                Line2D([], [], color=MUTED, ls="-", marker="o", ms=5, label="100% data"),
                Line2D([], [], color=INK, ls="-.", lw=1.2, label="released CNextU-Net"),
                Line2D([], [], color=PERSIST["color"], ls=":", lw=1.3, label="persistence")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Heat transport over 30-step rollouts from frames 80 and 140, {label} checkpoints: median over the "
                 "70 rollouts per Rayleigh number (7 Pr × 5 trajectories × 2 starts), mean over two seeds",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _finish(fig, stem)


# --- numerical solver (Dedalus) against the models -------------------------------------------------------

def _solvers(outputs: Path) -> dict[str, Path]:
    """Scored solver resolutions (outputs/baselines/solver_<res>/evaluations), finest first."""
    base = outputs / "baselines"
    found = {d.parent.name.removeprefix("solver_"): d for d in sorted(base.glob("solver_*/evaluations"))
             if (d / "rollout_na.parquet").exists()} if base.exists() else {}
    return dict(sorted(found.items(), key=lambda kv: -int(kv[0].split("x")[0])))


def _subset(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    idx = pd.MultiIndex.from_frame(df[["traj_id", "start"]])
    return df[idx.isin(list(keys))]


MATCH_STEPS = (1, 10, 30)          # the matching rule's steps (WP6, D29)
SU_PER_GPU_HOUR_EQ, SU_PER_CORE_HOUR_EQ = 32, 1  # SU equivalents of the plan (docs/final_plan.md WP6)


def _latency_seconds(clean_json: Path) -> float:
    return json.loads(clean_json.read_text())["latency"]["ms_per_rollout"] / 1000 if clean_json.exists() else np.nan


def solver_comparison(runs: pd.DataFrame, outputs: Path, label: str = "4h") -> pd.DataFrame:
    """Every source on the solver's rollouts, per Rayleigh number: VRMSE at the matching steps and its cost.

    Sources: the Dedalus grids (one core per rollout, build + stepping), our models at `label` (seed mean; one A100,
    batch 1), the released CNextU-Net and persistence. A diverged solver rollout scores inf from its first missing
    step (`solver_eval`), so a regime's mean is inf there: the rule counts it as a failure.
    """
    solvers = _solvers(outputs)
    if not solvers:
        return pd.DataFrame()
    rolls = {res: pd.read_parquet(d / "rollout_na.parquet") for res, d in solvers.items()}
    keys = set(map(tuple, next(iter(rolls.values()))[["traj_id", "start"]].drop_duplicates().values))
    rows = []

    def add(source, kind, tables, seconds, model="", fraction=np.nan, grid="", sec_range=(np.nan, np.nan),
            n_div=None, device=""):
        for ra in sorted({k for t in tables for k in t.rayleigh.unique()}):
            per = [t[t.rayleigh == ra] for t in tables]
            r = {"source": source, "kind": kind, "model": model, "fraction": fraction, "grid": grid, "rayleigh": ra,
                 "n_rollouts": len(per[0].groupby(["traj_id", "start"])), "n_seeds": len(per)}
            for k in MATCH_STEPS:
                r[f"vrmse_step{k}"] = float(np.mean([t[t.step == k].vrmse_mean.mean() for t in per]))
            r["vrmse_mean_30"] = float(np.mean([t.vrmse_mean.mean() for t in per]))
            r["n_diverged"] = n_div.get(ra, 0) if n_div else 0
            sec = seconds.get(ra, np.nan) if isinstance(seconds, dict) else seconds
            lo, hi = (sec_range[0].get(ra, np.nan), sec_range[1].get(ra, np.nan)) if isinstance(sec_range[0], dict) \
                else sec_range
            rows.append({**r, "seconds_per_rollout": sec, "seconds_min": lo, "seconds_max": hi, "device": device})

    for res, roll in rolls.items():
        roll = roll[roll.sigma == 0]
        rp = solvers[res] / "runs_na.parquet"
        sr = pd.read_parquet(rp) if rp.exists() else pd.DataFrame()
        div = roll.groupby(["rayleigh", "traj_id", "start"]).vrmse_mean.apply(lambda v: not np.isfinite(v).all())
        n_div = div.groupby(level="rayleigh").sum().astype(int).to_dict()
        g = sr.groupby("rayleigh").seconds_per_rollout if not sr.empty else None
        add(f"Dedalus {res}", "solver", [roll], g.median().to_dict() if g is not None else np.nan, grid=res,
            sec_range=(g.min().to_dict(), g.max().to_dict()) if g is not None else (np.nan, np.nan), n_div=n_div,
            device="1 CPU core: " + ", ".join(sorted(sr.cpu.unique())) if not sr.empty else "1 CPU core")
    evals = evaluated_runs(runs, outputs)
    for (model, fraction), ev in evals[evals.label == label].groupby(["model", "fraction"]):
        tables = [_subset(pd.read_parquet(e.dir / f"rollout_{label}.parquet").query("sigma == 0"), keys)
                  for _, e in ev.iterrows()]
        sec = float(np.nanmean([_latency_seconds(e.dir / f"clean_{label}.json") for _, e in ev.iterrows()]))
        add(f"{STYLE[model]['label']} {fraction:.0%} {label}", "model", tables, sec, model=model, fraction=fraction,
            device="1 A100, batch 1")
    for name, d in (("Released CNextU-Net", outputs / "baselines" / "released" / "cnext_unet" / "evaluations"),
                    ("Persistence", outputs / "baselines" / "persistence" / "evaluations")):
        if (d / "rollout_na.parquet").exists():
            t = _subset(pd.read_parquet(d / "rollout_na.parquet").query("sigma == 0"), keys)
            add(name, "reference", [t], _latency_seconds(d / "clean_na.json"),
                model="cnext_unet" if name.startswith("Released") else "persistence", device="1 A100, batch 1")
    return pd.DataFrame(rows)


def solver_matching(comp: pd.DataFrame, train_su: dict[tuple[str, float], float]) -> pd.DataFrame:
    """Per Rayleigh number and target model: the coarsest grid at or below the target's VRMSE at steps 1, 10, 30.

    The rule was fixed before the solver ran (docs/final_plan.md WP6, D29). The main target is our CNextU-Net at
    100% / 4 h; the 10% one and the released checkpoint are shown too. Cost per rollout in SU equivalents (1 per
    core-hour; 32 per GPU-hour); break-even: the number of rollouts after which training plus inference costs less
    than the solver at its matching grid.
    """
    if comp.empty:
        return pd.DataFrame()
    grids = comp[comp.kind == "solver"].copy()
    grids["cells"] = grids.grid.map(lambda g: int(g.split("x")[0]) * int(g.split("x")[1]))
    targets = [("CNextU-Net 100% 4h", ("cnext_unet", 1.0)), ("CNextU-Net 10% 4h", ("cnext_unet", 0.1)),
               ("Released CNextU-Net", None)]
    rows = []
    for ra in sorted(grids.rayleigh.unique()):
        for name, key in targets:
            t = comp[(comp.source == name) & (comp.rayleigh == ra)]
            if t.empty:
                continue
            t = t.iloc[0]
            g = grids[grids.rayleigh == ra].sort_values("cells")              # coarsest first
            ok = np.ones(len(g), bool)
            fails = {}
            for k in MATCH_STEPS:
                passed = (g[f"vrmse_step{k}"] <= t[f"vrmse_step{k}"]).to_numpy()
                ok &= passed
                for grid, p in zip(g.grid, passed):
                    if not p:
                        fails.setdefault(grid, []).append(k)
            match = g[ok].iloc[0] if ok.any() else None
            model_su = t.seconds_per_rollout / 3600 * SU_PER_GPU_HOUR_EQ
            row = {"rayleigh": ra, "target": name, "target_step1": t.vrmse_step1, "target_step10": t.vrmse_step10,
                   "target_step30": t.vrmse_step30, "target_seconds": t.seconds_per_rollout, "target_su": model_su,
                   "matching_grid": match.grid if match is not None else "none",
                   "failed_steps_by_grid": "; ".join(f"{gr}: {','.join(map(str, ks))}" for gr, ks in fails.items())}
            if match is not None:
                solver_su = match.seconds_per_rollout / 3600 * SU_PER_CORE_HOUR_EQ
                tr = train_su.get(key, np.nan) if key else np.nan
                row.update({"solver_step1": match.vrmse_step1, "solver_step10": match.vrmse_step10,
                            "solver_step30": match.vrmse_step30, "solver_seconds": match.seconds_per_rollout,
                            "solver_seconds_min": match.seconds_min, "solver_seconds_max": match.seconds_max,
                            "solver_su": solver_su, "train_su": tr,
                            "break_even_rollouts": tr / (solver_su - model_su) if solver_su > model_su else np.inf})
            rows.append(row)
    return pd.DataFrame(rows)


def solver_matching_figure(comp: pd.DataFrame, match: pd.DataFrame, outputs: Path, runs: pd.DataFrame,
                           label: str, stem: Path) -> list[str]:
    """Row 1: VRMSE against rollout step per regime (grids, CNextU-Net 10% / 100%, released, persistence).
    Row 2: each grid's VRMSE over CNextU-Net 100%'s at steps 1, 10, 30 against seconds per rollout; a grid
    matches when all three are at or below 1 (the rule); the matching grid is ringed."""
    if comp.empty:
        return []
    plt = _plt()
    from matplotlib.lines import Line2D

    solvers = _solvers(outputs)
    rolls = {res: pd.read_parquet(d / "rollout_na.parquet").query("sigma == 0") for res, d in solvers.items()}
    keys = set(map(tuple, next(iter(rolls.values()))[["traj_id", "start"]].drop_duplicates().values))
    evals = evaluated_runs(runs, outputs)
    curves = {}
    for frac in (1.0, 0.1):
        ev = evals[(evals.model == "cnext_unet") & (evals.fraction.round(3) == frac) & (evals.label == label)]
        curves[f"CNextU-Net {frac:.0%}"] = [_subset(pd.read_parquet(e.dir / f"rollout_{label}.parquet")
                                                    .query("sigma == 0"), keys) for _, e in ev.iterrows()]
    for name, d in (("released", outputs / "baselines" / "released" / "cnext_unet" / "evaluations"),
                    ("persistence", outputs / "baselines" / "persistence" / "evaluations")):
        if (d / "rollout_na.parquet").exists():
            curves[name] = [_subset(pd.read_parquet(d / "rollout_na.parquet").query("sigma == 0"), keys)]
    ras = sorted(comp.rayleigh.unique())
    grid_names = list(rolls)                                            # finest first
    shades = plt.get_cmap("Greys")(np.linspace(0.95, 0.35, len(grid_names)))
    gcol = dict(zip(grid_names, shades))
    curve_style = {"CNextU-Net 100%": dict(color=STYLE["cnext_unet"]["color"], lw=2.2),
                   "CNextU-Net 10%": dict(color=STYLE["cnext_unet"]["color"], lw=1.5, ls="--"),
                   "released": dict(color=INK, lw=1.3, ls="-."), "persistence": dict(color=PERSIST["color"], lw=1.3, ls=":")}
    fig, axes = plt.subplots(2, len(ras), figsize=(4.2 * len(ras), 7.2), squeeze=False)
    for j, ra in enumerate(ras):
        a, b = axes[0][j], axes[1][j]
        for res, roll in rolls.items():
            gm = roll[roll.rayleigh == ra].groupby("step").vrmse_mean.mean()
            a.plot(gm.index, gm.values, color=gcol[res], lw=1.2)
        for name, ts in curves.items():
            if ts:
                gm = pd.concat([t[t.rayleigh == ra].groupby("step").vrmse_mean.mean() for t in ts], axis=1).mean(axis=1)
                a.plot(gm.index, gm.values, **curve_style[name])
        a.set_yscale("log")
        a.set_xlabel("rollout step")
        a.set_title(f"Ra = 1e{int(round(np.log10(ra)))}, Pr = 1: error along the rollout", fontsize=9, color=INK)
        tgt = comp[(comp.source == f"CNextU-Net 100% {label}") & (comp.rayleigh == ra)]
        g = comp[(comp.kind == "solver") & (comp.rayleigh == ra)]
        if not tgt.empty:
            t = tgt.iloc[0]
            for k, mk in zip(MATCH_STEPS, ("o", "s", "^")):
                ratio = g[f"vrmse_step{k}"] / t[f"vrmse_step{k}"]
                fin = np.isfinite(ratio)
                b.plot(g.seconds_per_rollout[fin], ratio[fin], marker=mk, color=MUTED, ms=5, lw=0.8)
                b.plot(g.seconds_per_rollout[~fin], np.full((~fin).sum(), 30.0), marker=mk, color="#c62828", ms=6,
                       ls="none")
            for _, r in g.iterrows():                                  # label under each grid's step-30 point
                y30 = r.vrmse_step30 / t.vrmse_step30
                b.annotate(r.grid, (r.seconds_per_rollout, y30 if np.isfinite(y30) else 30.0), xytext=(0, -12),
                           textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
            b.axhline(1.0, color=STYLE["cnext_unet"]["color"], lw=1.4)
            b.axvline(t.seconds_per_rollout, color=STYLE["cnext_unet"]["color"], lw=1.0, ls="--")
            m = match[(match.rayleigh == ra) & (match.target == f"CNextU-Net 100% {label}")]
            if not m.empty and m.iloc[0].matching_grid != "none":
                mg = g[g.grid == m.iloc[0].matching_grid].iloc[0]
                b.plot([mg.seconds_per_rollout], [1.0], marker="o", ms=16, mfc="none", mec=STYLE["cnext_unet"]["color"],
                       mew=1.6)
                b.set_title(f"matching grid: {mg.grid} ({mg.seconds_per_rollout:.0f} s per rollout)", fontsize=9,
                            color=INK)
            else:
                b.set_title("no grid matches CNextU-Net at steps 1, 10 and 30", fontsize=9, color=INK)
        b.set_xscale("log")
        b.set_yscale("log")
        b.set_xlabel("seconds per 30-step rollout")
        for ax in (a, b):
            _style_axes(ax)
    axes[0][0].set_ylabel("VRMSE, mean over 10 rollouts")
    axes[1][0].set_ylabel("solver VRMSE / CNextU-Net 100% VRMSE")
    fig.align_ylabels(axes[:, 0])
    handles = [Line2D([], [], color=gcol[r], lw=1.4, label=f"Dedalus {r}") for r in grid_names]
    handles += [Line2D([], [], label=n if n.startswith("CNextU") else
                       {"released": "released CNextU-Net", "persistence": "persistence"}[n], **curve_style[n])
                for n in curves]
    handles += [Line2D([], [], marker=mk, color=MUTED, ls="none", label=f"step {k}")
                for k, mk in zip(MATCH_STEPS, ("o", "s", "^"))]
    handles.append(Line2D([], [], marker="o", color="#c62828", ls="none", label="diverged (plotted at 30)"))
    handles.append(Line2D([], [], color=STYLE["cnext_unet"]["color"], lw=1.0, ls="--",
                          label="CNextU-Net time per rollout (1 A100, batch 1)"))
    handles.append(Line2D([], [], marker="o", ms=10, mfc="none", mec=STYLE["cnext_unet"]["color"], ls="none",
                          label="matching grid (solver: 1 CPU core)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Numerical solver (Dedalus, restarted from the last input frame) against CNextU-Net ({label}) on the "
                 "same 10 rollouts per regime (5 test trajectories × starts 80, 140)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
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


TRAINING_PURPOSES = ["learning-rate search", "10% runs", "100% runs", "dedicated 1 h runs", "UQ ensemble training"]


def training_purpose(manifest: str, fraction: float) -> str:
    """What a training job was for, from its run's manifest (and data fraction for the core study)."""
    if manifest == "train":
        return f"{round(float(fraction) * 100)}% runs"
    return {"lr_search": "learning-rate search", "train_1h": "dedicated 1 h runs",
            "train_1h_f100": "dedicated 1 h runs", "uq": "UQ ensemble training"}.get(manifest, "training (other)")


def sacct_table(path: Path, attempts: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """One row per Slurm job (array task) with billed SU, GPU-hours, queue wait and what the job was for.

    RCIC bills `billing` x elapsed hours on paid partitions (32 per GPU + 1 per core); free partitions cost 0.
    Jobs cancelled before they started have no allocation and cost nothing.
    """
    df = pd.read_csv(path, sep="|", dtype=str)
    rows = []
    purpose_of = ({rid: training_purpose(m, f) for rid, m, f in zip(runs.run_id, runs.manifest, runs.fraction)}
                  if not runs.empty else {})
    task_of = {}
    if not attempts.empty:
        for _, a in attempts.iterrows():
            if isinstance(a.array_task, str) and a.array_task:
                task_of[a.array_task] = purpose_of.get(a.run_id, "")
            if pd.notna(a.job_id):
                task_of.setdefault(str(a.job_id), purpose_of.get(a.run_id, ""))
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
                "rb4-profile": "profiling", "rb4-eval": "evaluation", "rb4-visualize": "flow snapshots",
                "rb4-uq": "UQ evaluation", "rb4-solver": "numerical solver", "rb4-solver-score": "numerical solver",
                "rb4-onset": "data", "rb4-dedalus-setup": "environment"}.get(name, name)
        if name == "rb4-train":
            what = task_of.get(str(r.JobID)) or "training (other)"
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
    known = out[out.what.isin(TRAINING_PURPOSES)].groupby("array").what.first()
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

    colors = {"learning-rate search": "#1baf7a", "10% runs": "#2a78d6", "100% runs": "#0b3f8a",
              "dedicated 1 h runs": "#7fb2ea", "UQ ensemble training": "#c2185b", "UQ evaluation": "#f48fb1",
              "numerical solver": "#5d4037", "flow snapshots": "#9e9d24", "evaluation": "#eda100",
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
