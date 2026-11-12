#!/usr/bin/env python
"""Figures for the written reports (reports/*.tex), drawn at their printed size.

The exploratory figures in outputs/findings/ (scripts/findings.py) show everything; these show one message each,
sized for a 6.9 in text width so the 7-8 pt text stays 7-8 pt on the page. They carry no titles (the LaTeX
captions do), use a legend plus selective labels, and keep text in neutral ink. Run after scripts/findings.py:

    .venv/bin/python scripts/report_figures.py            # writes reports/figures/fig_*.pdf (and .png previews)

Inputs: outputs/findings/runs.csv, outputs/runs/*/evaluations, outputs/baselines/{persistence,released},
outputs/pilots/bench_data.json, outputs/lr_search/pilot_timings.csv, outputs/visuals/*.npz.
Figures whose inputs are missing are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import findings_figures as ff
import matplotlib.pyplot as plt
import matplotlib.ticker
from findings import load_metrics
from matplotlib.lines import Line2D

from rbbench import MODEL_NAMES
from rbbench.aggregate import GRID, INK, MUTED, ONSET_FRAMES, STYLE

FULL, HALF = 6.9, 3.35            # text width and a half column, inches (reports/midproject.tex: 0.8 in margins)
PERSIST = STYLE["persistence"]
TRUTH_LW = 2.0

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        p = out / f"{name}.{ext}"
        fig.savefig(p, **kw)
        paths.append(str(p))
    plt.close(fig)
    return paths


def onset_band(ax, label: bool = True):
    ax.axvspan(-2, ONSET_FRAMES, color=MUTED, alpha=0.08, lw=0)
    if label:
        ax.text(ONSET_FRAMES / 2, 0.97, "onset", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=MUTED)
        ax.text(ONSET_FRAMES + 8, 0.97, "developed convection", transform=ax.get_xaxis_transform(), ha="left",
                va="top", fontsize=7, color=MUTED)


# --- 1. released checkpoints by window (pipeline check) ----------------------------------------------------

def released_by_window(outputs: Path, out: Path, models=("cnext_unet", "tfno"),
                       name: str = "fig_released_by_window") -> list[str]:
    """Released checkpoints' one-step VRMSE by window (the released TFNO uses its paper-era forward pass)."""
    tables = {m: t for m in models if (t := ff._released(outputs, m, "one_step")) is not None}
    if not tables:
        return []
    persist = ff._persistence(outputs, "one_step")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35), sharey=True)
    for ax, col, title in ((axes[0], "normalised_vrmse_mean", "Paper protocol (normalised fields)"),
                           (axes[1], "vrmse_mean", "Physical fields")):
        onset_band(ax)
        for model, t in tables.items():
            g = t.groupby("start")[col].mean()
            ax.plot(g.index, g.values, color=STYLE[model]["color"])
        if persist is not None:
            g = persist.groupby("start")[col].mean()
            ax.plot(g.index, g.values, color=PERSIST["color"], ls=":", lw=1.3)
        ax.set_yscale("log")
        ax.set_xlim(-2, 197)
        ax.set_title(title)
        ax.set_xlabel("Input-window start frame")
    axes[0].set_ylabel("One-step VRMSE")
    axes[1].tick_params(labelleft=True)    # shared scale, but labelled on both panels
    handles = [Line2D([], [], color=STYLE[m]["color"], label=f"released {STYLE[m]['label']}") for m in tables]
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], ls=":", lw=1.3, label="persistence (copy last frame)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, name)


# --- 2. data loading against demand --------------------------------------------------------------------------

def loader(outputs: Path, out: Path) -> list[str]:
    bench, timings = outputs / "pilots" / "bench_data.json", outputs / "lr_search" / "pilot_timings.csv"
    if not bench.exists() or not timings.exists():
        return []
    b = pd.DataFrame(json.loads(bench.read_text())["rows"])
    pt = pd.read_csv(timings)
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    for _, r in pt.iterrows():
        ax.axhline(r.gpu_samples_per_s, color=STYLE[r.model]["color"], lw=1.1)
        ax.text(15.8, r.gpu_samples_per_s, f"{STYLE[r.model]['label']} needs {r.gpu_samples_per_s:.0f}/s",
                ha="left", va="center", fontsize=7, color=INK, clip_on=False)
    for loader_name, color, label in (("persistent", INK, "our loader"),
                                      ("upstream", MUTED, "upstream loader")):
        d = b[b.loader == loader_name].sort_values("workers")
        ax.plot(d.workers, d.samples_per_s, color=color, marker="o", ms=4, label=label)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.6))
    ax.set_xticks(sorted(b.workers.unique()))
    ax.set_xlim(3.3, 15.5)
    ax.set_ylim(0, pt.gpu_samples_per_s.max() * 1.1)
    ax.set_xlabel("Data-loader worker processes")
    ax.set_ylabel("Training samples per second")
    fig.tight_layout()
    return save(fig, out, "fig_loader")


# --- 3. training curves (all models on shared axes) -----------------------------------------------------------

def training(outputs: Path, runs: pd.DataFrame, out: Path) -> list[str]:
    runs = runs[(runs.manifest == "train") & (runs.fraction == 0.1)]
    if runs.empty:
        return []
    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(FULL, 2.5))
    seed_ls = {0: "-", 1: "--"}
    for model in [m for m in MODEL_NAMES if m in set(runs.model)]:
        st = STYLE[model]
        for _, r in runs[runs.model == model].sort_values("seed").iterrows():
            _, m = load_metrics(outputs / "runs" / r.run_id)
            tr = m[m.event == "train"].copy()
            tr["h"] = tr.train_seconds / 3600
            # smooth over ~3 minutes of training so the three models' noise levels are comparable
            smooth = tr.set_index(pd.to_timedelta(tr.h, unit="h")).loss.rolling("3min").mean()
            ax_loss.plot(tr.h, smooth.values, color=st["color"], ls=seed_ls.get(int(r.seed), ":"), lw=1.1)
            val = m[m.val_vrmse.notna()].drop_duplicates("step", keep="last")
            ax_val.plot(val.train_seconds / 3600, val.val_vrmse, color=st["color"], ls=seed_ls.get(int(r.seed), ":"),
                        marker=st["marker"], ms=3.5, lw=1.1)
    for ax in (ax_loss, ax_val):
        for x in (1, 4):
            ax.axvline(x, color=MUTED, lw=0.6, alpha=0.5)
        ax.set_xlim(0, 4.1)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("Active training time (h)")
    ax_loss.text(1.04, 0.97, "1 h checkpoint", transform=ax_loss.get_xaxis_transform(), fontsize=7, color=MUTED,
                 va="top")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Training MSE")
    ax_val.set_ylabel("Validation VRMSE")
    ax_val.set_ylim(0, None)
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3.5, label=STYLE[m]["label"])
               for m in MODEL_NAMES if m in set(runs.model)]
    handles += [Line2D([], [], color=MUTED, ls="-", label="seed 0"), Line2D([], [], color=MUTED, ls="--", label="seed 1")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_training")


# --- 4. test error by window: ours against the references --------------------------------------------------------

def test_by_window(outputs: Path, runs: pd.DataFrame, out: Path, label: str = "4h") -> list[str]:
    evals = ff.evaluated_runs(runs, outputs)
    tables = ff._load(evals, "one_step", label) if not evals.empty else {}
    if not tables:
        return []
    released = ff._released(outputs, "cnext_unet", "one_step")
    persist = ff._persistence(outputs, "one_step")
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    onset_band(ax)
    for model in [m for m in MODEL_NAMES if m in tables]:
        mean, lo, hi = ff._seed_mean(tables[model], "start", "vrmse_mean")
        if len(tables[model]) > 1:
            ax.fill_between(mean.index, lo, hi, color=STYLE[model]["color"], alpha=0.15, lw=0)
        ax.plot(mean.index, mean.values, color=STYLE[model]["color"])
    if released is not None:
        g = released.groupby("start").vrmse_mean.mean()
        ax.plot(g.index, g.values, color=INK, lw=1.1, ls="--")
    if persist is not None:
        g = persist.groupby("start").vrmse_mean.mean()
        ax.plot(g.index, g.values, color=PERSIST["color"], lw=1.3, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(-2, 197)
    ax.set_xlabel("Input-window start frame")
    ax.set_ylabel("One-step VRMSE")
    handles = [Line2D([], [], color=STYLE[m]["color"], label=f"{STYLE[m]['label']} (ours, 10%, {label})")
               for m in MODEL_NAMES if m in tables]
    if released is not None:
        handles.append(Line2D([], [], color=INK, lw=1.1, ls="--", label="released CNextU-Net (100%)"))
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_test_by_window")


# --- 4b. rollouts: error by step (developed convection) and survival below VRMSE 1 ---------------------------

def rollouts(outputs: Path, runs: pd.DataFrame, out: Path, label: str = "4h", threshold: float = 1.0) -> list[str]:
    from rbbench.evaluate import failure_step, kaplan_meier

    evals = ff.evaluated_runs(runs, outputs)
    tables = ff._load(evals, "rollout", label) if not evals.empty else {}
    if not tables:
        return []
    persist = ff._persistence(outputs, "rollout")
    sources = dict(tables)
    if persist is not None:
        sources["persistence"] = [persist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.45))
    for name, ts in sources.items():
        st = STYLE[name]
        ls = ":" if name == "persistence" else "-"
        dev = [t[(t.sigma == 0) & t.start.isin([80, 140])].groupby("step").vrmse_mean.mean() for t in ts]
        mean = pd.concat(dev, axis=1).mean(axis=1)
        a.plot(mean.index, mean.values, color=st["color"], ls=ls, marker=st.get("marker") or None, ms=3, markevery=5)
        curves = []
        for t in ts:
            piv = t[t.sigma == 0].pivot_table(index=["traj_id", "start"], columns="step", values="vrmse_mean")
            curves.append(kaplan_meier([failure_step(v.values, threshold) for _, v in piv.iterrows()], piv.shape[1]))
        c = np.mean(curves, axis=0)
        b.step(np.arange(0, len(c) + 1), np.r_[1.0, c], where="post", color=st["color"], ls=ls)
    a.set_title("(a) Error, rollouts from frames 80 and 140")
    a.set_ylabel("Rollout VRMSE")
    b.set_title(f"(b) Rollouts staying below VRMSE {threshold:g}, all starts")
    b.set_ylabel("Fraction of rollouts")
    b.set_ylim(0, 1.03)
    for ax in (a, b):
        ax.set_xlabel("Rollout step")
        ax.set_xlim(0, 30.5)
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3, label=f"{STYLE[m]['label']} ({label})")
               for m in MODEL_NAMES if m in tables]
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], ls=":", label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_rollouts")


# --- 4c. accuracy against cost: training time, allocated GPU time, inference time --------------------------------

def _developed(t: pd.DataFrame) -> float:
    """One-step VRMSE (physical fields) over windows whose first input frame is >= ONSET_FRAMES."""
    if "sigma" in t:
        t = t[t.sigma == 0]
    return float(t[t.start >= ONSET_FRAMES].vrmse_mean.mean())


def _rollout_end(t: pd.DataFrame, step: int = 30, starts=(80, 140)) -> float:
    """Rollout VRMSE at `step`, mean over rollouts from the developed-convection starts."""
    if "sigma" in t:
        t = t[t.sigma == 0]
    return float(t[t.start.isin(starts) & (t.step == step)].vrmse_mean.mean())


def accuracy_cost(outputs: Path, runs: pd.DataFrame, out: Path) -> list[str]:
    evals = ff.evaluated_runs(runs, outputs)
    res_csv = outputs / "tables" / "resource_use.csv"
    if evals.empty or not res_csv.exists():
        return []
    one = {lab: ff._load(evals, "one_step", lab) for lab in ("1h", "4h")}
    roll = ff._load(evals, "rollout", "4h")
    if not one["4h"] or not roll:
        return []
    res = pd.read_csv(res_csv)
    res = res[(res.budget == "4h") & (res.fraction.round(3) == 0.1)]
    train = runs[(runs.manifest == "train") & (runs.fraction.round(3) == 0.1)]
    released = ff._released(outputs, "cnext_unet", "one_step")
    p_one, p_roll = ff._persistence(outputs, "one_step"), ff._persistence(outputs, "rollout")
    models = [m for m in MODEL_NAMES if m in one["4h"]]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.35), gridspec_kw={"width_ratios": [0.8, 1, 1]})
    for m in models:
        st = dict(color=STYLE[m]["color"], marker=STYLE[m]["marker"])
        # (a) developed one-step error at the 1 h and 4 h checkpoints, seed mean with seed range
        xs, ys, lo, hi = [], [], [], []
        for h, lab in ((1, "1h"), (4, "4h")):
            v = [_developed(t) for t in one[lab].get(m, [])]
            if v:
                xs.append(h), ys.append(np.mean(v)), lo.append(np.mean(v) - min(v)), hi.append(max(v) - np.mean(v))
        a.errorbar(xs, ys, yerr=[lo, hi], ms=4, capsize=2, lw=1.4, **st)
        # (b) the same 4 h error against the GPU time each run held (training + data waits + overhead)
        dev4 = np.mean([_developed(t) for t in one["4h"][m]])
        wall = train[train.model == m].wall_hours
        b.errorbar([wall.mean()], [dev4], xerr=[[wall.mean() - wall.min()], [wall.max() - wall.mean()]],
                   ms=5, capsize=2, lw=1.0, ls="none", **st)
        b.annotate(STYLE[m]["label"], (wall.max(), dev4), xytext=(5, 0), textcoords="offset points",
                   va="center", fontsize=7, color=INK)
        # (c) error after 30 steps from developed flow against the time to compute those 30 steps
        r30 = np.mean([_rollout_end(t) for t in roll[m]])
        ms = res[res.model == m].rollout_ms.mean()
        c.plot([ms], [r30], ms=5, ls="none", **st)
        above = m == "avit"                                    # keeps its label clear of the persistence line
        c.annotate(f"{STYLE[m]['label']}\n{ms:.0f} ms", (ms, r30), xytext=(0, 7 if above else -7),
                   textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=7, color=INK)
    refs = []
    if released is not None:
        v = _developed(released)
        for ax in (a, b):
            ax.axhline(v, color=INK, lw=1.0, ls="--")
        refs.append(Line2D([], [], color=INK, lw=1.0, ls="--", label="released CNextU-Net (100% data)"))
    if p_one is not None:
        v = _developed(p_one)
        for ax in (a, b):
            ax.axhline(v, color=PERSIST["color"], lw=1.2, ls=":")
        refs.append(Line2D([], [], color=PERSIST["color"], lw=1.2, ls=":", label="persistence"))
    if p_roll is not None:
        c.axhline(_rollout_end(p_roll), color=PERSIST["color"], lw=1.2, ls=":")
    for ax in (a, b):
        ax.set_yscale("log")
        ax.set_ylim(0.03, 0.3)
        ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.3])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylabel("One-step VRMSE, developed")
    a.set_xticks([1, 4])
    a.set_xlim(0.4, 4.6)
    a.set_xlabel("Active training time (h)")
    a.set_title("(a) More training time")
    b.axvline(4, color=MUTED, lw=0.8)
    b.text(4.3, 0.16, "4 h of active\ntraining", fontsize=7, color=MUTED, va="center")
    b.set_xlim(0, 15.5)
    b.set_xlabel("Allocated GPU-hours per 4-h run")
    b.set_title("(b) Cost to train")
    c.set_xlim(0, 300)
    c.set_ylim(0.3, 0.9)
    c.set_xlabel("Inference time per 30-step rollout (ms)")
    c.set_ylabel("VRMSE after 30 steps, developed")
    c.set_title("(c) Cost to use (4-h models)")
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=4, label=f"{STYLE[m]['label']} (ours, 10%)")
               for m in models] + refs
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98),
               columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    return save(fig, out, "fig_accuracy_cost")


# --- 4d. error by Rayleigh number: one step, 30-step rollout, heat transport -------------------------------------

def _clean(t: pd.DataFrame) -> pd.DataFrame:
    return t[t.sigma == 0] if "sigma" in t else t


def _by_ra_one_step(t: pd.DataFrame) -> pd.Series:
    t = _clean(t)
    return t[t.start >= ONSET_FRAMES].groupby("rayleigh").vrmse_mean.mean()


def _by_ra_rollout(t: pd.DataFrame, step: int = 30, starts=(80, 140)) -> pd.Series:
    t = _clean(t)
    return t[t.start.isin(starts) & (t.step == step)].groupby("rayleigh").vrmse_mean.mean()


def _by_ra_nusselt(t: pd.DataFrame, starts=(80, 140)) -> pd.Series:
    """Median |relative error| of the Nusselt number averaged over each rollout, per Rayleigh number."""
    t = _clean(t)
    g = t[t.start.isin(starts)].groupby(["rayleigh", "traj_id", "start"])[["nu_pred", "nu_true"]].mean()
    return ((g.nu_pred - g.nu_true) / g.nu_true).abs().groupby(level="rayleigh").median()


def by_rayleigh(outputs: Path, runs: pd.DataFrame, out: Path, label: str = "4h") -> list[str]:
    evals = ff.evaluated_runs(runs, outputs)
    one, roll, phys = (ff._load(evals, k, label) for k in ("one_step", "rollout", "physics"))
    if not one or not roll or not phys:
        return []
    released = ff._released(outputs, "cnext_unet", "one_step")
    p_one, p_roll = ff._persistence(outputs, "one_step"), ff._persistence(outputs, "rollout")
    p_phys_path = outputs / "baselines" / "persistence" / "evaluations" / "physics_na.parquet"
    p_phys = pd.read_parquet(p_phys_path) if p_phys_path.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    panels = ((axes[0], one, _by_ra_one_step, p_one, "One-step VRMSE, developed", "(a) One step"),
              (axes[1], roll, _by_ra_rollout, p_roll, "VRMSE after 30 steps", "(b) 30 steps from frames 80, 140"),
              (axes[2], phys, _by_ra_nusselt, p_phys, "Nusselt-number error", "(c) Heat transport over 30 steps"))
    for ax, tables, fn, persist, ylabel, title in panels:
        for m in [m for m in MODEL_NAMES if m in tables]:
            s = pd.concat([fn(t) for t in tables[m]], axis=1).mean(axis=1)
            ax.plot(s.index, s.values, color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3.5)
        if persist is not None:
            s = fn(persist)
            ax.plot(s.index, s.values, color=PERSIST["color"], ls=":", lw=1.3)
        ax.set_xscale("log")
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
        ax.set_xlabel("Rayleigh number")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    if released is not None:
        s = _by_ra_one_step(released)
        axes[0].plot(s.index, s.values, color=INK, lw=1.0, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3.5,
                      label=f"{STYLE[m]['label']} ({label})") for m in MODEL_NAMES if m in one]
    if released is not None:
        handles.append(Line2D([], [], color=INK, lw=1.0, ls="--", label="released CNextU-Net (100%)"))
    handles.append(Line2D([], [], color=PERSIST["color"], ls=":", lw=1.3, label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98),
               columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    return save(fig, out, "fig_by_rayleigh")


# --- 5. spectra: where in scale the error is ----------------------------------------------------------------------

def spectra(outputs: Path, runs: pd.DataFrame, out: Path, label: str = "4h", regime=(1e8, 1.0)) -> list[str]:
    evals = ff.evaluated_runs(runs, outputs)
    tables = ff._load(evals, "spectra", label) if not evals.empty else {}
    if not tables:
        return []
    persist = ff._persistence(outputs, "spectra")
    sources = dict(tables)
    if persist is not None:
        sources["persistence"] = [persist]

    def clean(t):
        return t[(t.sigma == 0) & (t.k_index > 0) & (t.k_index < t.k_index.max())]

    def log_ratio(t, step):
        t = clean(t)
        t = t[t.step == step]
        key = ["rayleigh", "prandtl", "k_index", "k"]
        m = t[t.source == "pred"].merge(t[t.source == "true"], on=key, suffixes=("_p", "_t"))
        m["lr"] = np.log10(m.energy_p / m.energy_t)
        return m

    first = clean(next(iter(tables.values()))[0])
    last = int(first.step.max())
    kmax = float(first.k.max())
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    a, b, c = axes
    ra, pr = regime
    truth = first[(first.source == "true") & (first.step == last) & (first.rayleigh == ra) & (first.prandtl == pr)]
    a.loglog(truth.k, truth.energy, color=INK, lw=TRUTH_LW)
    for name, ts in sources.items():
        ls = ":" if name == "persistence" else "-"
        spec = pd.concat([clean(t)[(clean(t).source == "pred") & (clean(t).step == last) & (clean(t).rayleigh == ra)
                                   & (clean(t).prandtl == pr)].set_index("k").energy for t in ts], axis=1).mean(axis=1)
        a.loglog(spec.index, spec.values, color=STYLE[name]["color"], ls=ls, lw=1.2)
    a.set_title(f"(a) step {last}, Ra = 1e{int(np.log10(ra))}, Pr = {pr:g}")
    a.set_ylabel(r"$E(k_x)$")
    for ax, step, tag in ((b, 1, "(b)"), (c, last, "(c)")):
        for name, ts in sources.items():
            ls = ":" if name == "persistence" else "-"
            rs = [log_ratio(t, step) for t in ts]
            med = pd.concat([r.groupby("k").lr.median() for r in rs], axis=1).mean(axis=1)
            lo = pd.concat([r.groupby("k").lr.quantile(0.25) for r in rs], axis=1).mean(axis=1)
            hi = pd.concat([r.groupby("k").lr.quantile(0.75) for r in rs], axis=1).mean(axis=1)
            ax.fill_between(med.index, 10 ** lo, 10 ** hi, color=STYLE[name]["color"], alpha=0.12, lw=0)
            ax.plot(med.index, 10 ** med, color=STYLE[name]["color"], ls=ls, lw=1.2)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{tag} predicted / true, step {step}")
        ax.set_ylim(0.05, 5e3)
    c.sharey(b)
    c.tick_params(labelleft=True)          # same scale as (b), but labelled: the gap between panels hides the link
    b.set_ylabel(r"$E_{\mathrm{pred}}/E_{\mathrm{true}}$")
    for ax in axes:
        ax.axvspan(ff.SMALL_SCALES_K, kmax * 1.1, color=MUTED, alpha=0.08, lw=0)
        ax.set_xlabel(r"Horizontal wavenumber $k_x$")
    a.text(ff.SMALL_SCALES_K * 1.08, 0.04, "small\nscales", transform=a.get_xaxis_transform(), fontsize=7,
           color=MUTED, va="bottom")
    a.text(2.2, 0.04, "large scales", transform=a.get_xaxis_transform(), fontsize=7, color=MUTED, va="bottom")
    handles = [Line2D([], [], color=INK, lw=TRUTH_LW, label="true flow")]
    handles += [Line2D([], [], color=STYLE[m]["color"], lw=1.2, label=f"{STYLE[m]['label']} (ours, {label})")
                for m in MODEL_NAMES if m in tables]
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], ls=":", lw=1.2, label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_spectra")


# --- 6. flow snapshots ------------------------------------------------------------------------------------------

def snapshots(npz: Path, out: Path, field: str = "buoyancy", steps=(1, 12, 30)) -> list[str]:
    from rbbench.solver_grid import LX, LY, gauss_nodes, x_grid

    d = np.load(npz)
    meta = json.loads(str(d["meta"]))
    fi = meta["fields"].index(field)
    have = [int(s) for s in d["steps"]]
    rows = [s for s in steps if s in have]
    models = [m for m in MODEL_NAMES if m in d.files]
    cols = ["truth", *models]
    truth = d["truth"][..., fi]
    x, y = x_grid(truth.shape[1], LX), gauss_nodes(truth.shape[2], LY)
    if field == "buoyancy":
        # sequential (cold dark, hot bright), scaled to the interior so the plumes show; the hot wall saturates
        interior = truth[:, :, 8:-8]
        vmin, vmax = (float(v) for v in np.percentile(interior, [1, 99]))
        cmap = "inferno"
    else:
        lim = float(np.percentile(np.abs(truth), 99))
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(FULL, 0.52 * len(rows) + 0.75), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    mesh = None
    for i, step in enumerate(rows):
        k = have.index(step)
        for j, col in enumerate(cols):
            ax = axes[i][j]
            img = truth[k] if col == "truth" else d[col][k, ..., fi]
            mesh = ax.pcolormesh(x, y, img.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title("true flow" if col == "truth" else STYLE[col]["label"], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"step {step}", fontsize=7, color=INK)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=45, shrink=0.6)
    what = ff.FIELD_LABELS.get(field, field)
    cbar.set_label("Buoyancy $b$" if field == "buoyancy" else what, fontsize=7, color=INK)
    cbar.ax.tick_params(labelsize=6.5)
    tag = npz.stem.removeprefix("rollout_")
    return save(fig, out, f"fig_snapshots_{field}_{tag}")


def main() -> int:
    outputs, out = REPO / "outputs", REPO / "reports" / "figures"
    runs_csv = outputs / "findings" / "runs.csv"
    runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()
    written = released_by_window(outputs, out) + loader(outputs, out)
    if not runs.empty:
        written += (training(outputs, runs, out) + test_by_window(outputs, runs, out) + rollouts(outputs, runs, out)
                    + accuracy_cost(outputs, runs, out) + by_rayleigh(outputs, runs, out)
                    + spectra(outputs, runs, out))
    for npz in sorted((outputs / "visuals").glob("rollout_*.npz")) if (outputs / "visuals").exists() else []:
        written += snapshots(npz, out)
    print("\n".join(f"wrote {p}" for p in written if p.endswith(".pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
