#!/usr/bin/env python
"""Figures for the final report (WP7): both data fractions, onset split per trajectory (D29).

Same conventions as scripts/report_figures.py (printed size, 6.9 in text width, no titles beyond panel tags, neutral
text); the mid-project figures there stay as they were. Run after scripts/findings.py, which writes the tables read
here:

    .venv/bin/python scripts/final_figures.py          # writes reports/figures/final/fig_*.pdf (and .png previews)

Line and marker conventions in every figure: colour = model; solid line / filled marker = 100% of the training
data, dashed line / open marker = 10%; persistence dotted grey; the released CNextU-Net (The Well, 100% data, ~12 h
on an H100) dashed black. "Developed" one-step windows start at or after their trajectory's onset frame t_on
(tasks/onset_test.json, 10% of the developed kinetic energy); rollouts from frames 80 and 140 count as developed.
Figures whose inputs are missing are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_figures as rf  # noqa: E402  (sets the print rcParams)

import findings_figures as ff  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from rbbench import MODEL_NAMES  # noqa: E402
from rbbench.aggregate import INK, MUTED, STYLE  # noqa: E402
from rbbench.visuals import snapshot_files  # noqa: E402

REPO = rf.REPO
FULL, PERSIST = rf.FULL, rf.PERSIST
FRACS = (0.1, 1.0)
FRAC_TEXT = {0.1: "10%", 1.0: "100%"}
FRAC_LS = {0.1: "--", 1.0: "-"}
DEV_STARTS = (80, 140)
RELEASED = dict(color=INK, lw=1.0, ls="--")
PERSIST_LINE = dict(color=PERSIST["color"], lw=1.3, ls=":")


def _frac(x) -> float:
    return round(float(x), 3)


def marker_kw(model: str, fraction: float, ms: float = 4.0) -> dict:
    """Filled marker for 100% data, open for 10%."""
    c = STYLE[model]["color"]
    return dict(color=c, marker=STYLE[model]["marker"], ms=ms, mec=c, mfc=c if _frac(fraction) == 1.0 else "white",
                mew=1.0)


def fraction_handles() -> list[Line2D]:
    return [Line2D([], [], color=MUTED, ls="-", marker="o", ms=3.5, mfc=MUTED, label="100% data"),
            Line2D([], [], color=MUTED, ls="--", marker="o", ms=3.5, mfc="white", label="10% data")]


def model_handles(models) -> list[Line2D]:
    return [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3.5, label=STYLE[m]["label"])
            for m in models]


def reference_handles(released: bool = True, persistence: bool = True) -> list[Line2D]:
    h = []
    if released:
        h.append(Line2D([], [], label="released CNextU-Net", **RELEASED))
    if persistence:
        h.append(Line2D([], [], label="persistence", **PERSIST_LINE))
    return h


def row_major(handles: list, ncol: int) -> list:
    """Reorder so a legend with `ncol` columns reads row by row (matplotlib fills it column by column)."""
    n = len(handles)
    rows = -(-n // ncol)
    grid = [handles[r * ncol + c] if r * ncol + c < n else None for c in range(ncol) for r in range(rows)]
    return [h for h in grid if h is not None]


def legend(fig, handles, ncol=None, y=0.98):
    ncol = ncol or len(handles)
    fig.legend(handles=row_major(handles, ncol), loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y),
               columnspacing=1.1, handlelength=1.9)


def log_ticks(ax, ticks):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# --- data helpers (tested in tests/test_final_figures.py) ---------------------------------------------------------

def onset_frames(path: Path | None = None) -> dict[str, int | None]:
    from rbbench.onset import load_onset_frames

    return load_onset_frames(path or REPO / "tasks" / "onset_test.json")


def developed_mask(t: pd.DataFrame, t_on: dict[str, int | None]) -> np.ndarray:
    """True for windows whose first input frame is at or after their trajectory's t_on (never, if t_on is None)."""
    on = t.traj_id.map(lambda x: np.inf if t_on[x] is None else t_on[x]).astype(float).to_numpy()
    return t.start.to_numpy() >= on


def developed_mean(t: pd.DataFrame, t_on: dict, col: str = "vrmse_mean", by: str | None = None):
    t = rf._clean(t)
    d = t[developed_mask(t, t_on)]
    return d[col].mean() if by is None else d.groupby(by)[col].mean()


def released_study(outputs: Path, kind: str) -> pd.DataFrame | None:
    """The released CNextU-Net through the study evaluation (`eval-baseline --released cnext_unet`): same cohort,
    rollouts from frames 20 / 80 / 140 and physics as our runs; its one-step rows equal the pipeline check's."""
    p = outputs / "baselines" / "released" / "cnext_unet" / "evaluations" / f"{kind}_na.parquet"
    return pd.read_parquet(p) if p.exists() else None


def seed_stats(fc: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean, min and max over seeds of `metric` per model, fraction and budget (fraction_comparison.csv rows)."""
    runs = fc[fc.seed.notna()].copy()
    runs["fraction"] = runs.fraction.map(_frac)
    return runs.groupby(["model", "fraction", "budget"])[metric].agg(["mean", "min", "max"]).reset_index()


def with_rollout_end(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    fc["rollout_end_developed"] = fc[[f"rollout_step30_from{s}" for s in DEV_STARTS]].mean(axis=1)
    return fc


def reference(fc: pd.DataFrame, name: str, metric: str) -> float | None:
    r = fc[fc.model == name]
    return None if r.empty else float(r[metric].iloc[0])


def frames_from_onset(t: pd.DataFrame, t_on: dict) -> pd.DataFrame:
    """Adds `d` = window start - the trajectory's t_on (negative: onset); trajectories without t_on are dropped."""
    t = rf._clean(t)
    on = t.traj_id.map(t_on)
    t = t[on.notna()].assign(d=lambda x: x.start - x.traj_id.map(t_on).astype(int))
    return t


def common_offsets(t_on: dict, last_start: int = 195) -> tuple[int, int]:
    """The range of d = start - t_on that every trajectory reaches (starts 0..last_start)."""
    vals = [v for v in t_on.values() if v is not None]
    return -min(vals), last_start - max(vals)


def onset_end(t: pd.DataFrame, t_on: dict, factor: float = 2.0) -> pd.DataFrame:
    """Per Rayleigh number: the first window start (counted from frame 0, and from t_on) whose mean error over the
    regime's trajectories falls below `factor` times its median in developed convection: where the onset error ends."""
    t = frames_from_onset(t, t_on)
    lo, hi = common_offsets({k: v for k, v in t_on.items() if k in set(t.traj_id)})
    rows = []
    for ra, g in t.groupby("rayleigh"):
        level = factor * g[g.d >= 0].vrmse_mean.median()
        a = g.groupby("start").vrmse_mean.mean()
        r = g[(g.d >= lo) & (g.d <= hi)].groupby("d").vrmse_mean.mean()
        rows.append({"rayleigh": ra, "from_frame0": int(a[a < level].index.min()),
                     "from_onset": int(r[r < level].index.min())})
    return pd.DataFrame(rows)


# --- 1. data scaling: 10% against 100% at 1 h and 4 h ---------------------------------------------------------------

def data_scaling(fc: pd.DataFrame, out: Path) -> list[str]:
    if fc.empty:
        return []
    fc = with_rollout_end(fc)
    models = [m for m in MODEL_NAMES if m in set(fc.model)]
    panels = (("one_step_developed", "One-step VRMSE,\ndeveloped convection", "(a) One step, developed",
               [0.03, 0.05, 0.1, 0.2, 0.3]),
              ("one_step_onset", "One-step VRMSE, onset", "(b) One step, onset", [0.1, 0.2, 0.5, 1, 2, 3]),
              ("rollout_end_developed", "VRMSE after 30 steps", "(c) 30 steps from frames 80, 140",
               [0.3, 0.5, 1, 1.5]))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35))
    offsets = dict(zip(models, (0.93, 1.0, 1.075)))
    for ax, (metric, ylabel, title, ticks) in zip(axes, panels):
        st = seed_stats(fc, metric)
        for m in models:
            for budget, ls in (("1h", "--"), ("4h", "-")):
                s = st[(st.model == m) & (st.budget == budget)].sort_values("fraction")
                x = s.fraction.to_numpy() * offsets[m]
                ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], color=STYLE[m]["color"],
                            marker=STYLE[m]["marker"], ms=3.8, mfc=STYLE[m]["color"] if budget == "4h" else "white",
                            capsize=1.5, lw=1.2, ls=ls)
        for name, kw in (("released_cnext_unet", RELEASED), ("persistence", PERSIST_LINE)):
            v = reference(fc, name, metric)
            if v is not None:
                ax.axhline(v, **kw)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1.0], ["10%", "100%"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.065, 1.55)
        ax.set_xlabel("Training data")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        log_ticks(ax, ticks)
    handles = model_handles(models)
    handles += [Line2D([], [], color=MUTED, ls="-", marker="o", ms=3.5, mfc=MUTED, label="4 h"),
                Line2D([], [], color=MUTED, ls="--", marker="o", ms=3.5, mfc="white", label="1 h")]
    handles += reference_handles()
    legend(fig, handles)
    fig.tight_layout()
    return rf.save(fig, out, "fig_data_scaling")


# --- 2. accuracy against cost, both fractions ----------------------------------------------------------------------

def accuracy_cost(fc: pd.DataFrame, runs: pd.DataFrame, resource: pd.DataFrame, out: Path) -> list[str]:
    if fc.empty or runs.empty or resource.empty:
        return []
    fc = with_rollout_end(fc)
    models = [m for m in MODEL_NAMES if m in set(fc.model)]
    train = runs[runs.manifest == "train"].assign(fraction=lambda d: d.fraction.map(_frac))
    dev = seed_stats(fc, "one_step_developed")
    end = seed_stats(fc, "rollout_end_developed")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.4), gridspec_kw={"width_ratios": [0.8, 1, 1]})
    for m in models:
        for frac in FRACS:
            mk = marker_kw(m, frac)
            s = dev[(dev.model == m) & (dev.fraction == frac)].set_index("budget").reindex(["1h", "4h"])
            a.errorbar([1, 4], s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]], capsize=1.5, lw=1.2,
                       ls=FRAC_LS[frac], **mk)
            wall = train[(train.model == m) & (train.fraction == frac)].wall_hours
            v = s.loc["4h", "mean"]
            b.errorbar([wall.mean()], [v], xerr=[[wall.mean() - wall.min()], [wall.max() - wall.mean()]], capsize=1.5,
                       lw=0.9, ls="none", **mk)
            e = end[(end.model == m) & (end.fraction == frac) & (end.budget == "4h")]
            ms = resource[(resource.model == m) & (resource.budget == "4h")].rollout_ms.mean()
            c.plot([ms], e["mean"], ls="none", **marker_kw(m, frac, 4.5))
        wall100 = train[(train.model == m) & (train.fraction == 1.0)].wall_hours.mean()
        wall10 = train[(train.model == m) & (train.fraction == 0.1)].wall_hours.max()
        v = dev[(dev.model == m) & (dev.fraction == 0.1) & (dev.budget == "4h")]["mean"].iloc[0]
        b.annotate(STYLE[m]["label"], (max(wall100, wall10), v), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=7, color=INK)
        ms = resource[(resource.model == m) & (resource.budget == "4h")].rollout_ms.mean()
        e = end[(end.model == m) & (end.budget == "4h")]["mean"].max()
        above = m == "avit"                                  # the others sit just under the reference lines
        e = e if above else end[(end.model == m) & (end.budget == "4h")]["mean"].min()
        c.annotate(f"{STYLE[m]['label']}\n{ms:.0f} ms", (ms, e), xytext=(0, 6 if above else -6),
                   textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=7, color=INK)
    for name, kw in (("released_cnext_unet", RELEASED), ("persistence", PERSIST_LINE)):
        v = reference(fc, name, "one_step_developed")
        if v is not None:
            a.axhline(v, **kw)
            b.axhline(v, **kw)
        v = reference(fc, name, "rollout_end_developed")
        if v is not None:
            c.axhline(v, **kw)
    for ax in (a, b):
        log_ticks(ax, [0.03, 0.05, 0.1, 0.2, 0.3])
        ax.set_ylim(0.03, 0.3)
        ax.set_ylabel("One-step VRMSE, developed")
    a.set_xticks([1, 4])
    a.set_xlim(0.4, 4.6)
    a.set_xlabel("Active training time (h)")
    a.set_title("(a) More training time")
    b.axvline(4, color=MUTED, lw=0.8)
    b.text(4.25, 0.15, "4 h of active\ntraining", fontsize=7, color=MUTED, va="center")
    b.set_xlim(0, 14.5)
    b.set_xlabel("Allocated GPU-hours per 4-h run")
    b.set_title("(b) Cost to train")
    c.set_xlim(0, 300)
    c.set_ylim(0.3, 1.0)
    c.set_xlabel("Inference time per 30-step rollout (ms)")
    c.set_ylabel("VRMSE after 30 steps, developed")
    c.set_title("(c) Cost to use (4-h models)")
    legend(fig, model_handles(models) + fraction_handles() + reference_handles())
    fig.tight_layout()
    return rf.save(fig, out, "fig_accuracy_cost")


# --- 3. one-step test error by window, per fraction ---------------------------------------------------------------

def onset_zone(ax, t_on: dict, label: bool = True):
    """Every window before the earliest t_on is onset; between the earliest and latest it depends on the trajectory."""
    vals = [v for v in t_on.values() if v is not None]
    lo, hi = min(vals), max(vals)
    ax.axvspan(-2, lo, color=MUTED, alpha=0.10, lw=0)
    ax.axvspan(lo, hi, color=MUTED, alpha=0.05, lw=0)
    if label:
        ax.text(lo / 2, 0.97, "onset", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                color=MUTED)
        ax.text((lo + hi) / 2, 0.97, "onset ends\n(per trajectory)", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6.5, color=MUTED)
        ax.text(hi + 6, 0.97, "developed convection", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7, color=MUTED)


def test_by_window(outputs: Path, runs: pd.DataFrame, out: Path, t_on: dict, label: str = "4h") -> list[str]:
    released = released_study(outputs, "one_step")
    persist = ff._persistence(outputs, "one_step")
    tables = {f: ff._load(ff.evaluated_runs(runs, outputs, fraction=f), "one_step", label) for f in FRACS}
    if not any(tables.values()):
        return []
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.9), sharex=True, sharey=True)
    for ax, frac, tag in zip(axes, FRACS, ("(a)", "(b)")):
        onset_zone(ax, t_on, label=frac == FRACS[0])
        for m in [m for m in MODEL_NAMES if m in tables[frac]]:
            mean, lo, hi = ff._seed_mean(tables[frac][m], "start", "vrmse_mean")
            ax.fill_between(mean.index, lo, hi, color=STYLE[m]["color"], alpha=0.15, lw=0)
            ax.plot(mean.index, mean.values, color=STYLE[m]["color"])
        if released is not None:
            g = released.groupby("start").vrmse_mean.mean()
            ax.plot(g.index, g.values, **RELEASED)
        if persist is not None:
            g = persist.groupby("start").vrmse_mean.mean()
            ax.plot(g.index, g.values, **PERSIST_LINE)
        ax.set_yscale("log")
        ax.set_ylabel("One-step VRMSE")
        ax.set_title(f"{tag} ours trained on {FRAC_TEXT[frac]} of the data, {label[:-1]} h", loc="left")
    axes[1].set_xlim(-2, 197)
    axes[1].set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 195])
    axes[1].set_xlabel("Input-window start frame")
    legend(fig, model_handles([m for m in MODEL_NAMES if m in tables[1.0]]) + reference_handles())
    fig.tight_layout()
    return rf.save(fig, out, "fig_test_by_window")


def onset_aligned(outputs: Path, runs: pd.DataFrame, out: Path, t_on: dict, label: str = "4h") -> list[str]:
    """(a) One-step error against frames from each trajectory's onset frame; (b, c) our 100% CNextU-Net per Rayleigh
    number counted from frame 0 and from t_on: aligned at t_on the regimes' curves line up."""
    tables = {f: ff._load(ff.evaluated_runs(runs, outputs, fraction=f), "one_step", label) for f in FRACS}
    if not tables[1.0].get("cnext_unet"):
        return []
    lo, hi = common_offsets(t_on)
    released, persist = released_study(outputs, "one_step"), ff._persistence(outputs, "one_step")

    def curve(ts, by="d"):
        return pd.concat([frames_from_onset(t, t_on).groupby(by).vrmse_mean.mean() for t in ts], axis=1).mean(axis=1)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.45), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    drawn = []
    for frac in FRACS:
        for m in [m for m in MODEL_NAMES if m in tables[frac]]:
            g = curve(tables[frac][m]).loc[lo:hi]
            a.plot(g.index, g.values, color=STYLE[m]["color"], ls=FRAC_LS[frac], lw=1.2 if frac == 1.0 else 1.0)
            if m not in drawn:
                drawn.append(m)
    for t, kw in ((released, RELEASED), (persist, PERSIST_LINE)):
        if t is not None:
            g = curve([t]).loc[lo:hi]
            a.plot(g.index, g.values, **kw)
    a.axvline(0, color=INK, lw=0.7)
    a.set_xlim(lo, hi)
    a.set_xlabel(r"Window start $-$ onset frame $t_{\mathrm{on}}$")
    a.set_ylabel("One-step VRMSE")
    a.set_title(f"(a) Aligned at each trajectory's onset, {label[:-1]} h")
    log_ticks(a, [0.03, 0.05, 0.1, 0.2, 0.5, 1, 2])
    cn = STYLE["cnext_unet"]["color"]
    ras = sorted(pd.concat(tables[1.0]["cnext_unet"]).rayleigh.unique())
    shades = dict(zip(ras, [matplotlib.colors.to_rgba(cn, 1.0)[:3] + (al,) for al in np.linspace(0.3, 1.0, len(ras))]))
    ts = tables[1.0]["cnext_unet"]
    for ra in ras:
        sub = [t[t.rayleigh == ra] for t in ts]
        g0 = curve(sub, by="start")
        b.plot(g0.index, g0.values, color=shades[ra], lw=1.1)
        g1 = curve(sub).loc[lo:hi]
        c.plot(g1.index, g1.values, color=shades[ra], lw=1.1, label=f"1e{int(round(np.log10(ra)))}")
    b.set_xlim(-2, 150)
    b.set_xlabel("Window start frame")
    b.set_title("(b) CNextU-Net 100%, by Ra:\nfrom frame 0", fontsize=7.5)
    c.axvline(0, color=INK, lw=0.7)
    c.set_xlim(lo, 100)
    c.set_xlabel(r"Window start $-$ $t_{\mathrm{on}}$")
    c.set_title(r"(c) the same, from $t_{\mathrm{on}}$", fontsize=7.5)
    c.legend(title="Ra", loc="upper right", fontsize=6.5, title_fontsize=6.5, handlelength=1.2)
    for ax in (b, c):
        log_ticks(ax, [0.01, 0.03, 0.1, 0.3, 1, 3])
        ax.set_ylim(0.008, 6)
    b.set_ylabel("One-step VRMSE")
    c.tick_params(labelleft=False)
    legend(fig, model_handles(drawn) + fraction_handles() + reference_handles(), ncol=7)
    fig.tight_layout()
    return rf.save(fig, out, "fig_onset_aligned")


# --- 4. rollouts: error by step and survival below VRMSE 1, at 1 h and 4 h ----------------------------------------

def rollouts(outputs: Path, runs: pd.DataFrame, out: Path, threshold: float = 1.0) -> list[str]:
    from rbbench.evaluate import failure_step, kaplan_meier

    def curves(ts):
        dev = pd.concat([rf._clean(t)[rf._clean(t).start.isin(DEV_STARTS)].groupby("step").vrmse_mean.mean()
                         for t in ts], axis=1).mean(axis=1)
        km = []
        for t in ts:
            piv = rf._clean(t).pivot_table(index=["traj_id", "start"], columns="step", values="vrmse_mean")
            km.append(kaplan_meier([failure_step(v.values, threshold) for _, v in piv.iterrows()], piv.shape[1]))
        return dev, np.mean(km, axis=0)

    refs = {}
    for name, kind in (("released", "rollout"), ("persistence", "rollout")):
        t = released_study(outputs, kind) if name == "released" else ff._persistence(outputs, kind)
        if t is not None:
            refs[name] = curves([t])
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.3), sharex=True)
    drawn = set()
    for row, label in enumerate(("1h", "4h")):
        a, b = axes[row]
        for frac in FRACS:
            tables = ff._load(ff.evaluated_runs(runs, outputs, fraction=frac), "rollout", label)
            for m in [m for m in MODEL_NAMES if m in tables]:
                drawn.add(m)
                dev, km = curves(tables[m])
                st = dict(color=STYLE[m]["color"], ls=FRAC_LS[frac], lw=1.3 if frac == 1.0 else 1.1)
                a.plot(dev.index, dev.values, marker=STYLE[m]["marker"], ms=3,
                       mfc=STYLE[m]["color"] if frac == 1.0 else "white", markevery=5, **st)
                b.step(np.arange(0, len(km) + 1), np.r_[1.0, km], where="post", **st)
        for name, (dev, km) in refs.items():
            kw = RELEASED if name == "released" else PERSIST_LINE
            a.plot(dev.index, dev.values, **kw)
            b.step(np.arange(0, len(km) + 1), np.r_[1.0, km], where="post", **kw)
        tag = "ab" if row == 0 else "cd"
        h = label[:-1]
        a.set_title(f"({tag[0]}) {h} h: error, rollouts from frames 80 and 140", loc="left")
        b.set_title(f"({tag[1]}) {h} h: rollouts staying below VRMSE {threshold:g}, all starts", loc="left")
        a.set_ylabel("Rollout VRMSE")
        a.set_ylim(0, 1.6)
        b.set_ylabel("Fraction of rollouts")
        b.set_ylim(0, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("Rollout step")
        ax.set_xlim(0, 30.5)
    legend(fig, model_handles([m for m in MODEL_NAMES if m in drawn]) + fraction_handles() + reference_handles())
    fig.tight_layout()
    return rf.save(fig, out, "fig_rollouts")


# --- 5. by Rayleigh number: one step, 30 steps, heat transport ----------------------------------------------------

def by_rayleigh(outputs: Path, runs: pd.DataFrame, nu: pd.DataFrame, out: Path, t_on: dict,
                label: str = "4h") -> list[str]:
    released, persist = released_study(outputs, "one_step"), ff._persistence(outputs, "one_step")
    r_roll, p_roll = released_study(outputs, "rollout"), ff._persistence(outputs, "rollout")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.4))
    drawn = set()
    for frac in FRACS:
        evals = ff.evaluated_runs(runs, outputs, fraction=frac)
        one, roll = ff._load(evals, "one_step", label), ff._load(evals, "rollout", label)
        for m in [m for m in MODEL_NAMES if m in one]:
            drawn.add(m)
            kw = dict(ls=FRAC_LS[frac], lw=1.2, **marker_kw(m, frac, 3.5))
            s = pd.concat([developed_mean(t, t_on, by="rayleigh") for t in one[m]], axis=1).mean(axis=1)
            axes[0].plot(s.index, s.values, **kw)
            s = pd.concat([rf._by_ra_rollout(t) for t in roll[m]], axis=1).mean(axis=1)
            axes[1].plot(s.index, s.values, **kw)
            g = nu[(nu.model == m) & (nu.fraction.map(_frac) == frac) & (nu.budget == label)
                   & nu.rayleigh.notna()].sort_values("rayleigh")
            axes[2].plot(g.rayleigh, 100 * g.rel_err_signed, **kw)
    if released is not None:
        s = developed_mean(released, t_on, by="rayleigh")
        axes[0].plot(s.index, s.values, **RELEASED)
    if persist is not None:
        s = developed_mean(persist, t_on, by="rayleigh")
        axes[0].plot(s.index, s.values, **PERSIST_LINE)
    for t, kw in ((r_roll, RELEASED), (p_roll, PERSIST_LINE)):
        if t is not None:
            s = rf._by_ra_rollout(t)
            axes[1].plot(s.index, s.values, **kw)
    for name, kw in (("released_cnext_unet", RELEASED), ("persistence", PERSIST_LINE)):
        g = nu[(nu.model == name) & nu.rayleigh.notna()].sort_values("rayleigh")
        if not g.empty:
            axes[2].plot(g.rayleigh, 100 * g.rel_err_signed, **kw)
    axes[2].axhline(0, color=INK, lw=0.7)
    axes[2].set_ylim(-60, 140)
    off = nu[(nu.model == "released_cnext_unet") & (nu.rayleigh.notna()) & (100 * nu.rel_err_signed > 140)]
    if not off.empty:                       # the released model's heat transport runs off the scale at high Ra
        vals = ", ".join(f"+{100 * v:.0f}%" for v in off.sort_values("rayleigh").rel_err_signed)
        axes[2].text(0.98, 0.97, f"released, off scale:\n{vals}", transform=axes[2].transAxes, ha="right", va="top",
                     fontsize=6.5, color=INK)
    titles = (("One-step VRMSE, developed", "(a) One step"), ("VRMSE after 30 steps", "(b) 30 steps from 80, 140"),
              ("Nusselt-number error (%)", "(c) Heat transport over 30 steps"))
    for ax, (ylabel, title) in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Rayleigh number")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    log_ticks(axes[0], [0.02, 0.05, 0.1, 0.2, 0.4])
    legend(fig, model_handles([m for m in MODEL_NAMES if m in drawn]) + fraction_handles() + reference_handles(),
           ncol=7)
    fig.tight_layout()
    return rf.save(fig, out, "fig_by_rayleigh")


def nusselt(nu: pd.DataFrame, out: Path) -> list[str]:
    """Predicted against true rollout-averaged Nusselt number per Rayleigh number, after one hour (a) and four (b):
    medians over the 70 rollouts per Ra (seed mean) with the middle 50% of the rollouts; truth as a band."""
    s = nu[nu.rayleigh.notna()].assign(fraction=lambda d: d.fraction.map(lambda x: x if pd.isna(x) else _frac(x)))
    if s.empty:
        return []
    truth = s[s.model == "persistence"].sort_values("rayleigh")
    models = [m for m in MODEL_NAMES if m in set(s.model)]
    series = [(m, f) for m in models for f in FRACS] + [("released_cnext_unet", None)]
    shift = dict(zip(series, 10 ** (0.045 * (np.arange(len(series)) - (len(series) - 1) / 2))))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
    for ax, label, tag in zip(axes, ("1h", "4h"), ("(a)", "(b)")):
        ax.fill_between(truth.rayleigh, truth.nu_true_q25, truth.nu_true_q75, color=MUTED, alpha=0.22, lw=0)
        ax.plot(truth.rayleigh, truth.nu_true, color=INK, lw=1.6)
        for (m, frac) in series:
            if m == "released_cnext_unet":
                g = s[s.model == m].sort_values("rayleigh")
                kw = dict(color=INK, marker="D", ms=3.2, mfc="white", mec=INK, ls="--", lw=0.9)
            else:
                g = s[(s.model == m) & (s.fraction == frac) & (s.budget == label)].sort_values("rayleigh")
                kw = dict(ls=FRAC_LS[frac], lw=1.0, **marker_kw(m, frac, 3.4))
            if g.empty:
                continue
            x = g.rayleigh * shift[(m, frac)]
            ax.errorbar(x, g.nu_pred, yerr=[g.nu_pred - g.nu_pred_q25, g.nu_pred_q75 - g.nu_pred], capsize=1.2,
                        elinewidth=0.7, **kw)
        ax.set_xscale("log")
        ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(5e5, 2e10)
        ax.set_xlabel("Rayleigh number")
        ax.set_title(f"{tag} after {label[:-1]} {'hour' if label == '1h' else 'hours'} of training")
        log_ticks(ax, [5, 10, 20, 50, 100, 200, 500, 1000])
    axes[0].set_ylabel("Nusselt number, 30-step average")
    handles = [Line2D([], [], color=INK, lw=1.6, label="true (band: middle 50%)")] + model_handles(models)
    handles += fraction_handles()
    handles += [Line2D([], [], color=INK, marker="D", ms=3.2, mfc="white", ls="--", lw=0.9,
                       label="released CNextU-Net")]
    legend(fig, handles, ncol=7)
    fig.tight_layout()
    return rf.save(fig, out, "fig_nusselt")


# --- 6. the numerical solver against CNextU-Net ----------------------------------------------------------------------

def solver(outputs: Path, runs: pd.DataFrame, comp: pd.DataFrame, match: pd.DataFrame, out: Path,
           label: str = "4h") -> list[str]:
    solvers = ff._solvers(outputs)
    if comp.empty or not solvers:
        return []
    rolls = {res: pd.read_parquet(d / "rollout_na.parquet").query("sigma == 0") for res, d in solvers.items()}
    keys = set(map(tuple, next(iter(rolls.values()))[["traj_id", "start"]].drop_duplicates().values))
    ev = ff.evaluated_runs(runs, outputs, fraction=1.0)
    ev = ev[(ev.model == "cnext_unet") & (ev.label == label)]
    ours = [ff._subset(pd.read_parquet(e.dir / f"rollout_{label}.parquet").query("sigma == 0"), keys)
            for _, e in ev.iterrows()]
    refs = {}
    for name, d in (("released", outputs / "baselines" / "released" / "cnext_unet" / "evaluations"),
                    ("persistence", outputs / "baselines" / "persistence" / "evaluations")):
        if (d / "rollout_na.parquet").exists():
            refs[name] = ff._subset(pd.read_parquet(d / "rollout_na.parquet").query("sigma == 0"), keys)
    ras = sorted(comp.rayleigh.unique())
    grids = list(rolls)                                                   # finest first
    shade = dict(zip(grids, plt.get_cmap("Greys")(np.linspace(0.95, 0.35, len(grids)))))
    target = f"CNextU-Net 100% {label}"
    fig, axes = plt.subplots(1, len(ras) + 1, figsize=(FULL, 2.45),
                             gridspec_kw={"width_ratios": [1] * len(ras) + [1.15]})
    cn = STYLE["cnext_unet"]["color"]
    for ax, ra in zip(axes, ras):
        for res, roll in rolls.items():
            g = roll[roll.rayleigh == ra].groupby("step").vrmse_mean.mean()
            g = g.replace([np.inf], np.nan)
            ax.plot(g.index, g.values, color=shade[res], lw=1.0)
        g = pd.concat([t[t.rayleigh == ra].groupby("step").vrmse_mean.mean() for t in ours], axis=1).mean(axis=1)
        ax.plot(g.index, g.values, color=cn, lw=2.0)
        for name, t in refs.items():
            g = t[t.rayleigh == ra].groupby("step").vrmse_mean.mean()
            ax.plot(g.index, g.values, **(RELEASED if name == "released" else PERSIST_LINE))
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3)
        ax.set_xlim(0, 30.5)
        ax.set_xlabel("Rollout step")
        m = match[(match.rayleigh == ra) & (match.target == target)]
        mg = m.iloc[0].matching_grid if not m.empty else "none"
        sec = f" ({m.iloc[0].solver_seconds:.0f} s)" if mg != "none" else ""
        ax.set_title(f"Ra = 1e{int(round(np.log10(ra)))}\nmatching grid: {mg}{sec}", fontsize=7.5)
    axes[0].set_ylabel("Rollout VRMSE (10 rollouts)")
    for ax in axes[1:len(ras)]:
        ax.tick_params(labelleft=False)
    # last panel: each grid's worst ratio to CNextU-Net over steps 1, 10, 30, against its cost
    d = axes[-1]
    ra_style = dict(zip(ras, (dict(color=INK, marker="o"), dict(color=MUTED, marker="s"),
                              dict(color="#a8a7a3", marker="^"))))
    for ra in ras:
        t = comp[(comp.source == target) & (comp.rayleigh == ra)]
        g = comp[(comp.kind == "solver") & (comp.rayleigh == ra)].sort_values("seconds_per_rollout")
        if t.empty or g.empty:
            continue
        worst = np.max([g[f"vrmse_step{k}"] / t.iloc[0][f"vrmse_step{k}"] for k in ff.MATCH_STEPS], axis=0)
        fin = np.isfinite(worst)
        d.plot(g.seconds_per_rollout[fin], worst[fin], ms=3.5, lw=1.0, label=f"Ra = 1e{int(round(np.log10(ra)))}",
               **ra_style[ra])
        if (~fin).any():
            d.plot(g.seconds_per_rollout[~fin], np.full((~fin).sum(), 200.0), ls="none", ms=4, mfc="white",
                   **ra_style[ra])
    d.axhline(1.0, color=cn, lw=1.4)
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_ylim(1e-2, 400)
    d.set_xlabel("Solver seconds per rollout")
    d.set_ylabel("Solver / CNextU-Net VRMSE,\nworst of steps 1, 10, 30")
    d.set_title("Each grid's accuracy\nagainst its cost", fontsize=7.5)
    d.legend(loc="upper right", handlelength=1.5, fontsize=6.5)
    handles = [Line2D([], [], color=shade[r], lw=1.2, label=r) for r in grids]
    handles += [Line2D([], [], color=cn, lw=2.0, label="CNextU-Net 100%, 4 h")] + reference_handles()
    legend(fig, handles, ncol=5)
    fig.tight_layout()
    return rf.save(fig, out, "fig_solver")


# --- 7. uncertainty: calibration along the rollout -----------------------------------------------------------------

def uq(summ: pd.DataFrame, shifts: pd.DataFrame, out: Path) -> list[str]:
    if summ.empty or shifts.empty:
        return []
    cn = STYLE["cnext_unet"]["color"]
    r = summ[summ.window == "rollout"].assign(step=lambda d: d.group.astype(int))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.4))
    for frac in FRACS:
        g = r[(r.kind == "beta-NLL ensemble") & (r.fraction.map(_frac) == frac)].sort_values("step")
        a.plot(g.step, g.cov90, color=cn, ls=FRAC_LS[frac], lw=1.5)
        a.plot(g.step, g.cov50, color=cn, ls=FRAC_LS[frac], lw=0.9, alpha=0.55)
    a.axhline(0.9, color=INK, lw=0.7)
    a.axhline(0.5, color=INK, lw=0.7)
    a.text(29.5, 0.91, "nominal 90%", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    a.text(29.5, 0.51, "nominal 50%", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    a.set_ylim(0, 1.02)
    a.set_ylabel("Coverage on test")
    a.set_title("(a) β-NLL ensembles (CNextU-Net, 1 h)")
    s1 = shifts[(shifts["shift"] == "rollout: q-hat of step 1") & (shifts.nominal == 0.9)
                & (shifts.kind == "seed pair (4 h)") & (shifts.fraction.map(_frac) == 1.0)]
    for (family, scale), g in s1.groupby(["family", "scale"]):
        gg = g.groupby("step").coverage.mean()
        b.plot(gg.index, gg.values, color=STYLE[family]["color"], ls="-" if scale == "spread" else "--",
               lw=1.4 if scale == "spread" else 1.1)
    p = shifts[(shifts["shift"] == "rollout: q-hat of step 1") & (shifts.nominal == 0.9) & (shifts.kind == "persistence")]
    if not p.empty:
        gg = p.groupby("step").coverage.mean()
        b.plot(gg.index, gg.values, **PERSIST_LINE)
    b.axhline(0.9, color=INK, lw=0.7)
    b.set_ylim(0, 1.02)
    b.set_ylabel("Coverage of 90% intervals")
    b.set_title("(b) Conformal, calibrated at step 1")
    for family in ("cnext_unet", "tfno", "avit"):
        for frac in FRACS:
            g = r[(r.kind == "seed pair (4 h)") & (r.family == family) & (r.fraction.map(_frac) == frac)]
            g = g.sort_values("step")
            c.plot(g.step, g.ssr, color=STYLE[family]["color"], ls=FRAC_LS[frac], lw=1.2)
    for frac in FRACS:
        g = r[(r.kind == "beta-NLL ensemble") & (r.fraction.map(_frac) == frac)].sort_values("step")
        c.plot(g.step, g.ssr, color=cn, ls=FRAC_LS[frac], lw=2.6, alpha=0.4)
    c.axhline(1.0, color=INK, lw=0.7)
    c.set_yscale("log")
    c.set_yticks([0.2, 0.5, 1, 2])
    c.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    c.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c.set_ylabel("Spread / error")
    c.set_title("(c) Spread-skill ratio (1 = consistent)")
    for ax in (a, b, c):
        ax.set_xlabel("Rollout step (from 80, 140)")
        ax.set_xlim(0.5, 30.5)
    handles = model_handles(("cnext_unet", "tfno", "avit")) + fraction_handles()
    handles += [Line2D([], [], color=MUTED, ls="-", lw=1.4, label="(b) spread-scaled"),
                Line2D([], [], color=MUTED, ls="--", lw=1.1, label="(b) constant width"),
                Line2D([], [], color=cn, lw=2.6, alpha=0.4, label="(c) β-NLL ensemble"),
                Line2D([], [], label="persistence", **PERSIST_LINE)]
    legend(fig, handles, ncol=5)
    fig.tight_layout()
    return rf.save(fig, out, "fig_uq")


def uq_reliability(rel: pd.DataFrame, cov_ra: pd.DataFrame, out: Path) -> list[str]:
    """(a, b) RMSE of the ensemble mean against its predicted std in deciles of the predicted std (all four fields
    pooled, normalised units), one step and rollout step 10; (c) coverage of split-conformal 90% intervals per Rayleigh
    number, one step. All at 100% data: four-hour seed pairs and the one-hour beta-NLL ensemble."""
    if rel.empty or cov_ra.empty:
        return []
    cn = STYLE["cnext_unet"]["color"]
    rel = rel[(rel.field == "pooled")].assign(fraction=lambda d: d.fraction.map(_frac))
    beta = dict(color=cn, lw=2.6, alpha=0.4)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.45))
    for ax, (window, step, title) in zip((a, b), (("one_step", 1, "(a) One step"),
                                                  ("rollout", 10, "(b) Rollout step 10, from 80, 140"))):
        g = rel[(rel.window == window) & (rel.step == step) & (rel.fraction == 1.0)]
        for family in ("cnext_unet", "tfno", "avit"):
            gg = g[(g.kind == "seed pair (4 h)") & (g.family == family)].sort_values("bin")
            ax.plot(gg.sigma_rms, gg.rmse, color=STYLE[family]["color"], marker=STYLE[family]["marker"], ms=2.8, lw=1.1)
        gg = g[g.kind == "beta-NLL ensemble"].sort_values("bin")
        ax.plot(gg.sigma_rms, gg.rmse, marker="o", ms=3, **beta)
        lim = (3e-4, 3) if window == "one_step" else (1e-3, 3)
        ax.plot(lim, lim, color=INK, lw=0.7)
        ax.text(lim[1] / 1.6, lim[1] / 1.6 * 1.25, "reliable", rotation=0, ha="right", va="bottom", fontsize=6.5,
                color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lim)
        ax.set_ylim(lim[0] if window == "rollout" else 2e-3, lim[1])
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel("Predicted std (decile, RMS)")
        ax.set_ylabel("RMSE of the ensemble mean")
        ax.set_title(title)
    g = cov_ra[(cov_ra.window == "one_step") & (cov_ra.field == "mean") & (cov_ra.nominal == 0.9)]
    g = g.assign(fraction=g.fraction.map(lambda x: x if pd.isna(x) else _frac(x)))
    for family in ("cnext_unet", "tfno", "avit"):
        for scale, ls in (("spread", "-"), ("constant", "--")):
            gg = g[(g.kind == "seed pair (4 h)") & (g.family == family) & (g.fraction == 1.0)
                   & (g.scale == scale)].sort_values("rayleigh")
            c.plot(gg.rayleigh, gg.coverage, color=STYLE[family]["color"], ls=ls, lw=1.3 if ls == "-" else 1.0)
    gg = g[(g.kind == "beta-NLL ensemble") & (g.fraction == 1.0) & (g.scale == "gaussian")].sort_values("rayleigh")
    c.plot(gg.rayleigh, gg.coverage, **beta)
    gg = g[g.kind == "persistence"].sort_values("rayleigh")
    c.plot(gg.rayleigh, gg.coverage, **PERSIST_LINE)
    c.axhline(0.9, color=INK, lw=0.7)
    c.set_xscale("log")
    c.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10])
    c.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c.set_ylim(0.7, 1.0)
    c.set_xlabel("Rayleigh number")
    c.set_ylabel("Coverage of 90% intervals")
    c.set_title("(c) Conformal, one step")
    handles = model_handles(("cnext_unet", "tfno", "avit"))
    handles += [Line2D([], [], marker="o", ms=3, label="β-NLL ensemble", **beta),
                Line2D([], [], color=MUTED, ls="-", lw=1.3, label="(c) spread-scaled"),
                Line2D([], [], color=MUTED, ls="--", lw=1.0, label="(c) constant width"),
                Line2D([], [], label="persistence", **PERSIST_LINE)]
    legend(fig, handles, ncol=7)
    fig.tight_layout()
    return rf.save(fig, out, "fig_uq_reliability")


def correlation(corr: pd.DataFrame, out: Path) -> list[str]:
    """Spatial correlation between prediction and truth (field mean) along rollouts, four-hour checkpoints: (a) from
    frames 80 and 140, (b) from frame 20 (onset). Seed mean; the high-correlation thresholds 0.8 / 0.9 / 0.95 (D27)."""
    if corr.empty:
        return []
    corr = corr.assign(fraction=corr.fraction.map(lambda x: x if pd.isna(x) else _frac(x)))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    drawn = []
    for ax, (group, title) in zip(axes, (("developed", "(a) From frames 80 and 140 (developed)"),
                                         ("onset", "(b) From frame 20 (onset)"))):
        for thr in (0.8, 0.9, 0.95):
            ax.axhline(thr, color=MUTED, lw=0.6, ls="-", alpha=0.6)
        g = corr[corr.starts == group]
        for (family, frac), gg in g.groupby(["family", "fraction"], dropna=False):
            curve = gg.groupby("step").corr_mean.mean()
            if family == "persistence":
                ax.plot(curve.index, curve.values, **PERSIST_LINE)
                continue
            if family not in drawn:
                drawn.append(family)
            ax.plot(curve.index, curve.values, color=STYLE[family]["color"], ls=FRAC_LS[frac],
                    lw=1.3 if frac == 1.0 else 1.1, marker=STYLE[family]["marker"], ms=2.8, markevery=5,
                    mfc=STYLE[family]["color"] if frac == 1.0 else "white")
        ax.set_xlim(0.5, 30.5)
        ax.set_xlabel("Rollout step")
        ax.set_title(title)
    axes[0].set_ylim(0.3, 1.01)
    axes[0].set_ylabel("Correlation with the truth\n(field mean)")
    for thr in (0.8, 0.9, 0.95):
        axes[1].text(30.3, thr, f"{thr:g}", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    legend(fig, model_handles([m for m in MODEL_NAMES if m in drawn]) + fraction_handles()
           + [Line2D([], [], label="persistence", **PERSIST_LINE)])
    fig.tight_layout()
    return rf.save(fig, out, "fig_correlation")


# --- 8. input-noise robustness ------------------------------------------------------------------------------------

def noise(noise_t: pd.DataFrame, out: Path) -> list[str]:
    if noise_t.empty:
        return []
    sigmas = sorted(noise_t.sigma.unique())
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for ax, (window, group, title, ticks) in zip(axes, (
            ("one_step", "developed", "(a) One step, developed convection", [0.05, 0.1, 0.2, 0.3]),
            ("rollout", "10", "(b) Rollout step 10, from frames 80, 140", [0.2, 0.3, 0.5, 0.7]))):
        g = noise_t[(noise_t.window == window) & (noise_t.group.astype(str) == group)]
        for (family, frac), gg in g.groupby(["family", "fraction"], dropna=False):
            gg = gg.sort_values("sigma")
            if family == "persistence":
                ax.plot(gg.sigma, gg.vrmse, **PERSIST_LINE)
                continue
            ax.plot(gg.sigma, gg.vrmse, ls=FRAC_LS[_frac(frac)], lw=1.2, **marker_kw(family, frac, 3.5))
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(-0.001, 0.25)
        log_ticks(ax, ticks)
        ax.set_xlabel(r"Input noise $\sigma$ (normalised units)")
        ax.set_ylabel("Test VRMSE")
        ax.set_title(title)
    legend(fig, model_handles(("cnext_unet", "tfno", "avit")) + fraction_handles()
           + [Line2D([], [], label="persistence", **PERSIST_LINE)])
    fig.tight_layout()
    return rf.save(fig, out, "fig_noise")


# --- 9. where each run's GPU time went, both fractions ------------------------------------------------------------

def time_breakdown(runs: pd.DataFrame, out: Path) -> list[str]:
    r = runs[runs.manifest == "train"].assign(fraction=lambda d: d.fraction.map(_frac))
    if r.empty:
        return []
    rows = [row for m in MODEL_NAMES for f in FRACS
            for _, row in r[(r.model == m) & (r.fraction == f)].sort_values("seed").iterrows()]
    fig, ax = plt.subplots(figsize=(FULL, 3.0))
    other = "#c9c9c9"
    for i, row in enumerate(rows):
        yy = len(rows) - 1 - i + (0.4 if i < len(rows) // 3 else 0) + (0.4 if i < 2 * len(rows) // 3 else 0)
        c = STYLE[row.model]["color"]
        ax.barh(yy, row.train_hours, color=c, height=0.66, lw=0)
        ax.barh(yy, row.data_wait_hours, left=row.train_hours, height=0.66, facecolor="white", edgecolor=c,
                hatch="/////", lw=0)
        ax.barh(yy, row.other_hours, left=row.train_hours + row.data_wait_hours, height=0.66, color=other, lw=0)
        ax.text(row.wall_hours + 0.15, yy, f"{row.wall_hours:.1f} h, {row.su_estimate:.0f} SU", va="center",
                fontsize=6.5, color=INK)
        ax.text(-0.15, yy, f"{STYLE[row.model]['label']}, {FRAC_TEXT[row.fraction]}, seed {int(row.seed)}",
                ha="right", va="center", fontsize=7, color=INK)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.axvline(4, color=MUTED, lw=0.6, ls="--")
    ax.set_xlim(0, 14.6)
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlabel("GPU time held per 4-h run (h)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, lw=0, label="training (the 4 h budget)"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="/////", lw=0,
                             label="waiting for data"),
               plt.Rectangle((0, 0), 1, 1, color=other, lw=0,
                             label="start-up, data copy, validation, checkpoints, replayed steps")]
    legend(fig, handles, ncol=3)
    fig.tight_layout()
    return rf.save(fig, out, "fig_time")


# --- 10. flow snapshots, 10% against 100% --------------------------------------------------------------------------

def snapshots_by_fraction(outputs: Path, out: Path, field: str = "buoyancy", step: int = 30,
                          label: str = "4h", seed: int = 0) -> list[str]:
    """Per snapshot rollout: the true flow at `step` on top, then each model at 10% (left) and 100% (right)."""
    import json

    from rbbench.solver_grid import LX, LY, gauss_nodes, x_grid

    files = snapshot_files(outputs / "visuals")
    by_rollout: dict[str, dict[float, Path]] = {}
    for path, stem in files:
        for frac in FRACS:
            suffix = f"__f{int(round(frac * 100)):03d}_{label}_s{seed}"
            if stem.endswith(suffix):
                by_rollout.setdefault(stem[: -len(suffix)], {})[frac] = path
    written = []
    for rollout, paths in sorted(by_rollout.items()):
        if set(paths) != set(FRACS):
            continue
        data = {f: np.load(p) for f, p in paths.items()}
        meta = json.loads(str(data[1.0]["meta"]))
        fi = meta["fields"].index(field)
        k = [int(s) for s in data[1.0]["steps"]].index(step)
        truth = data[1.0]["truth"][k, ..., fi]
        if not np.allclose(truth, data[0.1]["truth"][k, ..., fi]):
            raise ValueError(f"{rollout}: the 10% and 100% snapshot files hold different true flows")
        models = [m for m in MODEL_NAMES if m in data[1.0].files and m in data[0.1].files]
        x, y = x_grid(truth.shape[0], LX), gauss_nodes(truth.shape[1], LY)
        interior = data[1.0]["truth"][..., fi][:, :, 8:-8]
        vmin, vmax = (float(v) for v in np.percentile(interior, [1, 99]))
        pw, gap, title_h, left = 3.3, 0.1, 0.17, 0.1
        ph = pw * LY / LX
        rows = 1 + len(models)
        cbar = 0.45
        H = rows * (ph + title_h) + cbar + 0.05
        fig = plt.figure(figsize=(FULL, H))
        mesh = None

        def panel(col: float, row: int, img, title):
            nonlocal mesh
            y0 = H - (row + 1) * (title_h + ph)
            ax = fig.add_axes(((left + col * (pw + gap)) / FULL, y0 / H, pw / FULL, ph / H))
            mesh = ax.pcolormesh(x, y, img.T, shading="nearest", cmap="inferno", vmin=vmin, vmax=vmax,
                                 rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(title, fontsize=7.5, pad=2)

        panel(0.5 + gap / (2 * pw), 0, truth, f"true flow, step {step}")
        for i, m in enumerate(models, start=1):
            for j, frac in enumerate((0.1, 1.0)):
                panel(j, i, data[frac][m][k, ..., fi], f"{STYLE[m]['label']}, {FRAC_TEXT[frac]} data")
        cax = fig.add_axes(((left + 0.6 * pw) / FULL, 0.22 / H, (0.8 * pw + gap) / FULL, 0.07 / H))
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label("Buoyancy $b$" if field == "buoyancy" else ff.FIELD_LABELS.get(field, field), fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
        written += rf.save(fig, out, f"fig_snapshots_fractions_{field}_step{step}_{rollout.removeprefix('rollout_')}",
                           pdf_dpi=300)
    return written


def main() -> int:
    outputs, out = REPO / "outputs", REPO / "reports" / "figures" / "final"
    find = outputs / "findings"

    def csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    runs, fc = csv(find / "runs.csv"), csv(find / "fraction_comparison.csv")
    resource, nu = csv(outputs / "tables" / "resource_use.csv"), csv(find / "nusselt_by_rayleigh.csv")
    comp, match = csv(find / "solver_comparison.csv"), csv(find / "solver_matching.csv")
    summ, shifts, noise_t = (csv(find / "uq" / f"{n}.csv") for n in ("uq_summary", "uq_shifts", "noise_robustness"))
    rel, cov_ra = (csv(find / "uq" / f"{n}.csv") for n in ("uq_reliability", "uq_coverage_by_rayleigh"))
    corr = csv(find / "uq" / "uq_correlation.csv")
    t_on = onset_frames()
    written = (data_scaling(fc, out) + uq(summ, shifts, out) + uq_reliability(rel, cov_ra, out) + noise(noise_t, out)
               + nusselt(nu, out) + correlation(corr, out))
    if not runs.empty:
        written += (accuracy_cost(fc, runs, resource, out) + test_by_window(outputs, runs, out, t_on)
                    + onset_aligned(outputs, runs, out, t_on)
                    + rollouts(outputs, runs, out) + by_rayleigh(outputs, runs, nu, out, t_on)
                    + solver(outputs, runs, comp, match, out) + time_breakdown(runs, out)
                    + rf.spectra(outputs, runs, out, fraction=1.0, tag="100%, 4 h")
                    + sum((rf.training(outputs, runs, out, fraction=f, name=f"fig_training_f{int(f * 100):03d}")
                           for f in FRACS), []))
    written += snapshots_by_fraction(outputs, out)
    for npz, tag in snapshot_files(outputs / "visuals"):    # steps 1, 12, 30 per fraction (old unsuffixed names skipped)
        if "__f010_" in tag or "__f100_" in tag:
            written += rf.snapshots(npz, out, tag)
    print("\n".join(f"wrote {p}" for p in written if p.endswith(".pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
