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

def released_by_window(outputs: Path, out: Path) -> list[str]:
    tables = {m: t for m in ("cnext_unet", "tfno") if (t := ff._released(outputs, m, "one_step")) is not None}
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
        ax.set_xlabel("first input frame of the window")
    axes[0].set_ylabel("one-step VRMSE")
    handles = [Line2D([], [], color=STYLE[m]["color"], label=f"released {STYLE[m]['label']}") for m in tables]
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], ls=":", lw=1.3, label="persistence (copy last frame)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_released_by_window")


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
    ax.set_xlabel("data-loader worker processes")
    ax.set_ylabel("training samples per second")
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
        ax.set_xlabel("GPU training time (h)")
    ax_loss.text(1.04, 0.97, "1 h checkpoint", transform=ax_loss.get_xaxis_transform(), fontsize=7, color=MUTED,
                 va="top")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("training loss (MSE, normalised)")
    ax_val.set_ylabel("validation VRMSE (64 windows)")
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
    ax.set_xlabel("first input frame of the window")
    ax.set_ylabel("one-step VRMSE (physical fields)")
    handles = [Line2D([], [], color=STYLE[m]["color"], label=f"{STYLE[m]['label']} (ours, 10%, {label})")
               for m in MODEL_NAMES if m in tables]
    if released is not None:
        handles.append(Line2D([], [], color=INK, lw=1.1, ls="--", label="released CNextU-Net (100%)"))
    if persist is not None:
        handles.append(Line2D([], [], color=PERSIST["color"], lw=1.3, ls=":", label="persistence"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    return save(fig, out, "fig_test_by_window")


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
    a.set_ylabel("kinetic energy E(k_x)")
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
    c.tick_params(labelleft=False)
    b.set_ylabel("E_pred / E_true")
    for ax in axes:
        ax.axvspan(ff.SMALL_SCALES_K, kmax * 1.1, color=MUTED, alpha=0.08, lw=0)
        ax.set_xlabel("horizontal wavenumber k_x")
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
    vmin, vmax = (0.0, 1.0) if field == "buoyancy" else (-np.abs(truth).max(), np.abs(truth).max())
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(FULL, 0.52 * len(rows) + 0.75), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    mesh = None
    for i, step in enumerate(rows):
        k = have.index(step)
        for j, col in enumerate(cols):
            ax = axes[i][j]
            img = truth[k] if col == "truth" else d[col][k, ..., fi]
            mesh = ax.pcolormesh(x, y, img.T, shading="nearest", cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=True)
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
    cbar.set_label(ff.FIELD_LABELS.get(field, field), fontsize=7, color=INK)
    cbar.ax.tick_params(labelsize=6.5)
    tag = npz.stem.removeprefix("rollout_")
    return save(fig, out, f"fig_snapshots_{field}_{tag}")


def main() -> int:
    outputs, out = REPO / "outputs", REPO / "reports" / "figures"
    runs_csv = outputs / "findings" / "runs.csv"
    runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()
    written = released_by_window(outputs, out) + loader(outputs, out)
    if not runs.empty:
        written += training(outputs, runs, out) + test_by_window(outputs, runs, out) + spectra(outputs, runs, out)
    for npz in sorted((outputs / "visuals").glob("rollout_*.npz")) if (outputs / "visuals").exists() else []:
        written += snapshots(npz, out)
    print("\n".join(f"wrote {p}" for p in written if p.endswith(".pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
