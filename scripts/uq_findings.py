#!/usr/bin/env python
"""Uncertainty and noise-robustness findings (WP8, WP9; D17, D30) from `outputs/uq/` -> `outputs/findings/uq/`.

Run by `scripts/findings.py` (or alone). Reads only the `eval-uq` outputs (`summary.parquet`, `members.parquet`,
`cells.npz`, `yprofile.npz`) and the frozen onset tables; the statistics are in `rbbench.uq_analysis`.

Tables (CSV):
  uq_summary.csv        per ensemble, test: accuracy, CRPS, Gaussian coverage, aleatoric / epistemic variance and the
                        spread-skill ratio, for one step (all, onset, developed) and rollout steps 1-30 from developed
                        flow (starts 80, 140)
  uq_conformal.csv      split conformal (validation -> test) per ensemble, scale, step and field: q-hat, coverage, width
  uq_shifts.csv         coverage under the three shifts: longer rollouts (q-hat of step 1), noisy inputs (clean q-hat),
                        Rayleigh number (calibrated on Ra <= 1e9, or on two Ra = 1e10 validation trajectories)
  uq_rank_pit.csv       rank histograms of the members and PIT histograms of the Gaussian mixtures
  noise_robustness.csv  each configuration's VRMSE at input noise sigma = 0-0.20 (seed mean), and its ratio to clean
  uq_yprofile.csv       squared error and predicted variance per y row (one step, developed): the boundary layers
Figures: uq_calibration, uq_variance_parts, uq_rank_pit, uq_conformal_width, noise_robustness.
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

from rbbench import uq_analysis as ua  # noqa: E402
from rbbench.onset import load_onset_frames  # noqa: E402

FIELDS = list(ua.FIELDS)
STEPS_SHOWN = (1, 10, 30)
DEV_STARTS = (80, 140)
SIGMAS = (0.0, 0.01, 0.05, 0.1, 0.2)


def ensembles() -> list[dict]:
    specs = json.loads((REPO / "tasks" / "uq_ensembles.json").read_text())["ensembles"]
    for i, e in enumerate(specs):
        e["index"] = i
        e["n_members"] = len(e["members"])
        e["family"] = ("persistence" if e["name"] == "persistence" else e["name"].split("_f")[0])
        e["fraction"] = (np.nan if e["name"] == "persistence" else int(e["name"].split("_f")[1][:3]) / 100)
        e["kind"] = ("beta-NLL ensemble" if "betanll" in e["name"] else "MSE 1 h pair" if e["name"].endswith("_mse")
                     else "persistence" if e["name"] == "persistence" else "seed pair (4 h)")
    return specs


def _fm(df: pd.DataFrame, metric: str) -> pd.Series:
    """Field mean of a summary metric."""
    return sum(df[f"{metric}_{f}"] for f in FIELDS) / len(FIELDS)


def _phase(df: pd.DataFrame, onset: dict) -> np.ndarray:
    return np.array([onset.get(t) is None or s < onset[t] for t, s in zip(df.traj_id, df.start)])


# --- 1. ensemble summary ---------------------------------------------------------------------------------------

def summary_table(outputs: Path, specs: list[dict], onset_test: dict) -> pd.DataFrame:
    rows = []
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "summary.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p)
        s = s[s.sigma == 0]
        one = s[s.kind == "one_step"]
        ph = _phase(one, onset_test)
        groups = [("one_step", "all", one), ("one_step", "onset", one[ph]), ("one_step", "developed", one[~ph])]
        roll = s[(s.kind == "rollout") & s.start.isin(DEV_STARTS)]
        groups += [("rollout", int(k), roll[roll.step == k]) for k in range(1, 31)]
        for kind, group, g in groups:
            rows.append({"ensemble": e["name"], "kind": e["kind"], "family": e["family"], "fraction": e["fraction"],
                         "n_members": e["n_members"], "window": kind, "group": group, "n_windows": len(g),
                         "vrmse_ens": g.vrmse_ens_mean.mean(), "crps": _fm(g, "crps").mean(),
                         "mae": _fm(g, "mae").mean(), "mae_member": _fm(g, "mae_member").mean(),
                         "rmse": np.sqrt(_fm(g, "mse").mean()), "cov50": _fm(g, "cov50").mean(),
                         "cov90": _fm(g, "cov90").mean(), "var_ep": _fm(g, "var_ep").mean(),
                         "var_al": _fm(g, "var_al").mean(),
                         "ssr": ua.spread_skill(g, e["n_members"]) if e["n_members"] > 1 or "betanll" in e["name"]
                         else np.nan})
    return pd.DataFrame(rows)


# --- 2. conformal, shifts, rank / PIT ---------------------------------------------------------------------------

def _scales(e: dict) -> list[str]:
    if "betanll" in e["name"]:
        return ["gaussian", "spread"]
    return ["constant", "spread"] if e["n_members"] > 1 else ["constant"]


def conformal_tables(outputs: Path, specs: list[dict], onset_valid: dict, onset_test: dict):
    conf, shifts, rp = [], [], []
    for e in specs:
        d = outputs / "uq" / e["name"]
        if not (d / "valid" / "cells.npz").exists() or not (d / "test" / "cells.npz").exists():
            continue
        tag = {"ensemble": e["name"], "kind": e["kind"], "family": e["family"], "fraction": e["fraction"]}
        cal1, te1 = ua.load_cells(d / "valid" / "cells.npz", "one_step"), ua.load_cells(d / "test" / "cells.npz", "one_step")
        calr, ter = ua.load_cells(d / "valid" / "cells.npz", "rollout"), ua.load_cells(d / "test" / "cells.npz", "rollout")
        w_cal1, w_te1 = ua.window_table(cal1, onset_valid), ua.window_table(te1, onset_test)
        w_calr, w_ter = ua.window_table(calr), ua.window_table(ter)
        dev_c, dev_t = w_calr.start.isin(DEV_STARTS).to_numpy(), w_ter.start.isin(DEV_STARTS).to_numpy()
        for scale in _scales(e):
            floor = ua.spread_floor(cal1) if scale == "spread" else None
            floor_r = ua.spread_floor(calr) if scale == "spread" else None
            # one step: one q-hat over all windows, coverage over all and by phase
            q1 = ua.conformal_by_group(cal1, te1, scale, group="pooled")
            conf.append(q1.assign(**tag, scale=scale, window="one_step", group="all"))
            for name, mask in (("onset", w_te1.onset.to_numpy()), ("developed", ~w_te1.onset.to_numpy())):
                c = ua.coverage_with(q1.drop(columns="step").assign(step=None), te1, scale, floor, test_mask=mask)
                conf.append(c.assign(**tag, scale=scale, window="one_step", group=name))
            # rollouts from developed flow: one q-hat per step
            qr = ua.conformal_by_group(calr, ter, scale, cal_mask=dev_c, test_mask=dev_t)
            conf.append(qr.assign(**tag, scale=scale, window="rollout", group="per step"))
            # shift 1: the q-hat of step 1 applied to every step
            q_s1 = qr[qr.step == 1].drop(columns="step").assign(step=None)
            c = ua.coverage_with(q_s1, ter, scale, floor_r, test_mask=dev_t)
            shifts.append(c.assign(**tag, scale=scale, shift="rollout: q-hat of step 1", sigma=0.0))
            # shift 2: noisy inputs, clean q-hat (one step pooled; rollouts per step)
            for sig in SIGMAS[1:]:
                n1 = ua.load_cells(d / "test" / "cells.npz", "one_step", sig)
                nr = ua.load_cells(d / "test" / "cells.npz", "rollout", sig)
                if n1 is None or nr is None:
                    continue
                c = ua.coverage_with(q1.drop(columns="step").assign(step=None), n1, scale, floor)
                shifts.append(c.assign(**tag, scale=scale, shift="noise: one step", sigma=sig))
                dev_n = ua.window_table(nr).start.isin(DEV_STARTS).to_numpy()
                c = ua.coverage_with(qr, nr, scale, floor_r, test_mask=dev_n)
                shifts.append(c.assign(**tag, scale=scale, shift="noise: rollout", sigma=sig))
            # shift 3: Rayleigh number (one step pooled)
            ra_c, ra_t = w_cal1.rayleigh.to_numpy(), w_te1.rayleigh.to_numpy()
            hi_ids = sorted({t for t, r in zip(w_cal1.traj_id, ra_c) if r == 1e10})[:2]
            for name, cmask in (("Ra: calibrated on all validation", np.ones(len(ra_c), bool)),
                                ("Ra: calibrated on Ra <= 1e9", ra_c <= 1e9),
                                ("Ra: calibrated on two Ra = 1e10 trajectories", w_cal1.traj_id.isin(hi_ids).to_numpy())):
                q = ua.conformal_by_group(cal1, te1, scale, group="pooled", cal_mask=cmask, test_mask=ra_t == 1e10)
                shifts.append(q.assign(**tag, scale=scale, shift=name, sigma=0.0))
        # rank histograms of the members, PIT of the Gaussian mixture
        if e["n_members"] > 1:
            for window, cells, mask, groups in (("one_step", te1, np.ones(len(w_te1), bool), [None]),
                                                ("rollout", ter, dev_t, STEPS_SHOWN)):
                for g in groups:
                    m = mask & ((w_ter.step == g).to_numpy() if g is not None else True)
                    h = ua.rank_histogram(cells["err"][:, m])
                    rp.append({**tag, "what": "rank", "window": window, "step": g, "histogram": json.dumps(h.round(4).tolist())})
                    if cells["std"] is not None:
                        pit = ua.pit_values(cells["err"][:, m], cells["std"][:, m])
                        hp = np.histogram(pit, bins=10, range=(0, 1))[0] / pit.size
                        rp.append({**tag, "what": "pit", "window": window, "step": g,
                                   "histogram": json.dumps(hp.round(4).tolist()),
                                   "tails_5pct": float(((pit < 0.05) | (pit > 0.95)).mean())})
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(conf), cat(shifts), pd.DataFrame(rp)


# --- 3. noise robustness (WP8) and boundary layers ---------------------------------------------------------------

def noise_table(outputs: Path, specs: list[dict], onset_test: dict) -> pd.DataFrame:
    rows = []
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "members.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        if m.sigma.nunique() < 2:
            continue
        one = m[m.kind == "one_step"]
        one = one.assign(onset=_phase(one, onset_test))
        roll = m[(m.kind == "rollout") & m.start.isin(DEV_STARTS)]
        groups = [("one_step", "all", one), ("one_step", "developed", one[~one.onset])]
        groups += [("rollout", str(k), roll[roll.step == k]) for k in STEPS_SHOWN]
        for kind, group, g in groups:
            per = g.groupby(["sigma", "member"]).vrmse_mean.mean().unstack("member")
            for sig, r in per.iterrows():
                rows.append({"ensemble": e["name"], "family": e["family"], "fraction": e["fraction"], "window": kind,
                             "group": group, "sigma": sig, "vrmse": r.mean(), "vrmse_min": r.min(),
                             "vrmse_max": r.max()})
    df = pd.DataFrame(rows)
    if not df.empty:
        clean = df[df.sigma == 0].set_index(["ensemble", "window", "group"]).vrmse
        df["ratio_to_clean"] = df.vrmse / df.set_index(["ensemble", "window", "group"]).index.map(clean)
    return df


def yprofile_table(outputs: Path, specs: list[dict], onset_test: dict) -> pd.DataFrame:
    rows = []
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "yprofile.npz"
        if not p.exists() or e["n_members"] < 2:
            continue
        with np.load(p) as y:
            ids, idx = y["traj_ids"].astype(str), y["index__one_step"]
            err2, var = y["err2__one_step"], y["var__one_step"]
        dev = np.array([not (onset_test.get(ids[r]) is None or s < onset_test[ids[r]]) for r, s, _ in idx])
        e2, v = err2[dev].mean(0), var[dev].mean(0)                   # [Y, F]
        for j in range(e2.shape[0]):
            rows.append({"ensemble": e["name"], "family": e["family"], "fraction": e["fraction"], "row": j,
                         **{f"rmse_{f}": float(np.sqrt(e2[j, i])) for i, f in enumerate(FIELDS)},
                         **{f"spread_{f}": float(np.sqrt(v[j, i])) for i, f in enumerate(FIELDS)}})
    return pd.DataFrame(rows)


# --- figures ------------------------------------------------------------------------------------------------------

def figures(out: Path, summ: pd.DataFrame, conf: pd.DataFrame, shifts: pd.DataFrame, rp: pd.DataFrame,
            noise: pd.DataFrame, yprof: pd.DataFrame) -> list[str]:
    import findings_figures as ff
    from matplotlib.lines import Line2D

    plt = ff._plt()
    from rbbench.aggregate import INK, MUTED, STYLE, _style_axes

    written = []
    col = {"cnext_unet": STYLE["cnext_unet"]["color"], "tfno": STYLE["tfno"]["color"], "avit": STYLE["avit"]["color"],
           "persistence": STYLE["persistence"]["color"]}
    lab = {"cnext_unet": "CNextU-Net", "tfno": "TFNO", "avit": "AViT", "persistence": "persistence"}

    # (1) calibration along the rollout: beta-NLL Gaussian coverage, conformal with the q-hat of step 1, SSR
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 3.9))
    r = summ[summ.window == "rollout"]
    for frac, ls in ((0.1, "--"), (1.0, "-")):
        g = r[(r.kind == "beta-NLL ensemble") & (r.fraction == frac)].sort_values("group")
        a.plot(g.group.astype(int), g.cov90, color=col["cnext_unet"], ls=ls, lw=1.8)
        a.plot(g.group.astype(int), g.cov50, color=col["cnext_unet"], ls=ls, lw=1.0, alpha=0.6)
    a.axhline(0.9, color=INK, lw=0.8)
    a.axhline(0.5, color=INK, lw=0.8, ls=":")
    a.set_title("(a) β-NLL ensembles: Gaussian 50% / 90% intervals", fontsize=9, color=INK)
    a.set_ylabel("coverage on test (field mean)")
    s1 = shifts[(shifts["shift"] == "rollout: q-hat of step 1") & (shifts.nominal == 0.9)]
    for (ens, scale), g in s1.groupby(["ensemble", "scale"]):
        meta = g.iloc[0]
        if meta.kind not in ("seed pair (4 h)", "persistence", "beta-NLL ensemble") or scale == "spread":
            continue
        if meta.kind == "seed pair (4 h)" and meta.fraction != 1.0:
            continue
        gg = g.groupby("step").coverage.mean()
        b.plot(gg.index, gg.values, color=col.get(meta.family, MUTED), lw=1.6,
               ls="-." if meta.kind == "beta-NLL ensemble" else ":" if meta.family == "persistence" else "-")
    b.axhline(0.9, color=INK, lw=0.8)
    b.set_title("(b) Conformal 90%, q-hat of step 1 kept (100% / 4 h; dash-dot: β-NLL)", fontsize=9, color=INK)
    for (ens, kind, family, frac), g in r.groupby(["ensemble", "kind", "family", "fraction"]):
        if kind == "persistence" or kind == "MSE 1 h pair":
            continue
        g = g.sort_values("group")
        c.plot(g.group.astype(int), g.ssr, color=col.get(family, MUTED), ls="-" if frac == 1.0 else "--",
               lw=1.6 if kind == "seed pair (4 h)" else 2.4, alpha=1.0 if kind == "seed pair (4 h)" else 0.5)
    c.axhline(1.0, color=INK, lw=0.8)
    c.set_yscale("log")
    c.set_title("(c) Spread-skill ratio (1 = consistent; thick: β-NLL)", fontsize=9, color=INK)
    for ax in (a, b, c):
        ax.set_xlabel("rollout step (from frames 80, 140)")
        _style_axes(ax)
    handles = [Line2D([], [], color=col[m], lw=1.6, label=lab[m]) for m in ("cnext_unet", "tfno", "avit", "persistence")]
    handles += [Line2D([], [], color=MUTED, ls="-", label="100% data"), Line2D([], [], color=MUTED, ls="--", label="10% data")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    written += ff._finish(fig, out / "uq_calibration")

    # (2) aleatoric and epistemic variance of the beta-NLL ensembles: along the rollout, and along y
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for frac, ls in ((0.1, "--"), (1.0, "-")):
        g = r[(r.kind == "beta-NLL ensemble") & (r.fraction == frac)].sort_values("group")
        a.plot(g.group.astype(int), g.var_al, color="#c2185b", ls=ls, lw=1.8)
        a.plot(g.group.astype(int), g.var_ep, color="#5e35b1", ls=ls, lw=1.8)
        a.plot(g.group.astype(int), g.rmse ** 2, color=INK, ls=ls, lw=1.0)
    a.set_yscale("log")
    a.set_xlabel("rollout step (from frames 80, 140)")
    a.set_ylabel("variance (normalised units², field mean)")
    a.set_title("(a) Aleatoric (pink), epistemic (purple), squared error (black)", fontsize=9, color=INK)
    if not yprof.empty:
        for frac, ls in ((0.1, "--"), (1.0, "-")):
            g = yprof[(yprof.family == "cnext_unet_f010_1h_betanll".split("_f")[0]) & (yprof.fraction == frac)
                      & yprof.ensemble.str.contains("betanll")].sort_values("row")
            if g.empty:
                continue
            b.plot(g.rmse_buoyancy, g.row, color=INK, ls=ls, lw=1.2)
            b.plot(g.spread_buoyancy, g.row, color="#c2185b", ls=ls, lw=1.6)
        b.set_xscale("log")
        b.set_xlabel("buoyancy: RMSE (black) and predicted std (pink)")
        b.set_ylabel("y row (0 = bottom wall, Gauss nodes)")
        b.set_title("(b) One step, developed flow, along the height", fontsize=9, color=INK)
    for ax in (a, b):
        _style_axes(ax)
    fig.legend(handles=[Line2D([], [], color=MUTED, ls="-", label="β-NLL ensemble, 100% data"),
                        Line2D([], [], color=MUTED, ls="--", label="β-NLL ensemble, 10% data")],
               loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    written += ff._finish(fig, out / "uq_variance_parts")

    # (3) rank histograms (seed pairs, one step) and PIT (beta-NLL, one step and step 10)
    if not rp.empty:
        rk = rp[(rp.what == "rank") & (rp.window == "one_step") & (rp.kind == "seed pair (4 h)")]
        pit = rp[(rp.what == "pit")]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
        width = 0.13
        for i, (_, row) in enumerate(rk.sort_values(["family", "fraction"]).iterrows()):
            h = json.loads(row["histogram"])
            axes[0].bar(np.arange(3) + (i - 2.5) * width, h, width, color=col.get(row.family, MUTED),
                        alpha=1.0 if row.fraction == 1.0 else 0.45)
        axes[0].axhline(1 / 3, color=INK, lw=0.8)
        axes[0].set_xticks(range(3), ["both above y", "one each side", "both below y"])
        axes[0].set_ylabel("share of grid points")
        axes[0].set_title("(a) Rank of the truth among the two seeds, one step (flat = 1/3)", fontsize=9, color=INK)
        for _, row in pit.iterrows():
            h = json.loads(row["histogram"])
            ls = "-" if row.fraction == 1.0 else "--"
            lw = 1.8 if row.window == "one_step" else 1.0
            axes[1].step(np.linspace(0.05, 0.95, 10), np.array(h) * 10, where="mid", color=col["cnext_unet"], ls=ls,
                         lw=lw, alpha=1.0 if row.window == "one_step" else 0.5)
        axes[1].axhline(1.0, color=INK, lw=0.8)
        axes[1].set_xlabel("PIT value")
        axes[1].set_ylabel("density (flat = 1)")
        axes[1].set_title("(b) PIT of the β-NLL mixtures (thick: one step; thin: rollout steps)", fontsize=9, color=INK)
        for ax in axes:
            _style_axes(ax)
        fig.tight_layout()
        written += ff._finish(fig, out / "uq_rank_pit")

    # (4) conformal widths: 90% interval width per step, constant scale, by model and fraction
    if not conf.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.9))
        cw = conf[(conf.window == "rollout") & (conf.nominal == 0.9) & (conf.scale.isin(["constant", "gaussian"]))]
        for (ens, kind, family, frac, scale), g in cw.groupby(["ensemble", "kind", "family", "fraction", "scale"],
                                                              dropna=False):
            if kind == "MSE 1 h pair":
                continue
            gg = g.groupby("step").width.mean()
            ax.plot(gg.index, gg.values, color=col.get(family, MUTED),
                    ls="-." if kind == "beta-NLL ensemble" else (":" if family == "persistence" else
                                                                 "-" if frac == 1.0 else "--"),
                    lw=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step (from frames 80, 140)")
        ax.set_ylabel("width of the 90% interval (normalised, field mean)")
        ax.set_title("Split-conformal 90% intervals: guaranteed coverage, width by model\n"
                     "(solid 100% / dashed 10% at 4 h; dash-dot β-NLL 1 h, Gaussian scale)", fontsize=9, color=INK)
        _style_axes(ax)
        fig.tight_layout()
        written += ff._finish(fig, out / "uq_conformal_width")

    # (5) noise robustness: VRMSE against input noise
    if not noise.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
        for ax, (window, group, title) in zip(axes, (("one_step", "developed", "(a) One step, developed flow"),
                                                     ("rollout", "10", "(b) Rollout step 10, from frames 80, 140"))):
            g = noise[(noise.window == window) & (noise.group == group)]
            for (ens, family, frac), gg in g.groupby(["ensemble", "family", "fraction"], dropna=False):
                gg = gg.sort_values("sigma")
                ax.plot(gg.sigma, gg.vrmse, color=col.get(family, MUTED), marker="o", ms=3.5,
                        ls=":" if family == "persistence" else "-" if frac == 1.0 else "--", lw=1.5)
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_xticks(list(SIGMAS), [f"{s:g}" for s in SIGMAS])
            ax.set_yscale("log")
            ax.set_xlabel("input noise σ (normalised units)")
            ax.set_ylabel("test VRMSE (field mean)")
            ax.set_title(title, fontsize=9, color=INK)
            _style_axes(ax)
        handles = [Line2D([], [], color=col[m], lw=1.6, label=lab[m]) for m in ("cnext_unet", "tfno", "avit", "persistence")]
        handles += [Line2D([], [], color=MUTED, ls="-", label="100% / 4 h"), Line2D([], [], color=MUTED, ls="--", label="10% / 4 h")]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        written += ff._finish(fig, out / "noise_robustness")
    return written


def main(outputs: Path | None = None, out: Path | None = None) -> list[str]:
    outputs = outputs or REPO / "outputs"
    out = out or outputs / "findings" / "uq"
    if not (outputs / "uq").exists():
        return []
    out.mkdir(parents=True, exist_ok=True)
    specs = ensembles()
    onset_test = load_onset_frames(REPO / "tasks" / "onset_test.json")
    onset_valid = load_onset_frames(REPO / "tasks" / "onset_valid.json")
    summ = summary_table(outputs, specs, onset_test)
    conf, shifts, rp = conformal_tables(outputs, specs, onset_valid, onset_test)
    noise = noise_table(outputs, specs, onset_test)
    yprof = yprofile_table(outputs, specs, onset_test)
    written = []
    for name, df in (("uq_summary", summ), ("uq_conformal", conf), ("uq_shifts", shifts), ("uq_rank_pit", rp),
                     ("noise_robustness", noise), ("uq_yprofile", yprof)):
        df.to_csv(out / f"{name}.csv", index=False)
        written.append(str(out / f"{name}.csv"))
    written += figures(out, summ, conf, shifts, rp, noise, yprof)
    return written


if __name__ == "__main__":
    for p in main():
        print(f"wrote {p}")
