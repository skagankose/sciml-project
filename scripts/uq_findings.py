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
  uq_reliability.csv    RMSE of the ensemble mean in deciles of its predicted std, per field and pooled (one step;
                        rollout steps 1, 10, 30 from developed flow)
  uq_coverage_by_rayleigh.csv  conformal (and the beta-NLL Gaussian) coverage per Rayleigh number
  uq_coverage_per_window.csv   distribution of each window's own coverage (conditional coverage), by phase
  uq_crps_fields.csv    per field: the beta-NLL ensembles' CRPS against the MSE point forecast's absolute error
  uq_variance_by_group.csv     aleatoric and epistemic variance against the squared error, by phase and Ra
  uq_correlation.csv    spatial correlation with the truth per field along rollouts (four-hour seed pairs, both
                        fractions, and persistence; from frames 80/140 and from 20), per member and step
  uq_correlation_time.csv      high-correlation time: first step below 0.8 / 0.9 / 0.95 (D27)
Figures: uq_calibration, uq_variance_parts, uq_rank_pit, uq_conformal_width, noise_robustness.
"""

from __future__ import annotations

import json
import re
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
        e["family"] = ("persistence" if e["name"] == "persistence" else re.split(r"_f\d{3}_", e["name"])[0])
        e["fraction"] = (np.nan if e["name"] == "persistence" else int(re.search(r"_f(\d{3})_", e["name"]).group(1)) / 100)
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


# --- 2b. reliability, coverage by Rayleigh number and per window, CRPS per field, variance by phase and Ra ---------

def _scored_windows(e: dict, d: Path, onset_valid: dict, onset_test: dict):
    """Validation and test cells (one step, and rollouts from developed flow) with their window tables."""
    cal1, te1 = (ua.load_cells(d / s / "cells.npz", "one_step") for s in ("valid", "test"))
    calr, ter = (ua.load_cells(d / s / "cells.npz", "rollout") for s in ("valid", "test"))
    return cal1, te1, calr, ter, ua.window_table(te1, onset_test), ua.window_table(ter)


def reliability_table(outputs: Path, specs: list[dict], onset_valid: dict, onset_test: dict) -> pd.DataFrame:
    """RMSE of the ensemble mean in deciles of its predicted std, per field and pooled over the fields (normalised
    units): one step (all test windows) and rollout steps 1, 10, 30 from frames 80 and 140."""
    rows = []
    for e in specs:
        d = outputs / "uq" / e["name"]
        if e["n_members"] < 2 or not (d / "test" / "cells.npz").exists():
            continue
        _, te1, _, ter, _, w_ter = _scored_windows(e, d, onset_valid, onset_test)
        tag = {"ensemble": e["name"], "kind": e["kind"], "family": e["family"], "fraction": e["fraction"]}
        dev = w_ter.start.isin(DEV_STARTS).to_numpy()
        for window, cells, mask, step in [("one_step", te1, None, 1)] + [
                ("rollout", ter, dev & (w_ter.step == k).to_numpy(), k) for k in STEPS_SHOWN]:
            err = cells["err"].mean(0) if mask is None else cells["err"][:, mask].mean(0)
            sub = cells if mask is None else {**cells, "err": cells["err"][:, mask],
                                              "std": None if cells["std"] is None else cells["std"][:, mask]}
            sig = ua.predicted_std(sub)
            for fi, f in enumerate(FIELDS + ["pooled"]):
                ee, ss = (err, sig) if f == "pooled" else (err[..., fi], sig[..., fi])
                r = ua.reliability(ee, ss)
                rows.append(r.assign(**tag, window=window, step=step, field=f))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def coverage_group_tables(outputs: Path, specs: list[dict], onset_valid: dict, onset_test: dict):
    """Split-conformal 90% / 50% coverage (validation -> test) by Rayleigh number, and per window (conditional
    coverage: the distribution of each window's own coverage), for one step and rollout steps 1, 10, 30 from
    developed flow. The beta-NLL ensembles' Gaussian intervals come from the exact per-window summary."""
    by_ra, per_win = [], []
    for e in specs:
        d = outputs / "uq" / e["name"]
        if not (d / "valid" / "cells.npz").exists() or not (d / "test" / "cells.npz").exists():
            continue
        tag = {"ensemble": e["name"], "kind": e["kind"], "family": e["family"], "fraction": e["fraction"]}
        cal1, te1, calr, ter, w_te1, w_ter = _scored_windows(e, d, onset_valid, onset_test)
        dev_c = ua.window_table(calr).start.isin(DEV_STARTS).to_numpy()
        dev_t = w_ter.start.isin(DEV_STARTS).to_numpy()
        for scale in _scales(e):
            floor1 = ua.spread_floor(cal1) if scale == "spread" else None
            floorr = ua.spread_floor(calr) if scale == "spread" else None
            q1 = ua.conformal_by_group(cal1, te1, scale, group="pooled").drop(columns="step").assign(step=None)
            qr = ua.conformal_by_group(calr, ter, scale, cal_mask=dev_c, test_mask=dev_t)
            for ra in sorted(w_te1.rayleigh.unique()):
                c = ua.coverage_with(q1, te1, scale, floor1, test_mask=(w_te1.rayleigh == ra).to_numpy())
                by_ra.append(c.assign(**tag, scale=scale, window="one_step", rayleigh=ra))
                c = ua.coverage_with(qr, ter, scale, floorr, test_mask=dev_t & (w_ter.rayleigh == ra).to_numpy())
                by_ra.append(c[c.step.isin(STEPS_SHOWN)].assign(**tag, scale=scale, window="rollout", rayleigh=ra))
            s1, _ = ua.scores(te1, scale, floor1)
            sr, _ = ua.scores(ter, scale, floorr)
            for nominal in (0.9, 0.5):
                q = q1[q1.nominal == nominal].set_index("field").q.reindex(FIELDS).to_numpy()
                cov = ua.window_coverage(s1, q)
                for group, m in (("all", np.ones(len(cov), bool)), ("onset", w_te1.onset.to_numpy()),
                                 ("developed", ~w_te1.onset.to_numpy())):
                    per_win.append(_win_stats(cov[m], **tag, scale=scale, window="one_step", step=1, group=group,
                                              nominal=nominal))
                for k in STEPS_SHOWN:
                    m = dev_t & (w_ter.step == k).to_numpy()
                    qk = qr[(qr.step == k) & (qr.nominal == nominal)].set_index("field").q.reindex(FIELDS).to_numpy()
                    per_win.append(_win_stats(ua.window_coverage(sr[m], qk), **tag, scale=scale, window="rollout",
                                              step=k, group="developed", nominal=nominal))
        if "betanll" in e["name"]:                         # Gaussian intervals, exact over every grid point
            s = pd.read_parquet(d / "test" / "summary.parquet")
            s = s[s.sigma == 0]
            one = s[s.kind == "one_step"].assign(onset=lambda x: _phase(x, onset_test))
            roll = s[(s.kind == "rollout") & s.start.isin(DEV_STARTS)]
            for nominal, col in ((0.9, "cov90"), (0.5, "cov50")):
                for ra, g in one.groupby("rayleigh"):
                    by_ra.append(pd.DataFrame([{"step": 1, "field": "mean", "nominal": nominal,
                                                "coverage": _fm(g, col).mean(), **tag, "scale": "gaussian interval",
                                                "window": "one_step", "rayleigh": ra}]))
                for (ra, k), g in roll[roll.step.isin(STEPS_SHOWN)].groupby(["rayleigh", "step"]):
                    by_ra.append(pd.DataFrame([{"step": k, "field": "mean", "nominal": nominal,
                                                "coverage": _fm(g, col).mean(), **tag, "scale": "gaussian interval",
                                                "window": "rollout", "rayleigh": ra}]))
                for group, g in (("all", one), ("onset", one[one.onset]), ("developed", one[~one.onset])):
                    per_win.append(_win_stats(_fm(g, col).to_numpy(), **tag, scale="gaussian interval",
                                              window="one_step", step=1, group=group, nominal=nominal))
                for k in STEPS_SHOWN:
                    per_win.append(_win_stats(_fm(roll[roll.step == k], col).to_numpy(), **tag,
                                              scale="gaussian interval", window="rollout", step=k, group="developed",
                                              nominal=nominal))
    ra_t = pd.concat(by_ra, ignore_index=True) if by_ra else pd.DataFrame()
    if not ra_t.empty:                                     # add the field mean of the conformal rows
        fm = (ra_t[ra_t.field != "mean"].groupby([c for c in ra_t.columns if c not in
                                                  ("field", "q", "coverage", "width")], dropna=False)
              [["coverage", "width"]].mean().reset_index().assign(field="mean"))
        ra_t = pd.concat([ra_t, fm], ignore_index=True)
    return ra_t, pd.DataFrame(per_win)


def _win_stats(cov: np.ndarray, **tag) -> dict:
    """Distribution of the per-window coverage of one interval: its mean (the marginal coverage), percentiles and the
    share of windows covering less than nominal - 0.1."""
    q = np.quantile(cov, [0.1, 0.25, 0.5, 0.75, 0.9]) if cov.size else [np.nan] * 5
    return {**tag, "n_windows": int(cov.size), "mean": float(cov.mean()) if cov.size else np.nan,
            "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4],
            "share_below_nominal_minus_0.1": float((cov < tag["nominal"] - 0.1).mean()) if cov.size else np.nan}


def crps_field_table(outputs: Path, specs: list[dict], onset_test: dict) -> pd.DataFrame:
    """Per field: the beta-NLL ensembles' CRPS against the absolute error of the MSE point forecast (the MSE 1 h
    pair's members, same fraction), normalised units; one step (all, onset, developed) and rollout steps 1, 10, 30."""
    summ = {}
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "summary.parquet"
        if p.exists() and e["family"] == "cnext_unet" and e["kind"] in ("beta-NLL ensemble", "MSE 1 h pair"):
            s = pd.read_parquet(p)
            summ[(e["kind"], e["fraction"])] = s[s.sigma == 0]
    rows = []
    for frac in sorted({f for _, f in summ}):
        b, m = summ.get(("beta-NLL ensemble", frac)), summ.get(("MSE 1 h pair", frac))
        if b is None or m is None:
            continue

        def groups(s):
            one = s[s.kind == "one_step"]
            ph = _phase(one, onset_test)
            roll = s[(s.kind == "rollout") & s.start.isin(DEV_STARTS)]
            return ([("one_step", "all", one), ("one_step", "onset", one[ph]), ("one_step", "developed", one[~ph])]
                    + [("rollout", str(k), roll[roll.step == k]) for k in STEPS_SHOWN])
        for (window, group, gb), (_, _, gm) in zip(groups(b), groups(m)):
            for f in FIELDS + ["mean"]:
                crps = _fm(gb, "crps").mean() if f == "mean" else gb[f"crps_{f}"].mean()
                mae_ens = _fm(gb, "mae").mean() if f == "mean" else gb[f"mae_{f}"].mean()
                mae_pt = _fm(gm, "mae_member").mean() if f == "mean" else gm[f"mae_member_{f}"].mean()
                rows.append({"fraction": frac, "window": window, "group": group, "field": f, "crps": crps,
                             "mae_ensemble_mean": mae_ens, "mae_mse_point": mae_pt, "crps_over_mae_point": crps / mae_pt})
    return pd.DataFrame(rows)


def variance_group_table(outputs: Path, specs: list[dict], onset_test: dict) -> pd.DataFrame:
    """Aleatoric and epistemic variance (field mean and per field) against the squared error, one step, by phase and
    Rayleigh number: where each part of the beta-NLL ensembles' uncertainty is large, and the seed pairs' spread."""
    rows = []
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "summary.parquet"
        if not p.exists() or e["kind"] not in ("beta-NLL ensemble", "seed pair (4 h)"):
            continue
        s = pd.read_parquet(p)
        one = s[(s.sigma == 0) & (s.kind == "one_step")]
        one = one.assign(phase=np.where(_phase(one, onset_test), "onset", "developed"))
        for (phase, ra), g in list(one.groupby(["phase", "rayleigh"])) + [((ph, np.nan), g) for ph, g in
                                                                          one.groupby("phase")]:
            for f in FIELDS + ["mean"]:
                col = (lambda c: _fm(g, c).mean()) if f == "mean" else (lambda c: g[f"{c}_{f}"].mean())
                al, ep, mse = col("var_al"), col("var_ep"), col("mse")
                rows.append({"ensemble": e["name"], "kind": e["kind"], "family": e["family"],
                             "fraction": e["fraction"], "phase": phase, "rayleigh": ra, "field": f, "var_al": al,
                             "var_ep": ep, "mse": mse, "aleatoric_share": al / (al + ep) if al + ep > 0 else np.nan,
                             "ssr": ua.spread_skill(g, e["n_members"], FIELDS if f == "mean" else [f])})
    return pd.DataFrame(rows)


# --- 2c. pattern correlation along rollouts (D27; computed in this pass for every four-hour run) ------------------

CORR_GROUPS = (("developed", (80, 140)), ("onset", (20,)))


def correlation_table(outputs: Path, specs: list[dict]) -> pd.DataFrame:
    """Per seed pair (four-hour checkpoints, both fractions) and persistence, per member, start group and rollout
    step: the mean spatial correlation with the truth per field and its field mean, the mean VRMSE, and the share of
    rollouts whose field-mean correlation exceeds 0.8 (clean inputs). The core evaluation computed the correlation
    only for the 100% runs; this pass has it for every run, and its means agree with the core evaluation's within
    0.001 where both exist."""
    rows = []
    for e in specs:
        p = outputs / "uq" / e["name"] / "test" / "members.parquet"
        if e["kind"] not in ("seed pair (4 h)", "persistence") or not p.exists():
            continue
        m = pd.read_parquet(p)
        r = m[(m.kind == "rollout") & (m.sigma == 0)]
        if "corr_mean" not in r:
            continue
        cols = [f"corr_{f}" for f in FIELDS] + ["corr_mean", "vrmse_mean"]
        for group, starts in CORR_GROUPS:
            for member, g in r[r.start.isin(starts)].groupby("member"):
                per = g.groupby("step")[cols].mean()
                share = g.groupby("step").corr_mean.apply(lambda c: float((c > 0.8).mean()))
                for step, row in per.iterrows():
                    rows.append({"ensemble": e["name"], "family": e["family"], "fraction": e["fraction"],
                                 "member": member, "starts": group, "step": int(step), **row.to_dict(),
                                 "share_corr_above_0.8": share[step]})
    return pd.DataFrame(rows)


def correlation_time_table(corr: pd.DataFrame) -> pd.DataFrame:
    """High-correlation time (D27, `rbbench.evaluate.high_correlation_steps`): the first rollout step at which the
    correlation averaged over the rollouts falls below 0.8, 0.9 or 0.95; NaN if it stays above through step 30."""
    from rbbench.evaluate import high_correlation_steps

    rows = []
    for (ens, fam, frac, member, group), g in corr.groupby(["ensemble", "family", "fraction", "member", "starts"],
                                                           dropna=False):
        for col in ("corr_mean", "corr_buoyancy"):
            steps = high_correlation_steps(g.set_index("step")[col].sort_index())
            rows.append({"ensemble": ens, "family": fam, "fraction": frac, "member": member, "starts": group,
                         "field": col.removeprefix("corr_"),
                         **{f"below_{t}": (np.nan if v is None else v) for t, v in steps.items()}})
    return pd.DataFrame(rows)


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
    rel = reliability_table(outputs, specs, onset_valid, onset_test)
    cov_ra, cov_win = coverage_group_tables(outputs, specs, onset_valid, onset_test)
    crps_f = crps_field_table(outputs, specs, onset_test)
    var_g = variance_group_table(outputs, specs, onset_test)
    corr = correlation_table(outputs, specs)
    corr_t = correlation_time_table(corr) if not corr.empty else pd.DataFrame()
    written = []
    for name, df in (("uq_summary", summ), ("uq_conformal", conf), ("uq_shifts", shifts), ("uq_rank_pit", rp),
                     ("noise_robustness", noise), ("uq_yprofile", yprof), ("uq_reliability", rel),
                     ("uq_coverage_by_rayleigh", cov_ra), ("uq_coverage_per_window", cov_win),
                     ("uq_crps_fields", crps_f), ("uq_variance_by_group", var_g), ("uq_correlation", corr),
                     ("uq_correlation_time", corr_t)):
        df.to_csv(out / f"{name}.csv", index=False)
        written.append(str(out / f"{name}.csv"))
    written += figures(out, summ, conf, shifts, rp, noise, yprof)
    return written


if __name__ == "__main__":
    for p in main():
        print(f"wrote {p}")
