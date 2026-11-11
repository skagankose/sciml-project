"""One command from copied run outputs to report tables and figures (runs on the Mac).

Reads tasks/train.jsonl, each run's COMPLETE.json, provenance.json and evaluations/, and writes
tables/{main_results,per_seed_results,paper_protocol,resource_use,gpu_hours,missing_runs}.csv and
figures/*.{pdf,png}.
Every number keeps its run_id, so it can be traced to commit, seed, manifest and raw file.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rbbench import METRIC_VERSION, MODEL_NAMES
from rbbench.data import FIELDS
from rbbench.evaluate import failure_step, kaplan_meier
from rbbench.manifests import read_json
from rbbench.noise import sigma_key
from rbbench.sweep import read_jsonl

# Fixed model -> colour/marker (reference categorical slots 1-4, light mode), never by rank.
STYLE = {
    "cnext_unet": {"color": "#2a78d6", "marker": "o", "label": "CNextU-Net"},
    "tfno": {"color": "#eb6834", "marker": "s", "label": "TFNO"},
    "avit": {"color": "#1baf7a", "marker": "^", "label": "AViT"},
    "convlstm": {"color": "#eda100", "marker": "D", "label": "ConvLSTM"},
    "persistence": {"color": "#52514e", "marker": "", "label": "Persistence"},
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _load_run(row: dict, output_root: Path, labels: list[str]) -> tuple[list[dict], dict | None, list[str]]:
    run_dir = output_root / "runs" / row["run_id"]
    problems, metrics = [], []
    complete = run_dir / "COMPLETE.json"
    if not complete.exists():
        return [], None, [f"{row['run_id']}: training not complete"]
    comp = read_json(complete)
    if not comp.get("budget_satisfied", False):
        problems.append(f"{row['run_id']}: milestones forced by max_steps (budget not satisfied)")
    for lab in labels:
        marker = run_dir / "evaluations" / f"EVAL_COMPLETE_{lab}.json"
        if not marker.exists():
            problems.append(f"{row['run_id']}: evaluation {lab} missing")
            continue
        info = read_json(marker)
        sigmas = {}
        for path in sorted((run_dir / "evaluations").glob(f"*_{lab}.json")):
            if path.name.startswith(("clean_", "noise_")):
                d = read_json(path)
                sigmas[sigma_key(d["sigma"])] = d
        metrics.append({"row": row, "label": lab, "info": info, "by_sigma": sigmas, "run_dir": run_dir,
                        "complete": comp})
    return metrics, comp, problems


def row_labels(row: dict) -> list[str]:
    """Milestone labels of one task, from the exact config the task trains with."""
    from rbbench.config import budget_labels
    from rbbench.sweep import task_config

    cfg = task_config(row["model"], row["experiment"], row["size"], row["subset_manifest"], row.get("overrides"))
    return budget_labels(cfg)


def collect(tasks_path: Path, output_root: Path, labels=None):
    """Load every task's evaluations; `labels` defaults to each task's own milestones."""
    rows = read_jsonl(tasks_path)
    loaded, problems, expected = [], [], 0
    for row in rows:
        want = list(labels) if labels else row_labels(row)
        expected += len(want)
        metrics, _, probs = _load_run(row, output_root, want)
        loaded += metrics
        problems += probs
    versions = {m["info"]["metric_version"] for m in loaded}
    cohorts = {m["info"]["cohort_sha256"] for m in loaded}
    if len(versions) > 1 or (versions and versions != {METRIC_VERSION}):
        raise RuntimeError(f"mixed metric versions {versions}; re-evaluate before aggregating")
    if len(cohorts) > 1:
        raise RuntimeError(f"runs were evaluated on different cohorts: {cohorts}")
    return rows, loaded, problems, expected


def per_seed_table(loaded: list[dict]) -> pd.DataFrame:
    recs = []
    for m in loaded:
        r = m["row"]
        key = {"run_id": r["run_id"], "model": r["model"], "fraction": r["fraction"], "seed": r["seed"],
               "budget": m["label"]}
        for skey, d in m["by_sigma"].items():
            s = d["sigma"]
            if "one_step" in d:
                for name, v in d["one_step"].items():
                    if name.startswith("vrmse"):
                        recs.append({**key, "metric": f"one_step_{name}", "sigma": s, "step": 1, "value": v})
            paper = d.get("paper_protocol")
            if paper:
                for name, v in paper.items():
                    if name.startswith("one_step_normalised_vrmse"):
                        recs.append({**key, "metric": f"paper_{name}", "sigma": s, "step": 1, "value": v})
                for units, windows in paper["rollout"]["window_vrmse_mean"].items():
                    for window, v in windows.items():
                        recs.append({**key, "metric": f"paper_rollout_{units}_vrmse_steps_{window.replace(':', '-')}",
                                     "sigma": s, "step": None, "value": np.nan if v is None else v})
            for step, v in d["rollout_vrmse_mean_at"].items():
                recs.append({**key, "metric": "rollout_vrmse_mean", "sigma": s, "step": int(step), "value": v})
            for thr, f in d["failure"].items():
                med = f["km_median_step"]
                recs.append({**key, "metric": f"km_median_failure_step_thr{thr}", "sigma": s, "step": None,
                             "value": np.nan if med is None else med})
                for step, v in f["survival_at"].items():
                    recs.append({**key, "metric": f"survival_thr{thr}", "sigma": s, "step": int(step), "value": v})
            for step, v in d.get("nusselt", {}).get("rel_err_mean_at", {}).items():
                recs.append({**key, "metric": "nusselt_rel_err", "sigma": s, "step": int(step), "value": v})
            for step, v in d.get("log_spectral_error_at", {}).items():
                recs.append({**key, "metric": "log_spectral_error", "sigma": s, "step": int(step), "value": v})
    return pd.DataFrame(recs)


def summary_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed
    keys = ["model", "fraction", "budget", "metric", "sigma", "step"]
    g = per_seed.groupby(keys, dropna=False)["value"]
    out = g.agg(mean="mean", std="std", n_seeds="count").reset_index()
    seeds = per_seed.groupby(keys, dropna=False).apply(
        lambda d: ";".join(f"s{int(s)}={v:.6g}" for s, v in zip(d.seed, d.value)), include_groups=False
    )
    out["per_seed"] = seeds.values
    return out


PAPER_TABLE_METRICS = (
    "paper_one_step_normalised_vrmse_mean",
    "paper_rollout_normalised_vrmse_steps_6-12", "paper_rollout_normalised_vrmse_steps_13-30",
    "paper_rollout_physical_vrmse_steps_6-12", "paper_rollout_physical_vrmse_steps_13-30",
)


def paper_protocol_table(summary: pd.DataFrame, released_check: dict | None) -> pd.DataFrame:
    """Our runs under The Well's protocol (Tables 2 and 3, test split; D20), next to the published values and to
    the released checkpoints scored by the same code (`eval-released`)."""
    from rbbench.paper import PUBLISHED_ROLLOUT_ABOVE, PUBLISHED_VRMSE

    if summary.empty:
        return summary
    d = summary[summary.metric.isin(PAPER_TABLE_METRICS) & (summary.sigma == 0)].copy()
    released = (released_check or {}).get("results", {})

    def published(r) -> str:
        if r.model not in PUBLISHED_VRMSE:
            return ""
        if r.metric == "paper_one_step_normalised_vrmse_mean":
            return f"{PUBLISHED_VRMSE[r.model]['test']:g}"
        return f"> {PUBLISHED_ROLLOUT_ABOVE:g}" if "_normalised_" in r.metric else ""

    def released_value(r) -> float:
        test = released.get(r.model, {}).get("test", {})
        if r.metric == "paper_one_step_normalised_vrmse_mean":
            return test.get("normalised_vrmse_mean", np.nan)
        units = "normalised" if "_normalised_" in r.metric else "physical"
        window = r.metric.rsplit("_steps_", 1)[1].replace("-", ":")
        v = test.get("paper_rollout", {}).get("window_vrmse_mean", {}).get(units, {}).get(window)
        return np.nan if v is None else v

    d["published"] = [published(r) for r in d.itertuples()]
    d["released_checkpoint"] = [released_value(r) for r in d.itertuples()]
    order = {m: i for i, m in enumerate(PAPER_TABLE_METRICS)}
    d = d.sort_values(["metric", "model", "fraction", "budget"], key=lambda c: c.map(order) if c.name == "metric"
                      else c.map({m: i for i, m in enumerate(MODEL_NAMES)}) if c.name == "model" else c)
    return d[["metric", "model", "fraction", "budget", "mean", "std", "n_seeds", "per_seed", "published",
              "released_checkpoint"]].reset_index(drop=True)


def resource_table(loaded: list[dict]) -> pd.DataFrame:
    recs = []
    for m in loaded:
        r, comp = m["row"], m["complete"]
        ms = comp["milestones"].get(m["label"], {})
        prov_path = m["run_dir"] / "provenance.json"
        prov = read_json(prov_path) if prov_path.exists() else {}
        clean = m["by_sigma"].get(sigma_key(0.0), {})
        lat = clean.get("latency", {})
        gpus = sorted({a.get("gpu", {}).get("name") or "none" for a in prov.get("attempts", [])})
        recs.append({
            "run_id": r["run_id"], "model": r["model"], "fraction": r["fraction"], "seed": r["seed"],
            "budget": m["label"], "train_seconds": ms.get("train_seconds"),
            "overshoot_seconds": ms.get("overshoot_seconds"), "data_wait_seconds": ms.get("data_wait_seconds"),
            "optimizer_steps": ms.get("step"),
            "samples_seen": ms.get("samples_seen"), "peak_train_mem_gb": ms.get("peak_mem_alloc_gb"),
            "n_parameters": prov.get("n_parameters"), "attempts": comp.get("attempts"),
            "val_overhead_s": comp.get("overhead_seconds", {}).get("val"),
            "ckpt_overhead_s": comp.get("overhead_seconds", {}).get("ckpt"),
            "infer_ms_per_frame": lat.get("ms_per_frame"), "rollout_ms": lat.get("ms_per_rollout"),
            "infer_peak_mem_gb": lat.get("peak_mem_alloc_gb"), "gpus": ";".join(gpus),
            "budget_satisfied": comp.get("budget_satisfied"),
        })
    return pd.DataFrame(recs)


def gpu_hours_table(rows: list[dict], output_root: Path) -> pd.DataFrame:
    """Wall and GPU hours per run, plus a TOTAL row (one GPU per job).

    Training: each attempt's wall time from train_run start to exit (provenance.json). An
    attempt killed without closing falls back to its last train_metrics.csv row, a lower
    bound, and is counted in `unclosed_attempts`. Evaluation: the wall time of each evaluation
    pass (EVAL_COMPLETE markers), counted once even when a pass covered several milestones.
    Neither includes queue time or the job's environment setup; `sacct` has Slurm's own figure.
    """
    recs = []
    for row in rows:
        run_dir = output_root / "runs" / row["run_id"]
        prov_path = run_dir / "provenance.json"
        if not prov_path.exists():
            continue
        attempts = read_json(prov_path).get("attempts", [])
        logged = {}
        if (run_dir / "train_metrics.csv").exists():
            log = pd.read_csv(run_dir / "train_metrics.csv", usecols=["attempt", "wall_seconds"])
            logged = log.groupby("attempt")["wall_seconds"].max().to_dict()
        train_wall = train_gpu = 0.0
        unclosed = 0
        for a in attempts:
            wall = a.get("wall_seconds")
            if wall is None:
                unclosed += 1
                wall = float(logged.get(a.get("attempt"), 0.0))
            train_wall += wall
            train_gpu += wall if str(a.get("device", "")).startswith("cuda") else 0.0
        passes = {}
        for marker in sorted((run_dir / "evaluations").glob("EVAL_COMPLETE_*.json")):
            p = read_json(marker).get("pass")
            if p:
                passes[(tuple(p["labels"]), p["wall_seconds"])] = p
        eval_wall = sum(p["wall_seconds"] for p in passes.values())
        eval_gpu = sum(p["wall_seconds"] for p in passes.values() if p["device"].startswith("cuda"))
        recs.append({
            "run_id": row["run_id"], "model": row["model"], "fraction": row["fraction"], "seed": row["seed"],
            "attempts": len(attempts), "unclosed_attempts": unclosed,
            "train_wall_hours": train_wall / 3600, "train_gpu_hours": train_gpu / 3600,
            "eval_passes": len(passes), "eval_wall_hours": eval_wall / 3600, "eval_gpu_hours": eval_gpu / 3600,
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    total = {c: df[c].sum() for c in df.columns if c.endswith("_hours") or c in ("attempts", "unclosed_attempts",
                                                                                  "eval_passes")}
    return pd.concat([df, pd.DataFrame([{"run_id": "TOTAL", **total}])], ignore_index=True)


# --- figures ------------------------------------------------------------------------------

def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _save(fig, figures_dir: Path, name: str) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = figures_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p))
    return paths


def _models_present(df):
    return [m for m in MODEL_NAMES if m in set(df.model)]


def make_figures(summary: pd.DataFrame, loaded: list[dict], resources: pd.DataFrame,
                 figures_dir: Path, baseline: dict | None = None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    written: list[str] = []
    if summary.empty:
        return written
    models = _models_present(summary)
    budgets = sorted(set(summary.budget))

    # 1. one-step VRMSE vs data fraction, per budget, average and per field
    names = ["one_step_vrmse_mean"] + [f"one_step_vrmse_{f}" for f in FIELDS]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), sharey=False)
    for ax, metric in zip(axes, names):
        for m in models:
            for b, ls in zip(budgets, ("--", "-", ":")):
                d = summary[(summary.model == m) & (summary.metric == metric) & (summary.budget == b)]
                d = d.sort_values("fraction")
                if d.empty:
                    continue
                st = STYLE[m]
                ax.errorbar(d.fraction * 100, d["mean"], yerr=d["std"].fillna(0), color=st["color"],
                            marker=st["marker"], ms=5, lw=1.5, ls=ls, capsize=2,
                            label=f"{st['label']} {b}")
        ax.set_xscale("log")
        ax.set_xticks([10, 30, 100], ["10", "30", "100"])
        ax.set_xlabel("training trajectories (%)")
        ax.set_title(metric.replace("one_step_vrmse_", "").replace("mean", "field mean"), fontsize=9, color=INK)
        _style_axes(ax)
    axes[0].set_ylabel("one-step test VRMSE")
    axes[0].legend(fontsize=6, frameon=False, ncol=1)
    fig.tight_layout()
    written += _save(fig, figures_dir, "fig1_one_step_vrmse_vs_fraction")
    plt.close(fig)

    # 2. rollout VRMSE vs horizon (clean), one panel per budget, full data
    full = max(summary.fraction)
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.2), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], budgets):
        for m in models:
            d = summary[(summary.model == m) & (summary.metric == "rollout_vrmse_mean") & (summary.budget == b)
                        & (summary.sigma == 0) & (summary.fraction == full)].sort_values("step")
            if d.empty:
                continue
            st = STYLE[m]
            ax.plot(d.step, d["mean"], color=st["color"], marker=st["marker"], ms=5, lw=2, label=st["label"])
            ax.fill_between(d.step, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0),
                            color=st["color"], alpha=0.15, lw=0)
        if baseline:
            curve = baseline.get("rollout_vrmse_mean_at", {})
            if curve:
                ks = sorted(int(k) for k in curve)
                ax.plot(ks, [curve[str(k)] for k in ks], color=STYLE["persistence"]["color"], lw=1.2, ls=":",
                        label="Persistence")
        ax.set_title(f"{b} budget, {int(full * 100)}% data, clean", fontsize=9, color=INK)
        ax.set_xlabel("rollout step")
        _style_axes(ax)
    axes[0][0].set_ylabel("test VRMSE (field mean)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    written += _save(fig, figures_dir, "fig2_rollout_vrmse_vs_step")
    plt.close(fig)

    # 3. survival (fraction of rollouts below threshold) vs step, recomputed from raw rollouts
    thr_metrics = sorted({m for m in summary.metric if m.startswith("survival_thr")})
    if thr_metrics:
        thr = thr_metrics[0].replace("survival_thr", "")
        fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.2), sharey=True, squeeze=False)
        for ax, b in zip(axes[0], budgets):
            for m in models:
                curves = []
                for item in loaded:
                    if item["row"]["model"] != m or item["label"] != b or item["row"]["fraction"] != full:
                        continue
                    roll = pd.read_parquet(item["run_dir"] / "evaluations" / f"rollout_{b}.parquet")
                    roll = roll[roll.sigma == 0]
                    piv = roll.pivot_table(index=["traj_id", "start"], columns="step", values="vrmse_mean")
                    events = [failure_step(v.values, float(thr)) for _, v in piv.iterrows()]
                    curves.append(kaplan_meier(events, piv.shape[1]))
                if not curves:
                    continue
                c = np.mean(curves, axis=0)
                st = STYLE[m]
                ax.step(np.arange(0, len(c) + 1), np.r_[1.0, c], where="post", color=st["color"], lw=2,
                        label=st["label"])
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{b}: rollouts with VRMSE <= {thr}", fontsize=9, color=INK)
            ax.set_xlabel("rollout step")
            _style_axes(ax)
        axes[0][0].set_ylabel("surviving fraction (KM, seed mean)")
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        written += _save(fig, figures_dir, "fig3_failure_survival")
        plt.close(fig)

    # 4. noise degradation at a fixed step (only when noise levels were evaluated)
    steps = sorted(set(summary[summary.metric == "rollout_vrmse_mean"].step.dropna()))
    if steps and summary.sigma.nunique() > 1:
        k = int(steps[min(2, len(steps) - 1)])
        fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.2), sharey=True, squeeze=False)
        for ax, b in zip(axes[0], budgets):
            for m in models:
                d = summary[(summary.model == m) & (summary.metric == "rollout_vrmse_mean") & (summary.budget == b)
                            & (summary.step == k) & (summary.fraction == full)].sort_values("sigma")
                if d.empty:
                    continue
                st = STYLE[m]
                ax.errorbar(d.sigma, d["mean"], yerr=d["std"].fillna(0), color=st["color"], marker=st["marker"],
                            ms=5, lw=1.5, capsize=2, label=st["label"])
            ax.set_title(f"{b}: VRMSE at step {k} vs initial noise", fontsize=9, color=INK)
            ax.set_xlabel("sigma (normalised units, initial 4 frames)")
            _style_axes(ax)
        axes[0][0].set_ylabel("test VRMSE (field mean)")
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        written += _save(fig, figures_dir, "fig4_noise_degradation")
        plt.close(fig)

    # 5. Nusselt relative error by regime (clean, final budget, full data, last spectra step)
    phys_frames = []
    for item in loaded:
        p = item["run_dir"] / "evaluations" / f"physics_{item['label']}.parquet"
        if p.exists() and item["row"]["fraction"] == full and item["label"] == budgets[-1]:
            df = pd.read_parquet(p)
            df["model"] = item["row"]["model"]
            phys_frames.append(df[df.sigma == 0])
    if phys_frames:
        phys = pd.concat(phys_frames)
        last = phys.step.max()
        d = phys[phys.step == last].groupby(["model", "rayleigh"])["nu_rel_err"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for m in _models_present(d):
            dm = d[d.model == m].sort_values("rayleigh")
            st = STYLE[m]
            ax.plot(dm.rayleigh, dm.nu_rel_err, color=st["color"], marker=st["marker"], ms=5, lw=2, label=st["label"])
        ax.set_xscale("log")
        ax.set_xlabel("Rayleigh number")
        ax.set_ylabel(f"|Nu_pred - Nu_true| / Nu_true at step {int(last)}")
        ax.set_title(f"{budgets[-1]}, {int(full * 100)}% data, clean (mean over Pr)", fontsize=9, color=INK)
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        written += _save(fig, figures_dir, "fig5_nusselt_rel_err_by_ra")
        plt.close(fig)

    # 6. horizontal KE spectra for one representative regime
    spec_frames = []
    for item in loaded:
        p = item["run_dir"] / "evaluations" / f"spectra_{item['label']}.parquet"
        if p.exists() and item["row"]["fraction"] == full and item["label"] == budgets[-1] and item["row"]["seed"] == 0:
            df = pd.read_parquet(p)
            df["model"] = item["row"]["model"]
            spec_frames.append(df[df.sigma == 0])
    if spec_frames:
        spec = pd.concat(spec_frames)
        regimes = sorted({(r, p) for r, p in zip(spec.rayleigh, spec.prandtl)})
        ra, pr = regimes[len(regimes) // 2]
        step = spec.step.max()
        # the Nyquist mode is dropped: the data hold almost no energy there, so it would set the axis range
        s = spec[(spec.rayleigh == ra) & (spec.prandtl == pr) & (spec.step == step) & (spec.k_index > 0)
                 & (spec.k_index < spec.k_index.max())]
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        truth = s[s.source == "true"].groupby("k")["energy"].mean()
        ax.loglog(truth.index, truth.values, color=INK, lw=2.2, label="reference")
        for m in _models_present(s):
            d = s[(s.model == m) & (s.source == "pred")].sort_values("k")
            st = STYLE[m]
            ax.loglog(d.k, d.energy, color=st["color"], lw=1.5, label=st["label"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("horizontal wavenumber k_x")
        ax.set_ylabel("E(k_x)")
        ax.set_title(f"Ra={ra:.0e}, Pr={pr:g}, step {int(step)}, seed 0", fontsize=9, color=INK)
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        written += _save(fig, figures_dir, "fig6_ke_spectra")
        plt.close(fig)

    # 7. resources: optimizer steps and inference latency (separate panels, one axis each)
    if not resources.empty:
        res = resources[resources.fraction == full]
        cols = [("optimizer_steps", "optimizer steps"), ("samples_seen", "training samples"),
                ("peak_train_mem_gb", "peak train memory (GB)"), ("infer_ms_per_frame", "inference ms / frame")]
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.0))
        for ax, (col, title) in zip(axes, cols):
            if res[col].dropna().empty:
                ax.text(0.5, 0.5, "not measured\n(CUDA only)", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED, fontsize=8)
            for j, b in enumerate(budgets):
                for i, m in enumerate(_models_present(res)):
                    v = res[(res.model == m) & (res.budget == b)][col].astype(float)
                    if v.dropna().empty:
                        continue
                    st = STYLE[m]
                    ax.bar(i + j * (len(MODEL_NAMES) + 1), v.mean(), yerr=v.std() if len(v) > 1 else None,
                           color=st["color"], width=0.85, edgecolor="white", linewidth=1.5,
                           label=st["label"] if j == 0 else None)
            ax.set_xticks([1.5 + j * (len(MODEL_NAMES) + 1) for j in range(len(budgets))], budgets)
            ax.set_title(title, fontsize=9, color=INK)
            _style_axes(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout()
        written += _save(fig, figures_dir, "fig7_resources")
        plt.close(fig)
    return written


def aggregate(tasks_path: Path, output_root: Path, tables_dir: Path, figures_dir: Path,
              labels=None, baseline_dir: Path | None = None) -> dict:
    rows, loaded, problems, expected = collect(tasks_path, output_root, labels)
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_seed = per_seed_table(loaded)
    summary = summary_table(per_seed)
    resources = resource_table(loaded)
    gpu_hours = gpu_hours_table(rows, output_root)
    per_seed.to_csv(tables_dir / "per_seed_results.csv", index=False)
    summary.to_csv(tables_dir / "main_results.csv", index=False)
    released_path = output_root / "baselines" / "released" / "released_check.json"
    paper_protocol_table(summary, read_json(released_path) if released_path.exists() else None).to_csv(
        tables_dir / "paper_protocol.csv", index=False)
    resources.to_csv(tables_dir / "resource_use.csv", index=False)
    gpu_hours.to_csv(tables_dir / "gpu_hours.csv", index=False)
    pd.DataFrame({"problem": problems}).to_csv(tables_dir / "missing_runs.csv", index=False)
    baseline = None
    if baseline_dir and (baseline_dir / "clean_na.json").exists():
        baseline = read_json(baseline_dir / "clean_na.json")
        if baseline["metric_version"] != METRIC_VERSION:
            raise RuntimeError("baseline evaluated with a different metric version")
    for p in problems:
        warnings.warn(p)
    figures = make_figures(summary, loaded, resources, figures_dir, baseline)
    totals = gpu_hours[gpu_hours.run_id == "TOTAL"].iloc[0] if not gpu_hours.empty else {}
    return {"tasks": len(rows), "expected_evaluations": expected, "found_evaluations": len(loaded),
            "train_gpu_hours": float(totals.get("train_gpu_hours", 0.0)),
            "eval_gpu_hours": float(totals.get("eval_gpu_hours", 0.0)),
            "problems": problems, "figures": figures}
