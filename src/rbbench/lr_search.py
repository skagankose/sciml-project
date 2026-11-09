"""Equal-budget learning-rate search: results, each model's rate, and the pilot timings (decisions D7, D24).

Reads only the search runs' training artifacts (COMPLETE.json, FAILED.json, provenance.json,
train_metrics.csv): the validation VRMSE at the search budget, the study's metric and the paper's selection
criterion (owner, 2026-11-08). Nothing here reads test data.

The same runs are the real-data pilots: GPU seconds per optimizer step and the time spent waiting for data,
which give the steps and epochs each model gets in the study's budgets (abstract milestone "budgets fixed
from pilot timings"). Step times come from every run that has trained, finished or not.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from rbbench.config import load_config
from rbbench.data import FIELDS
from rbbench.manifests import read_json
from rbbench.subsets import load_subset_ids, subset_filename

DONE = ("complete", "diverged")


def _last_attempt_status(prov: dict) -> str:
    attempts = prov.get("attempts") or []
    end = attempts[-1].get("end_status") if attempts else None
    if end == "preempted":
        return "preempted"
    if end == "error":
        return "stopped by an error"
    return "running, or stopped by an error"


def _train_log(metrics: Path) -> tuple[dict | None, float | None]:
    """The last logged training row, and the median data wait per step between log rows.

    Waits come mostly from storage stalls that last minutes (D24); the median over logging intervals is the
    wait between stalls.
    """
    if not metrics.exists():
        return None, None
    rows = {}
    with open(metrics, newline="") as f:
        for r in csv.DictReader(f):
            if r["event"] == "train":
                rows[int(r["step"])] = r  # a resumed run repeats steps: keep the later attempt
    ordered = [rows[k] for k in sorted(rows)]
    per_step = sorted(
        (float(b["data_wait_seconds"] or 0) - float(a["data_wait_seconds"] or 0)) / (int(b["step"]) - int(a["step"]))
        for a, b in zip(ordered, ordered[1:])
    )
    median = per_step[len(per_step) // 2] if per_step else None
    return (ordered[-1] if ordered else None), median


def run_results(rows: list[dict], output_root: Path) -> list[dict]:
    """One record per search row: status, validation at the budget, progress and timing."""
    out = []
    for row in rows:
        run = Path(output_root) / "runs" / row["run_id"]
        rec = {"task_index": row["task_index"], "model": row["model"], "lr": row["lr"], "run_id": row["run_id"]}
        prov = read_json(run / "provenance.json") if (run / "provenance.json").exists() else {}
        rec["wall_hours"] = sum(a.get("wall_seconds") or 0.0 for a in prov.get("attempts", [])) / 3600
        rec["attempts"] = len(prov.get("attempts", []))
        rec["n_train_windows"] = prov.get("n_train_windows")
        rec["n_train_trajectories"] = prov.get("n_train_trajectories")
        progress = None  # (step, train_seconds, data_wait_seconds, samples_seen)
        last, rec["median_wait_s_per_step"] = _train_log(run / "train_metrics.csv")
        if (run / "COMPLETE.json").exists():
            done = read_json(run / "COMPLETE.json")
            ms = max(done["milestones"].values(), key=lambda m: m["target_seconds"])
            val = ms["validation"]
            rec.update(status="complete", val_loss=val["val_loss"], val_vrmse=val["val_vrmse"],
                       **{f"val_vrmse_{f}": val.get(f"val_vrmse_{f}") for f in FIELDS})
            progress = (ms["step"], ms["train_seconds"], ms.get("data_wait_seconds") or 0.0, ms["samples_seen"])
        else:
            if (run / "FAILED.json").exists():
                rec.update(status="diverged", val_loss=math.inf, val_vrmse=math.inf)
            elif prov:
                rec["status"] = _last_attempt_status(prov)
            else:
                rec["status"] = "stopped before training" if run.exists() else "not started"
            if last is not None:
                progress = (int(last["step"]), float(last["train_seconds"]),
                            float(last["data_wait_seconds"] or 0.0), int(last["samples_seen"]))
        if progress is not None and progress[0] > 0:
            steps, gpu_s, wait_s, samples = progress
            rec.update(steps=steps, gpu_minutes=gpu_s / 60, data_wait_minutes=wait_s / 60,
                       batch_size=samples // steps, gpu_s_per_step=gpu_s / steps, wait_s_per_step=wait_s / steps)
            if rec["n_train_windows"]:
                rec["epochs"] = samples / rec["n_train_windows"]
        out.append(rec)
    return out


def training_curves(rows: list[dict], output_root: Path) -> dict[int, list[tuple[float, float]]]:
    """Per task index: (GPU training seconds, training loss) at every logged step.

    A step logged twice (a resumed run repeats the steps after its last saved state) keeps the later
    attempt's value.
    """
    curves = {}
    for row in rows:
        path = Path(output_root) / "runs" / row["run_id"] / "train_metrics.csv"
        if not path.exists():
            continue
        by_step = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["event"] == "train" and r["loss"]:
                    by_step[int(r["step"])] = (float(r["train_seconds"]), float(r["loss"]))
        if by_step:
            curves[row["task_index"]] = [by_step[s] for s in sorted(by_step)]
    return curves


def choose(results: list[dict]) -> dict:
    """Per model: the rate with the lowest validation VRMSE, once every rate of that model has finished.

    A diverged rate counts as infinitely bad. `at_grid_edge` flags a choice at the smallest or largest rate
    tried, where a wider grid might have done better.
    """
    chosen = {}
    for model in dict.fromkeys(r["model"] for r in results):
        mine = [r for r in results if r["model"] == model]
        pending = [r["lr"] for r in mine if r["status"] not in DONE]
        if pending:
            chosen[model] = {"lr": None, "pending_lrs": pending}
            continue
        finite = [r for r in mine if math.isfinite(r["val_vrmse"])]
        if not finite:
            chosen[model] = {"lr": None, "pending_lrs": [], "note": "every rate diverged"}
            continue
        best = min(finite, key=lambda r: r["val_vrmse"])
        grid = sorted(r["lr"] for r in mine)
        chosen[model] = {"lr": best["lr"], "val_vrmse": best["val_vrmse"], "val_loss": best["val_loss"],
                         "at_grid_edge": best["lr"] in (grid[0], grid[-1]), "pending_lrs": []}
    return chosen


def best_so_far(results: list[dict]) -> dict:
    """Per model: the finished rate with the lowest validation VRMSE (None if none has finished)."""
    best = {}
    for r in results:
        if r["status"] == "complete" and (r["model"] not in best or r["val_vrmse"] < best[r["model"]]["val_vrmse"]):
            best[r["model"]] = r
    return {m: r["lr"] for m, r in best.items()}


def pilot_timings(results: list[dict], tasks_dir: Path, experiment: str = "main") -> list[dict]:
    """Per model: measured step times and what the study's budgets buy, at 10% and 100% data.

    GPU time per step does not depend on the rate, so every run of a model that has trained is averaged,
    finished or not. The data wait is measured on the 10% subset; the 100% estimate assumes the same wait
    per step.
    """
    out = []
    for model in dict.fromkeys(r["model"] for r in results):
        timed = [r for r in results if r["model"] == model and r.get("gpu_s_per_step") and r.get("n_train_windows")]
        if not timed:
            continue
        gpu = sum(r["gpu_s_per_step"] for r in timed) / len(timed)
        wait = sum(r["wait_s_per_step"] for r in timed) / len(timed)
        medians = [r["median_wait_s_per_step"] for r in timed if r.get("median_wait_s_per_step") is not None]
        steady = sum(medians) / len(medians) if medians else wait
        batch = timed[0]["batch_size"]
        per_traj = timed[0]["n_train_windows"] / timed[0]["n_train_trajectories"]
        rec = {"model": model, "runs": len(timed), "runs_complete": sum(r["status"] == "complete" for r in timed),
               "batch_size": batch, "gpu_s_per_step": gpu, "wait_s_per_step": wait, "gpu_samples_per_s": batch / gpu,
               "wait_fraction_of_wall": wait / (gpu + wait), "median_wait_s_per_step": steady}
        cfg = load_config(model, experiment)
        for label, seconds in cfg["train"]["budgets_seconds"].items():
            steps = float(seconds) / gpu
            rec[f"steps_{label}"] = steps
            rec[f"est_wall_hours_{label}"] = float(seconds) * (gpu + wait) / gpu / 3600
            rec[f"est_wall_hours_{label}_no_stalls"] = float(seconds) * (gpu + steady) / gpu / 3600
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


# --- report: table and figures ------------------------------------------------------------

def _lr(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-")


def markdown_table(results: list[dict], chosen: dict) -> str:
    """Every search run: validation VRMSE (field mean and per field), loss, progress and timing."""
    from rbbench.aggregate import STYLE

    best = best_so_far(results)
    head = (["Model", "Learning rate", "Status", "Val VRMSE"] + [f"VRMSE {f}" for f in FIELDS]
            + ["Val loss", "Steps", "Epochs", "GPU min", "GPU s/step", "Data wait (min)", "Wall (h)"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def num(v, fmt):
        return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)

    for r in results:
        status = r["status"]
        mark = ""
        if status == "complete" and best.get(r["model"]) == r["lr"]:
            mark = " (chosen)" if chosen.get(r["model"], {}).get("lr") == r["lr"] else " (best so far)"
        vr = num(r.get("val_vrmse"), ".3f")
        lines.append("| " + " | ".join([
            STYLE[r["model"]]["label"], _lr(r["lr"]), status + mark, f"**{vr}**" if mark else vr,
            *[num(r.get(f"val_vrmse_{f}"), ".3f") for f in FIELDS], num(r.get("val_loss"), ".4f"),
            num(r.get("steps"), "d"), num(r.get("epochs"), ".1f"), num(r.get("gpu_minutes"), ".1f"),
            num(r.get("gpu_s_per_step"), ".3f"), num(r.get("data_wait_minutes"), ".1f"),
            num(r.get("wall_hours") or None, ".2f"),
        ]) + " |")
    return "\n".join(lines)


# Ordinal blue ramp (reference sequential hue, steps 250 -> 700): lighter = smaller rate.
_LR_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
_LR_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _ramp(n: int) -> list[str]:
    if n == 1:
        return [_LR_RAMP[2]]
    return [_LR_RAMP[round(i * (len(_LR_RAMP) - 1) / (n - 1))] for i in range(n)]


def _spread(values: list[float], gap: float) -> list[float]:
    """Positions for end labels: same order as `values`, at least `gap` apart (in the same units)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [0.0] * len(values)
    prev = -math.inf
    for i in order:
        pos[i] = max(values[i], prev + gap)
        prev = pos[i]
    return pos


def plot_lr_search(results: list[dict], curves: dict, chosen: dict, out_dir: Path,
                   budget_minutes: float = 20.0) -> list[str]:
    """Two figures: training loss against GPU time per rate (one panel per model), and validation VRMSE
    against the rate (all models)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rbbench.aggregate import INK, MUTED, STYLE, _save, _style_axes

    written: list[str] = []
    models = list(dict.fromkeys(r["model"] for r in results))

    # 1. training loss vs GPU minutes, one line per rate
    fig, axes = plt.subplots(1, len(models), figsize=(3.7 * len(models), 3.1), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        mine = sorted((r for r in results if r["model"] == model), key=lambda r: r["lr"])
        colors = _ramp(len(mine))
        ends = []
        for k, (r, color) in enumerate(zip(mine, colors)):
            pts = curves.get(r["task_index"])
            note = {"complete": "", "diverged": ", diverged"}.get(r["status"], ", not finished")
            if not pts:
                ax.plot([], [], color=color, lw=1.6, marker=_LR_MARKERS[k], ms=4,
                        label=f"{_lr(r['lr'])} (no data yet)")
                continue
            x = [s / 60 for s, _ in pts]
            y = [v for _, v in pts]
            ax.plot(x, y, color=color, lw=1.6, marker=_LR_MARKERS[k], ms=4, markevery=max(1, len(x) // 8),
                    label=f"{_lr(r['lr'])}{note}")
            if r["status"] == "diverged":
                ax.plot(x[-1], y[-1], marker="x", color=INK, ms=7, mew=1.6)
            if r["status"] == "complete":  # unfinished lines end mid-panel: the legend names them
                ends.append((x[-1], y[-1], _lr(r["lr"])))
        ax.set_yscale("log")
        ax.set_xlim(0, budget_minutes * 1.18)
        ax.set_title(STYLE[model]["label"], fontsize=9, color=INK)
        ax.set_xlabel("GPU training time (min)")
        _style_axes(ax)
        if ends:
            logs = [math.log10(e[1]) for e in ends]
            lo, hi = ax.get_ylim()
            ys = _spread(logs, 0.06 * (math.log10(hi) - math.log10(lo)))
            for (xe, _, text), yl in zip(ends, ys):
                ax.annotate(text, (xe, 10 ** yl), xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=MUTED, va="center")
    axes[0][0].set_ylabel("training loss (MSE, normalised)")
    axes[0][0].legend(fontsize=7, frameon=False, title="learning rate", title_fontsize=7)
    for ax in axes[0][1:]:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    written += _save(fig, out_dir, "lr_search_training_loss")
    plt.close(fig)

    # 2. validation VRMSE at the budget vs the rate
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    grid = sorted({r["lr"] for r in results})
    finite = [r["val_vrmse"] for r in results if r["status"] == "complete"]
    top = max(finite) * 1.12 if finite else 1.0
    ends = []
    for model in models:
        st = STYLE[model]
        done = sorted((r for r in results if r["model"] == model and r["status"] == "complete"), key=lambda r: r["lr"])
        div = [r for r in results if r["model"] == model and r["status"] == "diverged"]
        if done:
            ax.plot([r["lr"] for r in done], [r["val_vrmse"] for r in done], color=st["color"], marker=st["marker"],
                    ms=6, lw=1.6, label=st["label"])
            pick = chosen.get(model, {}).get("lr") or best_so_far(results).get(model)
            for r in done:
                if r["lr"] == pick:
                    ax.plot(r["lr"], r["val_vrmse"], marker="o", ms=13, mfc="none", mec=INK, mew=1.2)
            ends.append((done[-1]["lr"], done[-1]["val_vrmse"], st["label"]))
        else:
            ax.plot([], [], color=st["color"], marker=st["marker"], ms=6, lw=1.6, label=f"{st['label']} (pending)")
        for r in div:
            ax.plot(r["lr"], top, marker="x", color=st["color"], ms=8, mew=2)
            ax.annotate("diverged", (r["lr"], top), xytext=(0, 6), textcoords="offset points", fontsize=7,
                        color=MUTED, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(grid, [_lr(g) for g in grid])
    ax.minorticks_off()
    ax.set_xlim(grid[0] / 1.6, grid[-1] * 2.4)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("validation VRMSE (field mean)")
    ax.set_title(f"After {budget_minutes:g} min of GPU training, 10% data, seed 0", fontsize=9, color=INK)
    _style_axes(ax)
    if ends:
        lo, hi = ax.get_ylim()
        ys = _spread([e[1] for e in ends], 0.06 * (hi - lo))
        for (xe, _, text), y in zip(ends, ys):
            ax.annotate(text, (xe, y), xytext=(9, 0), textcoords="offset points", fontsize=7, color=MUTED,
                        va="center")
    ax.plot([], [], marker="o", ms=10, mfc="none", mec=INK, ls="none", label="chosen (or best so far)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    written += _save(fig, out_dir, "lr_search_val_vrmse")
    plt.close(fig)
    return written
