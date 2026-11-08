"""Per-model throughput, memory and latency at full resolution (Phase G feasibility).

Numbers measured on the Mac (CPU/MPS) are smoke checks only, never HPC3 estimates.
"""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rbbench.data import FieldWindows, ResumableRandomSampler, loader_settings, make_well_dataset
from rbbench.device import autocast_context, check_precision, synchronize
from rbbench.evaluate import measure_latency, model_predictor
from rbbench.manifests import atomic_torch_save, stable_int
from rbbench.models.factory import count_parameters, count_parameters_used, model_from_config
from rbbench.rollout import rollout

COLUMNS = (
    "model", "device", "gpu_name", "precision", "batch_size", "grid", "n_params", "n_params_used",
    "status", "error", "steps_timed", "step_s", "data_s", "forward_s", "backward_s", "optimizer_s",
    "samples_per_s", "budgets", "est_steps_first", "est_steps_final", "est_samples_final", "peak_alloc_gb",
    "peak_reserved_gb", "gpu_util_pct", "infer_ms_per_frame", "rollout30_ms", "rollout_finite",
    "ckpt_save_s", "ckpt_mb",
)


def _t(device):
    synchronize(device)
    return time.perf_counter()


def profile_model(cfg: dict, device: torch.device, dataset_dir: Path, batch_size: int,
                  steps: int, warmup: int, precision: str, rollout_steps: int = 30) -> dict:
    name = cfg["model"]["name"]
    row = {"model": name, "device": str(device), "precision": precision, "batch_size": batch_size,
           "grid": "x".join(map(str, cfg["task"]["spatial_resolution"])), "status": "ok", "error": ""}
    if device.type == "cuda":
        row["gpu_name"] = torch.cuda.get_device_name(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        check_precision(name, precision)
        ds = make_well_dataset(dataset_dir, "train", cfg["task"]["n_steps_input"], 1)
        windows = FieldWindows(ds, range(len(ds)))
        # A fresh random order per measurement: with one shared order, every measurement after the first
        # read the same windows from the page cache, and the 2026-11-08 profile missed the I/O cost.
        sampler = ResumableRandomSampler(len(windows), stable_int("profile", name, batch_size, precision))
        kwargs = loader_settings(cfg)
        kwargs["pin_memory"] &= device.type == "cuda"
        loader = DataLoader(windows, batch_size=batch_size, sampler=sampler, drop_last=True, **kwargs)
        model = model_from_config(cfg, seed=0)
        row["n_params"] = count_parameters(model)
        x0, _ = windows[0]
        row["n_params_used"] = count_parameters_used(model, x0[None])
        model.to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
        timings = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        it = iter(loader)
        total = 0.0
        for i in range(warmup + steps):
            t0 = _t(device)
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            t1 = _t(device)
            with autocast_context(device, precision):
                pred = model(x)
            loss = F.mse_loss(pred.float(), y)
            t2 = _t(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            t3 = _t(device)
            opt.step()
            t4 = _t(device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at profiling step {i}")
            if i >= warmup:
                for k, a, b in (("data", t0, t1), ("forward", t1, t2), ("backward", t2, t3), ("optimizer", t3, t4)):
                    timings[k] += b - a
                total += t4 - t0
        step_s = total / steps
        budgets = sorted((float(v), k) for k, v in cfg["train"]["budgets_seconds"].items())
        row.update(
            steps_timed=steps, step_s=step_s, data_s=timings["data"] / steps,
            forward_s=timings["forward"] / steps, backward_s=timings["backward"] / steps,
            optimizer_s=timings["optimizer"] / steps, samples_per_s=batch_size / step_s,
            budgets=";".join(f"{k}={v:g}s" for v, k in budgets),
            est_steps_first=int(budgets[0][0] / step_s), est_steps_final=int(budgets[-1][0] / step_s),
            est_samples_final=int(budgets[-1][0] / step_s) * batch_size,
        )
        if device.type == "cuda":
            row["peak_alloc_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            row["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
            try:
                row["gpu_util_pct"] = torch.cuda.utilization(device)
            except Exception:  # needs pynvml; optional
                row["gpu_util_pct"] = None
        predict = model_predictor(model, device, precision)
        lat = measure_latency(predict, x0[None], device, warmup=5, iters=20, rollout_steps=rollout_steps)
        row["infer_ms_per_frame"] = lat["ms_per_frame"]
        row["rollout30_ms"] = lat["ms_per_rollout"]
        preds = rollout(predict, x0[None].to(device), rollout_steps)
        row["rollout_finite"] = bool(torch.isfinite(preds).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pt"
            t = time.perf_counter()
            atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
            row["ckpt_save_s"] = time.perf_counter() - t
            row["ckpt_mb"] = path.stat().st_size / 1e6
    except torch.cuda.OutOfMemoryError as err:
        row.update(status="oom", error=str(err).splitlines()[0])
    except Exception as err:  # recorded, not hidden
        row.update(status="error", error=f"{type(err).__name__}: {err}")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def bench_loader(dataset_dir: Path, split: str, batch_size: int, batches: int, workers: int,
                 persistent: bool, seed: int, prefetch_factor: int = 4, n_steps_input: int = 4) -> dict:
    """Data-loading throughput alone (no model): random windows, as in training, fresh ones per call.

    Separates I/O from compute: training is I/O-bound when this is slower than the profiled step time.
    The first batch (worker start-up) is not timed.
    """
    ds = make_well_dataset(dataset_dir, split, n_steps_input, 1, persistent=persistent)
    windows = FieldWindows(ds, range(len(ds)))
    kwargs = {"num_workers": workers}
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(windows, batch_size=batch_size, sampler=ResumableRandomSampler(len(windows), seed),
                        drop_last=True, **kwargs)
    it = iter(loader)
    x, y = next(it)
    sample_bytes = (x[0].numel() + y[0].numel()) * x.element_size()
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    seconds = time.perf_counter() - t0
    del it
    return {"loader": "persistent" if persistent else "upstream", "workers": workers, "batch_size": batch_size,
            "batches": batches, "s_per_batch": seconds / batches,
            "samples_per_s": batches * batch_size / seconds,
            "mb_per_s": batches * batch_size * sample_bytes / seconds / 1e6}
