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

from rbbench.data import FieldWindows, ResumableRandomSampler, make_well_dataset
from rbbench.device import autocast_context, synchronize
from rbbench.evaluate import measure_latency, model_predictor
from rbbench.manifests import atomic_torch_save
from rbbench.models.factory import count_parameters, count_parameters_used, model_from_config
from rbbench.rollout import rollout

COLUMNS = (
    "model", "device", "gpu_name", "precision", "batch_size", "grid", "n_params", "n_params_used",
    "status", "error", "steps_timed", "step_s", "data_s", "forward_s", "backward_s", "optimizer_s",
    "samples_per_s", "est_steps_2h", "est_steps_6h", "est_samples_6h", "peak_alloc_gb",
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
        ds = make_well_dataset(dataset_dir, "train", cfg["task"]["n_steps_input"], 1)
        windows = FieldWindows(ds, range(len(ds)))
        loader = DataLoader(windows, batch_size=batch_size, sampler=ResumableRandomSampler(len(windows), 0),
                            drop_last=True, num_workers=cfg["data"]["num_workers"],
                            pin_memory=device.type == "cuda")
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
        row.update(
            steps_timed=steps, step_s=step_s, data_s=timings["data"] / steps,
            forward_s=timings["forward"] / steps, backward_s=timings["backward"] / steps,
            optimizer_s=timings["optimizer"] / steps, samples_per_s=batch_size / step_s,
            est_steps_2h=int(7200 / step_s), est_steps_6h=int(21600 / step_s),
            est_samples_6h=int(21600 / step_s) * batch_size,
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
