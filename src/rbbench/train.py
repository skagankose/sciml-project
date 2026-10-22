"""Budgeted one-step training shared by all four models.

Budget accounting: `train_seconds` sums, per optimizer step, the wall time from requesting
the batch to the end of the (synchronised) optimizer step. Startup, validation and
checkpoint writes fall outside these intervals and are logged separately. A milestone
checkpoint (e.g. 2h, 6h) is written after the first optimizer step whose cumulative
`train_seconds` reaches the milestone; the actual value and the overshoot are recorded.
The timer, optimizer, scheduler, RNG and sampler position all persist across resumes.
"""

from __future__ import annotations

import csv
import math
import os
import random
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rbbench.config import config_hash, dump_config, yaml_load
from rbbench.data import (
    FIELDS,
    FieldWindows,
    ResumableRandomSampler,
    evenly_spaced,
    list_trajectories,
    make_well_dataset,
    window_indices,
)
from rbbench.device import autocast_context, synchronize
from rbbench.manifests import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    now_utc,
    provenance,
    read_json,
    sha256_file,
)
from rbbench.metrics.well_metric import MeanAccumulator, vrmse
from rbbench.models.factory import count_parameters, model_from_config

CHECKPOINT_FORMAT = 1
METRIC_COLUMNS = (
    "event", "attempt", "step", "epoch", "train_seconds", "wall_seconds", "lr", "loss",
    "samples_seen", "val_loss", "val_vrmse", "gpu_mem_alloc_gb", "gpu_mem_reserved_gb", "note",
)


class TrainingDiverged(RuntimeError):
    pass


@dataclass
class RunPaths:
    root: Path

    def checkpoint(self, label: str) -> Path:
        return self.root / f"checkpoint_{label}.pt"

    @property
    def last_resume(self) -> Path:
        return self.root / "last_resume.pt"

    @property
    def complete(self) -> Path:
        return self.root / "COMPLETE.json"

    @property
    def failed(self) -> Path:
        return self.root / "FAILED.json"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def metrics(self) -> Path:
        return self.root / "train_metrics.csv"

    @property
    def config(self) -> Path:
        return self.root / "resolved_config.yaml"


class BudgetCosineLR:
    """Linear warmup over steps, then cosine decay over cumulative training seconds.

    The rate is a pure function of (step, train_seconds), so resuming is exact. With a
    wall-clock budget the total step count is unknown in advance; decaying on time makes
    the 6h checkpoint the end of the schedule and the 2h checkpoint an intermediate state.
    """

    def __init__(self, base_lr: float, warmup_steps: int, total_seconds: float,
                 min_lr_ratio: float = 0.0, name: str = "time_cosine"):
        if name not in ("time_cosine", "constant"):
            raise ValueError(f"unknown schedule {name!r}")
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_seconds = total_seconds
        self.min_lr_ratio = min_lr_ratio
        self.name = name

    def lr(self, step: int, train_seconds: float) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.name == "constant":
            return self.base_lr
        progress = min(train_seconds / self.total_seconds, 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos)

    def state_dict(self) -> dict:
        return dict(vars(self))


class PreemptionSignal:
    """Sets a flag on SIGUSR1/SIGTERM so the loop can checkpoint and exit cleanly."""

    SIGNALS = (signal.SIGUSR1, signal.SIGTERM)

    def __init__(self):
        self.received: int | None = None
        self._previous = {}

    def __enter__(self):
        for sig in self.SIGNALS:
            self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        self.received = signum

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _gpu_mem(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        round(torch.cuda.max_memory_reserved(device) / 1e9, 3),
    )


@torch.no_grad()
def evaluate_windows(model, well_ds, indices, device, precision: str, batch_size: int) -> dict:
    """One-step loss (normalised MSE) and VRMSE (physical units) on fixed windows."""
    model.eval()
    acc = MeanAccumulator()
    loader = DataLoader(FieldWindows(well_ds, indices), batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_context(device, precision):
            pred = model(x)
        pred = pred.float()
        acc.add("loss", ((pred - y) ** 2).mean(dim=(1, 2, 3)))
        pred_phys = well_ds.norm.denormalize_flattened(pred, "variable")
        true_phys = well_ds.norm.denormalize_flattened(y, "variable")
        acc.add("vrmse", vrmse(pred_phys, true_phys))
    model.train()
    per_field = acc.mean("vrmse")
    out = {"val_loss": float(acc.mean("loss")), "val_vrmse": float(per_field.mean()), "n": len(indices)}
    out.update({f"val_vrmse_{f}": float(v) for f, v in zip(FIELDS, per_field)})
    return out


class MetricsLog:
    def __init__(self, path: Path, attempt: int):
        self.path = path
        self.attempt = attempt
        self.t0 = time.time()
        new = not path.exists()
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=METRIC_COLUMNS)
        if new:
            self._w.writeheader()

    def write(self, **row) -> None:
        row.setdefault("attempt", self.attempt)
        row.setdefault("wall_seconds", round(time.time() - self.t0, 3))
        self._w.writerow({k: row.get(k) for k in METRIC_COLUMNS})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _select_train_indices(ds, dataset_dir: Path, subset_ids: list[str] | None) -> tuple[list[int], int]:
    refs = list_trajectories(dataset_dir, "train")
    if subset_ids is not None:
        by_id = {r.id: r for r in refs}
        missing = [i for i in subset_ids if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} subset ids not in the train split, e.g. {missing[:3]}")
        refs = [by_id[i] for i in subset_ids]
    return window_indices(ds, refs, dataset_dir), len(refs)


def verify_checkpoint(path: Path, model: torch.nn.Module) -> dict:
    """Load a checkpoint and check every tensor matches the model's keys and shapes."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(v.shape) for k, v in state["model"].items()}
    if expected != got:
        raise RuntimeError(f"{path}: state dict keys/shapes differ from the model")
    finite = all(torch.isfinite(v).all() for v in state["model"].values() if v.is_floating_point())
    if not finite:
        raise RuntimeError(f"{path}: non-finite weights")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "step": state["step"],
        "train_seconds": state["train_seconds"],
        "milestone": state.get("milestone"),
    }


def train_run(
    cfg: dict,
    run_dir: str | os.PathLike,
    dataset_dir: str | os.PathLike,
    seed: int,
    device: torch.device,
    subset_ids: list[str] | None = None,
    run_meta: dict | None = None,
    resume: bool = True,
    skip_if_complete: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    stop_after_steps: int | None = None,
    step_callback: Callable[[int], None] | None = None,
) -> dict:
    """Train one (model, subset, seed) run. Returns {"status": ...}.

    Statuses: "complete", "already_complete", "preempted" (state saved; rerun to resume).
    `stop_after_steps` and `step_callback` are test hooks that simulate preemption.
    """
    paths = RunPaths(Path(run_dir))
    paths.root.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_dir)
    run_meta = dict(run_meta or {})
    cfg_sha = config_hash(cfg)
    if paths.complete.exists() and skip_if_complete:
        return {"status": "already_complete", "run_dir": str(paths.root)}

    if paths.config.exists():
        previous = yaml_load(paths.config.read_text())
        if config_hash(previous) != cfg_sha:
            raise RuntimeError(f"{paths.config} holds a different config; refusing to mix runs")
    else:
        atomic_write_text(paths.config, dump_config(cfg))

    tcfg, ocfg, scfg = cfg["train"], cfg["optim"], cfg["schedule"]
    budgets = sorted(((float(v), k) for k, v in tcfg["budgets_seconds"].items()))
    final_seconds = budgets[-1][0]
    precision = tcfg["precision"]
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not tcfg.get("deterministic", False)

    # --- data
    n_in = cfg["task"]["n_steps_input"]
    train_ds = make_well_dataset(dataset_dir, "train", n_in, 1)
    if tuple(train_ds.size_tuple) != tuple(cfg["task"]["spatial_resolution"]):
        raise RuntimeError(f"data grid {train_ds.size_tuple} != config {cfg['task']['spatial_resolution']}")
    train_idx, n_traj = _select_train_indices(train_ds, dataset_dir, subset_ids)
    windows = FieldWindows(train_ds, train_idx)
    val_ds = make_well_dataset(dataset_dir, "valid", n_in, 1)
    val_idx = evenly_spaced(len(val_ds), tcfg["val_windows"])
    batch_size = tcfg["batch_size"]
    if len(windows) < batch_size:
        raise ValueError(f"{len(windows)} training windows < batch size {batch_size}")

    # --- model and optimizer (random init from `seed`)
    set_seed(seed)
    model = model_from_config(cfg, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ocfg["lr"], betas=tuple(ocfg["betas"]), weight_decay=ocfg["weight_decay"]
    )
    sched = BudgetCosineLR(ocfg["lr"], scfg["warmup_steps"], final_seconds, scfg["min_lr_ratio"], scfg["name"])
    sampler = ResumableRandomSampler(len(windows), seed)

    state = {
        "step": 0, "train_seconds": 0.0, "samples_seen": 0, "epoch": 0, "position": 0,
        "milestones": {}, "val_history": [], "attempt": 0, "overhead_seconds": {"val": 0.0, "ckpt": 0.0},
    }
    resumed_from = None
    if resume and paths.last_resume.exists():
        ckpt = torch.load(paths.last_resume, map_location="cpu", weights_only=False)
        if ckpt["config_sha256"] != cfg_sha or ckpt["seed"] != seed:
            raise RuntimeError("last_resume.pt belongs to a different config or seed")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for k in state:
            state[k] = ckpt["state"][k]
        load_rng_state(ckpt["rng"])
        resumed_from = state["step"]
    state["attempt"] += 1
    sampler.epoch, sampler.start = state["epoch"], state["position"]

    # --- provenance (one record per attempt)
    prov = read_json(paths.provenance) if paths.provenance.exists() else {"attempts": []}
    attempt_record = provenance(
        {"attempt": state["attempt"], "resumed_from_step": resumed_from, "device": str(device)}
    )
    if prov["attempts"]:
        first = prov["attempts"][0]["git"]["commit"]
        if first != attempt_record["git"]["commit"]:
            attempt_record["warning"] = f"git commit changed since attempt 1 ({first})"
    prov.update(
        run=run_meta, config_sha256=cfg_sha, seed=seed, model=cfg["model"]["name"],
        n_parameters=count_parameters(model), n_train_trajectories=n_traj,
        n_train_windows=len(windows), n_val_windows=len(val_idx), budgets_seconds=dict(tcfg["budgets_seconds"]),
    )
    prov["attempts"].append(attempt_record)
    atomic_write_json(paths.provenance, prov)

    log = MetricsLog(paths.metrics, state["attempt"])
    log.write(event="start" if resumed_from is None else "resume", step=state["step"],
              epoch=state["epoch"], train_seconds=state["train_seconds"], samples_seen=state["samples_seen"])

    def full_state(milestone: str | None = None) -> dict:
        return {
            "format": CHECKPOINT_FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": sched.state_dict(), "scaler": {"enabled": False}, "rng": rng_state(),
            "state": {k: v for k, v in state.items()}, "step": state["step"],
            "train_seconds": state["train_seconds"], "milestone": milestone, "seed": seed,
            "config_sha256": cfg_sha, "model_name": cfg["model"]["name"], "saved_utc": now_utc(),
        }

    def save(path: Path, milestone: str | None = None) -> None:
        t = time.perf_counter()
        atomic_torch_save(full_state(milestone), path)
        state["overhead_seconds"]["ckpt"] += time.perf_counter() - t

    def validate(event: str) -> dict:
        t = time.perf_counter()
        res = evaluate_windows(model, val_ds, val_idx, device, precision, batch_size)
        state["overhead_seconds"]["val"] += time.perf_counter() - t
        res.update(step=state["step"], train_seconds=state["train_seconds"], event=event)
        state["val_history"].append(res)
        alloc, reserved = _gpu_mem(device)
        log.write(event=event, step=state["step"], epoch=state["epoch"], train_seconds=state["train_seconds"],
                  val_loss=res["val_loss"], val_vrmse=res["val_vrmse"], samples_seen=state["samples_seen"],
                  gpu_mem_alloc_gb=alloc, gpu_mem_reserved_gb=reserved)
        return res

    def make_loader():
        return DataLoader(
            windows, batch_size=batch_size, sampler=sampler, drop_last=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
            persistent_workers=False,
        )

    max_steps = tcfg.get("max_steps")
    last_resume_wall = time.time()
    last_val_seconds = state["val_history"][-1]["train_seconds"] if state["val_history"] else 0.0
    loss_value = float("nan")
    model.train()
    try:
        with PreemptionSignal() as preempt:
            it = iter(make_loader())
            while len(state["milestones"]) < len(budgets):
                t0 = clock()
                try:
                    x, y = next(it)
                except StopIteration:
                    state["epoch"] += 1
                    state["position"] = 0
                    sampler.epoch, sampler.start = state["epoch"], 0
                    it = iter(make_loader())
                    continue
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lr = sched.lr(state["step"], state["train_seconds"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with autocast_context(device, precision):
                    pred = model(x)
                loss = F.mse_loss(pred.float(), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if tcfg.get("grad_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
                optimizer.step()
                synchronize(device)
                state["train_seconds"] += clock() - t0
                state["step"] += 1
                state["position"] += batch_size
                state["samples_seen"] += batch_size
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise TrainingDiverged(f"non-finite loss {loss_value} at step {state['step']}")

                if state["step"] % tcfg["log_every_steps"] == 0:
                    alloc, reserved = _gpu_mem(device)
                    log.write(event="train", step=state["step"], epoch=state["epoch"],
                              train_seconds=round(state["train_seconds"], 4), lr=lr, loss=loss_value,
                              samples_seen=state["samples_seen"], gpu_mem_alloc_gb=alloc,
                              gpu_mem_reserved_gb=reserved)

                forced = max_steps is not None and state["step"] >= max_steps
                for seconds, label in budgets:
                    if label in state["milestones"]:
                        continue
                    if state["train_seconds"] >= seconds or forced:
                        res = validate(f"milestone_{label}")
                        alloc, reserved = _gpu_mem(device)
                        state["milestones"][label] = {
                            "target_seconds": seconds, "train_seconds": state["train_seconds"],
                            "overshoot_seconds": state["train_seconds"] - seconds, "step": state["step"],
                            "samples_seen": state["samples_seen"], "forced_by_max_steps": forced and state["train_seconds"] < seconds,
                            "validation": res, "peak_mem_alloc_gb": alloc, "peak_mem_reserved_gb": reserved,
                        }
                        save(paths.checkpoint(label), milestone=label)
                        save(paths.last_resume)
                        last_resume_wall = time.time()
                if len(state["milestones"]) == len(budgets):
                    break

                if state["train_seconds"] - last_val_seconds >= tcfg["val_every_seconds"]:
                    validate("val")
                    last_val_seconds = state["train_seconds"]
                if step_callback is not None:
                    step_callback(state["step"])
                stop_now = preempt.received is not None or (
                    stop_after_steps is not None and state["step"] >= stop_after_steps
                )
                if stop_now or time.time() - last_resume_wall >= tcfg["resume_every_seconds"]:
                    save(paths.last_resume)
                    last_resume_wall = time.time()
                if stop_now:
                    note = f"signal {preempt.received}" if preempt.received else "stop_after_steps"
                    log.write(event="preempted", step=state["step"], train_seconds=state["train_seconds"], note=note)
                    return {"status": "preempted", "step": state["step"], "train_seconds": state["train_seconds"],
                            "run_dir": str(paths.root), "reason": note}
    except TrainingDiverged as err:
        atomic_write_json(paths.failed, {"reason": str(err), "state": {k: state[k] for k in ("step", "train_seconds")},
                                         "time_utc": now_utc()})
        log.write(event="failed", step=state["step"], train_seconds=state["train_seconds"], note=str(err))
        raise
    finally:
        log.close()

    return finalize_run(paths, model, state, cfg_sha, seed, run_meta)


def finalize_run(paths: RunPaths, model, state: dict, cfg_sha: str, seed: int, run_meta: dict) -> dict:
    """Write COMPLETE.json only after every milestone checkpoint verifies."""
    checkpoints = {label: verify_checkpoint(paths.checkpoint(label), model) for label in state["milestones"]}
    budget_satisfied = all(not m["forced_by_max_steps"] for m in state["milestones"].values())
    record = {
        "training_completed": True,
        "evaluation_completed": False,
        "budget_satisfied": budget_satisfied,
        "run": run_meta,
        "seed": seed,
        "config_sha256": cfg_sha,
        "milestones": state["milestones"],
        "checkpoints": checkpoints,
        "final_step": state["step"],
        "train_seconds": state["train_seconds"],
        "samples_seen": state["samples_seen"],
        "attempts": state["attempt"],
        "overhead_seconds": state["overhead_seconds"],
        "completed_utc": now_utc(),
    }
    atomic_write_json(paths.complete, record)
    return {"status": "complete", "run_dir": str(paths.root), **{k: record[k] for k in ("final_step", "train_seconds")}}
