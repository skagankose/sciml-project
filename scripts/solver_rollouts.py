#!/usr/bin/env python
"""Numerical-solver baseline: Dedalus rollouts from the frozen cohort's start frames (runs in the `rb-dedalus` env).

For one (Ra, Pr) regime per task, the first cohort trajectory of that regime, and each cohort start s, the solver is
restarted from frame s + 3 (the last frame the models see) at the chosen resolution and integrated over the same 30
output intervals the models predict. Each rollout is written to
    <out>/<resolution>/<file>__<index>__s<start>.h5
with the predicted frames at the solver's resolution (float32; buoyancy, pressure, velocity_x, velocity_y), the
wall time of building the solver and of time stepping, iterations and CFL steps. A finished file is skipped, so a
preempted task resumes. `rbbench eval-solver` (in `well-rb`) scores the files with the models' metric code.

    python scripts/solver_rollouts.py --selftest
    python scripts/solver_rollouts.py --task-index 0 --resolution 256x64
    python scripts/solver_rollouts.py --list              # the regimes, in task order
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rbbench.solver_grid import (  # the repo's src/; this env does not install the package
    NATIVE,
    RESOLUTIONS,
    rollout_frames,
    rollout_name,
    solver_cohort,
)

N_IN = 4


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def read_frames(path: Path, index: int, frames: list[int]) -> np.ndarray:
    """[len(frames), X, Y, 4] in physical units: buoyancy, pressure, velocity_x, velocity_y."""
    with h5py.File(path, "r") as h:
        b = h["t0_fields/buoyancy"][index, frames]
        p = h["t0_fields/pressure"][index, frames]
        u = h["t1_fields/velocity"][index, frames]
    return np.stack([b, p, u[..., 0], u[..., 1]], axis=-1).astype(np.float64)


def read_times(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    """The stored frame times of one trajectory, and the stored x coordinates."""
    with h5py.File(path, "r") as h:
        t = np.asarray(h["dimensions/time"][...], np.float64)
        if h["dimensions/time"].attrs.get("sample_varying", False):
            t = t[index]
        x = np.asarray(h["dimensions/x"][...], np.float64)
    return t, x


def run_rollout(entry: dict, start: int, shape: tuple[int, int], dataset_dir: Path, out_path: Path,
                horizon: int, max_seconds: float | None) -> dict:
    from rbbench.solver_dedalus import RBSolver
    from rbbench.solver_grid import resample, x_grid

    relpath, index = entry["id"].split("#")
    index = int(index)
    path = dataset_dir / relpath
    init, targets = rollout_frames(start, N_IN, horizon)
    times, x_stored = read_times(path, index)
    gaps = np.diff(times[[init, *targets]])
    state = read_frames(path, index, [init])[0]                           # [X, Y, 4] at 512 x 128
    if state.shape[:2] != NATIVE:
        raise RuntimeError(f"{path}: frames are {state.shape[:2]}, expected {NATIVE}")
    coarse = resample(np.moveaxis(state, -1, 0), shape)                   # [4, x, y]
    solver = RBSolver(entry["rayleigh"], entry["prandtl"], *shape)
    res = solver.rollout(coarse[0], coarse[2], coarse[3], output_dt=list(gaps), n_outputs=horizon,
                         max_seconds=max_seconds)
    attrs = {
        "traj_id": entry["id"], "start": start, "init_frame": init, "target_frames": targets,
        "rayleigh": entry["rayleigh"], "prandtl": entry["prandtl"], "resolution": f"{shape[0]}x{shape[1]}",
        "nx": shape[0], "ny": shape[1], "status": res.status, "steps_done": res.steps_done,
        "build_seconds": res.build_seconds, "step_seconds": res.step_seconds,
        "seconds_per_output": res.seconds_per_output, "iterations_per_output": res.iterations_per_output,
        "iterations": int(sum(res.iterations_per_output)), "dt_min": res.dt_min, "dt_max": res.dt_max,
        "initial_dt": res.initial_dt, "output_gaps": list(gaps),
        "stored_x_is_uniform_grid": bool(np.abs(x_stored - x_grid(NATIVE[0])).max() < 1e-6),
        "options": json.dumps(solver.opts), "cores": 1, "cpu": cpu_model(), "hostname": socket.gethostname(),
        "slurm_job": f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', ''))}_"
                     f"{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
        "dedalus_version": _dedalus_version(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("pred", data=res.frames.astype(np.float32), compression="gzip", compression_opts=1)
        for k, v in attrs.items():
            h.attrs[k] = v
        h.attrs["complete"] = True
    os.replace(tmp, out_path)
    return attrs


def _dedalus_version() -> str:
    try:
        from importlib.metadata import version

        return version("dedalus")
    except Exception:  # noqa: BLE001
        return "unknown"


def selftest() -> int:
    """On HPC3, without data: grids, restart reproducibility, float32 storage and a resolution change."""
    from rbbench.solver_dedalus import RBSolver, standard_initial_condition
    from rbbench.solver_grid import resample

    def vrmse(pred, true):
        mse = ((pred - true) ** 2).mean(axis=(-3, -2))
        var = true.var(axis=(-3, -2), ddof=1)
        return np.sqrt(mse / (var + 1e-7))

    ra, pr, nx, ny, report = 1e6, 1.0, 64, 16, {}
    b, ux, uy = standard_initial_condition(nx, ny, seed=1)
    t0 = time.perf_counter()
    ref = RBSolver(ra, pr, nx, ny).rollout(b, ux, uy, output_dt=0.25, n_outputs=100)   # to t = 25, past onset
    report["reference"] = {"status": ref.status, "seconds": time.perf_counter() - t0,
                           "max_speed_at_end": float(np.abs(ref.frames[-1, ..., 2:]).max())}
    k, h = 79, 20                                               # restart from t = 20, compare t = 20.25 .. 25
    truth = ref.frames[k + 1:k + 1 + h]
    for name, init, shape in (("float64 restart", ref.frames[k], (nx, ny)),
                              ("float32 restart", ref.frames[k].astype(np.float32).astype(np.float64), (nx, ny)),
                              ("half resolution", ref.frames[k], (nx // 2, ny // 2))):
        state = resample(np.moveaxis(init, -1, 0), shape)
        r = RBSolver(ra, pr, *shape).rollout(state[0], state[2], state[3], output_dt=0.25, n_outputs=h)
        pred = np.moveaxis(resample(np.moveaxis(r.frames, -1, 1), (nx, ny)), 1, -1)
        v = vrmse(pred, truth)
        report[name] = {"status": r.status, "vrmse_step1": v[0].tolist(), "vrmse_last": v[-1].tolist(),
                        "iterations": int(sum(r.iterations_per_output)), "initial_dt": r.initial_dt}
    ok = (ref.status == "ok" and report["reference"]["max_speed_at_end"] > 1e-3
          and max(report["float64 restart"]["vrmse_last"]) < 1e-3
          and max(report["float32 restart"]["vrmse_last"]) < 1e-2
          and report["half resolution"]["status"] == "ok")
    report["passed"] = bool(ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-dir", default=os.path.join(os.environ.get("DATA_ROOT", ""), "datasets",
                                                          "rayleigh_benard"))
    ap.add_argument("--cohort", default=str(REPO / "tasks" / "eval_cohort.json"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUTPUT_ROOT", str(REPO / "outputs")), "solver"))
    ap.add_argument("--resolution", default="512x128", choices=sorted(RESOLUTIONS))
    ap.add_argument("--task-index", type=int, help="regime index in the solver cohort (default: SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--per-regime", type=int, default=1, help="trajectories per (Ra, Pr) regime")
    ap.add_argument("--max-seconds", type=float, default=6 * 3600, help="wall-time cap per rollout")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.selftest:
        return selftest()

    cohort = json.loads(Path(args.cohort).read_text())
    entries = solver_cohort(cohort, args.per_regime)
    if args.list:
        for i, e in enumerate(entries):
            print(i, e["id"], e["rayleigh"], e["prandtl"], e["starts"])
        return 0
    index = args.task_index if args.task_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    entry = entries[index]
    shape = RESOLUTIONS[args.resolution]
    print(f"task {index}: {entry['id']} Ra={entry['rayleigh']:g} Pr={entry['prandtl']:g} at {args.resolution}, "
          f"starts {entry['starts']}, cohort sha256 {cohort['sha256'][:12]}", flush=True)
    for start in entry["starts"]:
        out = Path(args.out) / args.resolution / f"{rollout_name(entry['id'], start)}.h5"
        if out.exists():
            with h5py.File(out, "r") as h:
                if h.attrs.get("complete", False):
                    print(f"  start {start}: done ({h.attrs['status']}), skipped", flush=True)
                    continue
        a = run_rollout(entry, start, shape, Path(args.dataset_dir), out, cohort["rollout_steps"], args.max_seconds)
        print(f"  start {start}: {a['status']}, {a['steps_done']} outputs, build {a['build_seconds']:.1f} s, "
              f"stepping {a['step_seconds']:.1f} s, {a['iterations']} iterations, "
              f"CFL dt {a['dt_min']:.2e}..{a['dt_max']:.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
