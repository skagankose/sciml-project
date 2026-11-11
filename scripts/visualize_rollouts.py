#!/usr/bin/env python
"""Frames for the rollout figures: the true flow and our models' rollouts on a few test trajectories (well-rb env).

For each chosen trajectory and start frame, every trained run in the manifest at the given data fraction and seed
is loaded at the given checkpoint and rolled out over the cohort's 30 steps from the same 4 input frames, as in
the evaluation. Only the frames at a few steps are kept, in physical units, so the file is small (~5 MB per
trajectory) and the figure can be drawn on the Mac (`scripts/findings.py` -> `rollout_snapshots_*`):
    outputs/visuals/rollout_<file>__<index>__s<start>.npz   truth / <model> [steps, X, Y, 4], steps, meta (json)

    python scripts/visualize_rollouts.py --device cpu                  # defaults: the regimes and start below
    python scripts/visualize_rollouts.py --regimes 1e8,1 1e10,1 --start 80 --label 4h --fraction 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STEPS = [1, 3, 6, 12, 20, 30]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(REPO / "outputs")))
    ap.add_argument("--manifest", default=str(REPO / "tasks" / "train.jsonl"))
    ap.add_argument("--cohort", default=str(REPO / "tasks" / "eval_cohort.json"))
    ap.add_argument("--regimes", nargs="+", default=["1e8,1", "1e10,1"], help="Ra,Pr pairs; first cohort trajectory")
    ap.add_argument("--start", type=int, default=80, help="start frame (one of the cohort's starts)")
    ap.add_argument("--label", default="4h")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch

    from rbbench.cli import load_run_model
    from rbbench.data import full_trajectory_dataset, list_trajectories, load_trajectory
    from rbbench.evaluate import _physical, model_predictor
    from rbbench.rollout import rollout
    from rbbench.solver_grid import rollout_name
    from rbbench.sweep import read_jsonl

    dataset_dir = Path(args.data_root) / "datasets" / "rayleigh_benard"
    out_root = Path(args.output_root)
    cohort = json.loads(Path(args.cohort).read_text())
    if args.start not in cohort["starts"]:
        raise SystemExit(f"start {args.start} is not one of the cohort's starts {cohort['starts']}")
    horizon = cohort["rollout_steps"]
    steps = [s for s in STEPS if s <= horizon]
    device = torch.device(args.device)
    rows = [r for r in read_jsonl(Path(args.manifest))
            if abs(r["fraction"] - args.fraction) < 1e-9 and r["seed"] == args.seed
            and (out_root / "runs" / r["run_id"] / f"checkpoint_{args.label}.pt").exists()]
    if not rows:
        raise SystemExit("no finished runs with that fraction, seed and checkpoint")
    models = {}
    for r in rows:
        cfg, model, _ = load_run_model(out_root / "runs" / r["run_id"], args.label, device)
        models[r["model"]] = (model_predictor(model.eval(), device, cfg["train"]["precision"]), r["run_id"])
    n_in = cfg["task"]["n_steps_input"]
    fd, _ = full_trajectory_dataset(dataset_dir, "test", n_in)
    refs = {ref.id: ref for ref in list_trajectories(dataset_dir, "test")}

    for regime in args.regimes:
        ra, pr = (float(v) for v in regime.split(","))
        entry = next(e for e in cohort["trajectories"] if e["rayleigh"] == ra and e["prandtl"] == pr)
        traj = load_trajectory(fd, refs[entry["id"]], dataset_dir)                       # normalised [T, X, Y, F]
        t0 = args.start
        ctx = traj[None, t0:t0 + n_in].to(device)
        idx = [t0 + n_in - 1 + s for s in steps]
        out = {"truth": _physical(fd.norm, traj[idx]).numpy().astype(np.float32),
               "last_input": _physical(fd.norm, traj[t0 + n_in - 1:t0 + n_in]).numpy()[0].astype(np.float32)}
        timing = {}
        for name, (predict, _) in models.items():
            tic = time.perf_counter()
            with torch.no_grad():
                pred = rollout(predict, ctx, horizon)[0].cpu()                            # [H, X, Y, F]
            timing[name] = time.perf_counter() - tic
            out[name] = _physical(fd.norm, pred[[s - 1 for s in steps]]).numpy().astype(np.float32)
        meta = {"traj_id": entry["id"], "rayleigh": ra, "prandtl": pr, "start": t0, "steps": steps,
                "label": args.label, "fraction": args.fraction, "seed": args.seed,
                "runs": {k: v[1] for k, v in models.items()}, "fields": list(cfg["task"]["fields"]),
                "seconds_per_rollout": timing, "device": str(device)}
        path = out_root / "visuals" / f"rollout_{rollout_name(entry['id'], t0)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, steps=np.array(steps), meta=json.dumps(meta), **out)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {', '.join(models)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
