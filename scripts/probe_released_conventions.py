"""Which data convention were The Well's released checkpoints trained with? Read-only; HPC3 CPU node.

    python scripts/probe_released_conventions.py [--data-root "$DATA_ROOT"] [--out outputs/baselines/released/conventions.json]

The pipeline check (eval-released) assumes z-score inputs and a predicted next state. The released TFNO
scores VRMSE ~1.1 on every field even on late, fully developed windows (CNextU-Net: ~0.03), which is what a
mismatched convention gives. The pinned trainer supports z-score or RMS normalisation, and predicting the
next state or the change from the last input frame (DeltaWellDataset: during evaluation it adds the
denormalised delta to the last input, `Trainer.rollout_model`). This scores both checkpoints under all four
combinations on a few validation trajectories, using upstream's own normalisation classes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from the_well.data.normalization import RMSNormalization, ZScoreNormalization

from rbbench.config import load_config
from rbbench.data import full_trajectory_dataset, list_trajectories, load_trajectory
from rbbench.metrics.well_metric import vrmse
from rbbench.released import fetch, load_released

CONVENTIONS = ("zscore_state", "zscore_delta", "rms_state", "rms_delta")
PERSISTENCE = "persistence"  # x_{t+1} = x_t, the reference a delta-reading must clearly beat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--prandtl", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["cnext_unet", "tfno"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    dataset_dir = Path(args.data_root) / "datasets" / "rayleigh_benard"

    fd, _ = full_trajectory_dataset(dataset_dir, args.split, 4)
    stats = yaml.safe_load((dataset_dir / "stats.yaml").read_text())
    names = (fd.norm.core_field_names, fd.norm.core_constant_field_names)
    norms = {"zscore": ZScoreNormalization(stats, *names), "rms": RMSNormalization(stats, *names)}

    refs, seen = [], set()
    for r in list_trajectories(dataset_dir, args.split):
        if r.prandtl == args.prandtl and r.rayleigh not in seen:
            seen.add(r.rayleigh)
            refs.append(r)
    refs.sort(key=lambda r: r.rayleigh)

    models = {}
    for name in args.models:
        cfg = load_config(name, "main", "full")
        models[name] = load_released(name, cfg, fetch(name))

    # rows[model][convention] -> list of (start, vrmse per field)
    rows = {m: {c: [] for c in CONVENTIONS} for m in models}
    rows[PERSISTENCE] = {PERSISTENCE: []}
    for ref in refs:
        phys = fd.norm.denormalize_flattened(load_trajectory(fd, ref, dataset_dir).float(), "variable")
        starts = list(range(0, phys.shape[0] - 4, args.stride))
        print(f"{ref.id}: {len(starts)} windows", flush=True)
        for i in range(0, len(starts), args.batch_size):
            chunk = starts[i:i + args.batch_size]
            x = torch.stack([phys[s:s + 4] for s in chunk])          # [B, 4, X, Y, F] physical
            last = x[:, -1]
            true = torch.stack([phys[s + 4] for s in chunk])
            rows[PERSISTENCE][PERSISTENCE] += list(zip(chunk, vrmse(last, true).numpy()))
            for name, model in models.items():
                for kind, norm in norms.items():
                    with torch.no_grad():
                        out = model(norm.normalize_flattened(x, "variable")).float()
                    preds = {f"{kind}_state": norm.denormalize_flattened(out, "variable"),
                             f"{kind}_delta": last + norm.delta_denormalize_flattened(out, "variable")}
                    for conv, pred in preds.items():
                        v = vrmse(pred, true).numpy()
                        rows[name][conv] += [(s, vals) for s, vals in zip(chunk, v)]

    report = {"split": args.split, "trajectories": [r.id for r in refs], "stride": args.stride, "results": {}}
    print("\nVRMSE (field mean) per convention; 'late' = windows starting at frame >= 50")
    for name, convs in rows.items():
        for conv in convs:
            starts = np.array([s for s, _ in convs[conv]])
            vals = np.stack([v for _, v in convs[conv]])
            mean_f = vals.mean(1)
            late = starts >= 50
            if not late.any():  # short (synthetic) trajectories
                late = np.ones_like(late)
            res = {"mean": float(mean_f.mean()), "median": float(np.median(mean_f)),
                   "late_median": float(np.median(mean_f[late])),
                   "late_median_per_field": np.median(vals[late], 0).round(4).tolist(), "n_windows": len(starts)}
            report["results"].setdefault(name, {})[conv] = res
            print(f"  {name:11s} {conv:13s} mean {res['mean']:9.4f}  median {res['median']:8.4f}  "
                  f"late median {res['late_median']:8.4f}  late per field {res['late_median_per_field']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
