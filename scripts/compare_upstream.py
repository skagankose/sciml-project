"""Check this project's wrappers against upstream The Well code on a few real examples.

    python scripts/compare_upstream.py --data-root "$DATA_ROOT" [--size full] [--device cuda]

1. Window indexing: our trajectory/time index returns the same frames as a direct h5py read.
2. Model input: our adapter feeds each network exactly what upstream's formatter produces.
3. Rollout: our rollout equals upstream's moving-batch loop (Trainer.rollout_model, train=True
   branch, formatter-based) on the same model and window.
4. VRMSE: the wrapper equals the pinned upstream metric and an independent NumPy version.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("WANDB_MODE", "disabled")

from the_well.benchmark.metrics import VRMSE  # noqa: E402
from the_well.data.data_formatter import (  # noqa: E402
    DefaultChannelsFirstFormatter,
    DefaultChannelsLastFormatter,
)

from rbbench import MODEL_NAMES  # noqa: E402
from rbbench.config import load_config  # noqa: E402
from rbbench.data import list_trajectories, make_well_dataset, trajectory_window_range  # noqa: E402
from rbbench.device import pick_device  # noqa: E402
from rbbench.metrics.well_metric import vrmse, vrmse_reference  # noqa: E402
from rbbench.models.factory import build_model  # noqa: E402
from rbbench.rollout import rollout  # noqa: E402


def upstream_rollout(net, formatter, batch: dict, steps: int) -> torch.Tensor:
    """Upstream Trainer.rollout_model loop (normalised space), reduced to its formatter calls."""
    moving = dict(batch)
    preds = []
    for i in range(steps):
        inputs, _ = formatter.process_input(moving)
        y = formatter.process_output_channel_last(net(*inputs))
        y = formatter.process_output_expand_time(y)
        if i != steps - 1:
            moving["input_fields"] = torch.cat([moving["input_fields"][:, 1:], y], dim=1)
        preds.append(y)
    return torch.cat(preds, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--size", default="full", choices=["full", "tiny"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()
    dataset_dir = Path(args.data_root) / "datasets" / "rayleigh_benard"
    device = pick_device(args.device)
    report, ok = {}, True

    ds = make_well_dataset(dataset_dir, "test", 4, args.steps)
    ref = list_trajectories(dataset_dir, "test")[-1]
    rng = trajectory_window_range(ds, ref, dataset_dir)
    t0 = min(7, len(rng) - 1)
    sample = ds[rng.start + t0]
    with h5py.File(dataset_dir / ref.relpath) as h:
        raw = h["t0_fields/buoyancy"][ref.index, t0:t0 + 4]
    norm = ds.norm
    got = norm.denormalize_flattened(sample["input_fields"], "variable")[..., 0].numpy()
    report["indexing_max_abs_err"] = float(np.abs(got - raw).max())
    ok &= report["indexing_max_abs_err"] < 1e-4

    batch = {k: v[None].to(device) for k, v in sample.items() if torch.is_tensor(v)}
    resolution = tuple(ds.size_tuple)
    for name in MODEL_NAMES:
        args_ = load_config(name, "main", args.size)["model"]["args"]
        model = build_model(name, args_, resolution, seed=0).to(device).eval()
        entry = {}
        with torch.no_grad():
            ours = rollout(model, batch["input_fields"], args.steps)
            if name == "convlstm":
                entry["upstream"] = "n/a (not a Well model)"
            else:
                formatter = (DefaultChannelsLastFormatter if name == "avit" else DefaultChannelsFirstFormatter)(ds.metadata)
                theirs = upstream_rollout(model.net, formatter, batch, args.steps)
                entry["rollout_max_abs_diff"] = float((ours - theirs).abs().max())
                ok &= entry["rollout_max_abs_diff"] <= 1e-5 * max(1.0, float(theirs.abs().max()))
        report[name] = entry

    pred = norm.denormalize_flattened(ours.cpu(), "variable")
    true = norm.denormalize_flattened(sample["output_fields"][None], "variable")

    class Meta:
        n_spatial_dims = 2

    up = VRMSE.eval(pred, true, Meta())
    report["vrmse_vs_upstream_max_abs"] = float((vrmse(pred, true) - up).abs().max())
    report["vrmse_vs_numpy_max_rel"] = float(np.max(np.abs(vrmse(pred, true).numpy() / vrmse_reference(pred.numpy(), true.numpy()) - 1)))
    ok &= report["vrmse_vs_upstream_max_abs"] == 0.0 and report["vrmse_vs_numpy_max_rel"] < 1e-4
    report["passed"] = bool(ok)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
