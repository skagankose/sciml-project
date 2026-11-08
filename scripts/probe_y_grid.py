"""Which y grid are the rayleigh_benard values really on? Read-only; run on HPC3 (CPU node).

    python scripts/probe_y_grid.py [--data-root "$DATA_ROOT"] [--out outputs/audit/y_grid_probe.json]

The audit found that the files' `dimensions/y` are uniform, while The Well's README says the data are
"Sampled at Chebyshev nodes along the y axis". This prints the evidence for deciding which is true:
1. The stored y coordinates.
2. Near-wall rows of the mean buoyancy (over x and late times) for Pr = 1 at each Ra. At Ra = 1e10 the
   thermal boundary layer is ~0.005 thick: on Chebyshev nodes ~5 rows fall inside it, so b falls off
   gradually row by row; on a uniform 128-point grid (dy ~ 0.008) it drops within about one row.
3. The audit's physics checks (wall values, Nusselt) rerun with the values assumed to sit on
   Chebyshev-Gauss or Chebyshev-Lobatto nodes instead of the stored y. The correct grid passes them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from rbbench.audit import DEFAULT_TOL, physics_checks
from rbbench.config import load_config
from rbbench.data import list_trajectories, split_files
from rbbench.synthetic import chebyshev_gauss_nodes, chebyshev_lobatto_nodes

KEYS = ("b_wall_bottom_max_err", "b_wall_top_max_err", "u_wall_over_rms_max", "conductive_part_max_dev",
        "nu_volume_vs_wall_rel_median", "nu_volume_vs_wall_rel_max")


def near_wall_profiles(dataset_dir: Path, rows: int = 8, prandtl: float = 1.0) -> list[dict]:
    """Mean b and rms |u| per row next to each wall, one train trajectory per Ra at the given Pr."""
    out, seen = [], set()
    for ref in list_trajectories(dataset_dir, "train"):
        if ref.prandtl != prandtl or ref.rayleigh in seen:
            continue
        seen.add(ref.rayleigh)
        with h5py.File(dataset_dir / ref.relpath, "r") as h:
            n = h["t0_fields/buoyancy"].shape[1]
            late = slice(n // 2, n, max(1, n // 20))
            b = np.asarray(h["t0_fields/buoyancy"][ref.index, late], np.float64)  # [T, X, Y]
            u = np.asarray(h["t1_fields/velocity"][ref.index, late], np.float64)  # [T, X, Y, 2]
        b_mean = b.mean(axis=(0, 1))
        u_rms_row = np.sqrt((u**2).sum(-1).mean(axis=(0, 1)))
        u_rms = float(np.sqrt((u**2).sum(-1).mean()))
        out.append({
            "rayleigh": ref.rayleigh, "prandtl": ref.prandtl, "id": ref.id,
            "b_bottom_rows": b_mean[:rows].round(4).tolist(), "b_top_rows": b_mean[-rows:].round(4).tolist(),
            "b_mid": round(float(b_mean[len(b_mean) // 2]), 4),
            "u_rms_over_total_bottom_rows": (u_rms_row[:3] / u_rms).round(4).tolist(),
            "u_rms_over_total_top_rows": (u_rms_row[-3:] / u_rms).round(4).tolist(),
        })
    return sorted(out, key=lambda r: r["rayleigh"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--samples-per-regime", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    dataset_dir = Path(args.data_root) / "datasets" / "rayleigh_benard"
    cfg_physics = load_config("cnext_unet", "main", "full")["physics"]

    with h5py.File(split_files(dataset_dir, "train")[0], "r") as h:
        x = np.asarray(h["dimensions/x"][...], np.float64)
        y = np.asarray(h["dimensions/y"][...], np.float64)
    n = y.size
    report = {"stored_y": {
        "n": n, "first": y[:4].tolist(), "last": y[-4:].tolist(),
        "dy_min": float(np.diff(y).min()), "dy_max": float(np.diff(y).max()),
        "max_dev_from_linspace_0_1": float(np.abs(y - np.linspace(0, 1, n)).max()),
        "max_dev_from_cell_centres": float(np.abs(y - (np.arange(n) + 0.5) / n).max()),
    }}
    print("stored y:", json.dumps(report["stored_y"]))

    report["near_wall"] = near_wall_profiles(dataset_dir)
    print("\nmean b next to the walls (Pr=1, one train trajectory per Ra, late times):")
    for r in report["near_wall"]:
        print(f"  Ra={r['rayleigh']:.0e}  bottom rows 0..7: {r['b_bottom_rows']}")
        print(f"            top rows -8..-1: {r['b_top_rows']}   mid: {r['b_mid']}")
        print(f"            |u| rms / total, rows 0,1,2: {r['u_rms_over_total_bottom_rows']}"
              f"  rows -3,-2,-1: {r['u_rms_over_total_top_rows']}")

    report["hypotheses"] = {}
    print("\nphysics checks with the values placed on each node family (smaller is better):")
    for name, nodes in (("gauss", chebyshev_gauss_nodes), ("lobatto", chebyshev_lobatto_nodes)):
        res = physics_checks(dataset_dir, cfg_physics, x, nodes(n, 0.0, cfg_physics["Ly"]),
                             args.samples_per_regime, DEFAULT_TOL)
        failures = [f for f in res["failures"] if not f.startswith(("y grid", "y interval", "config y_grid"))]
        report["hypotheses"][name] = {"summary": res["summary"], "failures": failures}
        print(f"  {name:8s} " + "  ".join(f"{k}={res['summary'][k]:.3g}" for k in KEYS))
        print(f"  {'':8s} failures: {failures or 'none'}")
    print(f"\ntolerances: {DEFAULT_TOL}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
