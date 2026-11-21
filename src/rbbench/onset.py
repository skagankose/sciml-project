"""Convection onset per trajectory, from the true kinetic energy (`rbbench onset-frames`; D28, D29; WP4).

The mid-project report split one-step windows at a fixed frame 50 (D26). The professors asked for a cut per
trajectory, from the data: onset lasts until the kinetic energy reaches a fixed fraction of its developed-state
mean. Definition (fixed on the validation split before it is applied to any test result; docs/final_plan.md):

* KE(t): domain mean of (u^2 + w^2) / 2 of the stored (physical) fields, the x mean on the uniform periodic grid
  and the y mean with Chebyshev-Gauss quadrature (`physics.mean_kinetic_energy`, D18);
* developed-state mean: the mean of KE over the last quarter of the trajectory (frames 150-199 of 200);
* onset frame t_on(f): the first frame with KE >= f x the developed-state mean, for f in FRACTIONS; the primary
  fraction is 0.5, and None means the fluid never moves (no crossing then);
* a one-step window is onset if its first input frame is before t_on (the convention of the frame-50 cut), and a
  rollout starts in developed flow if its start frame is at or after t_on.

Only the true fields are read, never a model output, so the table carries no test result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from rbbench.data import list_trajectories
from rbbench.metrics import physics

FRACTIONS = (0.25, 0.5, 0.75)
PRIMARY_FRACTION = 0.5
DEVELOPED_SHARE = 0.25  # the last quarter of the frames
FIXED_CUT = 50          # the mid-project report's cut (D26), kept as the check


def developed_start(n_frames: int) -> int:
    """First frame of the developed-state window: the last quarter of the trajectory (150 for 200 frames)."""
    return n_frames - max(1, int(round(n_frames * DEVELOPED_SHARE)))


def onset_frame(ke: np.ndarray, fraction: float, dev_start: int | None = None) -> int | None:
    """First frame at which KE reaches `fraction` x its mean over the developed window.

    None if the fluid never moves (developed mean <= 0) or, for a fraction above 1, KE never gets there. For a
    fraction <= 1 and a moving fluid a crossing always exists (some developed frame is at least the mean).
    """
    ke = np.asarray(ke, np.float64)
    start = developed_start(ke.size) if dev_start is None else dev_start
    mean = ke[start:].mean()
    if not mean > 0:
        return None
    above = np.flatnonzero(ke >= fraction * mean)
    return int(above[0]) if above.size else None


def is_onset_window(first_input_frame: int, t_on: int | None) -> bool:
    """A window is onset if its first input frame precedes t_on (a trajectory that never develops is all onset)."""
    return t_on is None or first_input_frame < t_on


def kinetic_energy_series(velocity: np.ndarray, ops: physics.ChebyshevY) -> np.ndarray:
    """KE(t) of one trajectory's velocity [T, X, Y, 2] (stored units)."""
    return physics.mean_kinetic_energy(velocity[..., 0], velocity[..., 1], ops)


def onset_table(dataset_dir: Path, split: str, cfg_physics: dict, log=print) -> tuple[list[dict], dict]:
    """One row per trajectory of `split`, plus KE(t) per trajectory ({traj_id: array})."""
    from rbbench.evaluate import read_grid

    _, y_stored = read_grid(dataset_dir, split)
    ops = physics.ChebyshevY(physics.y_nodes(y_stored, cfg_physics), 0.0, cfg_physics["Ly"])
    rows, series = [], {}
    refs = list_trajectories(dataset_dir, split)
    for n, ref in enumerate(refs):
        with h5py.File(Path(dataset_dir) / ref.relpath, "r") as h:
            vel = np.asarray(h["t1_fields/velocity"][ref.index])  # [T, X, Y, 2]
        ke = kinetic_energy_series(vel, ops)
        start = developed_start(ke.size)
        rows.append({
            "traj_id": ref.id, "rayleigh": ref.rayleigh, "prandtl": ref.prandtl, "n_frames": int(ke.size),
            "ke_developed": float(ke[start:].mean()), "ke_max": float(ke.max()), "frame_ke_max": int(ke.argmax()),
            "onset_frame": {str(f): onset_frame(ke, f, start) for f in FRACTIONS},
        })
        series[ref.id] = ke
        if log and (n + 1) % 25 == 0:
            log(f"  {split}: {n + 1}/{len(refs)} trajectories")
    return rows, series


def definition() -> dict:
    return {
        "ke": "domain mean of (u^2 + w^2)/2, stored units; x mean uniform, y mean Chebyshev-Gauss quadrature",
        "developed_window": f"last {DEVELOPED_SHARE:g} of the frames (frames 150-199 of 200)",
        "onset_frame": "first frame with KE >= fraction x developed mean; null if the fluid never moves",
        "fractions": list(FRACTIONS), "primary_fraction": PRIMARY_FRACTION, "fixed_cut_check": FIXED_CUT,
        "window_rule": "a one-step window is onset if its first input frame < onset_frame",
    }


def write_onset(out_dir: Path, split: str, rows: list[dict], series: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / f"onset_{split}.json"
    table.write_text(json.dumps({"split": split, "definition": definition(), "n_trajectories": len(rows),
                                 "trajectories": rows}, indent=1))
    ke_csv = out_dir / f"ke_{split}.csv"
    with ke_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["traj_id", "frame", "ke"])
        for tid, ke in series.items():
            w.writerows((tid, t, f"{v:.9g}") for t, v in enumerate(ke))
    return table, ke_csv


def summarize(rows: list[dict], fraction: float = PRIMARY_FRACTION) -> dict:
    """t_on by Rayleigh number (min / median / max) and the trajectories that never develop."""
    by_ra: dict[float, list[int]] = {}
    never = []
    for r in rows:
        t = r["onset_frame"][str(fraction)]
        if t is None:
            never.append(r["traj_id"])
        else:
            by_ra.setdefault(r["rayleigh"], []).append(t)
    return {"fraction": fraction, "never": never,
            "by_rayleigh": {f"{ra:g}": {"min": int(min(v)), "median": float(np.median(v)), "max": int(max(v)),
                                        "n": len(v)} for ra, v in sorted(by_ra.items())}}
