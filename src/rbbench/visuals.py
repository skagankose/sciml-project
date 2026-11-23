"""File names of the flow snapshots (`scripts/visualize_rollouts.py`; read by `scripts/findings.py` and
`scripts/report_figures.py`).

A snapshot file holds the true flow and every model's rollout for one test trajectory and start frame, for the
models of one data fraction, checkpoint and seed. The stem carries all three, so the 10% and 100% snapshots of the
same rollout do not overwrite each other: `rollout_<file>__<index>__s<start>__f<fraction>_<label>_s<seed>.npz`.

Files written before the stem carried them (`rollout_<file>__<index>__s<start>.npz`; the mid-project report's
10% / 4 h / seed 0 snapshots) are read from their stored metadata and get the stem a new file of that set would get.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rbbench.solver_grid import rollout_name

PREFIX = "rollout_"


def snapshot_stem(traj_id: str, start: int, fraction: float, label: str, seed: int) -> str:
    """Stem without the prefix, e.g. rayleigh_benard_Rayleigh_1e8_Prandtl_1__0__s080__f100_4h_s0."""
    return f"{rollout_name(traj_id, start)}__f{int(round(fraction * 100)):03d}_{label}_s{seed}"


def snapshot_path(visuals_dir: Path, traj_id: str, start: int, fraction: float, label: str, seed: int) -> Path:
    return Path(visuals_dir) / f"{PREFIX}{snapshot_stem(traj_id, start, fraction, label, seed)}.npz"


def read_meta(npz: Path) -> dict:
    with np.load(npz) as d:
        return json.loads(str(d["meta"]))


def snapshot_files(visuals_dir: Path) -> list[tuple[Path, str]]:
    """(file, stem) of every distinct snapshot set in `visuals_dir`, sorted by stem.

    The stem comes from the stored metadata, not the file name. A set present under both the old and the new name
    (the old file pulled back from HPC3 next to a regenerated one) is listed once, under the new name.
    """
    visuals_dir = Path(visuals_dir)
    found: dict[str, Path] = {}
    for npz in sorted(visuals_dir.glob(f"{PREFIX}*.npz")) if visuals_dir.exists() else []:
        m = read_meta(npz)
        stem = snapshot_stem(m["traj_id"], m["start"], m["fraction"], m["label"], m["seed"])
        if stem not in found or npz.stem == PREFIX + stem:
            found[stem] = npz
    return sorted(((p, s) for s, p in found.items()), key=lambda ps: ps[1])
