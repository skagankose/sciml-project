"""Split manifest and nested, regime-stratified training-trajectory subsets.

Within each (Ra, Pr) group of the TRAIN split, trajectories are put in a seeded random order;
each fraction takes a prefix of that order, with per-group counts from a largest-remainder
allocation of round(fraction * N). Prefixes make S10 <= S30 <= S100 by construction (checked).
The same manifests serve every model and every training seed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from rbbench.data import SPLITS, TrajectoryRef, list_trajectories
from rbbench.manifests import now_utc, sha256_bytes, sha256_json, stable_int


def regime_key(regime: tuple[float, float]) -> str:
    ra, pr = regime
    return f"Ra={ra:.6g},Pr={pr:.6g}"


def trajectory_fingerprint(dataset_dir: Path, ref: TrajectoryRef) -> str:
    """Content hash of the last buoyancy frame plus (Ra, Pr), to detect duplicated trajectories."""
    with h5py.File(Path(dataset_dir) / ref.relpath, "r") as h:
        last = np.ascontiguousarray(h["t0_fields/buoyancy"][ref.index, -1], dtype=np.float32)
    return sha256_bytes(last.tobytes() + f"|{ref.rayleigh!r}|{ref.prandtl!r}".encode())


def build_split_manifest(dataset_dir: Path, fingerprints: bool = True) -> dict:
    dataset_dir = Path(dataset_dir)
    splits = {}
    for split in SPLITS:
        refs = list_trajectories(dataset_dir, split)
        rows = []
        for ref in refs:
            row = ref.to_dict()
            if fingerprints:
                row["fingerprint"] = trajectory_fingerprint(dataset_dir, ref)
            rows.append(row)
        counts = defaultdict(int)
        for ref in refs:
            counts[regime_key(ref.regime)] += 1
        splits[split] = {
            "n_trajectories": len(refs),
            "n_files": len({r.relpath for r in refs}),
            "counts_by_regime": dict(sorted(counts.items())),
            "trajectories": rows,
        }
    leaks = check_disjoint(splits) if fingerprints else None
    return {
        "dataset": dataset_dir.name,
        "created_utc": now_utc(),
        "splits": splits,
        "cross_split_duplicates": leaks,
    }


def check_disjoint(splits: dict) -> dict:
    """Trajectory ids and content fingerprints shared between splits (must all be empty)."""
    out = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids_a = {r["id"] for r in splits[a]["trajectories"]}
            ids_b = {r["id"] for r in splits[b]["trajectories"]}
            fp_a = {r.get("fingerprint") for r in splits[a]["trajectories"]} - {None}
            fp_b = {r.get("fingerprint") for r in splits[b]["trajectories"]} - {None}
            out[f"{a}|{b}"] = {"ids": sorted(ids_a & ids_b), "fingerprints": len(fp_a & fp_b)}
    return out


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment of `total` items proportionally to group sizes.

    Ties are broken by group key, so the result is deterministic.
    """
    n = sum(group_sizes.values())
    if not 0 <= total <= n:
        raise ValueError(f"cannot allocate {total} of {n}")
    quotas = {g: total * s / n for g, s in group_sizes.items()}
    counts = {g: math.floor(q) for g, q in quotas.items()}
    remaining = total - sum(counts.values())
    order = sorted(quotas, key=lambda g: (-(quotas[g] - counts[g]), g))
    for g in order[:remaining]:
        counts[g] += 1
    return counts


def make_nested_subsets(
    train_refs: list[TrajectoryRef],
    fractions: list[float],
    seed: int = 0,
    holdout_regimes: list[tuple[float, float]] | None = None,
) -> dict[float, dict]:
    holdout = {regime_key(tuple(r)) for r in holdout_regimes or []}
    groups: dict[str, list[TrajectoryRef]] = defaultdict(list)
    for ref in train_refs:
        key = regime_key(ref.regime)
        if key not in holdout:
            groups[key].append(ref)
    if holdout and not holdout <= {regime_key(r.regime) for r in train_refs}:
        raise ValueError(f"held-out regimes not found in train split: {sorted(holdout)}")
    ordered = {}
    for key, refs in sorted(groups.items()):
        rng = np.random.default_rng(stable_int("subset", seed, key))
        refs = sorted(refs)
        ordered[key] = [refs[i] for i in rng.permutation(len(refs))]
    sizes = {k: len(v) for k, v in ordered.items()}
    n_total = sum(sizes.values())

    result = {}
    previous: dict[str, int] | None = None
    for frac in sorted(fractions):
        if not 0 < frac <= 1:
            raise ValueError(f"fraction {frac} not in (0, 1]")
        total = int(math.floor(frac * n_total + 0.5))
        counts = allocate_counts(sizes, total)
        if previous and any(counts[k] < previous[k] for k in counts):
            raise RuntimeError(f"allocation for {frac} is not nested in the previous fraction")
        previous = counts
        chosen = sorted(r for k, refs in ordered.items() for r in refs[: counts[k]])
        ids = [r.id for r in chosen]
        result[frac] = {
            "fraction": frac,
            "seed": seed,
            "n_trajectories": len(ids),
            "n_pool": n_total,
            "holdout_regimes": sorted(holdout),
            "counts_by_regime": dict(sorted(counts.items())),
            "ids_sha256": sha256_json(ids),
            "trajectory_ids": ids,
        }
    return result


def subset_filename(fraction: float, holdout: bool = False) -> str:
    return f"subset_{int(round(fraction * 100))}{'_heldout' if holdout else ''}.json"


def load_subset_ids(manifest: dict) -> list[str]:
    ids = manifest["trajectory_ids"]
    if sha256_json(ids) != manifest["ids_sha256"]:
        raise ValueError("subset manifest ids do not match their recorded hash")
    return ids
