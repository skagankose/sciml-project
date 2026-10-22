"""Verify a split downloaded with the official `the-well-download` CLI.

A split is marked complete only when every registry file exists locally with exactly the
remote Content-Length, opens with h5py, and reports its trajectories. Upstream publishes no
checksums, so the SHA-256 manifest written here proves local integrity and lets later
copies be compared; it does not prove equality with the server's bytes.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import urllib.request
from pathlib import Path

import h5py
import yaml

from rbbench.manifests import atomic_write_json, now_utc, sha256_file

DATASET = "rayleigh_benard"


def registry_urls(split: str) -> list[str]:
    from the_well.utils.download import WELL_REGISTRY

    with open(WELL_REGISTRY) as f:
        return yaml.safe_load(f)[DATASET][split]


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers["Content-Length"])
    except Exception:
        return None


def verify_split(data_root: Path, split: str, compute_sha: bool = True, check_remote: bool = True) -> dict:
    dataset_dir = Path(data_root) / "datasets" / DATASET
    folder = dataset_dir / "data" / split
    urls = registry_urls(split)
    names = [os.path.basename(u) for u in urls]
    sizes = {}
    if check_remote:
        with cf.ThreadPoolExecutor(8) as ex:
            sizes = dict(zip(names, ex.map(remote_size, urls)))
    files, problems = [], []
    for name in names:
        path = folder / name
        entry = {"name": name, "exists": path.exists(), "remote_bytes": sizes.get(name)}
        if not path.exists():
            problems.append(f"missing {name}")
            files.append(entry)
            continue
        entry["bytes"] = path.stat().st_size
        if entry["remote_bytes"] is not None and entry["bytes"] != entry["remote_bytes"]:
            problems.append(f"{name}: {entry['bytes']} bytes, remote {entry['remote_bytes']}")
        try:
            with h5py.File(path, "r") as h:
                entry["n_trajectories"] = int(h.attrs["n_trajectories"])
                entry["dataset_name"] = str(h.attrs["dataset_name"])
                entry["buoyancy_shape"] = list(h["t0_fields/buoyancy"].shape)
        except Exception as err:
            problems.append(f"{name}: unreadable ({err})")
        if compute_sha and not problems:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
    extra = sorted({p.name for p in folder.glob("*.hdf5")} - set(names)) if folder.exists() else []
    if extra:
        problems.append(f"unexpected files: {extra}")
    if check_remote and any(v is None for v in sizes.values()):
        problems.append("could not read remote sizes for every file")
    if any(f.get("dataset_name") not in (None, DATASET) for f in files):
        problems.append("a file is not rayleigh_benard (wrong dataset?)")
    report = {
        "dataset": DATASET, "split": split, "folder": str(folder), "checked_utc": now_utc(),
        "n_registry_files": len(names), "files": files, "problems": problems,
        "total_bytes": sum(f.get("bytes", 0) for f in files),
        "n_trajectories": sum(f.get("n_trajectories", 0) for f in files),
        "complete": not problems,
    }
    atomic_write_json(dataset_dir / f"rbbench_verify_{split}.json", report)
    if report["complete"] and compute_sha:
        atomic_write_json(dataset_dir / f"rbbench_complete_{split}.json",
                          {k: report[k] for k in ("split", "checked_utc", "total_bytes", "n_trajectories")}
                          | {"sha256": {f["name"]: f["sha256"] for f in files}})
    return report
