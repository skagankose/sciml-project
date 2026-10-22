"""Task manifests: one JSONL row per (model, fraction, seed), parsed by index (never `eval`)."""

from __future__ import annotations

import json
from pathlib import Path

from rbbench import MODEL_NAMES
from rbbench.config import config_hash, load_config
from rbbench.manifests import read_json, sha256_file
from rbbench.subsets import load_subset_ids, subset_filename

REQUIRED_KEYS = (
    "task_index", "run_id", "model", "fraction", "seed", "experiment", "size",
    "subset_manifest", "subset_sha256", "config_sha256",
)


def run_id(model: str, fraction: float, seed: int, subset_sha: str, cfg_sha: str) -> str:
    return f"{model}/f{int(round(fraction * 100)):03d}/s{seed}/{subset_sha[:8]}-{cfg_sha[:8]}"


def task_config(model: str, experiment: str, size: str, subset_manifest: str, extra: list[str] | None = None) -> dict:
    return load_config(model, experiment, size, [f"data.subset_manifest={subset_manifest}", *(extra or [])])


def make_sweep(experiment: str, tasks_dir: Path, size: str = "full", models=MODEL_NAMES,
               fractions=None, seeds=None, extra_overrides: list[str] | None = None,
               repo_root: Path | None = None) -> list[dict]:
    repo_root = repo_root or tasks_dir.parent
    base = load_config(models[0], experiment, size)
    fractions = fractions or base["sweep"]["fractions"]
    seeds = seeds or base["sweep"]["seeds"]
    rows = []
    for model in models:
        for frac in fractions:
            manifest_path = tasks_dir / subset_filename(frac)
            rel = manifest_path.resolve().relative_to(repo_root.resolve()).as_posix()
            manifest = read_json(manifest_path)
            load_subset_ids(manifest)  # validates the id hash
            if abs(manifest["fraction"] - frac) > 1e-12:
                raise ValueError(f"{manifest_path} is for fraction {manifest['fraction']}, not {frac}")
            subset_sha = sha256_file(manifest_path)
            for seed in seeds:
                cfg = task_config(model, experiment, size, rel, extra_overrides)
                cfg_sha = config_hash(cfg)
                rows.append({
                    "task_index": len(rows), "run_id": run_id(model, frac, seed, subset_sha, cfg_sha),
                    "model": model, "fraction": frac, "seed": seed, "experiment": experiment, "size": size,
                    "subset_manifest": rel, "subset_sha256": subset_sha, "config_sha256": cfg_sha,
                    "overrides": list(extra_overrides or []),
                })
    validate_rows(rows, expect=len(models) * len(fractions) * len(seeds))
    return rows


def validate_rows(rows: list[dict], expect: int | None = None) -> None:
    if expect is not None and len(rows) != expect:
        raise ValueError(f"{len(rows)} tasks, expected {expect}")
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(f"task {i} lacks {missing}")
        if r["task_index"] != i:
            raise ValueError(f"task_index {r['task_index']} on line {i}")
    ids = [r["run_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids")
    combos = [(r["model"], r["fraction"], r["seed"]) for r in rows]
    if len(set(combos)) != len(combos):
        raise ValueError("duplicate (model, fraction, seed) combinations")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    validate_rows(rows)
    return rows


def get_task(path: Path, index: int) -> dict:
    rows = read_jsonl(path)
    if not 0 <= index < len(rows):
        raise IndexError(f"task index {index} out of range 0..{len(rows) - 1}")
    return rows[index]
