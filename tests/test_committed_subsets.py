"""Gate E on the committed real manifests: rerunning the subset selection from the committed split manifest gives
the committed S10, S30 and S100 exactly (same ids, same hashes), nested, stratified 4 / 12 / 40 per regime, with no
trajectory shared with validation or test; and every task manifest points at those files by hash.

The selection (`make_nested_subsets`) needs only the train split's (file, index, Ra, Pr), which the split manifest
records; reading Ra and Pr from the files and the content fingerprints were checked when it was built on HPC3
(`rbbench make-subsets`, 2026-11-07). No dataset is needed here (rule 2)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from rbbench.data import TrajectoryRef
from rbbench.manifests import sha256_file
from rbbench.subsets import load_subset_ids, make_nested_subsets

TASKS = Path(__file__).resolve().parents[1] / "tasks"
FRACTIONS = {10: 0.1, 30: 0.3, 100: 1.0}


@pytest.fixture(scope="module")
def committed():
    split = json.loads((TASKS / "split_manifest.json").read_text())
    subsets = {f: json.loads((TASKS / f"subset_{f}.json").read_text()) for f in FRACTIONS}
    return split, subsets


def test_rerun_reproduces_the_committed_subsets(committed):
    split, subsets = committed
    train = [TrajectoryRef(r["relpath"], r["index"], r["rayleigh"], r["prandtl"])
             for r in split["splits"]["train"]["trajectories"]]
    assert all(r.id == row["id"] for r, row in zip(train, split["splits"]["train"]["trajectories"]))
    seeds = {m["seed"] for m in subsets.values()}
    assert seeds == {0}
    rerun = make_nested_subsets(train, sorted(FRACTIONS.values()), seed=0)
    for pct, frac in FRACTIONS.items():
        old, new = subsets[pct], rerun[frac]
        assert new["trajectory_ids"] == old["trajectory_ids"], pct
        assert new["ids_sha256"] == old["ids_sha256"], pct
        assert new["counts_by_regime"] == old["counts_by_regime"], pct
        assert load_subset_ids(old) == old["trajectory_ids"]          # recorded hash matches the ids


def test_committed_subsets_nested_stratified_and_disjoint_from_eval(committed):
    split, subsets = committed
    s10, s30, s100 = (set(subsets[p]["trajectory_ids"]) for p in (10, 30, 100))
    assert s10 < s30 < s100
    assert s100 == {r["id"] for r in split["splits"]["train"]["trajectories"]}
    for pct, per_regime in ((10, 4), (30, 12), (100, 40)):
        counts = subsets[pct]["counts_by_regime"]
        assert len(counts) == 35 and set(counts.values()) == {per_regime}, pct
        per_file = Counter(r.split("#")[0] for r in subsets[pct]["trajectory_ids"])  # one file per regime
        assert len(per_file) == 35 and set(per_file.values()) == {per_regime}, pct
    held_out = {r["id"] for s in ("valid", "test") for r in split["splits"][s]["trajectories"]}
    assert len(held_out) == 350 and not held_out & s100
    assert all(v == {"ids": [], "fingerprints": 0} for v in split["cross_split_duplicates"].values())


def test_task_manifests_point_at_the_committed_subsets_by_hash(committed):
    _, subsets = committed
    split_sha = sha256_file(TASKS / "split_manifest.json")
    assert {m["split_manifest_sha256"] for m in subsets.values()} == {split_sha}
    file_sha = {f"tasks/subset_{p}.json": sha256_file(TASKS / f"subset_{p}.json") for p in FRACTIONS}
    rows = [json.loads(line) for p in sorted(TASKS.glob("*.jsonl")) for line in p.read_text().splitlines()
            if line.strip()]
    assert rows
    for row in rows:
        assert row["subset_sha256"] == file_sha[row["subset_manifest"]], row.get("run_id")
