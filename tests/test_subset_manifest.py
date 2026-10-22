from collections import Counter

import pytest

from rbbench.data import TrajectoryRef, list_trajectories
from rbbench.subsets import (
    allocate_counts,
    build_split_manifest,
    load_subset_ids,
    make_nested_subsets,
    regime_key,
)


def _fake_train(per_regime=40):
    regimes = [(ra, pr) for ra in (1e6, 1e7, 1e8, 1e9, 1e10) for pr in (0.1, 0.2, 0.5, 1, 2, 5, 10)]
    return [TrajectoryRef(f"data/train/f_{ra:g}_{pr:g}.hdf5", i, ra, pr) for ra, pr in regimes for i in range(per_regime)]


def test_real_shaped_counts_are_exact_and_nested():
    refs = _fake_train()  # 35 regimes x 40 = 1400, like the real train split
    subs = make_nested_subsets(refs, [0.1, 0.3, 1.0], seed=0)
    assert [subs[f]["n_trajectories"] for f in (0.1, 0.3, 1.0)] == [140, 420, 1400]
    s10, s30, s100 = (set(subs[f]["trajectory_ids"]) for f in (0.1, 0.3, 1.0))
    assert s10 < s30 < s100
    assert set(subs[0.1]["counts_by_regime"].values()) == {4}
    assert set(subs[0.3]["counts_by_regime"].values()) == {12}


def test_deterministic_and_seed_dependent():
    refs = _fake_train()
    a = make_nested_subsets(refs, [0.1, 0.3, 1.0], seed=0)
    b = make_nested_subsets(list(reversed(refs)), [0.1, 0.3, 1.0], seed=0)
    c = make_nested_subsets(refs, [0.1, 0.3, 1.0], seed=1)
    assert a[0.1]["ids_sha256"] == b[0.1]["ids_sha256"]
    assert a[0.1]["ids_sha256"] != c[0.1]["ids_sha256"]


def test_uneven_groups_use_largest_remainder():
    # quotas 1.5, 0.9, 0.6 -> floors 1, 0, 0 -> two extra seats to the largest remainders (b, c)
    assert allocate_counts({"a": 5, "b": 3, "c": 2}, 3) == {"a": 1, "b": 1, "c": 1}
    assert allocate_counts({"a": 5, "b": 3, "c": 2}, 5) == {"a": 3, "b": 1, "c": 1}
    counts = allocate_counts({"a": 7, "b": 7, "c": 7}, 10)
    assert sum(counts.values()) == 10 and sorted(counts.values()) == [3, 3, 4]


def test_holdout_excludes_regime():
    refs = _fake_train(4)
    subs = make_nested_subsets(refs, [1.0], seed=0, holdout_regimes=[(1e8, 1.0)])
    assert regime_key((1e8, 1.0)) not in subs[1.0]["counts_by_regime"]
    assert regime_key((1e8, 10.0)) in subs[1.0]["counts_by_regime"]
    assert subs[1.0]["n_trajectories"] == 34 * 4


def test_tampered_manifest_rejected():
    m = make_nested_subsets(_fake_train(4), [0.5], seed=0)[0.5]
    m["trajectory_ids"] = m["trajectory_ids"][1:]
    with pytest.raises(ValueError):
        load_subset_ids(m)


def test_split_manifest_disjoint_and_regimes_read_from_file(dataset_dir):
    manifest = build_split_manifest(dataset_dir)
    for pair, leak in manifest["cross_split_duplicates"].items():
        assert leak == {"ids": [], "fingerprints": 0}, pair
    train = list_trajectories(dataset_dir, "train")
    assert Counter(r.regime for r in train) == {(1e6, 1.0): 4, (1e7, 1.0): 4, (1e6, 10.0): 4, (1e7, 10.0): 4}
    subs = make_nested_subsets(train, [0.5, 1.0], seed=0)
    ids = {r["id"] for s in ("valid", "test") for r in manifest["splits"][s]["trajectories"]}
    assert not ids & set(subs[1.0]["trajectory_ids"])
