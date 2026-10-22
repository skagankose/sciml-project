import json

import pytest
from test_subset_manifest import _fake_train

from rbbench.manifests import atomic_write_json
from rbbench.subsets import make_nested_subsets, subset_filename
from rbbench.sweep import make_sweep, read_jsonl, validate_rows, write_jsonl


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / "tasks"
    for frac, m in make_nested_subsets(_fake_train(), [0.1, 0.3, 1.0], seed=0).items():
        atomic_write_json(d / subset_filename(frac), m)
    return d


def test_main_sweep_has_36_unique_tasks(tasks_dir, tmp_path):
    rows = make_sweep("main", tasks_dir, repo_root=tmp_path)
    assert len(rows) == 36
    assert len({r["run_id"] for r in rows}) == 36
    assert {(r["model"], r["fraction"], r["seed"]) for r in rows} == {
        (m, f, s) for m in ("cnext_unet", "tfno", "avit", "convlstm") for f in (0.1, 0.3, 1.0) for s in (0, 1, 2)
    }
    path = tmp_path / "train.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
    # same subset manifest for every model and seed of a fraction
    for f in (0.1, 0.3, 1.0):
        assert len({r["subset_sha256"] for r in rows if r["fraction"] == f}) == 1


def test_duplicates_rejected(tasks_dir, tmp_path):
    rows = make_sweep("main", tasks_dir, repo_root=tmp_path)
    bad = [dict(r) for r in rows]
    bad[1]["run_id"] = bad[0]["run_id"]
    with pytest.raises(ValueError):
        validate_rows(bad)


def test_rows_are_json_not_shell(tasks_dir, tmp_path):
    rows = make_sweep("main", tasks_dir, repo_root=tmp_path)
    line = json.dumps(rows[0])
    assert json.loads(line)["seed"] == 0
