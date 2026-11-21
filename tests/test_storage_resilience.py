"""Surviving /pub trouble (docs/final_plan.md WP1): read retries, the node-local copy, restarts in the allocation."""

import filecmp
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from rbbench.cli import main
from rbbench.data import list_trajectories, make_well_dataset, window_indices
from rbbench.stage import StagingError, stage_dataset

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "cluster" / "run_with_restarts.sh"


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- read retries -------------------------------------------------------------------------

def test_read_retries_return_the_same_sample(dataset_dir, monkeypatch):
    monkeypatch.setenv("RBBENCH_READ_RETRIES", "3")
    monkeypatch.setenv("RBBENCH_READ_RETRY_SECONDS", "0")
    up = make_well_dataset(dataset_dir, "train", 4, 1, persistent=False)
    fast = make_well_dataset(dataset_dir, "train", 4, 1)
    fast[0]  # handles open before the failures
    original, calls, drops = fast._reconstruct_fields, {"n": 0}, {"n": 0}
    original_drop = fast._drop_handles

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(70, "Communication error on send")
        return original(*args, **kwargs)

    def counting_drop():
        drops["n"] += 1
        original_drop()

    monkeypatch.setattr(fast, "_reconstruct_fields", flaky)
    monkeypatch.setattr(fast, "_drop_handles", counting_drop)
    assert _equal(up[7], fast[7])
    assert calls["n"] == 3 and drops["n"] == 2


def test_read_retries_give_up(dataset_dir, monkeypatch):
    monkeypatch.setenv("RBBENCH_READ_RETRIES", "1")
    monkeypatch.setenv("RBBENCH_READ_RETRY_SECONDS", "0")
    fast = make_well_dataset(dataset_dir, "train", 4, 1)

    def broken(*args, **kwargs):
        raise OSError(70, "Communication error on send")

    monkeypatch.setattr(fast, "_reconstruct_fields", broken)
    with pytest.raises(OSError):
        fast[0]


# --- node-local copy ----------------------------------------------------------------------

def test_staged_copy_gives_identical_samples(dataset_dir, tmp_path):
    report = stage_dataset(dataset_dir.parents[1], tmp_path / "local", ["train"], workers=2)
    staged = tmp_path / "local" / "datasets" / "rayleigh_benard"
    assert report["files"] == len(list((dataset_dir / "data" / "train").glob("*.hdf5")))
    for f in (dataset_dir / "data" / "train").glob("*.hdf5"):
        copy = staged / "data" / "train" / f.name
        assert not copy.is_symlink() and filecmp.cmp(f, copy, shallow=False)
    for split in ("valid", "test"):
        for f in (dataset_dir / "data" / split).glob("*.hdf5"):
            link = staged / "data" / split / f.name
            assert link.is_symlink() and link.resolve() == f.resolve()
    for entry in dataset_dir.iterdir():
        if entry.name != "data":
            assert (staged / entry.name).is_symlink()
    for split in ("train", "valid"):
        refs = list_trajectories(dataset_dir, split)
        assert [r.id for r in list_trajectories(staged, split)] == [r.id for r in refs]
        up, st = make_well_dataset(dataset_dir, split, 4, 1), make_well_dataset(staged, split, 4, 1)
        assert window_indices(st, refs, staged) == window_indices(up, refs, dataset_dir)
        for i in (0, len(up) // 2, len(up) - 1):
            assert _equal(up[i], st[i])
    again = stage_dataset(dataset_dir.parents[1], tmp_path / "local", ["train"])  # idempotent
    assert again["files"] == report["files"]


def _copy_of_fixture(dataset_dir, tmp_path) -> Path:
    root = tmp_path / "pub"
    shutil.copytree(dataset_dir.parents[1], root, symlinks=True)
    return root


def test_staging_checks_the_download_record(dataset_dir, tmp_path):
    root = _copy_of_fixture(dataset_dir, tmp_path)
    src = root / "datasets" / "rayleigh_benard"
    files = sorted((src / "data" / "train").glob("*.hdf5"))
    record = {"total_bytes": sum(f.stat().st_size for f in files), "sha256": {f.name: "x" for f in files}}
    (src / "rbbench_complete_train.json").write_text(json.dumps(record))
    report = stage_dataset(root, tmp_path / "ok", ["train"])
    assert report["checked"]["train"] == "matches rbbench_complete record"
    record["total_bytes"] += 1
    (src / "rbbench_complete_train.json").write_text(json.dumps(record))
    with pytest.raises(StagingError, match="download record"):
        stage_dataset(root, tmp_path / "bad", ["train"])


def test_staging_refuses_without_space(dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("rbbench.stage.shutil.disk_usage", lambda p: shutil._ntuple_diskusage(10, 10, 0))
    with pytest.raises(StagingError, match="free"):
        stage_dataset(dataset_dir.parents[1], tmp_path / "full", ["train"])


def test_stage_data_command(dataset_dir, tmp_path, capsys):
    assert main(["stage-data", "--data-root", str(dataset_dir.parents[1]), "--dest", str(tmp_path / "d")]) == 0
    assert json.loads(capsys.readouterr().out)["splits"] == ["train"]
    assert main(["stage-data", "--data-root", str(tmp_path / "missing"), "--dest", str(tmp_path / "e")]) == 1


# --- restarts inside the allocation -------------------------------------------------------

COUNTING_CHILD = """
import pathlib, sys
p = pathlib.Path(sys.argv[1]); n = int(p.read_text()) if p.exists() else 0; p.write_text(str(n + 1))
codes = [int(c) for c in sys.argv[2:]]; sys.exit(codes[min(n, len(codes) - 1)])
"""


def _wrapper_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update({"RESTART_DELAY": "0", "MAX_RESTARTS": "2"}, **extra)
    return env


def _run_counting(tmp_path, codes, **env) -> tuple[int, int]:
    counter = tmp_path / "count"
    rc = subprocess.run(["bash", str(WRAPPER), sys.executable, "-c", COUNTING_CHILD, str(counter), *map(str, codes)],
                        env=_wrapper_env(**env), capture_output=True, timeout=60).returncode
    return rc, int(counter.read_text())


@pytest.mark.parametrize("codes, expected", [
    ([1, 0], (0, 2)),        # a crash, then the resumed run finishes
    ([1, 1, 1, 1], (1, 3)),  # gives up after MAX_RESTARTS restarts
    ([3], (3, 1)),           # state saved after a signal: no restart
    ([0], (0, 1)),
])
def test_wrapper_restarts_after_a_crash(tmp_path, codes, expected):
    assert _run_counting(tmp_path, codes) == expected


def test_wrapper_does_not_restart_near_the_time_limit(tmp_path):
    end = str(int(time.time()) + 10 * 60)
    assert _run_counting(tmp_path, [1, 0], SLURM_JOB_END_TIME=end, MIN_MINUTES_LEFT="30") == (1, 1)


SIGNALLED_CHILD = """
import pathlib, signal, sys, time
def saved(sig, frame):
    pathlib.Path(sys.argv[1]).write_text("saved")
    sys.exit(3)
signal.signal(signal.SIGUSR1, saved)
pathlib.Path(sys.argv[2]).write_text("ready")
time.sleep(60)
sys.exit(1)
"""


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX signals")
def test_wrapper_forwards_usr1_and_cleans_up(tmp_path):
    marker, ready, staged = tmp_path / "saved", tmp_path / "ready", tmp_path / "staged"
    staged.mkdir()
    (staged / "file").write_text("x")
    proc = subprocess.Popen(["bash", str(WRAPPER), sys.executable, "-c", SIGNALLED_CHILD, str(marker), str(ready)],
                            env=_wrapper_env(CLEANUP_DIR=str(staged)))
    deadline = time.time() + 30
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert ready.exists()
    proc.send_signal(signal.SIGUSR1)
    assert proc.wait(timeout=30) == 3
    assert marker.read_text() == "saved"
    assert not staged.exists()
