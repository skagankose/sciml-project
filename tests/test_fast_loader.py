"""The persistent-handle reader returns exactly what upstream WellDataset returns, only faster."""

import random

import pytest
import torch
from conftest import GRID
from torch.utils.data import DataLoader

from rbbench.cli import main
from rbbench.config import config_hash, load_config
from rbbench.data import FieldWindows, loader_settings, make_well_dataset


def _equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("split", ["train", "test"])
def test_samples_are_identical(dataset_dir, split):
    up = make_well_dataset(dataset_dir, split, 4, 1, persistent=False)
    fast = make_well_dataset(dataset_dir, split, 4, 1)
    assert type(fast).__name__ == "PersistentWellDataset" and len(fast) == len(up)
    for i in random.Random(0).sample(range(len(up)), 25) + [0, len(up) - 1]:
        assert _equal(up[i], fast[i]), i


def test_whole_trajectories_are_identical(dataset_dir):
    up = make_well_dataset(dataset_dir, "test", 4, 20, persistent=False)
    fast = make_well_dataset(dataset_dir, "test", 4, 20)
    for i in range(len(up)):
        assert _equal(up[i], fast[i])


def test_workers_after_the_parent_opened_files(dataset_dir):
    """The parent reads first (handles open), then forked workers must reopen and read the same batches."""
    up = FieldWindows(make_well_dataset(dataset_dir, "train", 4, 1, persistent=False), range(40))
    fast = FieldWindows(make_well_dataset(dataset_dir, "train", 4, 1), range(40))
    fast[0]
    for (xa, ya), (xb, yb) in zip(DataLoader(up, batch_size=8), DataLoader(fast, batch_size=8, num_workers=2)):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_runtime_settings_do_not_change_the_run(monkeypatch):
    a = load_config("tfno")
    b = load_config("tfno", overrides=["runtime.num_workers=15", "runtime.prefetch_factor=8"])
    assert config_hash(a) == config_hash(b)
    assert loader_settings(b) == {"num_workers": 15, "pin_memory": True, "prefetch_factor": 8}
    monkeypatch.setenv("RBBENCH_NUM_WORKERS", "7")
    assert loader_settings(a)["num_workers"] == 7
    assert loader_settings(load_config("tfno", "smoke")) == {"num_workers": 7, "pin_memory": False, "prefetch_factor": 4}
    monkeypatch.setenv("RBBENCH_NUM_WORKERS", "0")
    assert loader_settings(a) == {"num_workers": 0, "pin_memory": True}


def test_bench_data_command(dataset_dir, tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["bench-data", "--data-root", str(dataset_dir.parents[1]), "--batch-size", "4", "--batches", "3",
                 "--workers", "0", "2", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.count("samples/s") == 4 and "upstream" in text and "persistent" in text
    assert GRID == (64, 32) and out.exists()
