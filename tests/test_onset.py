"""Convection onset per trajectory from the true kinetic energy (rbbench.onset; D28, D29)."""

import csv
import json
from pathlib import Path

import numpy as np

from rbbench.cli import main
from rbbench.config import load_config
from rbbench.data import list_trajectories
from rbbench.evaluate import read_grid
from rbbench.metrics import physics
import pytest

from rbbench.onset import (
    FRACTIONS,
    PRIMARY_FRACTION,
    developed_start,
    is_onset_window,
    kinetic_energy_series,
    load_onset_frames,
    onset_frame,
)

REPO = Path(__file__).resolve().parents[1]


def test_developed_window_is_the_last_quarter():
    assert developed_start(200) == 150
    assert developed_start(24) == 18


def test_onset_frame():
    ke = np.array([0.0, 0.0, 0.1, 0.6, 2.0, 1.0, 1.0, 1.0])  # overshoot at 4, developed mean 1 over frames 5-7
    assert onset_frame(ke, 0.5, dev_start=5) == 3
    assert onset_frame(ke, 0.75, dev_start=5) == 4
    assert onset_frame(ke, 0.05, dev_start=5) == 2
    assert onset_frame(np.zeros(8), 0.5, dev_start=5) is None  # a fluid that never moves never develops
    assert onset_frame(ke, 3.0, dev_start=5) is None


def test_window_rule_matches_the_fixed_cut_convention():
    assert is_onset_window(49, 50) and not is_onset_window(50, 50)
    assert is_onset_window(150, None)


def test_kinetic_energy_uses_chebyshev_weights(dataset_dir):
    cfg = load_config("cnext_unet")["physics"]
    _, y = read_grid(dataset_dir, "test")
    ops = physics.ChebyshevY(physics.y_nodes(y, cfg), 0.0, cfg["Ly"])
    vel = np.random.default_rng(0).normal(size=(5, 8, y.size, 2))
    manual = (0.5 * (vel[..., 0] ** 2 + vel[..., 1] ** 2)).mean(axis=1) @ ops.weights / cfg["Ly"]
    np.testing.assert_allclose(kinetic_energy_series(vel, ops), manual, rtol=1e-12)


def test_onset_frames_command(dataset_dir, tmp_path, capsys):
    out = tmp_path / "onset"
    assert main(["onset-frames", "--data-root", str(dataset_dir.parents[1]), "--splits", "valid", "test",
                 "--out-dir", str(out)]) == 0
    for split in ("valid", "test"):
        table = json.loads((out / f"onset_{split}.json").read_text())
        refs = list_trajectories(dataset_dir, split)
        # the same ids as the evaluation's per-window parquet (traj_id), so the cut can be joined on them
        assert [r["traj_id"] for r in table["trajectories"]] == [r.id for r in refs]
        assert table["definition"]["primary_fraction"] == PRIMARY_FRACTION == 0.1
        with (out / f"ke_{split}.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        for row in table["trajectories"]:
            ke = np.array([float(r["ke"]) for r in rows if r["traj_id"] == row["traj_id"]])
            assert ke.size == row["n_frames"]
            assert row["onset_frame"]["0.5"] == onset_frame(ke, 0.5)
            assert row["onset_frame"]["0.1"] == onset_frame(ke, 0.1)
    assert "by_rayleigh" in capsys.readouterr().out
    # recomputed from the stored KE series alone (no data): the same rows, and only the tables are written
    again = tmp_path / "again"
    assert main(["onset-frames", "--from-ke", str(out), "--splits", "valid", "test", "--out-dir", str(again)]) == 0
    for split in ("valid", "test"):
        rows_again = json.loads((again / f"onset_{split}.json").read_text())["trajectories"]
        rows = json.loads((out / f"onset_{split}.json").read_text())["trajectories"]
        assert [r["onset_frame"] for r in rows_again] == [r["onset_frame"] for r in rows]
        for a, b in zip(rows_again, rows):  # the CSV keeps 9 significant digits
            assert a["ke_developed"] == pytest.approx(b["ke_developed"], rel=1e-8)
        assert not (again / f"ke_{split}.csv").exists()


def test_frozen_onset_tables_cover_the_cohort():
    """tasks/onset_<split>.json (D29: fraction 0.1, fixed on validation) cover every trajectory the evaluation uses."""
    cohort = json.loads((REPO / "tasks" / "eval_cohort.json").read_text())
    for split, n in (("valid", 175), ("test", 175)):
        table = json.loads((REPO / "tasks" / f"onset_{split}.json").read_text())
        assert table["definition"]["primary_fraction"] == 0.1 and table["n_trajectories"] == n
        assert all(set(r["onset_frame"]) == {str(f) for f in FRACTIONS} for r in table["trajectories"])
        frames = load_onset_frames(REPO / "tasks" / f"onset_{split}.json")
        assert None not in frames.values()
    test = load_onset_frames(REPO / "tasks" / "onset_test.json")
    assert {e["id"] for e in cohort["trajectories"]} <= set(test)
    # every rollout start of the cohort falls clearly on one side: 20 in onset, 80 and 140 in developed flow
    assert all(20 < test[e["id"]] <= 80 for e in cohort["trajectories"])


def test_load_onset_frames_needs_the_fraction(tmp_path):
    path = tmp_path / "onset_test.json"
    path.write_text(json.dumps({"trajectories": [{"traj_id": "a", "onset_frame": {"0.5": 60}}]}))
    assert load_onset_frames(path, 0.5) == {"a": 60}
    with pytest.raises(KeyError):
        load_onset_frames(path, 0.1)
