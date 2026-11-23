"""Flow-snapshot file names (rbbench.visuals): 10% and 100% snapshots of one rollout must not overwrite each other."""

import json

import numpy as np

from rbbench.solver_grid import rollout_name
from rbbench.visuals import PREFIX, snapshot_files, snapshot_path, snapshot_stem

TRAJ = "data/test/rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5#0"


def _write(path, fraction, label="4h", seed=0, start=80, traj=TRAJ):
    meta = {"traj_id": traj, "start": start, "fraction": fraction, "label": label, "seed": seed}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, steps=np.array([1, 30]), meta=json.dumps(meta), truth=np.zeros((2, 4, 2, 4), np.float32))
    return path


def test_stem_carries_fraction_label_and_seed():
    assert snapshot_stem(TRAJ, 80, 0.1, "4h", 0) == "rayleigh_benard_Rayleigh_1e8_Prandtl_1__0__s080__f010_4h_s0"
    assert snapshot_stem(TRAJ, 80, 1.0, "4h", 0) == "rayleigh_benard_Rayleigh_1e8_Prandtl_1__0__s080__f100_4h_s0"
    assert snapshot_stem(TRAJ, 80, 1.0, "1h", 1).endswith("__f100_1h_s1")
    stems = {snapshot_stem(TRAJ, 80, f, lab, s) for f in (0.1, 1.0) for lab in ("1h", "4h") for s in (0, 1)}
    assert len(stems) == 8


def test_writer_and_reader_agree(tmp_path):
    p10 = _write(snapshot_path(tmp_path, TRAJ, 80, 0.1, "4h", 0), 0.1)
    p100 = _write(snapshot_path(tmp_path, TRAJ, 80, 1.0, "4h", 0), 1.0)
    assert p10 != p100
    assert snapshot_files(tmp_path) == [(p10, p10.stem.removeprefix(PREFIX)), (p100, p100.stem.removeprefix(PREFIX))]


def test_old_names_are_read_from_their_metadata(tmp_path):
    """The mid-project report's 10% files have no fraction in the name; they get the stem a new file would get."""
    old = _write(tmp_path / f"{PREFIX}{rollout_name(TRAJ, 80)}.npz", 0.1)
    new100 = _write(snapshot_path(tmp_path, TRAJ, 80, 1.0, "4h", 0), 1.0)
    assert snapshot_files(tmp_path) == [(old, snapshot_stem(TRAJ, 80, 0.1, "4h", 0)),
                                        (new100, snapshot_stem(TRAJ, 80, 1.0, "4h", 0))]
    # a regenerated 10% file next to the old one (the old one pulled back from HPC3): listed once, under the new name
    new10 = _write(snapshot_path(tmp_path, TRAJ, 80, 0.1, "4h", 0), 0.1)
    files = snapshot_files(tmp_path)
    assert [s for _, s in files] == [snapshot_stem(TRAJ, 80, 0.1, "4h", 0), snapshot_stem(TRAJ, 80, 1.0, "4h", 0)]
    assert files[0][0] == new10


def test_missing_folder(tmp_path):
    assert snapshot_files(tmp_path / "nope") == []
