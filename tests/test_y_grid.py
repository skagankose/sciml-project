"""rayleigh_benard stores a placeholder uniform `dimensions/y`; the values are on Chebyshev-Gauss nodes (D18).

These tests rebuild that situation on synthetic data: values on Gauss nodes, stored y overwritten with
linspace(0, 1, n).
"""

import h5py
import numpy as np
import pytest
from conftest import GRID

from rbbench.audit import DEFAULT_TOL, physics_checks
from rbbench.config import load_config
from rbbench.data import split_files
from rbbench.evaluate import physics_status
from rbbench.manifests import atomic_write_json
from rbbench.metrics import physics
from rbbench.noise import wall_rows
from rbbench.synthetic import chebyshev_gauss_nodes, chebyshev_lobatto_nodes, make_synthetic_dataset

PHYS = load_config("cnext_unet", "main", "full")["physics"]


def test_config_uses_gauss_nodes():
    assert PHYS["y_grid"] == "gauss"


def test_y_nodes():
    placeholder = np.linspace(0, 1, 128)
    assert np.allclose(physics.y_nodes(placeholder, PHYS), chebyshev_gauss_nodes(128))
    assert np.allclose(physics.y_nodes(placeholder, {**PHYS, "y_grid": "lobatto"}), chebyshev_lobatto_nodes(128))
    assert np.array_equal(physics.y_nodes(placeholder, {**PHYS, "y_grid": "auto"}), placeholder)
    with pytest.raises(ValueError, match="y_grid"):
        physics.y_nodes(placeholder, {**PHYS, "y_grid": "uniform"})


def test_stored_y_status():
    tol = DEFAULT_TOL["grid"]
    assert physics.stored_y_status(chebyshev_gauss_nodes(128).astype(np.float32), PHYS, tol) == "matches"
    assert physics.stored_y_status(np.linspace(0, 1, 128).astype(np.float32), PHYS, tol) == "placeholder"
    assert physics.stored_y_status(chebyshev_lobatto_nodes(128), PHYS, tol) == "unexpected"


def test_no_wall_rows_on_gauss_nodes():
    """With the placeholder y, rows 0 and -1 sit on y = 0 and 1 and noise would skip them."""
    placeholder = np.linspace(0, 1, 128)
    assert wall_rows(placeholder, 0.0, 1.0).sum() == 2
    assert not wall_rows(physics.y_nodes(placeholder, PHYS), 0.0, 1.0).any()


@pytest.fixture(scope="module")
def placeholder_dataset(tmp_path_factory):
    d = make_synthetic_dataset(tmp_path_factory.mktemp("placeholder_y"), nx=GRID[0], ny=GRID[1], n_steps=24,
                               trajectories={"train": 1})
    for f in split_files(d, "train"):
        with h5py.File(f, "r+") as h:
            y = h["dimensions/y"]
            y[...] = np.linspace(0, 1, y.shape[0]).astype(y.dtype)
    with h5py.File(split_files(d, "train")[0], "r") as h:
        x, y = np.asarray(h["dimensions/x"][...]), np.asarray(h["dimensions/y"][...])
    return d, x, y


def test_audit_passes_with_gauss_on_placeholder(placeholder_dataset):
    d, x, y = placeholder_dataset
    res = physics_checks(d, PHYS, x, y, 1, DEFAULT_TOL)
    assert res["passed"], res["failures"]
    assert (res["y_grid"]["best"], res["stored_y_status"], res["y_grid_used"]) == ("uniform", "placeholder", "gauss")


def test_audit_fails_when_trusting_placeholder(placeholder_dataset):
    d, x, y = placeholder_dataset
    res = physics_checks(d, {**PHYS, "y_grid": "auto"}, x, y, 1, DEFAULT_TOL)
    assert not res["passed"]
    assert any(f.startswith("y grid is not Chebyshev") for f in res["failures"])


def test_audit_fails_on_wrong_family(placeholder_dataset):
    d, x, y = placeholder_dataset
    res = physics_checks(d, {**PHYS, "y_grid": "lobatto"}, x, y, 1, DEFAULT_TOL)
    assert not res["passed"]


def test_physics_status_requires_the_same_family(tmp_path):
    audit = tmp_path / "audit.json"
    atomic_write_json(audit, {"physics_checks": {"passed": True, "failures": [], "y_grid_used": "gauss"}})
    assert physics_status(audit, "gauss")[0]
    ok, note = physics_status(audit, "lobatto")
    assert not ok and "lobatto" in note
