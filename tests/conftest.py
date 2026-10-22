import os
import warnings

import pytest
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

from rbbench.config import load_config
from rbbench.synthetic import make_synthetic_dataset

GRID = (64, 32)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    """Small Well-format synthetic dataset, written once per test session."""
    root = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(root, nx=GRID[0], ny=GRID[1], n_steps=24,
                                  trajectories={"train": 4, "valid": 2, "test": 2})


@pytest.fixture
def tiny_cfg():
    def make(model, *overrides, experiment="smoke"):
        return load_config(model, experiment, "tiny", [f"task.spatial_resolution=[{GRID[0]},{GRID[1]}]", *overrides])

    return make


@pytest.fixture(autouse=True)
def _cpu_threads():
    torch.set_num_threads(2)
