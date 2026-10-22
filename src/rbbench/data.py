"""Data access through the upstream `WellDataset`, plus trajectory-level bookkeeping.

Canonical layout (documented in docs/methodology.md):
  * a window is `[T, X, Y, F]` exactly as `WellDataset` returns it, `[B, T, X, Y, F]` batched;
  * X is the horizontal, periodic axis (512), Y the vertical, wall-bounded Chebyshev axis (128);
  * F is ordered (buoyancy, pressure, velocity_x, velocity_y);
  * stacked channels are time-major: channel = t * F + f (upstream formatter convention).
HDF5 parsing and z-score normalization are upstream's; this module only selects windows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, Sampler

from rbbench.manifests import stable_int

FIELDS = ("buoyancy", "pressure", "velocity_x", "velocity_y")
DATASET_NAME = "rayleigh_benard"
SPLITS = ("train", "valid", "test")


# --- layout -------------------------------------------------------------------------------

def stack_channels(x: torch.Tensor) -> torch.Tensor:
    """[B, T, X, Y, F] -> [B, T*F, X, Y] (time-major channels)."""
    return rearrange(x, "b t x y f -> b (t f) x y")


def unstack_channels(x: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Inverse of `stack_channels`."""
    return rearrange(x, "b (t f) x y -> b t x y f", t=n_steps)


def to_sequence(x: torch.Tensor) -> torch.Tensor:
    """[B, T, X, Y, F] -> [B, T, F, X, Y] (recurrent models)."""
    return rearrange(x, "b t x y f -> b t f x y")


def from_sequence(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b t f x y -> b t x y f")


# --- paths --------------------------------------------------------------------------------

def dataset_dir_from_env(cfg_paths: dict | None = None, data_root: str | None = None) -> Path:
    env = (cfg_paths or {}).get("data_root_env", "DATA_ROOT")
    root = data_root or os.environ.get(env)
    if not root:
        raise RuntimeError(f"set ${env} (or pass --data-root) to the directory holding datasets/")
    return Path(root) / "datasets" / DATASET_NAME


# --- trajectory catalog -------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class TrajectoryRef:
    """One physical trajectory: `relpath` is relative to the dataset dir."""

    relpath: str
    index: int
    rayleigh: float
    prandtl: float

    @property
    def id(self) -> str:
        return f"{self.relpath}#{self.index}"

    @property
    def regime(self) -> tuple[float, float]:
        return (self.rayleigh, self.prandtl)

    def to_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


def _read_scalar(h: h5py.File, name: str, sample: int) -> float:
    if "scalars" in h and name in h["scalars"]:
        ds = h["scalars"][name]
        arr = np.asarray(ds[...])
        if ds.attrs.get("sample_varying", False):
            arr = arr[sample]
        return float(np.asarray(arr).reshape(-1)[0])
    if name in h.attrs:
        return float(np.asarray(h.attrs[name]).reshape(-1)[0])
    raise KeyError(f"{h.filename}: no scalar or attribute {name!r}")


def split_files(dataset_dir: Path, split: str) -> list[Path]:
    folder = Path(dataset_dir) / "data" / split
    files = sorted(list(folder.glob("*.hdf5")) + list(folder.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"no HDF5 files in {folder}")
    return files


def list_trajectories(dataset_dir: Path, split: str) -> list[TrajectoryRef]:
    """All trajectories of a split, with (Ra, Pr) read from each file (not its name)."""
    dataset_dir = Path(dataset_dir)
    refs = []
    for path in split_files(dataset_dir, split):
        rel = path.relative_to(dataset_dir).as_posix()
        with h5py.File(path, "r") as h:
            for i in range(int(h.attrs["n_trajectories"])):
                refs.append(
                    TrajectoryRef(rel, i, _read_scalar(h, "Rayleigh", i), _read_scalar(h, "Prandtl", i))
                )
    return sorted(refs)


# --- WellDataset wrappers -----------------------------------------------------------------

def make_well_dataset(
    dataset_dir: Path,
    split: str,
    n_steps_input: int = 4,
    n_steps_output: int = 1,
    use_normalization: bool = True,
    return_grid: bool = False,
):
    """Upstream `WellDataset` over one split, z-score normalised with the dataset's stats.yaml."""
    from the_well.data.datasets import WellDataset
    from the_well.data.normalization import ZScoreNormalization
    from the_well.data.utils import flatten_field_names

    ds = WellDataset(
        path=str(dataset_dir),
        well_split_name=split,
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        max_rollout_steps=n_steps_output,
        use_normalization=use_normalization,
        normalization_type=ZScoreNormalization if use_normalization else None,
        boundary_return_type=None,
        return_grid=return_grid,
    )
    names = tuple(flatten_field_names(ds.metadata, include_constants=False))
    if names != FIELDS:
        raise RuntimeError(f"unexpected field order {names}; expected {FIELDS}")
    return ds


def _file_index(ds, ref: TrajectoryRef, dataset_dir: Path) -> int:
    target = (Path(dataset_dir) / ref.relpath).resolve()
    for i, p in enumerate(ds.files_paths):
        if Path(p).resolve() == target:
            return i
    raise KeyError(f"{ref.relpath} is not part of {ds}")


def trajectory_window_range(ds, ref: TrajectoryRef, dataset_dir: Path) -> range:
    """Global `WellDataset` indices of every window of one trajectory, in time order."""
    f = _file_index(ds, ref, dataset_dir)
    if ref.index >= ds.n_trajectories_per_file[f]:
        raise IndexError(f"{ref.id}: file has {ds.n_trajectories_per_file[f]} trajectories")
    n_windows = ds.n_windows_per_trajectory[f]
    file_start = 0 if f == 0 else int(ds.file_index_offsets[f])  # upstream stores -1 for file 0
    start = file_start + ref.index * n_windows
    return range(start, start + n_windows)


def window_indices(ds, refs: Sequence[TrajectoryRef], dataset_dir: Path) -> list[int]:
    out: list[int] = []
    for ref in sorted(refs):
        out.extend(trajectory_window_range(ds, ref, dataset_dir))
    return out


class FieldWindows(Dataset):
    """(input [T,X,Y,F], target [X,Y,F]) pairs from selected `WellDataset` windows."""

    def __init__(self, well_dataset, indices: Sequence[int]):
        if well_dataset.n_steps_output != 1:
            raise ValueError("FieldWindows expects n_steps_output == 1")
        self.ds = well_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        sample = self.ds[self.indices[i]]
        return sample["input_fields"], sample["output_fields"][0]


def load_trajectory(ds_full, ref: TrajectoryRef, dataset_dir: Path) -> torch.Tensor:
    """Every frame of one trajectory, normalised, `[T_total, X, Y, F]`.

    `ds_full` must have n_steps_input + n_steps_output == steps per trajectory, so each
    trajectory is exactly one window and the file is read once.
    """
    rng = trajectory_window_range(ds_full, ref, dataset_dir)
    if len(rng) != 1:
        raise ValueError("load_trajectory needs a dataset whose single window spans the trajectory")
    sample = ds_full[rng.start]
    return torch.cat([sample["input_fields"], sample["output_fields"]], dim=0)


def full_trajectory_dataset(dataset_dir: Path, split: str, n_steps_input: int = 4):
    probe = make_well_dataset(dataset_dir, split, n_steps_input, 1)
    steps = set(probe.n_steps_per_trajectory)
    if len(steps) != 1:
        raise RuntimeError(f"trajectories of unequal length in {split}: {sorted(steps)}")
    total = steps.pop()
    return make_well_dataset(dataset_dir, split, n_steps_input, total - n_steps_input), total


# --- resumable sampler --------------------------------------------------------------------

class ResumableRandomSampler(Sampler[int]):
    """Epoch-seeded permutation that can restart mid-epoch at `start`.

    The permutation of epoch e depends only on (seed, e), never on worker count or batch
    size, so a resumed run visits exactly the samples an uninterrupted run would.
    """

    def __init__(self, n: int, seed: int, epoch: int = 0, start: int = 0):
        self.n = n
        self.seed = seed
        self.epoch = epoch
        self.start = start

    def permutation(self, epoch: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(stable_int("sampler", self.seed, epoch))
        return torch.randperm(self.n, generator=g)

    def __iter__(self) -> Iterator[int]:
        return iter(self.permutation(self.epoch)[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start


def evenly_spaced(n_total: int, k: int) -> list[int]:
    if n_total <= 0:
        return []
    k = min(k, n_total)
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, k)})
