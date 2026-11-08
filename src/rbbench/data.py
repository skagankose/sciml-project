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

H5_CHUNK_CACHE_BYTES = 64 * 1024 * 1024


def loader_settings(cfg: dict) -> dict:
    """DataLoader keyword arguments from `cfg["runtime"]`; RBBENCH_NUM_WORKERS overrides the worker count."""
    rt = cfg.get("runtime", {})
    workers = int(os.environ.get("RBBENCH_NUM_WORKERS") or rt.get("num_workers", 0))
    kwargs = {"num_workers": workers, "pin_memory": bool(rt.get("pin_memory", False))}
    if workers > 0:
        kwargs["prefetch_factor"] = int(rt.get("prefetch_factor", 2))
    return kwargs


def _persistent_well_dataset_class():
    """`WellDataset` that keeps each HDF5 file open per process and reads it with HDF5's native driver.

    Upstream `_load_one_sample` (the_well e63dec8) opens the file anew for every sample, through fsspec with
    a fresh 8 MB block cache (`IO_PARAMS`). On HPC3's BeeGFS that made training I/O-bound: 5.75 s per
    CNextU-Net step and 7.8 s per TFNO step at batch 64, against 1.42 s and 0.55 s of compute (2026-11-08).
    This subclass overrides only how the file is opened. The body below is upstream's `_load_one_sample`
    with the `with h5.File(self.fs.open(...))` replaced by a cached handle. Field reading, normalisation and
    sample assembly stay upstream's (`_reconstruct_fields`, `__getitem__`), so samples are identical
    (tests/test_fast_loader.py). Handles are reopened after a fork (DataLoader workers) by checking the pid.
    """
    cached = globals().get("_PERSISTENT_CLASS")
    if cached is not None:
        return cached
    from the_well.data.datasets import WellDataset
    from the_well.data.utils import maximum_stride_for_initial_index

    class PersistentWellDataset(WellDataset):
        def _h5_file(self, file_idx: int):
            pid = os.getpid()
            if getattr(self, "_h5_pid", None) != pid:
                self._h5_pid, self._h5_handles = pid, {}
            handle = self._h5_handles.get(file_idx)
            if handle is None:
                handle = h5py.File(self.files_paths[file_idx], "r", rdcc_nbytes=H5_CHUNK_CACHE_BYTES)
                self._h5_handles[file_idx] = handle
            return handle

        def __getstate__(self):  # never pickle open handles (spawned workers reopen their own)
            state = self.__dict__.copy()
            state.pop("_h5_handles", None)
            state.pop("_h5_pid", None)
            return state

        def _load_one_sample(self, index):  # upstream body; only the file opening differs
            if self.restriction_set is not None:
                index = self.restriction_set[index]
            file_idx = int(np.searchsorted(self.file_index_offsets, index, side="right") - 1)
            windows_per_trajectory = self.n_windows_per_trajectory[file_idx]
            local_idx = index - max(self.file_index_offsets[file_idx], 0)  # first offset is -1
            sample_idx = local_idx // windows_per_trajectory
            time_idx = local_idx % windows_per_trajectory
            file = self._h5_file(file_idx)
            dt = self.min_dt_stride
            if self.max_dt_stride > self.min_dt_stride:
                effective_max_dt = maximum_stride_for_initial_index(
                    time_idx, self.n_steps_per_trajectory[file_idx], self.n_steps_input, self.n_steps_output)
                effective_max_dt = min(effective_max_dt, self.max_dt_stride)
                if effective_max_dt > self.min_dt_stride:
                    dt = np.random.randint(self.min_dt_stride, effective_max_dt + 1)
            data = {}
            output_steps = min(self.n_steps_output, self.max_rollout_steps)
            if self.full_trajectory_mode and self.start_output_steps_at_t >= 0:
                time_idx = self.start_output_steps_at_t - (self.n_steps_input) * dt
            n_steps = self.n_steps_input + output_steps
            cache = self.caches[file_idx]
            data["variable_fields"], data["constant_fields"] = self._reconstruct_fields(
                file, cache, sample_idx, time_idx, n_steps, dt)
            data["variable_scalars"], data["constant_scalars"] = self._reconstruct_scalars(
                file, cache, sample_idx, time_idx, n_steps, dt)
            if self.boundary_return_type is not None:
                data["boundary_conditions"] = self._reconstruct_bcs(file, cache, sample_idx, time_idx, n_steps, dt)
            if self.return_grid:
                data["space_grid"], data["time_grid"] = self._reconstruct_grids(
                    file, cache, sample_idx, time_idx, n_steps, dt)
            return data, file_idx, sample_idx, time_idx, dt

    # Importable as rbbench.data.PersistentWellDataset, so spawned workers (macOS) can unpickle it.
    PersistentWellDataset.__module__ = __name__
    PersistentWellDataset.__qualname__ = "PersistentWellDataset"
    globals()["_PERSISTENT_CLASS"] = PersistentWellDataset
    return PersistentWellDataset


def __getattr__(name: str):
    if name == "PersistentWellDataset":
        return _persistent_well_dataset_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_well_dataset(
    dataset_dir: Path,
    split: str,
    n_steps_input: int = 4,
    n_steps_output: int = 1,
    use_normalization: bool = True,
    return_grid: bool = False,
    persistent: bool = True,
):
    """Upstream `WellDataset` over one split, z-score normalised with the dataset's stats.yaml.

    `persistent` (default) keeps files open per process (`_persistent_well_dataset_class`); samples are the
    same as upstream's, only faster on network storage.
    """
    from the_well.data.datasets import WellDataset
    from the_well.data.normalization import ZScoreNormalization
    from the_well.data.utils import flatten_field_names

    ds = (_persistent_well_dataset_class() if persistent else WellDataset)(
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
