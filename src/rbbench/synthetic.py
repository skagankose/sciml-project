"""Deterministic synthetic data in The Well HDF5 format, for tests and Mac smoke runs.

Fields are analytic convection rolls on a periodic-x / Chebyshev-y grid that satisfy the RB
boundary conditions of the real dataset (b = 1 at y = 0, b = 0 at y = Ly = 1, u = 0 at walls),
so physics diagnostics have closed-form answers:

    theta = k x - omega t + phi,                  k = 2 pi m / Lx
    psi   = a sin^2(pi y) sin(theta)              (stream function)
    u_x   = d psi/dy =  a pi sin(2 pi y) sin(theta)
    u_y   = -d psi/dx = -a k sin^2(pi y) cos(theta)
    b     = (1 - y) + beta sin(2 pi y) - c sin(pi y) cos(theta)
    p     = d sin(pi y) cos(2 theta)

Volume-averaged Nusselt number (Delta b = 1, Ly = 1):
    Nu = 1 + <u_y b> / kappa = 1 + 2 a k c / (3 pi kappa),   kappa = (Ra Pr)^(-1/2).
The mean-profile term with beta = -(Nu - 1) / (2 pi) makes the wall Nusselt number
-d<b>/dy at both walls equal the volume value, as in a statistically steady flow.
This is NOT a solution of the Boussinesq equations; it only exercises shapes and diagnostics.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import yaml

DATASET_NAME = "rayleigh_benard"
DEFAULT_REGIMES = ((1.0e6, 1.0), (1.0e7, 1.0), (1.0e6, 10.0), (1.0e7, 10.0))
DEFAULT_TRAJECTORIES = {"train": 6, "valid": 2, "test": 2}
SPLIT_SEED_OFFSET = {"train": 0, "valid": 1000, "test": 2000}


def chebyshev_gauss_nodes(n: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev-Gauss (roots of T_n) nodes on [a, b], ascending, walls excluded."""
    x = -np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return a + (b - a) * (x + 1) / 2


def chebyshev_lobatto_nodes(n: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto (extrema) nodes on [a, b], ascending, walls included."""
    x = -np.cos(np.pi * np.arange(n) / (n - 1))
    return a + (b - a) * (x + 1) / 2


def analytic_fields(x, y, t, params: dict) -> dict[str, np.ndarray]:
    """Fields on the meshgrid of 1-D `x`, `y` at times `t`; arrays shaped [T, X, Y]."""
    a, c, d, m, omega, phi, lx, beta = (
        params[k] for k in ("a", "c", "d", "m", "omega", "phi", "Lx", "beta")
    )
    k = 2 * math.pi * m / lx
    tt, xx, yy = np.meshgrid(np.asarray(t), np.asarray(x), np.asarray(y), indexing="ij")
    theta = k * xx - omega * tt + phi
    s = np.sin(np.pi * yy)
    return {
        "buoyancy": (1 - yy) + beta * np.sin(2 * np.pi * yy) - c * s * np.cos(theta),
        "pressure": d * s * np.cos(2 * theta),
        "velocity_x": a * np.pi * np.sin(2 * np.pi * yy) * np.sin(theta),
        "velocity_y": -a * k * s**2 * np.cos(theta),
    }


def analytic_nusselt(params: dict, rayleigh: float, prandtl: float) -> float:
    kappa = (rayleigh * prandtl) ** -0.5
    k = 2 * math.pi * params["m"] / params["Lx"]
    return 1 + 2 * params["a"] * k * params["c"] / (3 * math.pi * kappa)


def trajectory_params(rng: np.random.Generator, rayleigh: float, prandtl: float, lx: float) -> dict:
    kappa = (rayleigh * prandtl) ** -0.5
    a = float(rng.uniform(0.05, 0.2))
    m = int(rng.integers(1, 4))
    nu = float(rng.uniform(1.5, 4.0))
    k = 2 * math.pi * m / lx
    return {
        "a": a,
        "c": (nu - 1) * 3 * math.pi * kappa / (2 * a * k),
        "d": float(rng.uniform(0.01, 0.05)),
        "m": m,
        "omega": float(rng.uniform(0.2, 1.0)),
        "phi": float(rng.uniform(0, 2 * math.pi)),
        "Lx": lx,
        "beta": -(nu - 1) / (2 * math.pi),
    }


def file_name(rayleigh: float, prandtl: float) -> str:
    return f"{DATASET_NAME}_Rayleigh_{rayleigh:.0e}_Prandtl_{prandtl:g}.hdf5"


def write_file(
    path: Path,
    rayleigh: float,
    prandtl: float,
    n_traj: int,
    nx: int,
    ny: int,
    n_steps: int,
    dt: float,
    seed: int,
    noise: float,
    lx: float = 4.0,
    ly: float = 1.0,
    y_grid: str = "gauss",
) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = (np.arange(nx) * lx / nx).astype(np.float32)
    y_nodes = chebyshev_gauss_nodes if y_grid == "gauss" else chebyshev_lobatto_nodes
    y = y_nodes(ny, 0.0, ly).astype(np.float32)
    t = (np.arange(n_steps) * dt).astype(np.float32)
    b = np.empty((n_traj, n_steps, nx, ny), np.float32)
    p = np.empty_like(b)
    u = np.empty((n_traj, n_steps, nx, ny, 2), np.float32)
    params = []
    for i in range(n_traj):
        prm = trajectory_params(rng, rayleigh, prandtl, lx)
        params.append(prm)
        f = analytic_fields(x, y, t, prm)
        jitter = lambda shape: noise * rng.standard_normal(shape)
        b[i] = f["buoyancy"] + jitter(b[i].shape)
        p[i] = f["pressure"] + jitter(p[i].shape)
        u[i, ..., 0] = f["velocity_x"] + jitter(b[i].shape)
        u[i, ..., 1] = f["velocity_y"] + jitter(b[i].shape)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["dataset_name"] = DATASET_NAME
        h.attrs["grid_type"] = "cartesian"
        h.attrs["n_spatial_dims"] = 2
        h.attrs["n_trajectories"] = n_traj
        h.attrs["simulation_parameters"] = ["Rayleigh", "Prandtl"]
        h.attrs["Rayleigh"] = rayleigh
        h.attrs["Prandtl"] = prandtl
        h.attrs["synthetic"] = True

        dims = h.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        for key, val in (("time", t), ("x", x), ("y", y)):
            ds = dims.create_dataset(key, data=val)
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"] = False

        bcs = h.create_group("boundary_conditions")
        x_mask = np.zeros(nx, bool)
        x_mask[[0, -1]] = True
        y_mask = np.zeros(ny, bool)
        y_mask[[0, -1]] = True
        for name, dim, kind, mask in (
            ("x_periodic", "x", "PERIODIC", x_mask),
            ("y_wall", "y", "WALL", y_mask),
        ):
            g = bcs.create_group(name)
            g.attrs["associated_dims"] = [dim]
            g.attrs["associated_fields"] = []
            g.attrs["bc_type"] = kind
            g.attrs["sample_varying"] = False
            g.attrs["time_varying"] = False
            g.create_dataset("mask", data=mask)

        scalars = h.create_group("scalars")
        scalars.attrs["field_names"] = ["Rayleigh", "Prandtl"]
        for key, val in (("Rayleigh", rayleigh), ("Prandtl", prandtl)):
            ds = scalars.create_dataset(key, data=np.array(val, dtype=np.float32))
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"] = False

        t0 = h.create_group("t0_fields")
        t0.attrs["field_names"] = ["buoyancy", "pressure"]
        for key, val in (("buoyancy", b), ("pressure", p)):
            ds = t0.create_dataset(key, data=val)
            ds.attrs["dim_varying"] = [True, True]
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True
        t1 = h.create_group("t1_fields")
        t1.attrs["field_names"] = ["velocity"]
        ds = t1.create_dataset("velocity", data=u)
        ds.attrs["dim_varying"] = [True, True]
        ds.attrs["sample_varying"] = True
        ds.attrs["time_varying"] = True
        h.create_group("t2_fields").attrs["field_names"] = []
    return params


def write_stats(dataset_dir: Path, split: str = "train") -> dict:
    """stats.yaml in the upstream layout, computed from the given split."""
    acc: dict[str, list] = {"buoyancy": [], "pressure": [], "velocity": []}
    deltas: dict[str, list] = {"buoyancy": [], "pressure": [], "velocity": []}
    for path in sorted((dataset_dir / "data" / split).glob("*.hdf5")):
        with h5py.File(path, "r") as h:
            for name, arr in (
                ("buoyancy", h["t0_fields/buoyancy"][...]),
                ("pressure", h["t0_fields/pressure"][...]),
                ("velocity", h["t1_fields/velocity"][...]),
            ):
                acc[name].append(arr)
                deltas[name].append(np.diff(arr, axis=1))

    def summarize(arrays: list, vector: bool) -> tuple:
        a = np.concatenate([x.reshape(-1, 2) if vector else x.reshape(-1) for x in arrays])
        mean = a.mean(0)
        std = a.std(0)
        rms = np.sqrt((a**2).mean(0))
        conv = (lambda v: [float(e) for e in v]) if vector else float
        return conv(mean), conv(std), conv(rms)

    stats: dict[str, dict] = {k: {} for k in ("mean", "std", "rms", "mean_delta", "std_delta", "rms_delta")}
    for name in acc:
        vector = name == "velocity"
        stats["mean"][name], stats["std"][name], stats["rms"][name] = summarize(acc[name], vector)
        (
            stats["mean_delta"][name],
            stats["std_delta"][name],
            stats["rms_delta"][name],
        ) = summarize(deltas[name], vector)
    with open(dataset_dir / "stats.yaml", "w") as f:
        yaml.safe_dump(stats, f)
    return stats


def make_synthetic_dataset(
    root: str | Path,
    nx: int = 64,
    ny: int = 32,
    n_steps: int = 40,
    regimes=DEFAULT_REGIMES,
    trajectories: dict | None = None,
    dt: float = 0.25,
    seed: int = 0,
    noise: float = 1e-5,
    y_grid: str = "gauss",
) -> Path:
    """Write <root>/datasets/rayleigh_benard/{stats.yaml,data/<split>/*.hdf5}; return the dataset dir."""
    trajectories = trajectories or DEFAULT_TRAJECTORIES
    dataset_dir = Path(root) / "datasets" / DATASET_NAME
    for split, n_traj in trajectories.items():
        for j, (ra, pr) in enumerate(regimes):
            write_file(
                dataset_dir / "data" / split / file_name(ra, pr),
                ra, pr, n_traj, nx, ny, n_steps, dt,
                seed=seed + SPLIT_SEED_OFFSET[split] + j,
                noise=noise, y_grid=y_grid,
            )
    write_stats(dataset_dir)
    return dataset_dir
