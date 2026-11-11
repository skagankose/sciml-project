"""Grids, resolutions and spectral resampling for the numerical-solver baseline (NumPy and SciPy only).

The rayleigh_benard frames are Dedalus grid values at scale 1 (README at the pinned commit: "based on the software
Dedalus"; the setup is Dedalus's own `ivp_2d_rayleigh_benard` example at 512 x 128): x on the uniform grid
x_j = Lx j / Nx of a RealFourier basis, which has no Nyquist mode, and y on the Chebyshev-Gauss nodes of [0, Ly] in
ascending order (D18). This module must stay importable without torch or Dedalus: the solver runs in its own
environment (`rb-dedalus`) and imports it from there.

Changing resolution is done in spectral space, as Dedalus does when it changes a field's resolution: Fourier
truncation or zero-padding in x, Chebyshev truncation or zero-padding in y. It is exact for any field the smaller
basis can represent.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

FIELDS = ("buoyancy", "pressure", "velocity_x", "velocity_y")  # == rbbench.data.FIELDS (tests/test_solver.py)
NATIVE = (512, 128)
RESOLUTIONS = {"512x128": (512, 128), "256x64": (256, 64), "128x32": (128, 32)}
LX, LY = 4.0, 1.0


def resolution_name(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def x_grid(n: int, lx: float = LX) -> np.ndarray:
    return lx * np.arange(n) / n


def gauss_nodes(n: int, ly: float = LY) -> np.ndarray:
    """Chebyshev-Gauss nodes on [0, ly], ascending (Dedalus's ChebyshevT grid at scale 1)."""
    return ly * (1.0 - np.cos(np.pi * (np.arange(n) + 0.5) / n)) / 2.0


def resample_x(f: np.ndarray, n_new: int, axis: int = -2) -> np.ndarray:
    """Periodic values on x_j = Lx j / n -> the same function on n_new points, by Fourier truncation/padding.

    Modes up to (but excluding) the smaller grid's Nyquist mode are kept; Dedalus's RealFourier basis holds no
    Nyquist mode either.
    """
    f = np.asarray(f, np.float64)
    n = f.shape[axis]
    if n_new == n:
        return f.copy()
    c = np.fft.rfft(f, axis=axis)
    keep = min(n, n_new) // 2
    shape = list(c.shape)
    shape[axis] = n_new // 2 + 1
    d = np.zeros(shape, complex)
    sl = [slice(None)] * c.ndim
    sl[axis] = slice(0, keep)
    d[tuple(sl)] = c[tuple(sl)]
    return np.fft.irfft(d, n=n_new, axis=axis) * (n_new / n)


def resample_y(f: np.ndarray, n_new: int, axis: int = -1) -> np.ndarray:
    """Values on n ascending Chebyshev-Gauss nodes -> the same Chebyshev series on n_new such nodes.

    With theta_j = pi (j + 1/2) / n the ascending nodes are s_j = -cos(theta_j), and
    f(s_j) = sum_k b_k cos(k theta_j) with b_k = (-1)^k a_k for the Chebyshev coefficients a_k. A type-2 DCT gives
    the b_k (up to 2n) and a type-3 DCT evaluates them on the new nodes; the sign convention is the same on both
    grids, so it cancels.
    """
    f = np.asarray(f, np.float64)
    n = f.shape[axis]
    if n_new == n:
        return f.copy()
    coef = sfft.dct(f, type=2, axis=axis) / (2.0 * n)
    keep = min(n, n_new)
    shape = list(coef.shape)
    shape[axis] = n_new
    padded = np.zeros(shape)
    sl = [slice(None)] * coef.ndim
    sl[axis] = slice(0, keep)
    padded[tuple(sl)] = coef[tuple(sl)]
    return sfft.dct(padded, type=3, axis=axis)


def resample(f: np.ndarray, shape: tuple[int, int], x_axis: int = -2, y_axis: int = -1) -> np.ndarray:
    """Change the resolution of fields stored as [..., X, Y] (or with the axes given)."""
    return resample_y(resample_x(f, shape[0], axis=x_axis), shape[1], axis=y_axis)


def chebyshev_coefficients(f: np.ndarray, axis: int = -1) -> np.ndarray:
    """Chebyshev coefficients a_k of values on ascending Gauss nodes (for tests and diagnostics)."""
    f = np.asarray(f, np.float64)
    n = f.shape[axis]
    b = sfft.dct(f, type=2, axis=axis) / n
    idx = [slice(None)] * b.ndim
    idx[axis] = 0
    b[tuple(idx)] /= 2.0
    sign = (-1.0) ** np.arange(n)
    shape = [1] * b.ndim
    shape[axis] = n
    return b * sign.reshape(shape)


def rollout_frames(start: int, n_in: int, horizon: int) -> tuple[int, list[int]]:
    """The solver starts from the last input frame the models see and predicts the same target frames."""
    init = start + n_in - 1
    return init, list(range(start + n_in, start + n_in + horizon))


def solver_cohort(cohort: dict, per_regime: int = 1) -> list[dict]:
    """The first `per_regime` trajectories of each (Ra, Pr) regime in the frozen cohort, in cohort order."""
    seen: dict[tuple, int] = {}
    out = []
    for entry in cohort["trajectories"]:
        key = (entry["rayleigh"], entry["prandtl"])
        if seen.get(key, 0) < per_regime:
            seen[key] = seen.get(key, 0) + 1
            out.append(entry)
    return out


def rollout_name(traj_id: str, start: int) -> str:
    """File stem for one rollout: data/test/<file>.hdf5#<i> at start s -> <file>__<i>__s<s>."""
    path, index = traj_id.split("#")
    stem = path.rsplit("/", 1)[-1].removesuffix(".hdf5").removesuffix(".h5")
    return f"{stem}__{index}__s{start:03d}"
