"""Deterministic perturbation of the observed initial window (robustness evaluation only).

Noise is drawn on the CPU from a generator seeded by a stable hash of
(trajectory id, start index, sigma, noise seed, field), so every model, worker count and
batch order sees identical draws. It is added in training-normalised units to the four
input frames only; targets and later predicted steps are never perturbed.
"""

from __future__ import annotations

import numpy as np
import torch

from rbbench.manifests import stable_int


def sigma_key(sigma: float) -> str:
    return f"{float(sigma):.6g}"


def noise_seed(traj_id: str, start: int, sigma: float, seed: int, field: int) -> int:
    return stable_int("noise", traj_id, int(start), sigma_key(sigma), int(seed), int(field))


def perturb_window(
    window: torch.Tensor,
    traj_id: str,
    start: int,
    sigma: float,
    seed: int,
    keep_y: np.ndarray | None = None,
) -> torch.Tensor:
    """Return a perturbed copy of one normalised window `[T, X, Y, F]`.

    `keep_y` (bool mask over Y) marks rows left unchanged, e.g. grid points lying exactly on
    a wall where boundary values are prescribed. With Chebyshev-Gauss nodes no point lies on
    a wall and the mask is empty.
    """
    if sigma == 0:
        return window.clone()
    t, nx, ny, nf = window.shape
    out = window.clone()
    for f in range(nf):
        g = torch.Generator().manual_seed(noise_seed(traj_id, start, sigma, seed, f))
        eps = torch.randn((t, nx, ny), generator=g, dtype=torch.float32)
        if keep_y is not None and np.any(keep_y):
            eps[..., torch.as_tensor(np.asarray(keep_y, bool))] = 0.0
        out[..., f] += float(sigma) * eps.to(out.device, out.dtype)
    return out


def wall_rows(y: np.ndarray, a: float, b: float, tol: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, np.float64)
    return (np.abs(y - a) < tol) | (np.abs(y - b) < tol)
