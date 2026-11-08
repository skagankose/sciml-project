"""Physics diagnostics on physical-unit fields shaped [..., X, Y].

x is horizontal and periodic (uniform grid); y is wall-bounded on Chebyshev nodes. Vertical
derivatives and integrals use the Chebyshev interpolant through the y nodes (spectral
accuracy, matching the Dedalus discretisation), never unweighted index differences.

Conventions (rayleigh_benard README, to be confirmed by scripts/inspect_dataset.py):
  b(y=0) = b_bottom = Ly, b(y=Ly) = b_top = 0, kappa = (Ra Pr)^(-1/2), Delta b = b_bottom - b_top.
  Nu = < u_y b - kappa db/dy > / (kappa Delta b / Ly)   (volume average; 1 for pure conduction)
  Nu_wall = -d<b>_x/dy |_wall / (Delta b / Ly)          (independent check, at each wall)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as C

# --- y grid -------------------------------------------------------------------------------

def _gauss(n):
    return -np.cos(np.pi * (np.arange(n) + 0.5) / n)


def _lobatto(n):
    return -np.cos(np.pi * np.arange(n) / (n - 1))


Y_FAMILIES = {"gauss": _gauss, "lobatto": _lobatto}


def y_nodes(y_stored: np.ndarray, cfg_physics: dict) -> np.ndarray:
    """The y coordinates the field values actually sit on (decisions D18).

    `y_grid: auto` trusts the file's `dimensions/y`. `gauss` or `lobatto` generates that Chebyshev family on
    [0, Ly] instead: rayleigh_benard stores a placeholder linspace(0, 1, 128) in `dimensions/y`, although its
    values are on Chebyshev-Gauss nodes (scripts/probe_y_grid.py on the real data, 2026-11-07).
    """
    y_stored = np.asarray(y_stored, np.float64)
    family = cfg_physics.get("y_grid", "auto")
    if family == "auto":
        return y_stored
    if family not in Y_FAMILIES:
        raise ValueError(f"physics.y_grid must be auto, gauss or lobatto, not {family!r}")
    return cfg_physics["Ly"] * (Y_FAMILIES[family](y_stored.size) + 1) / 2


def stored_y_status(y_stored: np.ndarray, cfg_physics: dict, tol: float) -> str:
    """How the file's y relates to the configured family: "matches", "placeholder" (uniform on [0, Ly]) or
    "unexpected". Only "unexpected" is a problem when the family is set explicitly."""
    y_stored = np.asarray(y_stored, np.float64)
    if np.abs(y_stored - y_nodes(y_stored, cfg_physics)).max() <= tol:
        return "matches"
    if np.abs(y_stored - np.linspace(0.0, cfg_physics["Ly"], y_stored.size)).max() <= tol:
        return "placeholder"
    return "unexpected"


def classify_y_grid(y: np.ndarray, a: float = 0.0, b: float | None = None) -> dict:
    """Compare y to Chebyshev-Gauss, Chebyshev-Lobatto and uniform grids on [a, b].

    If `b` is None it is estimated per family (Lobatto: y[-1]; Gauss: from the end nodes).
    """
    y = np.asarray(y, np.float64)
    n = y.size
    out = {"n": int(n), "monotonic_increasing": bool(np.all(np.diff(y) > 0))}
    candidates = {}
    for name, ref in (("gauss", _gauss(n)), ("lobatto", _lobatto(n)), ("uniform", None)):
        if ref is None:
            fitted = np.linspace(y[0], y[-1], n)
            lo, hi = float(y[0]), float(y[-1])
        else:
            # least-squares affine map of the reference nodes onto y gives the interval
            A = np.stack([np.ones(n), ref], 1)
            (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
            fitted = c0 + c1 * ref
            lo, hi = float(c0 - c1), float(c0 + c1)
        if b is not None and ref is not None:
            lo, hi = a, b
            fitted = lo + (hi - lo) * (ref + 1) / 2
        candidates[name] = {
            "max_abs_dev": float(np.max(np.abs(fitted - y))),
            "interval": [lo, hi],
        }
    best = min(candidates, key=lambda k: candidates[k]["max_abs_dev"])
    out.update(candidates=candidates, best=best)
    return out


@dataclass
class ChebyshevY:
    """Interpolation-based operators for samples at Chebyshev nodes y in [a, b]."""

    y: np.ndarray
    a: float
    b: float

    def __post_init__(self):
        self.y = np.asarray(self.y, np.float64)
        n = self.y.size
        s = 2 * (self.y - self.a) / (self.b - self.a) - 1  # map to [-1, 1]
        V = C.chebvander(s, n - 1)  # V[i, j] = T_j(s_i)
        self._Vinv = np.linalg.inv(V)  # coefficients = Vinv @ values
        dV = np.stack([C.chebval(s, C.chebder(np.eye(n)[j])) for j in range(n)], 1)
        self.D = dV @ self._Vinv * (2 / (self.b - self.a))  # d/dy at the nodes
        # Exact integral of each T_j over [-1, 1]: 2/(1-j^2) for even j, 0 for odd j.
        j = np.arange(n)
        tint = np.where(j % 2 == 0, 2.0 / (1 - j.astype(float) ** 2 + (j % 2 == 1)), 0.0)
        self.weights = (tint @ self._Vinv) * (self.b - self.a) / 2  # quadrature on [a, b]
        self._wall_value = {
            "bottom": C.chebvander(np.array([-1.0]), n - 1)[0] @ self._Vinv,
            "top": C.chebvander(np.array([1.0]), n - 1)[0] @ self._Vinv,
        }
        dwall = lambda s0: np.array([C.chebval(s0, C.chebder(np.eye(n)[k])) for k in range(n)])
        self._wall_deriv = {
            "bottom": dwall(-1.0) @ self._Vinv * (2 / (self.b - self.a)),
            "top": dwall(1.0) @ self._Vinv * (2 / (self.b - self.a)),
        }

    def ddy(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(f, np.float64) @ self.D.T

    def integrate(self, f: np.ndarray) -> np.ndarray:
        """Integral over y in [a, b] along the last axis."""
        return np.asarray(f, np.float64) @ self.weights

    def mean(self, f: np.ndarray) -> np.ndarray:
        return self.integrate(f) / (self.b - self.a)

    def wall_value(self, f: np.ndarray, wall: str) -> np.ndarray:
        return np.asarray(f, np.float64) @ self._wall_value[wall]

    def wall_ddy(self, f: np.ndarray, wall: str) -> np.ndarray:
        return np.asarray(f, np.float64) @ self._wall_deriv[wall]


# --- Nusselt ------------------------------------------------------------------------------

def kappa_from(rayleigh, prandtl):
    return (np.asarray(rayleigh, np.float64) * np.asarray(prandtl, np.float64)) ** -0.5


def nusselt(b, uy, ops: ChebyshevY, kappa, delta_b: float = 1.0) -> dict:
    """Volume and wall Nusselt numbers for fields [..., X, Y] (x-average is a plain mean)."""
    b = np.asarray(b, np.float64)
    uy = np.asarray(uy, np.float64)
    ly = ops.b - ops.a
    kappa = np.asarray(kappa, np.float64)
    scale = kappa * delta_b / ly
    conv = ops.mean((uy * b).mean(axis=-2))
    cond = ops.mean((-kappa[..., None] if kappa.ndim else -kappa) * ops.ddy(b).mean(axis=-2))
    b_profile = b.mean(axis=-2)
    return {
        "nu_volume": (conv + cond) / scale,
        "nu_conductive_part": cond / scale,  # == 1 exactly up to quadrature/derivative error
        "nu_convective_part": conv / scale,
        "nu_wall_bottom": -ops.wall_ddy(b_profile, "bottom") / (delta_b / ly),
        "nu_wall_top": -ops.wall_ddy(b_profile, "top") / (delta_b / ly),
        "b_wall_bottom": ops.wall_value(b_profile, "bottom"),
        "b_wall_top": ops.wall_value(b_profile, "top"),
    }


# --- horizontal kinetic-energy spectrum ---------------------------------------------------

def kinetic_energy_spectrum(ux, uy, ops: ChebyshevY, lx: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided horizontal spectrum E(k_x), vertically averaged with Chebyshev quadrature.

    Normalised so that sum_k E(k) equals the domain-mean kinetic energy 0.5 <u_x^2 + u_y^2>
    (Parseval). Only x is Fourier-transformed; y is never treated as periodic.
    """
    ux = np.asarray(ux, np.float64)
    uy = np.asarray(uy, np.float64)
    nx = ux.shape[-2]
    fx = np.fft.rfft(ux, axis=-2) / nx
    fy = np.fft.rfft(uy, axis=-2) / nx
    e = 0.5 * (np.abs(fx) ** 2 + np.abs(fy) ** 2)  # [..., K, Y]
    mult = np.full(e.shape[-2], 2.0)
    mult[0] = 1.0
    if nx % 2 == 0:
        mult[-1] = 1.0
    e = e * mult[:, None]
    spectrum = ops.mean(e)  # [..., K]
    k = 2 * np.pi * np.arange(e.shape[-2]) / lx
    return k, spectrum


def mean_kinetic_energy(ux, uy, ops: ChebyshevY) -> np.ndarray:
    ke = 0.5 * (np.asarray(ux, np.float64) ** 2 + np.asarray(uy, np.float64) ** 2)
    return ops.mean(ke.mean(axis=-2))


def log_spectral_error(e_pred, e_true, floor: float = 1e-12, skip_mean_mode: bool = True) -> dict:
    """Mean |log10 E_pred - log10 E_true| over bins where both spectra exceed `floor`."""
    e_pred = np.asarray(e_pred, np.float64)
    e_true = np.asarray(e_true, np.float64)
    valid = (e_true > floor) & (e_pred > floor)
    if skip_mean_mode:
        valid[..., 0] = False
    diff = np.abs(np.log10(np.where(valid, e_pred, 1.0)) - np.log10(np.where(valid, e_true, 1.0)))
    n_valid = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where(n_valid > 0, (diff * valid).sum(axis=-1) / np.maximum(n_valid, 1), np.nan)
    return {"log_spectral_error": err, "n_bins_used": n_valid, "n_bins_excluded": valid.shape[-1] - n_valid}
