"""Dedalus solver for rayleigh_benard, restarted from a stored frame (numerical baseline; runs in `rb-dedalus`).

The problem is Dedalus 3's `examples/ivp_2d_rayleigh_benard/rayleigh_benard.py` (v3.0.3), which is the setup The
Well's README describes for this dataset: Boussinesq equations with kappa = (Ra Pr)^-1/2, nu = (Ra/Pr)^-1/2,
b(y=0) = Ly, b(y=Ly) = 0, no-slip walls, integ(p) = 0, a 4 x 1 box, RealFourier x ChebyshevT bases with 3/2
dealiasing, RK222 and a CFL-limited step capped at 0.125. Only b and u carry state: p and the tau variables are
solved for in every stage, and RK222 keeps no history, so a restart from (b, u) is the same initial value problem.

The solver lands exactly on each output time (the frames are 0.25 apart) by shortening the last step before it.
A restart from a developed flow would otherwise take its first ~10 steps at `initial_dt` (Dedalus's CFL returns
it until its first update), so the initial step is computed from the initial velocity with the same safety factor.

Serial only (one MPI rank): the grid is then the global grid and runs are trivially parallel across rollouts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rbbench.solver_grid import LX, LY, gauss_nodes, x_grid

DEFAULTS = {
    "dealias": 1.5,
    "timestepper": "RK222",
    "max_dt": 0.125,
    "safety": 0.5,
    "threshold": 0.05,
    "cadence": 10,
    "max_change": 1.5,
    "min_change": 0.5,
}


@dataclass
class RolloutResult:
    frames: np.ndarray                       # [H, X, Y, 4]: buoyancy, pressure, velocity_x, velocity_y
    status: str                              # ok | diverged | timeout
    steps_done: int
    build_seconds: float
    step_seconds: float
    seconds_per_output: list[float] = field(default_factory=list)
    iterations_per_output: list[int] = field(default_factory=list)
    dt_min: float = np.inf                   # CFL steps (before shortening to land on an output time)
    dt_max: float = 0.0
    initial_dt: float = np.nan


class RBSolver:
    def __init__(self, rayleigh: float, prandtl: float, nx: int, ny: int, lx: float = LX, ly: float = LY,
                 **options):
        import dedalus.public as d3

        self.opts = {**DEFAULTS, **options}
        self.nx, self.ny, self.lx, self.ly = nx, ny, lx, ly
        t0 = time.perf_counter()
        coords = d3.CartesianCoordinates("x", "z")
        dist = d3.Distributor(coords, dtype=np.float64)
        if dist.comm.size != 1:
            raise RuntimeError("RBSolver runs serially (one MPI rank); parallelise across rollouts instead")
        dealias = self.opts["dealias"]
        xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, lx), dealias=dealias)
        zbasis = d3.ChebyshevT(coords["z"], size=ny, bounds=(0, ly), dealias=dealias)

        p = dist.Field(name="p", bases=(xbasis, zbasis))
        b = dist.Field(name="b", bases=(xbasis, zbasis))
        u = dist.VectorField(coords, name="u", bases=(xbasis, zbasis))
        tau_p = dist.Field(name="tau_p")
        tau_b1 = dist.Field(name="tau_b1", bases=xbasis)
        tau_b2 = dist.Field(name="tau_b2", bases=xbasis)
        tau_u1 = dist.VectorField(coords, name="tau_u1", bases=xbasis)
        tau_u2 = dist.VectorField(coords, name="tau_u2", bases=xbasis)

        kappa = (rayleigh * prandtl) ** (-1 / 2)
        nu = (rayleigh / prandtl) ** (-1 / 2)
        Lz = ly
        _, ez = coords.unit_vector_fields(dist)
        lift_basis = zbasis.derivative_basis(1)
        def lift(A):
            return d3.Lift(A, lift_basis, -1)

        grad_u = d3.grad(u) + ez * lift(tau_u1)
        grad_b = d3.grad(b) + ez * lift(tau_b1)
        namespace = {"p": p, "b": b, "u": u, "tau_p": tau_p, "tau_b1": tau_b1, "tau_b2": tau_b2,
                     "tau_u1": tau_u1, "tau_u2": tau_u2, "kappa": kappa, "nu": nu, "Lz": Lz, "ez": ez,
                     "lift": lift, "grad_u": grad_u, "grad_b": grad_b}  # Dedalus adds its operators after these
        problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=namespace)
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
        problem.add_equation("b(z=0) = Lz")
        problem.add_equation("u(z=0) = 0")
        problem.add_equation("b(z=Lz) = 0")
        problem.add_equation("u(z=Lz) = 0")
        problem.add_equation("integ(p) = 0")
        self.solver = problem.build_solver(getattr(d3, self.opts["timestepper"]))
        self.d3, self.dist, self.p, self.b, self.u = d3, dist, p, b, u
        self.xbasis, self.zbasis = xbasis, zbasis
        self.rayleigh, self.prandtl, self.kappa, self.nu = rayleigh, prandtl, kappa, nu

        # the grid at scale 1 must be the one the data is stored on (or its resolution's equivalent)
        x, z = dist.local_grids(xbasis, zbasis)
        x, z = np.ravel(x), np.ravel(z)
        if np.abs(x - x_grid(nx, lx)).max() > 1e-12:
            raise RuntimeError("Dedalus x grid differs from Lx j / Nx")
        self.z_order = np.argsort(z)
        if np.abs(z[self.z_order] - gauss_nodes(ny, ly)).max() > 1e-12:
            raise RuntimeError("Dedalus z grid is not the Chebyshev-Gauss nodes")
        self.z_inverse = np.argsort(self.z_order)
        self.build_seconds = time.perf_counter() - t0
        self.used = False

    # --- state in the data's layout: [X, Y] with y ascending ---------------------------------------------
    def set_state(self, buoyancy: np.ndarray, velocity_x: np.ndarray, velocity_y: np.ndarray) -> None:
        for f in (self.b, self.u, self.p):
            f.change_scales(1)
        self.b["g"] = np.asarray(buoyancy, np.float64)[:, self.z_inverse]
        self.u["g"][0] = np.asarray(velocity_x, np.float64)[:, self.z_inverse]
        self.u["g"][1] = np.asarray(velocity_y, np.float64)[:, self.z_inverse]
        self.p["g"] = 0.0

    def get_state(self) -> np.ndarray:
        for f in (self.b, self.u, self.p):
            f.change_scales(1)
        o = self.z_order
        return np.stack([self.b["g"][:, o], self.p["g"][:, o], self.u["g"][0][:, o], self.u["g"][1][:, o]], axis=-1)

    def cfl_dt(self, velocity_x: np.ndarray, velocity_y: np.ndarray) -> float:
        """Advective CFL step of a velocity on the dealiased grid, approximated from the scale-1 grid values."""
        dealias = self.opts["dealias"]
        dx = self.lx / (self.nx * dealias)
        zg = gauss_nodes(round(self.ny * dealias), self.ly)
        dz = np.diff(zg).min()
        freq = np.abs(velocity_x).max() / dx + np.abs(velocity_y).max() / dz
        return float(min(self.opts["max_dt"], self.opts["safety"] / freq)) if freq > 0 else self.opts["max_dt"]

    def rollout(self, buoyancy: np.ndarray, velocity_x: np.ndarray, velocity_y: np.ndarray,
                output_dt: float | list[float], n_outputs: int, max_seconds: float | None = None,
                check_every: int = 50) -> RolloutResult:
        """Integrate from the given state and return the frames at the n_outputs output times.

        One rollout per solver: Dedalus's CFL keeps a frequency handler and its first-step logic keys on the
        solver's initial iteration, so a reused solver could take its first step from the previous run's flow.
        """
        if self.used:
            raise RuntimeError("RBSolver.rollout runs once per solver; build a new RBSolver for the next rollout")
        self.used = True
        d3, solver, o = self.d3, self.solver, self.opts
        self.set_state(buoyancy, velocity_x, velocity_y)
        solver.sim_time = 0.0
        initial_dt = 0.5 * self.cfl_dt(velocity_x, velocity_y)  # conservative: the CFL grows it within ~30 steps
        cfl = d3.CFL(solver, initial_dt=initial_dt, cadence=o["cadence"], safety=o["safety"],
                     threshold=o["threshold"], max_change=o["max_change"], min_change=o["min_change"],
                     max_dt=o["max_dt"])
        cfl.add_velocity(self.u)
        gaps = [output_dt] * n_outputs if np.isscalar(output_dt) else list(output_dt)
        targets = np.cumsum(gaps)
        frames = np.full((n_outputs, self.nx, self.ny, 4), np.nan)
        res = RolloutResult(frames=frames, status="ok", steps_done=0, build_seconds=self.build_seconds,
                            step_seconds=0.0, initial_dt=initial_dt)
        t_start = time.perf_counter()
        for k, target in enumerate(targets):
            t_out, it0 = time.perf_counter(), solver.iteration
            while solver.sim_time < target * (1 - 1e-12):
                dt_cfl = cfl.compute_timestep()
                solver.step(min(dt_cfl, target - solver.sim_time))
                res.dt_min, res.dt_max = min(res.dt_min, dt_cfl), max(res.dt_max, dt_cfl)
                if (solver.iteration % check_every == 0) and not np.isfinite(self.b["g"]).all():
                    res.status = "diverged"
                    break
                if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
                    res.status = "timeout"
                    break
            if res.status != "ok":
                break
            state = self.get_state()
            if not np.isfinite(state).all():
                res.status = "diverged"
                break
            frames[k] = state
            res.steps_done = k + 1
            res.seconds_per_output.append(time.perf_counter() - t_out)
            res.iterations_per_output.append(solver.iteration - it0)
        res.step_seconds = time.perf_counter() - t_start
        return res


def standard_initial_condition(nx: int, ny: int, seed: int = 42, scale: float = 1e-3, db0: float = 1.0,
                               lx: float = LX, ly: float = LY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dataset's initial condition, b = (Ly - y) db0 + y (Ly - y) noise, u = 0 (for self-tests; the noise
    differs from Dedalus's fill_random)."""
    rng = np.random.default_rng(seed)
    y = gauss_nodes(ny, ly)[None, :]
    b = (ly - y) * db0 + y * (ly - y) * scale * rng.standard_normal((nx, ny))
    return b, np.zeros((nx, ny)), np.zeros((nx, ny))
