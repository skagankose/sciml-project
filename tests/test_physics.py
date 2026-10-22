import numpy as np
import pytest

from rbbench.metrics import physics
from rbbench.synthetic import (
    analytic_fields,
    analytic_nusselt,
    chebyshev_gauss_nodes,
    chebyshev_lobatto_nodes,
)

PARAMS = dict(a=0.1, c=0.01, d=0.03, m=2, omega=0.5, phi=0.3, Lx=4.0, beta=-0.2)


@pytest.fixture(params=[chebyshev_gauss_nodes, chebyshev_lobatto_nodes], ids=["gauss", "lobatto"])
def ops(request):
    return physics.ChebyshevY(request.param(64), 0.0, 1.0)


def test_grid_classification():
    assert physics.classify_y_grid(chebyshev_gauss_nodes(128))["best"] == "gauss"
    assert physics.classify_y_grid(chebyshev_lobatto_nodes(128))["best"] == "lobatto"
    assert physics.classify_y_grid(np.linspace(0, 1, 128))["best"] == "uniform"


def test_nonuniform_quadrature_and_derivative(ops):
    y = ops.y
    for p in range(10):
        assert ops.integrate(y**p) == pytest.approx(1 / (p + 1), abs=1e-12)
    assert ops.integrate(np.exp(y)) == pytest.approx(np.e - 1, abs=1e-12)
    assert np.abs(ops.ddy(np.sin(3 * y)) - 3 * np.cos(3 * y)).max() < 1e-8
    # unweighted index averaging would be wrong on a Chebyshev grid
    assert abs(np.mean(y**2) - 1 / 3) > 1e-3


def test_wall_values_and_gradients(ops):
    f = 1 - ops.y + 0.3 * np.sin(2 * np.pi * ops.y)
    assert ops.wall_value(f, "bottom") == pytest.approx(1.0, abs=1e-10)
    assert ops.wall_value(f, "top") == pytest.approx(0.0, abs=1e-10)
    assert ops.wall_ddy(f, "bottom") == pytest.approx(-1 + 0.6 * np.pi, abs=1e-8)


def test_conduction_state_has_unit_nusselt(ops):
    x = np.arange(32) * 4 / 32
    b = np.broadcast_to(1 - ops.y, (32, ops.y.size))
    nu = physics.nusselt(b, np.zeros_like(b), ops, kappa=1e-3)
    for key in ("nu_volume", "nu_wall_bottom", "nu_wall_top", "nu_conductive_part"):
        assert nu[key] == pytest.approx(1.0, abs=1e-10), key
    assert x.size == 32


def test_analytic_convective_nusselt(ops):
    x = np.arange(128) * 4 / 128
    f = analytic_fields(x, ops.y, [0.0, 0.7], PARAMS)
    ra, pr = 1e6, 1.0
    nu = physics.nusselt(f["buoyancy"], f["velocity_y"], ops, physics.kappa_from(ra, pr))
    np.testing.assert_allclose(nu["nu_volume"], analytic_nusselt(PARAMS, ra, pr), rtol=1e-8)
    np.testing.assert_allclose(nu["nu_conductive_part"], 1.0, atol=1e-10)
    # beta sets the wall value: Nu_wall = 1 - 2 pi beta
    np.testing.assert_allclose(nu["nu_wall_bottom"], 1 - 2 * np.pi * PARAMS["beta"], rtol=1e-8)


def test_spectrum_parseval_and_horizontal_only(ops):
    rng = np.random.default_rng(0)
    ux = rng.standard_normal((3, 64, ops.y.size))
    uy = rng.standard_normal((3, 64, ops.y.size))
    k, e = physics.kinetic_energy_spectrum(ux, uy, ops, lx=4.0)
    assert e.shape == (3, 33) and k[1] == pytest.approx(2 * np.pi / 4)
    np.testing.assert_allclose(e.sum(-1), physics.mean_kinetic_energy(ux, uy, ops), rtol=1e-12)


def test_spectrum_peak_at_forcing_mode(ops):
    x = np.arange(64) * 4 / 64
    f = analytic_fields(x, ops.y, [0.0], PARAMS)
    _, e = physics.kinetic_energy_spectrum(f["velocity_x"], f["velocity_y"], ops, 4.0)
    assert int(e[0].argmax()) == PARAMS["m"]


def test_log_spectral_error_excludes_zero_bins():
    e_true = np.array([1.0, 1e-3, 0.0, 1e-2])
    e_pred = np.array([5.0, 1e-2, 1e-4, 0.0])
    out = physics.log_spectral_error(e_pred, e_true)
    assert out["n_bins_used"] == 1 and out["log_spectral_error"] == pytest.approx(1.0)
