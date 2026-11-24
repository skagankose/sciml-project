"""rbbench.uq_analysis on synthetic samples where the answer is known (calibrated or shifted ensembles)."""

import numpy as np
import pandas as pd
import pytest

from rbbench import uq_analysis as ua

F = len(ua.FIELDS)


def _cells(err, std=None, steps=(1,), n_traj=20, starts=(80,)):
    """err [M, N, K, F] with N = n_traj * len(starts) * len(steps) windows in that order."""
    index = np.array([(t, s, k) for t in range(n_traj) for s in starts for k in steps], np.int32)
    ids = np.array([f"data/test/rb_Rayleigh_{'1e10' if t % 5 == 0 else '1e8'}_Prandtl_1.hdf5#{t}" for t in range(n_traj)])
    assert err.shape[1] == len(index)
    return {"err": err.astype(np.float32), "std": None if std is None else std.astype(np.float32), "index": index,
            "traj_ids": ids, "members": np.array([f"s{i}" for i in range(err.shape[0])])}


def _calibrated(rng, m, n_traj=40, k=64, steps=(1,), scale=1.0, member_scale=None):
    """Members' means mu_m = truth-centred noise of the same size as the truth's spread: a perfect ensemble.
    y = c + e, mu_m = c + e_m with e ~ N(0, scale^2), e_m ~ N(0, member_scale^2) (default: scale): err_m = y - mu_m."""
    n = n_traj * len(steps)
    c = rng.normal(size=(n, k, F))
    y = c + rng.normal(scale=scale, size=(n, k, F))
    mu = c[None] + rng.normal(scale=scale if member_scale is None else member_scale, size=(m, n, k, F))
    return y[None] - mu, _cells(y[None] - mu, n_traj=n_traj, steps=steps)


def test_conformal_quantile_rank():
    s = np.arange(1, 100, dtype=float)            # n = 99 scores
    assert ua.conformal_quantile(s, 0.1) == 90.0  # ceil(100 * 0.9) = 90th smallest
    assert ua.conformal_quantile(s[:5], 0.1) == np.inf


def test_split_conformal_covers_exchangeable_data():
    rng = np.random.default_rng(0)
    _, cal = _calibrated(rng, 1, n_traj=200, k=32, steps=(1, 2))
    _, te = _calibrated(rng, 1, n_traj=200, k=32, steps=(1, 2))
    res = ua.conformal_by_group(cal, te, "constant")
    assert set(res.step) == {1, 2}
    for nominal in (0.5, 0.9):
        cov = res[res.nominal == nominal].coverage
        assert cov.min() > nominal - 0.02 and cov.max() < nominal + 0.02


def test_spread_scale_adapts_when_the_spread_grows_with_the_error():
    rng = np.random.default_rng(1)
    _, cal = _calibrated(rng, 2, n_traj=200, k=32)
    floor = ua.spread_floor(cal)
    # errors and spread both twice as large: the spread scale keeps its coverage, a constant scale does not
    _, both = _calibrated(rng, 2, n_traj=200, k=32, scale=2.0)
    q_sp = ua.conformal_by_group(cal, cal, "spread", alphas=(0.1,))
    q_c = ua.conformal_by_group(cal, cal, "constant", alphas=(0.1,))
    assert (ua.coverage_with(q_sp, both, "spread", floor=floor).coverage > 0.87).all()
    assert (ua.coverage_with(q_c, both, "constant").coverage < 0.7).all()
    # errors twice as large but the same spread: the spread scale under-covers too (a shift it cannot see)
    _, err_only = _calibrated(rng, 2, n_traj=200, k=32, scale=2.0, member_scale=1.0)
    assert (ua.coverage_with(q_sp, err_only, "spread", floor=floor).coverage < 0.86).all()   # 0.90 nominal


def test_rank_histogram_flat_for_a_perfect_ensemble_and_u_shaped_when_too_narrow():
    rng = np.random.default_rng(2)
    err, _ = _calibrated(rng, 2, n_traj=100, k=64)
    np.testing.assert_allclose(ua.rank_histogram(err), [1 / 3] * 3, atol=0.01)
    narrow = err + rng.normal(scale=3.0, size=err.shape[1:])[None]   # truth far outside the members
    h = ua.rank_histogram(narrow)
    assert h[0] + h[2] > 0.8


def test_pit_uniform_for_calibrated_gaussians():
    rng = np.random.default_rng(3)
    sd = rng.uniform(0.5, 2.0, size=(5, 2000, 8, F))
    err = rng.normal(size=sd.shape) * sd           # y - mu_m ~ N(0, sd_m^2) for each member
    pit = ua.pit_values(err[:1], sd[:1])            # one member: exactly uniform
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / pit.size
    np.testing.assert_allclose(hist, 0.1, atol=0.01)


def test_spread_skill_one_for_a_perfect_finite_ensemble():
    rng = np.random.default_rng(4)
    m, n = 2, 20000
    rows = []
    c = rng.normal(size=(n, F))
    y = c + rng.normal(size=(n, F))
    mu = c[None] + rng.normal(size=(m, n, F))
    mean = mu.mean(0)
    for f_i, f in enumerate(ua.FIELDS):
        rows.append(pd.DataFrame({f"mse_{f}": (y[:, f_i] - mean[:, f_i]) ** 2,
                                  f"var_ep_{f}": mu[:, :, f_i].var(0), f"var_al_{f}": 0.0}))
    summary = pd.concat(rows, axis=1)
    assert ua.spread_skill(summary, m) == pytest.approx(1.0, abs=0.03)


def test_window_table_phase_and_rayleigh():
    rng = np.random.default_rng(5)
    _, cells = _calibrated(rng, 1, n_traj=10, k=4)
    w = ua.window_table(cells, onset={t: 100 for t in cells["traj_ids"]})
    assert w.onset.all()                            # start 80 < t_on 100
    assert set(w.rayleigh) == {1e8, 1e10}


def test_predicted_std_matches_the_error_of_a_perfect_finite_ensemble():
    rng = np.random.default_rng(6)
    err, cells = _calibrated(rng, 2, n_traj=400, k=32)
    sig = ua.predicted_std(cells)
    assert sig.shape == err.shape[1:]
    rmse = np.sqrt((err.mean(0) ** 2).mean())
    assert np.sqrt((sig ** 2).mean()) == pytest.approx(rmse, rel=0.03)    # c = 3 for two members
    # predicted variances add to the spread: one member with its own Gaussian std
    sd = np.full((1,) + err.shape[1:], 0.5)
    one = _cells(err[:1], std=sd, n_traj=400)
    np.testing.assert_allclose(ua.predicted_std(one), 0.5)


def test_reliability_bins_follow_a_calibrated_std_and_flatten_an_uninformative_one():
    rng = np.random.default_rng(7)
    sigma = rng.uniform(0.1, 3.0, size=200_000)
    err = rng.normal(size=sigma.size) * sigma                  # calibrated: error drawn with the predicted std
    r = ua.reliability(err, sigma, n_bins=10)
    assert r.n.sum() == sigma.size and (r.n > 0).all()
    np.testing.assert_allclose(r.rmse, r.sigma_rms, rtol=0.03)
    assert r.rmse.iloc[-1] / r.rmse.iloc[0] > 5                # the std separates small from large errors
    flat = ua.reliability(rng.normal(size=sigma.size), sigma)  # error independent of the std
    assert flat.rmse.max() / flat.rmse.min() < 1.05


def test_window_coverage_averages_to_the_marginal_and_shows_its_spread():
    s = np.zeros((4, 10, 2))
    s[0] = 2.0                                                 # one window entirely outside the interval
    s[1, :5] = 2.0                                             # one half outside
    cov = ua.window_coverage(s, [1.0, 1.0])
    np.testing.assert_allclose(cov, [0.0, 0.5, 1.0, 1.0])
    assert cov.mean() == pytest.approx((s <= 1.0).mean())
