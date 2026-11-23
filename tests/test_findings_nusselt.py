"""Heat transport per Rayleigh number (WP5): rollout averages, relative errors, medians over rollouts, seed means."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import findings_figures as ff  # noqa: E402


def _physics(scale, starts=(20, 80, 140), ras=(1e6, 1e10), steps=30):
    """nu_true = 10 * Ra-index + step / 10 (varies along the rollout); nu_pred = scale * nu_true."""
    rows = []
    for k, ra in enumerate(ras):
        for traj in range(3):
            for s in starts:
                for step in range(1, steps + 1):
                    true = 10.0 * (k + 1) + step / 10
                    rows.append({"traj_id": f"t{k}{traj}", "rayleigh": ra, "prandtl": 1.0, "start": s, "sigma": 0.0,
                                 "step": step, "nu_true": true, "nu_pred": scale * true})
                    rows.append({**rows[-1], "sigma": 0.05, "nu_pred": 99.0})       # noisy inputs: ignored
    return pd.DataFrame(rows)


def test_rollout_means_use_developed_starts_and_clean_inputs():
    g = ff._nu_rollout_means(_physics(1.2))
    assert set(g.start) == {80, 140} and (g.steps == 30).all()
    assert len(g) == 2 * 3 * 2                                                       # Ra x trajectories x starts
    np.testing.assert_allclose(g.rel_err, 0.2)
    np.testing.assert_allclose(g[g.rayleigh == 1e6].nu_true, 10 + 1.55)             # mean of step / 10 over 1..30


def test_summary_medians_and_seed_mean():
    rolls = []
    for seed, scale in ((0, 0.8), (1, 1.0)):
        rolls.append(ff._nu_rollout_means(_physics(scale)).assign(model="avit", fraction=0.1, seed=seed, budget="4h"))
    s = ff.nusselt_by_rayleigh(pd.concat(rolls, ignore_index=True))
    per_ra = s[s.rayleigh.notna()].set_index("rayleigh")
    assert (per_ra.n_seeds == 2).all() and (per_ra.n_rollouts == 6).all()
    np.testing.assert_allclose(per_ra.rel_err_signed, -0.1)                         # mean of -0.2 and 0
    np.testing.assert_allclose(per_ra.rel_err_abs, 0.1)
    overall = s[s.rayleigh.isna()].iloc[0]
    assert overall.n_rollouts == 12
    np.testing.assert_allclose(overall.rel_err_signed, -0.1)
