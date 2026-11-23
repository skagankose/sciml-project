"""The WP6 matching rule (D29): per regime, the coarsest solver grid at or below the target's VRMSE at steps 1, 10, 30."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import findings_figures as ff  # noqa: E402


def _row(source, kind, ra, s1, s10, s30, seconds, grid=""):
    return {"source": source, "kind": kind, "grid": grid, "rayleigh": ra, "vrmse_step1": s1, "vrmse_step10": s10,
            "vrmse_step30": s30, "seconds_per_rollout": seconds, "seconds_min": seconds, "seconds_max": seconds}


def _comp():
    rows = [_row("CNextU-Net 100% 4h", "model", 1e6, 0.01, 0.10, 0.30, 0.25),
            # finest passes, middle passes, coarsest fails only at step 1: the middle one is the match
            _row("Dedalus 256x64", "solver", 1e6, 0.001, 0.01, 0.02, 100.0, "256x64"),
            _row("Dedalus 128x32", "solver", 1e6, 0.005, 0.05, 0.30, 20.0, "128x32"),
            _row("Dedalus 64x16", "solver", 1e6, 0.02, 0.05, 0.10, 4.0, "64x16"),
            _row("CNextU-Net 100% 4h", "model", 1e10, 0.10, 0.30, 0.60, 0.25),
            # one fails at step 10, the coarse one diverged (inf): nothing matches
            _row("Dedalus 256x64", "solver", 1e10, 0.05, 0.40, 0.55, 150.0, "256x64"),
            _row("Dedalus 64x16", "solver", 1e10, 0.05, 0.20, np.inf, 8.0, "64x16")]
    return pd.DataFrame(rows)


def test_coarsest_passing_grid_and_cost():
    m = ff.solver_matching(_comp(), {("cnext_unet", 1.0): 180.0}).set_index("rayleigh")
    assert m.loc[1e6, "matching_grid"] == "128x32"          # ties count as passing (0.30 <= 0.30)
    assert "64x16: 1" in m.loc[1e6, "failed_steps_by_grid"]
    solver_su = 20.0 / 3600                                  # one core-hour = 1 SU
    model_su = 0.25 / 3600 * 32                              # one GPU-hour = 32 SU
    assert m.loc[1e6, "solver_su"] == pytest.approx(solver_su)
    assert m.loc[1e6, "target_su"] == pytest.approx(model_su)
    assert m.loc[1e6, "break_even_rollouts"] == pytest.approx(180.0 / (solver_su - model_su))


def test_no_match_when_every_grid_fails_or_diverges():
    m = ff.solver_matching(_comp(), {}).set_index("rayleigh")
    assert m.loc[1e10, "matching_grid"] == "none"
    assert "64x16: 30" in m.loc[1e10, "failed_steps_by_grid"]        # inf fails
    assert "256x64: 10" in m.loc[1e10, "failed_steps_by_grid"]
