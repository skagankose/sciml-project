"""Data helpers of scripts/final_figures.py: the per-trajectory onset split, seed statistics and legend order."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_figures as F  # noqa: E402


def test_developed_mask_uses_each_trajectorys_onset_frame():
    t = pd.DataFrame({"traj_id": ["a", "a", "b", "b", "c"], "start": [40, 41, 77, 78, 150],
                      "vrmse_mean": [1.0, 0.1, 2.0, 0.2, 9.0]})
    t_on = {"a": 41, "b": 78, "c": None}                    # c never convects: all onset
    np.testing.assert_array_equal(F.developed_mask(t, t_on), [False, True, False, True, False])
    assert F.developed_mean(t, t_on) == pytest.approx(0.15)


def test_developed_mean_drops_noisy_rows_and_groups():
    t = pd.DataFrame({"traj_id": ["a"] * 4, "start": [60, 60, 70, 70], "sigma": [0.0, 0.1, 0.0, 0.1],
                      "rayleigh": [1e6, 1e6, 1e8, 1e8], "vrmse_mean": [0.1, 5.0, 0.3, 5.0]})
    g = F.developed_mean(t, {"a": 50}, by="rayleigh")
    assert g.to_dict() == {1e6: pytest.approx(0.1), 1e8: pytest.approx(0.3)}


def test_seed_stats_and_references():
    fc = pd.DataFrame({"model": ["avit", "avit", "avit", "persistence"], "fraction": [0.1, 0.1, 1.0, np.nan],
                       "seed": [0, 1, 0, np.nan], "budget": ["4h", "4h", "4h", "-"],
                       "one_step_developed": [0.07, 0.08, 0.06, 0.23],
                       "rollout_step30_from80": [1.0, 1.0, 1.0, 0.8], "rollout_step30_from140": [0.5, 0.7, 0.6, 0.6]})
    st = F.seed_stats(fc, "one_step_developed").set_index(["model", "fraction", "budget"])
    assert st.loc[("avit", 0.1, "4h")].tolist() == pytest.approx([0.075, 0.07, 0.08])
    assert len(st) == 2                                      # the reference row is not a seed
    assert F.reference(fc, "persistence", "one_step_developed") == 0.23
    assert F.reference(fc, "released_cnext_unet", "one_step_developed") is None
    assert F.with_rollout_end(fc).rollout_end_developed.tolist() == pytest.approx([0.75, 0.85, 0.8, 0.7])


def test_legend_reads_row_by_row():
    # matplotlib fills legend columns first: with 3 columns, row-major [a b c / d e] must be passed as a d b e c
    assert F.row_major(list("abcde"), 3) == list("adbec")
    assert F.row_major(list("abc"), 3) == list("abc")


def test_frames_from_onset_and_the_offsets_every_trajectory_reaches():
    t = pd.DataFrame({"traj_id": ["a", "a", "b", "c"], "start": [0, 60, 60, 10], "sigma": [0.0, 0.0, 0.0, 0.0],
                      "vrmse_mean": [1.0, 0.1, 0.2, 0.3]})
    d = F.frames_from_onset(t, {"a": 50, "b": 41, "c": None})   # c never convects: dropped
    assert d.d.tolist() == [-50, 10, 19]
    assert F.common_offsets({"a": 50, "b": 41, "c": None}) == (-41, 195 - 50)


def test_onset_end_counts_from_frame0_and_from_each_onset():
    # two trajectories of one regime whose error drops 10 frames before their own onset (t_on 50 and 70)
    rows = []
    for traj, on in (("a", 50), ("b", 70)):
        for s in range(196):
            rows.append({"traj_id": traj, "rayleigh": 1e8, "start": s, "vrmse_mean": 1.0 if s < on - 10 else 0.05})
    t = pd.DataFrame(rows)
    r = F.onset_end(t, {"a": 50, "b": 70}).iloc[0]
    assert r.from_onset == -10                  # aligned: both drop at t_on - 10
    assert r.from_frame0 == 60                  # from frame 0 the mean stays above 2 x 0.05 until b has dropped too
