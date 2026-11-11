import numpy as np
import pandas as pd
import pytest

from rbbench.aggregate import summary_table
from rbbench.evaluate import failure_step, kaplan_meier, km_median


def test_failure_step_and_censoring():
    curve = np.array([0.2, 0.5, 1.2, 0.8, 3.0])
    assert failure_step(curve, 1.0) == 3
    assert failure_step(curve, 2.0) == 5
    assert failure_step(curve, 5.0) is None  # right-censored, not 0 and not 5


def test_kaplan_meier_with_censoring():
    s = kaplan_meier([2, None, 4, None], horizon=5)
    np.testing.assert_allclose(s, [1.0, 0.75, 0.75, 0.5, 0.5])
    assert km_median(s) == 4
    assert km_median(kaplan_meier([None, None], 5)) is None


def test_summary_mean_std_and_seed_listing():
    df = pd.DataFrame({
        "model": ["tfno"] * 3, "fraction": [0.1] * 3, "budget": ["1h"] * 3, "metric": ["m"] * 3,
        "sigma": [0.0] * 3, "step": [1] * 3, "seed": [0, 1, 2], "value": [1.0, 2.0, 3.0], "run_id": ["a", "b", "c"],
    })
    out = summary_table(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["mean"] == 2.0 and row["std"] == pytest.approx(1.0) and row["n_seeds"] == 3
    assert row["per_seed"] == "s0=1;s1=2;s2=3"


def test_missing_seed_is_visible():
    df = pd.DataFrame({
        "model": ["avit"] * 2, "fraction": [1.0] * 2, "budget": ["4h"] * 2, "metric": ["m"] * 2,
        "sigma": [0.0] * 2, "step": [1] * 2, "seed": [0, 2], "value": [1.0, 1.5], "run_id": ["a", "c"],
    })
    assert summary_table(df).iloc[0]["n_seeds"] == 2


def test_paper_protocol_table_sets_references_beside_our_runs():
    from rbbench.aggregate import paper_protocol_table

    rows = []
    for model in ("cnext_unet", "avit"):
        for seed, v in ((0, 0.5), (1, 0.7)):
            rows += [{"model": model, "fraction": 0.1, "budget": "4h", "seed": seed, "sigma": 0.0, "step": 1,
                      "metric": "paper_one_step_normalised_vrmse_mean", "value": v},
                     {"model": model, "fraction": 0.1, "budget": "4h", "seed": seed, "sigma": 0.0, "step": None,
                      "metric": "paper_rollout_normalised_vrmse_steps_13-30", "value": 10 * v},
                     {"model": model, "fraction": 0.1, "budget": "4h", "seed": seed, "sigma": 0.0, "step": 1,
                      "metric": "one_step_vrmse_mean", "value": v}]
    released = {"results": {"cnext_unet": {"test": {
        "normalised_vrmse_mean": 0.6688,
        "paper_rollout": {"window_vrmse_mean": {"normalised": {"6:12": 36.1, "13:30": 34.7}}}}}}}
    t = paper_protocol_table(summary_table(pd.DataFrame(rows)), released)
    assert set(t.metric) == {"paper_one_step_normalised_vrmse_mean", "paper_rollout_normalised_vrmse_steps_13-30"}
    one = t[(t.metric == "paper_one_step_normalised_vrmse_mean")].set_index("model")
    assert one.loc["cnext_unet", "mean"] == pytest.approx(0.6) and one.loc["cnext_unet", "n_seeds"] == 2
    assert one.loc["cnext_unet", "published"] == "0.6699" and one.loc["cnext_unet", "released_checkpoint"] == 0.6688
    assert one.loc["avit", "published"] == "" and np.isnan(one.loc["avit", "released_checkpoint"])
    roll = t[t.metric == "paper_rollout_normalised_vrmse_steps_13-30"].set_index("model")
    assert roll.loc["cnext_unet", "published"] == "> 10" and roll.loc["cnext_unet", "released_checkpoint"] == 34.7
    assert list(t.model[:2]) == ["cnext_unet", "avit"]  # fixed model order, not by value


def test_one_step_is_split_into_onset_and_developed_windows():
    """D26: one-step VRMSE is also reported for onset (first input frame < 50) and developed convection."""
    import pandas as pd

    from rbbench.aggregate import ONSET_FRAMES, one_step_phase_records

    one = pd.DataFrame({"start": [0, 10, ONSET_FRAMES - 1, ONSET_FRAMES, 120],
                        "vrmse_mean": [3.0, 2.0, 1.0, 0.2, 0.1],
                        "normalised_vrmse_mean": [9.0, 6.0, 3.0, 0.2, 0.1]})
    recs = {r["metric"]: r["value"] for r in one_step_phase_records(one, {"model": "m"})}
    assert recs["one_step_vrmse_mean_onset"] == 2.0
    assert abs(recs["one_step_vrmse_mean_developed"] - 0.15) < 1e-12
    assert recs["one_step_normalised_vrmse_mean_onset"] == 6.0
    assert all(r["step"] == 1 and r["sigma"] == 0.0 for r in one_step_phase_records(one, {}))
