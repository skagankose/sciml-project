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
