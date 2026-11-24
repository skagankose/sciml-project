"""scripts/uq_findings.py: high-correlation time from the per-step mean correlation (D27)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import uq_findings as U  # noqa: E402


def test_correlation_time_is_the_first_step_below_each_threshold():
    steps = np.arange(1, 31)
    rows = []
    for member, curve in (("s0", 1.0 - 0.01 * steps), ("s1", np.full(30, 0.99))):   # s1 never falls below
        for k, r in zip(steps, curve):
            rows.append({"ensemble": "e", "family": "cnext_unet", "fraction": 1.0, "member": member,
                         "starts": "developed", "step": int(k), "corr_mean": r, "corr_buoyancy": r})
    t = U.correlation_time_table(pd.DataFrame(rows))
    s0 = t[(t.member == "s0") & (t.field == "mean")].iloc[0]
    assert (s0["below_0.95"], s0["below_0.9"], s0["below_0.8"]) == (6, 11, 21)    # 1 - 0.01 k < thr
    s1 = t[(t.member == "s1") & (t.field == "mean")].iloc[0]
    assert np.isnan(s1["below_0.8"]) and np.isnan(s1["below_0.95"])                # above through step 30
    assert set(t.field) == {"mean", "buoyancy"}
