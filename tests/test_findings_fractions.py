"""The findings / report figures pool a model's runs as seeds, so each figure must see one data fraction only.

With the 100% runs evaluated next to the 10% runs, an unfiltered `evaluated_runs` mixed both into one seed mean
under a "10% data" label (found 2026-11-23). These tests pin the fraction filter and the per-fraction labels.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import findings_figures as ff  # noqa: E402


def _runs(tmp_path):
    rows = []
    for model in ("cnext_unet", "tfno"):
        for frac, tag in ((0.1, "f010"), (1.0, "f100")):
            for seed in (0, 1):
                run_id = f"{model}/{tag}/s{seed}/x-y"
                ev = tmp_path / "runs" / run_id / "evaluations"
                ev.mkdir(parents=True)
                for lab in ("1h", "4h"):
                    (ev / f"EVAL_COMPLETE_{lab}.json").write_text(json.dumps({"label": lab}))
                    pd.DataFrame({"start": [0, 60], "vrmse_mean": [frac, frac]}).to_parquet(ev / f"one_step_{lab}.parquet")
                rows.append({"run_id": run_id, "model": model, "seed": seed, "fraction": frac, "manifest": "train"})
    rows.append({"run_id": "cnext_unet/f010/s0/lr", "model": "cnext_unet", "seed": 0, "fraction": 0.1,
                 "manifest": "lr_search"})
    return pd.DataFrame(rows)


def test_fraction_filter(tmp_path):
    runs = _runs(tmp_path)
    assert len(ff.evaluated_runs(runs, tmp_path)) == 16                      # 2 models x 2 fractions x 2 seeds x 2
    for frac in (0.1, 1.0):
        ev = ff.evaluated_runs(runs, tmp_path, fraction=frac)
        assert len(ev) == 8 and set(ev.fraction) == {frac}
        tables = ff._load(ev, "one_step", "4h")
        assert {m: len(ts) for m, ts in tables.items()} == {"cnext_unet": 2, "tfno": 2}   # the two seeds only
        assert all(float(t.vrmse_mean.iloc[0]) == frac for ts in tables.values() for t in ts)


def test_fraction_text(tmp_path):
    runs = _runs(tmp_path)
    assert ff.fraction_text(ff.evaluated_runs(runs, tmp_path, fraction=0.1)) == "10% data"
    assert ff.fraction_text(ff.evaluated_runs(runs, tmp_path, fraction=1.0)) == "100% data"
    assert ff.fraction_text(ff.evaluated_runs(runs, tmp_path)) == "10% / 100% data"


def test_training_purpose():
    assert ff.training_purpose("train", 0.1) == "10% runs"
    assert ff.training_purpose("train", 1.0) == "100% runs"
    assert ff.training_purpose("uq", 1.0) == "UQ ensemble training"
    assert ff.training_purpose("train_1h_f100", 1.0) == "dedicated 1 h runs"
    assert ff.training_purpose("lr_search", 0.1) == "learning-rate search"
