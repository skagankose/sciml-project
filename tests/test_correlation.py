"""Correlation and high-correlation time in the rollout evaluation (D27)."""

import json

import numpy as np
import pandas as pd
import torch

from rbbench.config import load_config
from rbbench.data import FIELDS, full_trajectory_dataset
from rbbench.evaluate import CORRELATION_THRESHOLDS, evaluate, high_correlation_steps, make_cohort
from rbbench.rollout import Persistence

GRID = "task.spatial_resolution=[64,32]"


def test_high_correlation_steps():
    curve = pd.Series([0.99, 0.95, 0.85, 0.7], index=[1, 2, 3, 4])
    assert high_correlation_steps(curve) == {"0.8": 4, "0.9": 3, "0.95": 3}  # first step strictly below
    assert high_correlation_steps(pd.Series([0.99, 0.98], index=[1, 2])) == {"0.8": None, "0.9": None, "0.95": None}


def test_correlation_is_off_by_default(dataset_dir, tmp_path):
    """Opt-in (owner, 2026-11-13): the default evaluation writes no correlation columns or summaries."""
    cfg = load_config("cnext_unet", "smoke", "tiny", [GRID])
    _, total = full_trajectory_dataset(dataset_dir, "test")
    cohort = make_cohort(dataset_dir, cfg["eval"], 4, total)
    evaluate({"p": Persistence()}, cfg, dataset_dir, cohort, tmp_path, torch.device("cpu"))
    roll = pd.read_parquet(tmp_path / "rollout_p.parquet")
    assert not [c for c in roll.columns if c.startswith("corr_")]
    clean = json.loads((tmp_path / "clean_p.json").read_text())
    assert "high_correlation_time" not in clean and "rollout_corr_mean_at" not in clean


def test_rollouts_carry_correlation_and_high_correlation_time(dataset_dir, tmp_path):
    cfg = load_config("cnext_unet", "smoke", "tiny", [GRID])
    _, total = full_trajectory_dataset(dataset_dir, "test")
    cohort = make_cohort(dataset_dir, cfg["eval"], 4, total)
    evaluate({"p": Persistence()}, cfg, dataset_dir, cohort, tmp_path, torch.device("cpu"), with_correlation=True)
    roll = pd.read_parquet(tmp_path / "rollout_p.parquet")
    corr_cols = [f"corr_{f}" for f in FIELDS]
    assert set(corr_cols + ["corr_mean"]) <= set(roll.columns)
    values = roll[corr_cols].to_numpy()
    assert np.all((values[np.isfinite(values)] >= -1 - 1e-9) & (values[np.isfinite(values)] <= 1 + 1e-9))
    np.testing.assert_allclose(roll.corr_mean, np.nanmean(values, axis=1))

    clean = json.loads((tmp_path / "clean_p.json").read_text())
    hct = clean["high_correlation_time"]
    assert hct["thresholds"] == list(CORRELATION_THRESHOLDS)
    r = roll[roll.sigma == 0]
    assert hct["field_mean"]["all"] == high_correlation_steps(r.groupby("step").corr_mean.mean())
    assert hct["buoyancy"]["all"] == high_correlation_steps(r.groupby("step").corr_buoyancy.mean())
    assert set(hct["field_mean"]) == {"all"} | {str(s) for e in cohort["trajectories"] for s in e["starts"]}
    curve = r.groupby("step").corr_mean.mean()
    assert clean["rollout_corr_mean_at"] == {str(k): float(curve.loc[k]) for k in cfg["eval"]["report_steps"]
                                             if k in curve.index}
