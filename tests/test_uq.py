"""Uncertainty quantification (D30, WP9): variance output, beta-NLL, scores, the ensemble evaluation."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import FakeClock

from rbbench.config import config_hash, load_config
from rbbench.data import full_trajectory_dataset
from rbbench.evaluate import make_cohort
from rbbench.models.factory import GaussianOutput, count_parameters, model_from_config, training_loss
from rbbench.rollout import rollout
from rbbench.sweep import read_jsonl
from rbbench.train import beta_nll_loss, train_run
from rbbench.uq import (
    crps_mixture,
    default_ensembles,
    ensemble_stats,
    evaluate_uq,
    member_rollout,
    model_member,
    persistence_member,
)

REPO = Path(__file__).resolve().parents[1]
OVERRIDES = ["train.budgets_seconds={1h: 3.0}", "train.val_every_seconds=1.0e9", "train.resume_every_seconds=1.0e9",
             "schedule.warmup_steps=3", "train.batch_size=4"]


# --- scores against independent calculations ------------------------------------------------

def _crps_numeric(y, mus, sds):
    """CRPS by integrating (F(z) - 1{y <= z})^2 for a mixture of Gaussians."""
    z = np.linspace(-30, 30, 600001)
    cdf = np.mean([0.5 * (1 + np.vectorize(math.erf)((z - m) / (s * math.sqrt(2)))) for m, s in zip(mus, sds)], 0)
    return np.trapezoid((cdf - (z >= y)) ** 2, z)


def test_crps_of_a_gaussian_mixture_matches_integration():
    y, mus, sds = 0.7, [0.0, 1.5, -0.4], [1.0, 0.5, 2.0]
    got = crps_mixture(torch.tensor(y, dtype=torch.float64), torch.tensor(mus, dtype=torch.float64)[:, None],
                       torch.tensor(sds, dtype=torch.float64)[:, None])
    assert float(got) == pytest.approx(_crps_numeric(y, mus, sds), abs=1e-5)
    # one N(0, 1) at y = 0.7: the lecture's example value 0.42
    one = crps_mixture(torch.tensor(0.7), torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    assert float(one) == pytest.approx(0.42, abs=0.005)


def test_crps_of_point_forecasts():
    y = torch.tensor([0.3, -1.0])
    one = torch.tensor([[1.0, 0.5]])
    assert torch.allclose(crps_mixture(y, one, None), (y - one[0]).abs())  # one point forecast: absolute error
    two = torch.tensor([[1.0, 0.5], [0.0, -2.0]])
    expected = (y - two).abs().mean(0) - 0.5 * (two[0] - two[1]).abs() * 2 / 4
    assert torch.allclose(crps_mixture(y, two, None), expected)


def test_ensemble_stats_split_the_variance_and_cover_calibrated_draws():
    g = torch.Generator().manual_seed(0)
    m, b, nx, ny, nf = 5, 2, 64, 64, 4
    means = torch.randn(m, b, nx, ny, nf, generator=g) * 0.3
    stds = torch.full((m, b, nx, ny, nf), 0.5)
    mix = means[torch.randint(0, m, (b, nx, ny, nf), generator=g), torch.arange(b)[:, None, None, None],
                torch.arange(nx)[None, :, None, None], torch.arange(ny)[None, None, :, None],
                torch.arange(nf)[None, None, None, :]]
    y = mix + 0.5 * torch.randn(b, nx, ny, nf, generator=g)  # a draw from the mixture itself
    s = ensemble_stats(y, means, stds)
    assert torch.allclose(s["var_al"], torch.full((b, nf), 0.25))
    assert torch.allclose(s["var_ep"], means.var(0, unbiased=False).mean((1, 2)))
    # the Gaussian approximation of a mixture is close to calibrated here
    assert (s["cov90"] - 0.9).abs().max() < 0.03 and (s["cov50"] - 0.5).abs().max() < 0.04
    point = ensemble_stats(y, means[:1], None)
    assert torch.all(point["var_ep"] == 0) and torch.allclose(point["crps"], point["mae"])


# --- variance output and beta-NLL -------------------------------------------------------------

def test_variance_output_and_beta_nll(tiny_cfg):
    cfg = tiny_cfg("cnext_unet", "train.loss=beta_nll")
    assert training_loss(cfg) == "beta_nll" and training_loss(tiny_cfg("cnext_unet")) == "mse"
    model = model_from_config(cfg, seed=0)
    assert isinstance(model, GaussianOutput)
    plain = model_from_config(tiny_cfg("cnext_unet"), seed=0)
    assert 0 < count_parameters(model) - count_parameters(plain) < 0.01 * count_parameters(plain)
    x = torch.randn(2, 4, 64, 32, 4)
    mean, log_var = model.forward_dist(x)
    assert mean.shape == log_var.shape == (2, 64, 32, 4) and torch.equal(model(x), mean)
    # beta = 1: the mean's gradient is the MSE's up to the factor 1/2
    mu = torch.randn(3, 5, requires_grad=True)
    y, lv = torch.randn(3, 5), torch.randn(3, 5)
    beta_nll_loss(mu, lv, y, 1.0).backward()
    assert torch.allclose(mu.grad, (mu - y).detach() / mu.numel())


def test_beta_nll_training_run(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("cnext_unet", *OVERRIDES, "train.loss=beta_nll")
    assert config_hash(cfg) != config_hash(tiny_cfg("cnext_unet", *OVERRIDES))
    res = train_run(cfg, tmp_path / "r", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert res["status"] == "complete"
    ckpt = torch.load(tmp_path / "r" / "checkpoint_1h.pt", map_location="cpu", weights_only=False)
    model = model_from_config(cfg)
    model.load_state_dict(ckpt["model"])
    assert hasattr(model, "forward_dist")


def test_uq_experiment_and_manifests_keep_the_core_hashes():
    cfg = load_config("cnext_unet", "uq")
    assert training_loss(cfg) == "beta_nll" and cfg["train"]["budgets_seconds"] == {"1h": 3600}
    assert cfg["optim"]["lr"] == load_config("cnext_unet", "main")["optim"]["lr"] == 1e-3
    assert training_loss(load_config("cnext_unet", "main")) == "mse"
    rows = read_jsonl(REPO / "tasks" / "uq.jsonl")
    assert [(r["fraction"], r["seed"]) for r in rows] == [(f, s) for f in (0.1, 1.0) for s in range(5)]
    assert all(r["experiment"] == "uq" and r["model"] == "cnext_unet" for r in rows)


def test_default_ensembles_match_the_manifests():
    spec = default_ensembles()
    assert json.loads((REPO / "tasks" / "uq_ensembles.json").read_text()) == json.loads(json.dumps(spec))
    for ens in spec["ensembles"]:
        rows = [read_jsonl(REPO / m["manifest"])[m["row"]] for m in ens["members"] if "manifest" in m]
        if not rows:
            continue
        assert len({(r["model"], r["fraction"]) for r in rows}) == 1, ens["name"]
        assert len({r["seed"] for r in rows}) == len(rows), ens["name"]
        name = ens["name"]
        assert name.startswith(rows[0]["model"]) and f"f{int(round(rows[0]['fraction'] * 100)):03d}" in name
        assert ens["splits"]["valid"] == [0.0] and ens["splits"]["test"][0] == 0.0


# --- the evaluation pass --------------------------------------------------------------------

def _cohort(dataset_dir):
    cfg = load_config("cnext_unet", "smoke", "tiny", ["task.spatial_resolution=[64,32]"])
    _, total = full_trajectory_dataset(dataset_dir, "test")
    return cfg, make_cohort(dataset_dir, cfg["eval"], 4, total)


def test_member_rollout_equals_the_shared_rollout(tiny_cfg):
    model = model_from_config(tiny_cfg("cnext_unet"), seed=0).eval()
    ctx = torch.randn(2, 4, 64, 32, 4)
    means, stds = member_rollout(model_member("a", model), ctx, 3)
    assert stds is None and torch.allclose(means, rollout(model, ctx, 3))


def test_evaluate_uq_end_to_end(dataset_dir, tiny_cfg, tmp_path):
    cfg, cohort = _cohort(dataset_dir)
    gcfg = tiny_cfg("cnext_unet", "train.loss=beta_nll")
    members = [model_member(f"s{i}", model_from_config(gcfg, seed=i)) for i in range(3)]
    out = tmp_path / "ens"
    res = evaluate_uq(members, cfg, dataset_dir, "test", cohort, [0.0, 0.1], out, torch.device("cpu"),
                      one_step_stride=8, cells=(16, 4))
    assert res["gaussian"] and res["members"] == ["s0", "s1", "s2"]
    summary = pd.read_parquet(out / "summary.parquet")
    horizon = cohort["rollout_steps"]
    n_roll = sum(len(e["starts"]) for e in cohort["trajectories"]) * horizon
    assert (summary.query("kind == 'rollout'").groupby("sigma").size() == n_roll).all()
    assert (summary.filter(like="var_al_") > 0).all().all() and (summary.filter(like="cov90_") <= 1).all().all()
    members_df = pd.read_parquet(out / "members.parquet")
    assert set(members_df.member) == {"s0", "s1", "s2"} and "corr_mean" in members_df
    cells = np.load(out / "cells.npz")
    err, std, idx = cells["err__rollout__s0"], cells["std__rollout__s0"], cells["index__rollout__s0"]
    assert err.shape == (3, n_roll, 16, 4) and std.shape == err.shape and idx.shape == (n_roll, 3)
    assert cells["err__rollout__s0.1"].shape[2] == 4  # fewer cells under noise
    # the sampled errors average to the exact means (mae of a member <-> mean |err|), roughly
    assert np.isfinite(err.astype(np.float32)).all()
    prof = np.load(out / "yprofile.npz")
    assert prof["err2__one_step"].shape[1:] == (32, 4) and prof["var__rollout"].shape[1:] == (32, 4)
    assert json.loads((out / "UQ_COMPLETE.json").read_text())["n_trajectories"] == len(cohort["trajectories"])


def test_persistence_ensemble_on_validation(dataset_dir, tmp_path):
    cfg, cohort = _cohort(dataset_dir)
    out = tmp_path / "p"
    res = evaluate_uq([persistence_member()], cfg, dataset_dir, "valid", cohort, [0.0], out, torch.device("cpu"),
                      one_step_stride=4, cells=(8, 2))
    assert res["split"] == "valid" and not res["gaussian"]
    s = pd.read_parquet(out / "summary.parquet")
    assert np.allclose(s.filter(like="crps_").to_numpy(), s.filter(regex="^mae_(?!member)").to_numpy())
    assert (s.filter(like="var_ep_") == 0).all().all()
