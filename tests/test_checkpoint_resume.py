import os
import signal

import pandas as pd
import torch
from helpers import FakeClock

from rbbench.train import train_run

OVERRIDES = ["train.budgets_seconds={1h: 3.0, 4h: 5.0}", "train.val_every_seconds=1.0e9",
             "train.resume_every_seconds=1.0e9", "schedule.warmup_steps=3", "train.batch_size=4"]


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_interrupted_equals_uninterrupted(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("cnext_unet", *OVERRIDES)
    dev = torch.device("cpu")
    train_run(cfg, tmp_path / "a", dataset_dir, 1, dev, clock=FakeClock(0.5))
    r1 = train_run(cfg, tmp_path / "b", dataset_dir, 1, dev, clock=FakeClock(0.5), stop_after_steps=7)
    assert r1["status"] == "preempted"
    mid = _weights(tmp_path / "b" / "last_resume.pt")
    assert mid["step"] == 7 and mid["state"]["position"] == 28  # sampler position persisted
    r2 = train_run(cfg, tmp_path / "b", dataset_dir, 1, dev, clock=FakeClock(0.5))
    assert r2["status"] == "complete"
    a, b = _weights(tmp_path / "a" / "checkpoint_4h.pt"), _weights(tmp_path / "b" / "checkpoint_4h.pt")
    assert a["step"] == b["step"] == 10
    assert _same(a["model"], b["model"])  # bitwise on CPU
    oa, ob = a["optimizer"]["state"], b["optimizer"]["state"]
    assert all(torch.equal(oa[k]["exp_avg_sq"], ob[k]["exp_avg_sq"]) for k in oa)
    assert a["state"]["epoch"] == b["state"]["epoch"] and a["state"]["samples_seen"] == b["state"]["samples_seen"]


def test_full_state_saved(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("tfno", *OVERRIDES)
    train_run(cfg, tmp_path / "r", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    ck = _weights(tmp_path / "r" / "checkpoint_1h.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "state", "config_sha256", "seed"):
        assert key in ck
    assert {"python", "numpy", "torch"} <= set(ck["rng"])
    assert {"epoch", "position", "train_seconds", "milestones", "attempt"} <= set(ck["state"])
    assert ck["milestone"] == "1h"


def test_sigusr1_checkpoints_and_exits(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("convlstm", *OVERRIDES)

    def fire(step):
        if step == 3:
            os.kill(os.getpid(), signal.SIGUSR1)

    res = train_run(cfg, tmp_path / "r", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5), step_callback=fire)
    assert res["status"] == "preempted" and "signal" in res["reason"]
    assert _weights(tmp_path / "r" / "last_resume.pt")["step"] == 3
    log = pd.read_csv(tmp_path / "r" / "train_metrics.csv")
    assert "preempted" in set(log.event)
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL  # handler restored


def test_overfit_small_set_loss_decreases(dataset_dir, tiny_cfg, tmp_path):
    from rbbench.data import list_trajectories

    cfg = tiny_cfg("cnext_unet", "train.budgets_seconds={1h: 20.0, 4h: 40.0}", "optim.lr=3.0e-3",
                   "schedule.name=constant", "train.val_every_seconds=1.0e9", "train.log_every_steps=1")
    one = [list_trajectories(dataset_dir, "train")[0].id]
    train_run(cfg, tmp_path / "r", dataset_dir, 0, torch.device("cpu"), subset_ids=one, clock=FakeClock(0.5))
    log = pd.read_csv(tmp_path / "r" / "train_metrics.csv")
    loss = log[log.event == "train"].loss.to_numpy()
    assert loss[-10:].mean() < 0.5 * loss[:5].mean()
