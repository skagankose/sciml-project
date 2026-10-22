import json
import math

import pytest
import torch
from helpers import FakeClock

from rbbench.train import BudgetCosineLR, train_run

BUDGET = ["train.budgets_seconds={2h: 2.0, 6h: 6.0}", "train.val_every_seconds=1.0e9",
          "train.resume_every_seconds=1.0e9", "train.max_steps=null"]


def test_milestones_at_cumulative_training_time(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    res = train_run(cfg, tmp_path / "run", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert res["status"] == "complete"
    done = json.loads((tmp_path / "run" / "COMPLETE.json").read_text())
    ms = done["milestones"]
    # each step lasts 0.5 s: 2 s is reached after step 4, 6 s after step 12
    assert (ms["2h"]["step"], ms["2h"]["train_seconds"]) == (4, 2.0)
    assert (ms["6h"]["step"], ms["6h"]["train_seconds"]) == (12, 6.0)
    assert done["budget_satisfied"] and ms["2h"]["overshoot_seconds"] == 0.0
    assert set(done["checkpoints"]) == {"2h", "6h"}


def test_overshoot_recorded(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    train_run(cfg, tmp_path / "run", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.75))
    ms = json.loads((tmp_path / "run" / "COMPLETE.json").read_text())["milestones"]
    assert ms["2h"]["step"] == 3 and ms["2h"]["overshoot_seconds"] == pytest.approx(0.25)


def test_milestones_survive_preemption(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    run = tmp_path / "run"
    first = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5), stop_after_steps=6)
    assert first["status"] == "preempted" and not (run / "COMPLETE.json").exists()
    two_h = (run / "checkpoint_2h.pt").read_bytes()
    second = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert second["status"] == "complete"
    ms = json.loads((run / "COMPLETE.json").read_text())["milestones"]
    assert (ms["6h"]["step"], ms["6h"]["train_seconds"]) == (12, 6.0)  # the first 3 s were not recounted
    assert (run / "checkpoint_2h.pt").read_bytes() == two_h
    again = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert again["status"] == "already_complete"


def test_config_change_refused(dataset_dir, tiny_cfg, tmp_path):
    run = tmp_path / "run"
    train_run(tiny_cfg("avit", *BUDGET), run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(), stop_after_steps=1)
    with pytest.raises(RuntimeError):
        train_run(tiny_cfg("avit", *BUDGET, "optim.lr=0.1"), run, dataset_dir, 0, torch.device("cpu"))


def test_schedule_shape():
    s = BudgetCosineLR(1.0, warmup_steps=4, total_seconds=100.0)
    assert [s.lr(i, 0.0) for i in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert s.lr(10, 0.0) == 1.0 and s.lr(10, 50.0) == pytest.approx(0.5)
    assert s.lr(10, 100.0) == pytest.approx(0.0, abs=1e-12) and s.lr(10, 1e6) == s.lr(10, 100.0)
    assert math.isclose(BudgetCosineLR(1.0, 0, 100.0, name="constant").lr(5, 99.0), 1.0)
