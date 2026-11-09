import json
import math

import pytest
import torch
from helpers import FakeClock

from rbbench.train import BudgetCosineLR, train_run

BUDGET = ["train.budgets_seconds={1h: 2.0, 4h: 6.0}", "train.val_every_seconds=1.0e9",
          "train.resume_every_seconds=1.0e9", "train.max_steps=null"]


def test_milestones_at_cumulative_training_time(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    res = train_run(cfg, tmp_path / "run", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert res["status"] == "complete"
    done = json.loads((tmp_path / "run" / "COMPLETE.json").read_text())
    ms = done["milestones"]
    # each step lasts 0.5 s: 2 s is reached after step 4, 6 s after step 12
    assert (ms["1h"]["step"], ms["1h"]["train_seconds"]) == (4, 2.0)
    assert (ms["4h"]["step"], ms["4h"]["train_seconds"]) == (12, 6.0)
    assert done["budget_satisfied"] and ms["1h"]["overshoot_seconds"] == 0.0
    assert set(done["checkpoints"]) == {"1h", "4h"}


def test_budget_counts_gpu_time_not_data_wait(dataset_dir, tiny_cfg, tmp_path):
    """compute (default): waiting for batches is logged, not charged; wall charges both (D24)."""
    import csv

    train_run(tiny_cfg("avit", *BUDGET), tmp_path / "c", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    done = json.loads((tmp_path / "c" / "COMPLETE.json").read_text())
    ms = done["milestones"]
    assert (ms["4h"]["step"], ms["4h"]["train_seconds"]) == (12, 6.0)
    assert ms["1h"]["data_wait_seconds"] == pytest.approx(2.0) and done["data_wait_seconds"] == pytest.approx(6.0)
    rows = [r for r in csv.DictReader(open(tmp_path / "c" / "train_metrics.csv")) if r["event"] == "train"]
    assert rows and all(r["data_wait_seconds"] != "" for r in rows)

    train_run(tiny_cfg("avit", *BUDGET, "train.budget_clock=wall"), tmp_path / "w", dataset_dir, 0,
              torch.device("cpu"), clock=FakeClock(0.5))
    ms = json.loads((tmp_path / "w" / "COMPLETE.json").read_text())["milestones"]
    assert (ms["1h"]["step"], ms["4h"]["step"]) == (2, 6) and ms["4h"]["data_wait_seconds"] == pytest.approx(3.0)

    with pytest.raises(ValueError, match="budget_clock"):
        train_run(tiny_cfg("avit", *BUDGET, "train.budget_clock=gpu"), tmp_path / "x", dataset_dir, 0,
                  torch.device("cpu"), clock=FakeClock(0.5))


def test_study_configs_use_the_compute_clock():
    from rbbench.config import load_config

    for experiment in ("main", "replication", "lr_search"):
        assert load_config("avit", experiment)["train"]["budget_clock"] == "compute"


def test_overshoot_recorded(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    train_run(cfg, tmp_path / "run", dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.75))
    ms = json.loads((tmp_path / "run" / "COMPLETE.json").read_text())["milestones"]
    assert ms["1h"]["step"] == 3 and ms["1h"]["overshoot_seconds"] == pytest.approx(0.25)


def test_milestones_survive_preemption(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("avit", *BUDGET)
    run = tmp_path / "run"
    first = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5), stop_after_steps=6)
    assert first["status"] == "preempted" and not (run / "COMPLETE.json").exists()
    one_h = (run / "checkpoint_1h.pt").read_bytes()
    second = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert second["status"] == "complete"
    ms = json.loads((run / "COMPLETE.json").read_text())["milestones"]
    assert (ms["4h"]["step"], ms["4h"]["train_seconds"]) == (12, 6.0)  # the first 3 s were not recounted
    assert (run / "checkpoint_1h.pt").read_bytes() == one_h
    again = train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    assert again["status"] == "already_complete"
    attempts = json.loads((run / "provenance.json").read_text())["attempts"]
    assert [a["end_status"] for a in attempts] == ["preempted", "trained"]
    assert all(a["wall_seconds"] > 0 for a in attempts)


def test_gpu_hours_table_sums_attempts(dataset_dir, tiny_cfg, tmp_path):
    from rbbench.aggregate import gpu_hours_table

    cfg = tiny_cfg("avit", *BUDGET)
    out = tmp_path / "outputs"
    run = out / "runs" / "avit" / "r0"
    train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5), stop_after_steps=6)
    train_run(cfg, run, dataset_dir, 0, torch.device("cpu"), clock=FakeClock(0.5))
    prov = json.loads((run / "provenance.json").read_text())
    prov["attempts"][1]["device"] = "cuda"  # pretend the resumed attempt ran on a GPU
    del prov["attempts"][0]["wall_seconds"]  # and the first was killed before it could close
    (run / "provenance.json").write_text(json.dumps(prov))
    (run / "evaluations").mkdir()
    for lab in ("1h", "4h"):  # one evaluation pass covering both milestones
        marker = {"pass": {"wall_seconds": 1800.0, "labels": ["1h", "4h"], "device": "cuda"}}
        (run / "evaluations" / f"EVAL_COMPLETE_{lab}.json").write_text(json.dumps(marker))
    rows = [{"run_id": "avit/r0", "model": "avit", "fraction": 1.0, "seed": 0}]
    df = gpu_hours_table(rows, out).set_index("run_id")
    r = df.loc["avit/r0"]
    assert r["attempts"] == 2 and r["unclosed_attempts"] == 1
    assert r["train_gpu_hours"] == pytest.approx(prov["attempts"][1]["wall_seconds"] / 3600)
    assert r["train_wall_hours"] > r["train_gpu_hours"]  # the unclosed attempt's logged wall time counts
    assert r["eval_passes"] == 1 and r["eval_gpu_hours"] == pytest.approx(0.5)
    assert df.loc["TOTAL", "eval_gpu_hours"] == pytest.approx(0.5)


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
