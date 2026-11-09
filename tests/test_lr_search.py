"""Equal-budget learning-rate search: manifest, per-model rates and the selection (decisions D7, D24)."""

import json

import pytest
from test_subset_manifest import _fake_train

from rbbench.cli import main
from rbbench.config import config_hash, load_config
from rbbench.manifests import atomic_write_json
from rbbench.subsets import make_nested_subsets, subset_filename
from rbbench.sweep import make_sweep, read_jsonl, write_jsonl

CORE = ("cnext_unet", "tfno", "avit")
GRID = (1e-4, 5e-4, 1e-3)


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / "tasks"
    for frac, m in make_nested_subsets(_fake_train(), [0.1, 0.3, 1.0], seed=0).items():
        atomic_write_json(d / subset_filename(frac), m)
    return d


def test_search_manifest_has_one_row_per_model_and_rate(tasks_dir, tmp_path):
    rows = make_sweep("lr_search", tasks_dir, models=CORE, repo_root=tmp_path)
    assert len(rows) == 9 and len({r["run_id"] for r in rows}) == 9
    assert {(r["model"], r["lr"]) for r in rows} == {(m, lr) for m in CORE for lr in GRID}
    assert {(r["fraction"], r["seed"]) for r in rows} == {(0.1, 0)}
    for r in rows:
        cfg = load_config(r["model"], "lr_search", overrides=r["overrides"])
        assert cfg["optim"]["lr"] == r["lr"] and cfg["train"]["budgets_seconds"] == {"20m": 1200}
        assert cfg["schedule"]["warmup_steps"] == 100
    write_jsonl(rows, tmp_path / "lr.jsonl")
    assert read_jsonl(tmp_path / "lr.jsonl") == rows


def test_core_sweep_rows_have_no_lr(tasks_dir, tmp_path):
    rows = make_sweep("main", tasks_dir, models=CORE, fractions=[0.1, 1.0], seeds=[0, 1], repo_root=tmp_path)
    assert len(rows) == 12 and all("lr" not in r and r["overrides"] == [] for r in rows)


def test_rates_per_experiment():
    """The replication keeps The Well's rates whatever the search picks for the study."""
    assert load_config("cnext_unet", "replication")["optim"]["lr"] == 5e-4
    assert load_config("tfno", "replication")["optim"]["lr"] == 1e-4
    for m in CORE:
        cfg = load_config(m, "main")
        assert "model_overrides" not in cfg and cfg["optim"]["lr"] == 5e-4  # until the search is done


def test_model_overrides_touch_only_their_model(tmp_path):
    import shutil

    from rbbench.config import CONFIG_DIR

    d = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, d)
    exp = d / "experiments" / "main.yaml"
    before = {m: config_hash(load_config(m, config_dir=d)) for m in CORE}
    exp.write_text("model_overrides:\n  avit:\n    optim:\n      lr: 1.0e-3\n")
    after = {m: config_hash(load_config(m, config_dir=d)) for m in CORE}
    assert load_config("avit", config_dir=d)["optim"]["lr"] == 1e-3
    assert after["cnext_unet"] == before["cnext_unet"] and after["tfno"] == before["tfno"]
    assert after["avit"] != before["avit"]
    exp.write_text("model_overrides:\n  fno:\n    optim:\n      lr: 1.0e-3\n")
    with pytest.raises(ValueError, match="unknown models"):
        load_config("avit", config_dir=d)


def _fake_run(out, row, val_loss, val_vrmse, gpu_s=0.5, wait_s=0.25, windows=27440, trajectories=140):
    run = out / "runs" / row["run_id"]
    run.mkdir(parents=True)
    steps = int(1200 / gpu_s)
    milestone = {"target_seconds": 1200.0, "train_seconds": steps * gpu_s, "step": steps,
                 "samples_seen": steps * 64, "data_wait_seconds": steps * wait_s,
                 "validation": {"val_loss": val_loss, "val_vrmse": val_vrmse}}
    atomic_write_json(run / "COMPLETE.json", {"milestones": {"20m": milestone}})
    atomic_write_json(run / "provenance.json", {"attempts": [{"wall_seconds": 1800.0}],
                                                "n_train_windows": windows, "n_train_trajectories": trajectories})
    return run


def test_select_lr(tasks_dir, tmp_path, capsys):
    rows = make_sweep("lr_search", tasks_dir, models=("cnext_unet", "tfno"), repo_root=tmp_path)
    manifest = tmp_path / "lr.jsonl"
    write_jsonl(rows, manifest)
    out = tmp_path / "outputs"
    # (val loss, val VRMSE): CNextU-Net's lowest loss (1e-4) is not its lowest VRMSE (5e-4); VRMSE decides
    scores = {("cnext_unet", 1e-4): (0.10, 0.40), ("cnext_unet", 5e-4): (0.20, 0.30), ("cnext_unet", 1e-3): (0.25, 0.35),
              ("tfno", 1e-4): (0.40, 0.60), ("tfno", 5e-4): (0.35, 0.50)}
    for r in rows:
        if (r["model"], r["lr"]) in scores:
            _fake_run(out, r, *scores[r["model"], r["lr"]])
    args = ["select-lr", "--manifest", str(manifest), "--output-root", str(out), "--tasks-dir", str(tasks_dir)]

    # TFNO's 1e-3 has not run yet: no choice for TFNO
    assert main(args) == 1
    sel = json.loads((out / "lr_search" / "selection.json").read_text())
    assert sel["chosen"]["cnext_unet"]["lr"] == 5e-4 and not sel["chosen"]["cnext_unet"]["at_grid_edge"]
    assert sel["chosen"]["tfno"] == {"lr": None, "pending_lrs": [1e-3]}

    # it diverges: counted as the worst, TFNO takes 5e-4
    tfno_hi = next(r for r in rows if (r["model"], r["lr"]) == ("tfno", 1e-3))
    run = out / "runs" / tfno_hi["run_id"]
    run.mkdir(parents=True)
    atomic_write_json(run / "FAILED.json", {"reason": "non-finite loss"})
    capsys.readouterr()
    assert main(args) == 0
    text = capsys.readouterr().out
    sel = json.loads((out / "lr_search" / "selection.json").read_text())
    assert sel["chosen"]["tfno"]["lr"] == 5e-4 and "diverged" in text
    assert "cnext_unet:\n    optim:\n      lr: 5.0e-04" in text

    # pilot timings: 0.5 GPU s/step -> 7200 steps in 1 h, 28800 in 4 h; waiting 0.25 s is a third of wall
    t = {x["model"]: x for x in sel["pilot_timings"]}["cnext_unet"]
    assert t["steps_1h"] == pytest.approx(7200) and t["steps_4h"] == pytest.approx(28800)
    assert t["wait_fraction_of_wall"] == pytest.approx(1 / 3) and t["est_wall_hours_4h"] == pytest.approx(6.0)
    n10 = len(json.loads((tasks_dir / subset_filename(0.1)).read_text())["trajectory_ids"])
    assert t["epochs_4h_f010"] == pytest.approx(28800 * 64 / (n10 * 27440 / 140))
    assert (out / "lr_search" / "lr_search.csv").exists() and (out / "lr_search" / "pilot_timings.csv").exists()


def test_select_lr_refuses_a_core_manifest(tasks_dir, tmp_path):
    rows = make_sweep("main", tasks_dir, models=("avit",), fractions=[0.1], seeds=[0], repo_root=tmp_path)
    write_jsonl(rows, tmp_path / "train.jsonl")
    with pytest.raises(SystemExit, match="not a learning-rate search"):
        main(["select-lr", "--manifest", str(tmp_path / "train.jsonl"), "--output-root", str(tmp_path / "o")])
