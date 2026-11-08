"""Pipeline check with The Well's released checkpoints, tested offline: `fetch` is replaced by
a local tiny checkpoint, so no download happens."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_model

from rbbench import released
from rbbench.cli import main
from rbbench.config import load_config
from rbbench.data import full_trajectory_dataset, list_trajectories
from rbbench.evaluate import evaluate, evaluate_one_step, make_cohort
from rbbench.models.factory import model_from_config
from rbbench.rollout import Persistence

GRID = "task.spatial_resolution=[64,32]"


def _released_config(cfg):
    task = cfg["task"]
    return {"dim_in": 16, "dim_out": 4, "n_spatial_dims": 2, "spatial_resolution": list(task["spatial_resolution"]),
            **cfg["model"]["args"]}


@pytest.fixture
def fake_fetch(tmp_path, monkeypatch):
    """Replace the download with tiny local checkpoints in the released format."""
    def fetch(name):
        cfg = load_config(name, "main", "tiny", [GRID])
        path = tmp_path / f"{name}.safetensors"
        save_model(model_from_config(cfg, seed=7).net, str(path))
        return {"repo_id": f"local/{name}", "revision": "test", "sha256": "test",
                "config": _released_config(cfg), "weights_path": str(path)}

    monkeypatch.setattr(released, "fetch", fetch)
    return fetch


def test_full_configs_match_the_released_ones():
    # The released config.json files (revisions pinned in released.py), checked 2026-11-07.
    cnext = {"blocks_at_neck": 1, "blocks_per_stage": 2, "dim_in": 16, "dim_out": 4, "gradient_checkpointing": False,
             "init_features": 42, "n_spatial_dims": 2, "spatial_resolution": [512, 128], "stages": 4}
    tfno = {"dim_in": 16, "dim_out": 4, "gradient_checkpointing": False, "hidden_channels": 128, "modes1": 16,
            "modes2": 16, "modes3": 16, "n_spatial_dims": 2, "spatial_resolution": [512, 128]}
    assert released.config_differences(cnext, load_config("cnext_unet")) == []
    assert released.config_differences(tfno, load_config("tfno")) == []
    assert released.config_differences({**cnext, "init_features": 32}, load_config("cnext_unet")) == [
        "init_features: released 32, ours 42"]


def test_load_released_is_strict_and_refuses_other_architectures(fake_fetch):
    cfg = load_config("cnext_unet", "main", "tiny", [GRID])
    fetched = fake_fetch("cnext_unet")
    model = released.load_released("cnext_unet", cfg, fetched)
    expected = model_from_config(cfg, seed=7)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), expected.state_dict().values()))
    with pytest.raises(ValueError, match="init_features"):
        released.load_released("cnext_unet", cfg, {**fetched, "config": {**fetched["config"], "init_features": 99}})


def test_fetch_rejects_a_wrong_file(tmp_path, monkeypatch):
    import huggingface_hub

    bad = tmp_path / "model.safetensors"
    bad.write_bytes(b"")  # what the conda downloads on /pub looked like
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, name, revision=None: str(tmp_path / name))
    with pytest.raises(RuntimeError, match="sha256"):
        released.fetch("tfno")


def test_one_step_matches_the_main_evaluation(dataset_dir, tmp_path):
    """The pipeline check scores with exactly the code that scores our own runs."""
    cfg = load_config("cnext_unet", "smoke", "tiny", [GRID])
    _, total = full_trajectory_dataset(dataset_dir, "test")
    cohort = make_cohort(dataset_dir, cfg["eval"], 4, total)
    evaluate({"p": Persistence()}, cfg, dataset_dir, cohort, tmp_path, torch.device("cpu"))
    main_rows = pd.read_parquet(tmp_path / "one_step_p.parquet")
    alone = evaluate_one_step({"p": Persistence()}, dataset_dir, "test", 4, cfg["eval"], torch.device("cpu"))["p"]
    pd.testing.assert_frame_equal(main_rows, alone)


def test_eval_released_command(dataset_dir, fake_fetch, tmp_path):
    args = ["eval-released", "--data-root", str(dataset_dir.parents[1]), "--output-root", str(tmp_path / "out"),
            "--device", "cpu", "--size", "tiny", "--override", GRID]
    with pytest.raises(SystemExit, match="not verified"):
        main(args)
    code = main(args + ["--allow-unverified-data"])
    record = json.loads((tmp_path / "out" / "baselines" / "released" / "released_check.json").read_text())
    assert code == 1 and record["passed"] is False  # random tiny weights are worse than the paper
    for name in ("cnext_unet", "tfno"):
        for split in ("valid", "test"):
            r = record["results"][name][split]
            assert r["n_windows"] == len(list_trajectories(dataset_dir, split)) * (24 - 4)  # every window
            assert r["published_vrmse_mean"] == released.PUBLISHED_VRMSE[name][split]
            assert r["paper_protocol"] == released.PAPER_PROTOCOL[split]
            assert r["at_least_as_good"] == (r["vrmse_mean"] <= r["published_vrmse_mean"])
            df = pd.read_parquet(tmp_path / "out" / "baselines" / "released" / name / f"one_step_{split}.parquet")
            assert {"vrmse_mean", "paper_valid_vrmse_mean", "paper_test_vrmse_mean"} <= set(df.columns)
    assert record["checkpoints"]["tfno"]["repo_id"] == "local/tfno" and record["provenance"]["git"]
    assert record["checkpoints"]["tfno"]["forward"] == released.PAPER_FORWARD["tfno"]
    assert record["checkpoints"]["cnext_unet"]["forward"] == "pinned the_well"
    partial = main(args + ["--allow-unverified-data", "--max-trajectories", "1"])
    record = json.loads((tmp_path / "out" / "baselines" / "released" / "released_check.json").read_text())
    assert partial == 0 and record["passed"] is None  # a partial run is never reported as a check


def test_paper_forward_skips_the_fno_blocks(fake_fetch):
    """the_well <= v1.2.0 dropped the FNO block outputs (fixed in e430160), so the released TFNO is
    lifting -> projection; only TFNO gets that forward back."""
    cfg = load_config("tfno", "main", "tiny", [GRID])
    fetched = fake_fetch("tfno")
    model = released.load_released("tfno", cfg, fetched)
    pinned = released.load_released("tfno", cfg, fetched, paper_forward=False)
    assert model.released_forward == released.PAPER_FORWARD["tfno"] and pinned.released_forward is None
    x = torch.randn(2, 4, 64, 32, 4)
    with torch.no_grad():
        inner = pinned.net.model
        expected = inner.projection(inner.lifting(torch.einsum("btxyf->btfxy", x).reshape(2, 16, 64, 32)))
        assert torch.allclose(model(x), expected.permute(0, 2, 3, 1), atol=1e-6)
        assert not torch.allclose(model(x), pinned(x), atol=1e-3)  # the blocks do change the output
    cnext = released.load_released("cnext_unet", load_config("cnext_unet", "main", "tiny", [GRID]),
                                   fake_fetch("cnext_unet"))
    assert cnext.released_forward is None


def test_paper_protocols(dataset_dir):
    """Persistence has known answers: on raw input it returns the raw last frame, so the paper-test score
    equals ours; VRMSE is scale-invariant, so the normalised-units score matches too (healthy variance)."""
    cfg = load_config("cnext_unet", "smoke", "tiny", [GRID])
    plain = evaluate_one_step({"p": Persistence()}, dataset_dir, "valid", 4, cfg["eval"], torch.device("cpu"))["p"]
    both = evaluate_one_step({"p": Persistence()}, dataset_dir, "valid", 4, cfg["eval"], torch.device("cpu"),
                             paper_protocols=True)["p"]
    pd.testing.assert_frame_equal(plain, both[plain.columns])  # our scores are unchanged
    assert np.allclose(both.paper_test_vrmse_mean, both.vrmse_mean, rtol=1e-5)
    assert np.allclose(both.paper_valid_vrmse_mean, both.vrmse_mean, rtol=1e-3)
