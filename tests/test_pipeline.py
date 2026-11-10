"""End-to-end smoke: the same command the README asks you to run before every push."""

import pandas as pd

from rbbench.cli import main
from rbbench.manifests import read_json


def test_smoke_all_models(tmp_path):
    assert main(["smoke", "--device", "cpu", "--synthetic", "--grid", "64", "32", "--steps", "2",
                 "--all-models", "--workdir", str(tmp_path)]) == 0
    assert read_json(tmp_path / "audit.json")["physics_checks"]["passed"]
    rows = [l for l in (tmp_path / "tasks" / "train.jsonl").read_text().splitlines() if l]
    assert len(rows) == 4
    main_results = pd.read_csv(tmp_path / "tables" / "main_results.csv")
    assert set(main_results.model) == {"cnext_unet", "tfno", "avit", "convlstm"}
    assert {"one_step_vrmse_mean", "rollout_vrmse_mean", "nusselt_rel_err", "log_spectral_error",
            "paper_one_step_normalised_vrmse_mean", "paper_rollout_normalised_vrmse_steps_6-12",
            "paper_rollout_normalised_vrmse_steps_13-30"} <= set(main_results.metric)
    paper = pd.read_csv(tmp_path / "tables" / "paper_protocol.csv")
    assert set(paper.model) == {"cnext_unet", "tfno", "avit", "convlstm"} and paper["mean"].notna().all()
    for run in (tmp_path / "outputs" / "runs").glob("*/*/*/*"):
        assert (run / "COMPLETE.json").exists()
        assert (run / "evaluations" / "EVAL_COMPLETE_4h.json").exists()
        prov = read_json(run / "provenance.json")
        assert prov["attempts"][0]["git"] and prov["n_parameters"] > 0
        assert prov["attempts"][0]["end_status"] == "trained" and prov["attempts"][0]["wall_seconds"] > 0
        assert read_json(run / "evaluations" / "EVAL_COMPLETE_1h.json")["pass"]["labels"] == ["1h", "4h"]
    gpu_hours = pd.read_csv(tmp_path / "tables" / "gpu_hours.csv")
    assert len(gpu_hours) == 5 and gpu_hours.run_id.iloc[-1] == "TOTAL"
    assert (gpu_hours.train_wall_hours > 0).all() and (gpu_hours.train_gpu_hours == 0).all()  # CPU only
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 7


def test_cli_help():
    import pytest

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
