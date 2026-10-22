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
    assert {"one_step_vrmse_mean", "rollout_vrmse_mean", "nusselt_rel_err", "log_spectral_error"} <= set(
        main_results.metric
    )
    for run in (tmp_path / "outputs" / "runs").glob("*/*/*/*"):
        assert (run / "COMPLETE.json").exists()
        assert (run / "evaluations" / "EVAL_COMPLETE_6h.json").exists()
        prov = read_json(run / "provenance.json")
        assert prov["attempts"][0]["git"] and prov["n_parameters"] > 0
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 7


def test_cli_help():
    import pytest

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
