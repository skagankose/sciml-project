"""The corrected AViT in the findings and the final figures (D32): it joins the core evaluations, replaces AViT as
released in the main figures, and its UQ ensembles and Slurm jobs are labelled correctly."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_figures as F  # noqa: E402
import findings_figures as ff  # noqa: E402
import uq_findings  # noqa: E402


def _runs(tmp_path):
    rows = []
    for manifest, model in (("train", "cnext_unet"), ("train", "tfno"), ("train", "avit"),
                            ("train_avit_fixed", "avit_fixed"), ("lr_search", "avit")):
        run_id = f"{model}/f010/s0/{manifest}"
        ev = tmp_path / "runs" / run_id / "evaluations"
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "EVAL_COMPLETE_4h.json").write_text("{}")
        rows.append({"manifest": manifest, "run_id": run_id, "model": model, "fraction": 0.1, "seed": 0})
    return pd.DataFrame(rows)


def test_core_evaluations_include_the_corrected_avit(tmp_path):
    ev = ff.evaluated_runs(_runs(tmp_path), tmp_path)
    assert sorted(ev.model) == ["avit", "avit_fixed", "cnext_unet", "tfno"]          # not the lr-search run
    assert sorted(ff.evaluated_runs(_runs(tmp_path), tmp_path, manifest="train").model) == ["avit", "cnext_unet", "tfno"]


def test_main_runs_replace_avit_as_released(tmp_path):
    main = F.main_runs(_runs(tmp_path))
    assert sorted(main.model) == ["avit_fixed", "cnext_unet", "tfno"]
    assert set(main.manifest) == {"train"}                 # helpers that select manifest == "train" take all three


def test_main_style_labels(monkeypatch):
    from rbbench.aggregate import STYLE

    saved = {k: dict(STYLE[k]) for k in ("avit", "avit_fixed")}
    try:
        F.use_main_style()
        assert STYLE["avit_fixed"]["label"] == "AViT" and STYLE["avit_fixed"]["color"] == saved["avit"]["color"]
        assert STYLE["avit"]["label"] == "AViT as released"
    finally:
        STYLE.update(saved)


def test_uq_ensemble_names_parse_family_and_fraction():
    specs = {e["name"]: e for e in uq_findings.ensembles()}
    e = specs["avit_fixed_f010_4h_seeds"]
    assert e["family"] == "avit_fixed" and e["fraction"] == 0.1 and e["kind"] == "seed pair (4 h)"
    e = specs["cnext_unet_f100_1h_betanll"]
    assert e["family"] == "cnext_unet" and e["fraction"] == 1.0 and e["kind"] == "beta-NLL ensemble"


def test_slurm_purpose_of_the_corrected_avit():
    assert ff.training_purpose("train_avit_fixed", 1.0) == "AViT (fixed) runs"
    assert "AViT (fixed) runs" in ff.TRAINING_PURPOSES
