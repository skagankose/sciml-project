from rbbench.config import budget_labels, deep_merge, load_config


def test_core_budgets_are_1h_and_4h():
    cfg = load_config("cnext_unet")
    assert cfg["train"]["budgets_seconds"] == {"1h": 3600, "4h": 14400}
    assert budget_labels(cfg) == ["1h", "4h"]
    assert cfg["eval"]["noise_sigmas"] == [0.0]  # noise robustness is an extension


def test_budget_overrides_replace_the_mapping():
    cfg = load_config("tfno", "pilot")
    assert set(cfg["train"]["budgets_seconds"]) == {"1h", "4h"}
    cfg = load_config("tfno", "main", "full", ["train.budgets_seconds={b: 20, a: 10}"])
    assert cfg["train"]["budgets_seconds"] == {"b": 20, "a": 10}
    assert budget_labels(cfg) == ["a", "b"]  # ordered by seconds, not by name or file order


def test_other_mappings_still_merge():
    assert deep_merge({"train": {"x": 1, "y": 2}}, {"train": {"y": 3}}) == {"train": {"x": 1, "y": 3}}


def test_noise_experiment_adds_levels():
    assert load_config("avit", "noise")["eval"]["noise_sigmas"] == [0.0, 0.01, 0.05, 0.10, 0.20]
