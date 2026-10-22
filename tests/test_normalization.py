import torch
import yaml

from rbbench.data import FIELDS, make_well_dataset
from rbbench.synthetic import write_stats


def test_round_trip(dataset_dir):
    ds = make_well_dataset(dataset_dir, "valid")
    x = torch.randn(3, 4, 64, 32, 4)
    back = ds.norm.normalize_flattened(ds.norm.denormalize_flattened(x, "variable"), "variable")
    assert torch.allclose(back, x, atol=1e-5)


def test_uses_train_stats_for_every_split(dataset_dir):
    stats = yaml.safe_load((dataset_dir / "stats.yaml").read_text())
    for split in ("train", "valid", "test"):
        norm = make_well_dataset(dataset_dir, split).norm
        assert torch.allclose(norm.flattened_means["variable"][0], torch.tensor(stats["mean"]["buoyancy"]))


def test_stats_not_fitted_on_test(dataset_dir, tmp_path):
    """stats.yaml equals statistics of the train split, not of test."""
    import shutil

    copy = tmp_path / "rb"
    shutil.copytree(dataset_dir, copy)
    train = write_stats(copy, "train")
    test = write_stats(copy, "test")
    on_disk = yaml.safe_load((dataset_dir / "stats.yaml").read_text())
    assert on_disk == train
    assert train != test


def test_field_names_stable(dataset_dir):
    from the_well.data.utils import flatten_field_names

    ds = make_well_dataset(dataset_dir, "test")
    assert tuple(flatten_field_names(ds.metadata, include_constants=False)) == FIELDS
