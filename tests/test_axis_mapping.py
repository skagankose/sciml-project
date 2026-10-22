import h5py
import torch

from rbbench.data import (
    FIELDS,
    from_sequence,
    list_trajectories,
    make_well_dataset,
    stack_channels,
    to_sequence,
    trajectory_window_range,
    unstack_channels,
)
from rbbench.models.well_models import WellModelAdapter


def test_round_trips():
    x = torch.randn(2, 4, 8, 6, 4)
    assert torch.equal(unstack_channels(stack_channels(x), 4), x)
    assert torch.equal(from_sequence(to_sequence(x)), x)


def test_channel_order_is_time_major():
    x = torch.zeros(1, 4, 8, 6, 4)
    x[0, 2, :, :, 3] = 1.0  # t=2, f=velocity_y
    c = stack_channels(x)
    assert c[0, 2 * 4 + 3].eq(1).all() and c.sum() == 8 * 6


def test_adapter_matches_upstream_formatter():
    """Our adapter feeds the network exactly what upstream's formatter would."""
    from einops import rearrange

    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, z):
            seen["z"] = z
            return z[:, :4]

    x = torch.randn(2, 4, 8, 6, 4)
    WellModelAdapter(Probe(), "channels_first", 4, 4)(x)
    assert torch.equal(seen["z"], rearrange(x, "b t ... c -> b (t c) ..."))


def test_x_is_horizontal_and_fields_in_order(dataset_dir):
    ds = make_well_dataset(dataset_dir, "train")
    assert ds.size_tuple == (64, 32)  # x (periodic, 64) then y (Chebyshev, 32)
    ref = list_trajectories(dataset_dir, "train")[5]
    idx = trajectory_window_range(ds, ref, dataset_dir).start + 3
    sample = ds[idx]
    with h5py.File(dataset_dir / ref.relpath) as h:
        raw = {
            "buoyancy": h["t0_fields/buoyancy"][ref.index, 3:8],
            "pressure": h["t0_fields/pressure"][ref.index, 3:8],
            "velocity_x": h["t1_fields/velocity"][ref.index, 3:8, ..., 0],
            "velocity_y": h["t1_fields/velocity"][ref.index, 3:8, ..., 1],
        }
    frames = torch.cat([sample["input_fields"], sample["output_fields"]])
    phys = ds.norm.denormalize_flattened(frames, "variable")
    for f, name in enumerate(FIELDS):
        assert torch.allclose(phys[..., f], torch.as_tensor(raw[name]), atol=1e-5), name
