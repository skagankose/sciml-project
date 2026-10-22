import numpy as np
import torch

from rbbench.noise import perturb_window, wall_rows
from rbbench.rollout import Persistence, rollout


def test_deterministic_and_order_independent():
    w = torch.randn(4, 16, 8, 4)
    a = perturb_window(w, "data/test/f.hdf5#3", 20, 0.05, 0)
    _ = perturb_window(w, "other#1", 7, 0.05, 0)  # unrelated draw in between
    b = perturb_window(w, "data/test/f.hdf5#3", 20, 0.05, 0)
    assert torch.equal(a, b)
    assert not torch.equal(a, perturb_window(w, "data/test/f.hdf5#3", 20, 0.05, 1))
    assert not torch.equal(a, perturb_window(w, "data/test/f.hdf5#3", 21, 0.05, 0))


def test_sigma_zero_is_clean_and_input_untouched():
    w = torch.randn(4, 16, 8, 4)
    keep = w.clone()
    assert torch.equal(perturb_window(w, "t#0", 0, 0.0, 0), w)
    perturb_window(w, "t#0", 0, 0.2, 0)
    assert torch.equal(w, keep)


def test_scale_and_fields_independent():
    w = torch.zeros(4, 64, 32, 4)
    d = perturb_window(w, "t#0", 0, 0.1, 0)
    assert abs(d.std().item() - 0.1) < 0.005
    assert not torch.allclose(d[..., 0], d[..., 1])


def test_only_initial_frames_are_noisy():
    """Rollout from a noisy context: predictions come from the model, not from new noise."""
    ctx = torch.randn(1, 4, 8, 6, 4)
    noisy = perturb_window(ctx[0], "t#0", 0, 0.1, 0)[None]
    out = rollout(Persistence(), noisy, 5)
    assert torch.equal(out[:, -1], noisy[:, -1])


def test_wall_mask():
    y = np.array([0.0, 0.1, 0.5, 1.0])
    assert wall_rows(y, 0.0, 1.0).tolist() == [True, False, False, True]
    w = torch.zeros(4, 8, 4, 4)
    d = perturb_window(w, "t#0", 0, 0.1, 0, keep_y=wall_rows(y, 0.0, 1.0))
    assert d[:, :, [0, 3]].abs().sum() == 0 and d[:, :, 1:3].abs().sum() > 0
