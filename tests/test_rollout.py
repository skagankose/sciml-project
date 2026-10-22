import torch

from rbbench.rollout import Persistence, rollout


def test_persistence_repeats_last_frame():
    ctx = torch.randn(2, 4, 8, 6, 4)
    out = rollout(Persistence(), ctx, 30)
    assert out.shape == (2, 30, 8, 6, 4)
    assert torch.equal(out, ctx[:, -1:].expand_as(out))


def test_linear_extrapolation_hand_computed():
    """x_{t+1} = 2 x_t - x_{t-1} on a linear sequence continues it exactly."""
    ctx = torch.arange(4.0).view(1, 4, 1, 1, 1).expand(1, 4, 3, 2, 4).clone()
    out = rollout(lambda w: 2 * w[:, -1] - w[:, -2], ctx, 30)
    assert torch.allclose(out[0, :, 0, 0, 0], torch.arange(4.0, 34.0))


def test_window_shifts_and_uses_only_predictions():
    seen = []

    def predictor(w):
        seen.append(w[0, :, 0, 0, 0].tolist())
        return w[:, -1] + 100

    ctx = torch.arange(4.0).view(1, 4, 1, 1, 1)
    rollout(predictor, ctx, 3)
    assert seen == [[0, 1, 2, 3], [1, 2, 3, 103], [2, 3, 103, 203]]


def test_shape_mismatch_rejected():
    import pytest

    with pytest.raises(ValueError):
        rollout(lambda w: w[:, -1, :2], torch.randn(1, 4, 8, 6, 4), 1)
