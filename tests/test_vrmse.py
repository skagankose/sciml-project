import numpy as np
import pytest
import torch

from rbbench.metrics.well_metric import MeanAccumulator, vrmse, vrmse_reference


def test_matches_independent_reference():
    g = torch.Generator().manual_seed(0)
    true = torch.randn(3, 5, 16, 8, 4, generator=g)
    pred = true + 0.3 * torch.randn(3, 5, 16, 8, 4, generator=g)
    ours = vrmse(pred, true).numpy()
    assert ours.shape == (3, 5, 4)
    np.testing.assert_allclose(ours, vrmse_reference(pred.numpy(), true.numpy()), rtol=1e-5)


def test_known_values():
    true = torch.zeros(1, 4, 2, 1)
    true[0, :, :, 0] = torch.tensor([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])  # var = 8/7
    assert vrmse(true.clone(), true)[0, 0].item() == pytest.approx(0.0)
    mean_pred = torch.zeros_like(true)  # predicting the mean
    expected = np.sqrt(1.0 / (8 / 7 + 1e-7))
    assert vrmse(mean_pred, true)[0, 0].item() == pytest.approx(expected, rel=1e-6)


def test_differs_from_nrmse_and_rmse():
    g = torch.Generator().manual_seed(0)
    true = 2.0 * torch.randn(1, 16, 8, 1, generator=g) + 20.0  # large mean: NRMSE << VRMSE; std 2: VRMSE ~ RMSE/2
    pred = true + 0.1
    v = vrmse(pred, true).item()
    nrmse = (0.1**2 / (true**2).mean()).sqrt().item()
    assert v > 5 * nrmse and abs(v - 0.1) > 1e-3


def test_sample_weighted_accumulation_no_last_batch_bias():
    values = torch.arange(10.0)
    acc = MeanAccumulator()
    for chunk in values.split(4):  # batches of 4, 4, 2
        acc.add("m", chunk)
    assert acc.mean("m") == pytest.approx(4.5)
    batch_mean = np.mean([c.mean().item() for c in values.split(4)])
    assert batch_mean != pytest.approx(4.5)  # the bias the accumulator avoids
