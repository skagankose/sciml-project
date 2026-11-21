import numpy as np
import pytest
import torch

from rbbench.metrics.well_metric import MeanAccumulator, correlation, vrmse, vrmse_reference


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


def test_correlation_matches_numpy_and_known_values():
    g = torch.Generator().manual_seed(0)
    true = torch.randn(2, 3, 16, 8, 4, generator=g)
    pred = true + 0.5 * torch.randn(2, 3, 16, 8, 4, generator=g)
    r = correlation(pred, true).numpy()
    assert r.shape == (2, 3, 4)
    ref = np.corrcoef(pred[1, 2, ..., 3].numpy().ravel(), true[1, 2, ..., 3].numpy().ravel())[0, 1]
    assert r[1, 2, 3] == pytest.approx(ref, rel=1e-6)
    np.testing.assert_allclose(correlation(true, true).numpy(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(correlation(3.0 * true + 2.0, true).numpy(), 1.0, rtol=1e-12)  # affine: still 1
    np.testing.assert_allclose(correlation(-true, true).numpy(), -1.0, rtol=1e-12)
    assert np.isnan(correlation(torch.zeros_like(true), true).numpy()).all()  # a constant field has no correlation


def test_vrmse_and_correlation_of_a_displaced_structure():
    """The double penalty: a sharp structure in the wrong place scores VRMSE ~ sqrt(2) (correlation ~ 0), worse than
    predicting the mean (VRMSE 1), which is why the failure thresholds are read as 1 = the mean, ~1.41 = displaced."""
    x = torch.arange(64.0)
    blob = lambda c: torch.exp(-((x[:, None] - c) ** 2 + (x[None, :] - 32) ** 2) / 8.0)
    true = blob(16.0)[None, :, :, None]
    displaced = blob(48.0)[None, :, :, None]
    assert vrmse(displaced, true).item() == pytest.approx(np.sqrt(2), rel=0.02)
    assert abs(correlation(displaced, true).item()) < 0.05
    assert vrmse(torch.full_like(true, true.mean().item()), true).item() == pytest.approx(1.0, rel=0.02)
