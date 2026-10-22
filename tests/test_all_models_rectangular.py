import pytest
import torch

from rbbench import MODEL_NAMES
from rbbench.config import load_config
from rbbench.models.factory import build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("grid", [(64, 32), (128, 32)])
def test_tiny_models_rectangular(name, grid):
    cfg = load_config(name, "smoke", "tiny")
    model = build_model(name, cfg["model"]["args"], grid, seed=0)
    x = torch.randn(2, 4, *grid, 4)
    y = model(x)
    assert y.shape == (2, *grid, 4)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_rejects_wrong_history_length(name):
    cfg = load_config(name, "smoke", "tiny")
    model = build_model(name, cfg["model"]["args"], (64, 32), seed=0)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 64, 32, 4))


@pytest.mark.slow
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_full_size_forward_512x128(name):
    """Full-size models at the real resolution (CPU, batch 1); A100 memory is checked in Phase G."""
    cfg = load_config(name, "main", "full")
    model = build_model(name, cfg["model"]["args"], (512, 128), seed=0)
    with torch.no_grad():
        y = model(torch.randn(1, 4, 512, 128, 4))
    assert y.shape == (1, 512, 128, 4)
    n = sum(p.numel() for p in model.parameters())
    assert 12e6 < n < 21e6, n
