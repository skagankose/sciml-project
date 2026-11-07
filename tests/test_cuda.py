"""Runs only on HPC3 GPU nodes (`pytest -m cuda`); skipped elsewhere."""

import pytest
import torch

from rbbench import MODEL_NAMES
from rbbench.config import load_config
from rbbench.device import FP32_ONLY_MODELS
from rbbench.models.factory import build_model
from rbbench.rollout import rollout

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_arch_supported():
    cap = "sm_%d%d" % torch.cuda.get_device_capability()
    assert cap in torch.cuda.get_arch_list()


RUNNABLE = [(n, p) for p in ("fp32", "bf16") for n in MODEL_NAMES if not (p == "bf16" and n in FP32_ONLY_MODELS)]


@pytest.mark.parametrize(("name", "precision"), RUNNABLE)
def test_full_resolution_step_and_rollout(name, precision):
    cfg = load_config(name, "main", "full")
    model = build_model(name, cfg["model"]["args"], (512, 128), seed=0).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = torch.randn(1, 4, 512, 128, 4, device="cuda"), torch.randn(1, 512, 128, 4, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        pred = model(x)
    loss = torch.nn.functional.mse_loss(pred.float(), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        out = rollout(lambda w: model(w).float(), x, 30)
    assert out.shape == (1, 30, 512, 128, 4) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(FP32_ONLY_MODELS))
def test_bf16_fft_still_unsupported(name):
    """The A100 finding behind FP32_ONLY_MODELS (2026-11-07). If this starts failing, the installed torch
    runs bf16 FFTs and the fp32-only rule can be revisited (decisions D8)."""
    cfg = load_config(name, "main", "full")
    model = build_model(name, cfg["model"]["args"], (512, 128), seed=0).cuda()
    x = torch.randn(1, 4, 512, 128, 4, device="cuda")
    with pytest.raises(RuntimeError, match="Unsupported dtype BFloat16"), torch.autocast("cuda", dtype=torch.bfloat16):
        model(x)
