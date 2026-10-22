"""Runs only on HPC3 GPU nodes (`pytest -m cuda`); skipped elsewhere."""

import pytest
import torch

from rbbench import MODEL_NAMES
from rbbench.config import load_config
from rbbench.models.factory import build_model
from rbbench.rollout import rollout

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_arch_supported():
    cap = "sm_%d%d" % torch.cuda.get_device_capability()
    assert cap in torch.cuda.get_arch_list()


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
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
