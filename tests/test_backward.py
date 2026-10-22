import ast
from pathlib import Path

import pytest
import torch

from rbbench import MODEL_NAMES
from rbbench.config import load_config
from rbbench.models.factory import build_model

SRC = Path(__file__).resolve().parents[1] / "src" / "rbbench"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_gradients_and_update(name):
    cfg = load_config(name, "smoke", "tiny")
    model = build_model(name, cfg["model"]["args"], (64, 32), seed=0)
    x, y = torch.randn(2, 4, 64, 32, 4), torch.randn(2, 64, 32, 4)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    opt.step()
    changed = sum(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_random_init_depends_only_on_seed(name):
    args = load_config(name, "smoke", "tiny")["model"]["args"]
    a = build_model(name, args, (64, 32), seed=0).state_dict()
    b = build_model(name, args, (64, 32), seed=0).state_dict()
    c = build_model(name, args, (64, 32), seed=1).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a if a[k].is_floating_point() and a[k].numel() > 1)


def test_no_pretrained_weight_loading_in_source():
    """Weights are only ever loaded from this project's own checkpoints."""
    for path in SRC.rglob("*.py"):
        if "vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("from_pretrained", "hf_hub_download"):
                raise AssertionError(f"{path}: uses {node.attr}")
