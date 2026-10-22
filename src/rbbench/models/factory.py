"""One entry point for the four model families. All weights are randomly initialised."""

from __future__ import annotations

import torch
from torch import nn

from rbbench import MODEL_NAMES

UPSTREAM_CLASS = {
    "cnext_unet": "the_well.benchmark.models.UNetConvNext",
    "tfno": "the_well.benchmark.models.TFNO -> neuralop.models.TFNO (FNO, factorization='Tucker')",
    "avit": "the_well.benchmark.models.AViT",
    "convlstm": "rbbench.vendor.openstl.convlstm_modules.ConvLSTMCell (OpenSTL)",
}


def build_model(
    name: str,
    args: dict,
    resolution,
    n_steps_input: int = 4,
    n_fields: int = 4,
    seed: int | None = None,
) -> nn.Module:
    """Build `name` from scratch. With `seed`, initialisation is reproducible."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    if seed is not None:
        torch.manual_seed(seed)
    if name == "cnext_unet":
        from rbbench.models.well_models import build_cnext_unet as builder
    elif name == "tfno":
        from rbbench.models.well_models import build_tfno as builder
    elif name == "avit":
        from rbbench.models.well_models import build_avit as builder
    else:
        from rbbench.models.convlstm import build_convlstm as builder
    return builder(dict(args), resolution, n_steps_input, n_fields)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_used(model: nn.Module, example: torch.Tensor) -> int:
    """Parameters that receive a gradient from one forward/backward pass."""
    model.zero_grad(set_to_none=True)
    model(example).float().pow(2).mean().backward()
    used = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return used


def model_from_config(cfg: dict, seed: int | None = None) -> nn.Module:
    task = cfg["task"]
    return build_model(
        cfg["model"]["name"],
        cfg["model"]["args"],
        task["spatial_resolution"],
        task["n_steps_input"],
        len(task["fields"]),
        seed=seed,
    )
