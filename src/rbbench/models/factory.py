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
    n_out: int | None = None,
) -> nn.Module:
    """Build `name` from scratch. With `seed`, initialisation is reproducible. `n_out` output channels per
    pixel (default `n_fields`; 2 x `n_fields` for a mean and a log-variance, D30)."""
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
    return builder(dict(args), resolution, n_steps_input, n_fields, n_out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_used(model: nn.Module, example: torch.Tensor) -> int:
    """Parameters that receive a gradient from one forward/backward pass."""
    model.zero_grad(set_to_none=True)
    model(example).float().pow(2).mean().backward()
    used = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return used


LOSSES = ("mse", "beta_nll")


def training_loss(cfg: dict) -> str:
    """`train.loss` (default mse). The core runs never set it, so their config hashes are unchanged (D30)."""
    loss = cfg.get("train", {}).get("loss", "mse")
    if loss not in LOSSES:
        raise ValueError(f"train.loss must be one of {LOSSES}, not {loss!r}")
    return loss


class GaussianOutput(nn.Module):
    """A mean and a log-variance per field and pixel, for training with the beta-NLL (D30, Lecture 17).

    The wrapped network outputs 2F channels: the first F are the mean, the last F the log-variance (clamped to
    [LOG_VAR_MIN, LOG_VAR_MAX] in normalised units). `forward` returns the mean only, so rollouts, validation and
    the standard evaluation treat the model like any other; `forward_dist` returns (mean, log-variance).
    """

    LOG_VAR_MIN, LOG_VAR_MAX = -16.0, 6.0

    def __init__(self, net: nn.Module, n_fields: int):
        super().__init__()
        self.net = net
        self.n_fields = n_fields

    def forward_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        if out.shape[-1] != 2 * self.n_fields:
            raise ValueError(f"expected {2 * self.n_fields} output channels, got {out.shape[-1]}")
        mean, log_var = out[..., :self.n_fields], out[..., self.n_fields:]
        return mean, log_var.clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[..., :self.n_fields]


def model_from_config(cfg: dict, seed: int | None = None) -> nn.Module:
    task = cfg["task"]
    n_fields = len(task["fields"])
    gaussian = training_loss(cfg) == "beta_nll"
    model = build_model(
        cfg["model"]["name"],
        cfg["model"]["args"],
        task["spatial_resolution"],
        task["n_steps_input"],
        n_fields,
        seed=seed,
        n_out=2 * n_fields if gaussian else None,
    )
    return GaussianOutput(model, n_fields) if gaussian else model
