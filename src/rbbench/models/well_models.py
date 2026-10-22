"""Adapters around the upstream The Well benchmark models.

All project models take the canonical window `[B, T, X, Y, F]` (The Well sample layout with
a batch axis) and return the next frame `[B, X, Y, F]`. Time and fields are stacked into
channels exactly like upstream `DefaultChannels{First,Last}Formatter`: channel = t * F + f.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class WellModelAdapter(nn.Module):
    def __init__(self, net: nn.Module, layout: str, n_steps_input: int, n_fields: int):
        super().__init__()
        if layout not in ("channels_first", "channels_last"):
            raise ValueError(layout)
        self.net = net
        self.layout = layout
        self.n_steps_input = n_steps_input
        self.n_fields = n_fields

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.n_steps_input or x.shape[-1] != self.n_fields:
            raise ValueError(
                f"expected [B, {self.n_steps_input}, X, Y, {self.n_fields}], got {tuple(x.shape)}"
            )
        if self.layout == "channels_first":
            out = self.net(rearrange(x, "b t x y f -> b (t f) x y"))
            return rearrange(out, "b f x y -> b x y f")
        return self.net(rearrange(x, "b t x y f -> b x y (t f)"))


def build_cnext_unet(args: dict, resolution, n_steps_input: int, n_fields: int) -> nn.Module:
    from the_well.benchmark.models import UNetConvNext

    net = UNetConvNext(
        dim_in=n_steps_input * n_fields,
        dim_out=n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    return WellModelAdapter(net, "channels_first", n_steps_input, n_fields)


def build_tfno(args: dict, resolution, n_steps_input: int, n_fields: int) -> nn.Module:
    from the_well.benchmark.models import TFNO

    net = TFNO(
        dim_in=n_steps_input * n_fields,
        dim_out=n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    return WellModelAdapter(net, "channels_first", n_steps_input, n_fields)


def build_avit(args: dict, resolution, n_steps_input: int, n_fields: int) -> nn.Module:
    from the_well.benchmark.models import AViT

    if any(r % 16 for r in resolution):
        raise ValueError(f"AViT needs both spatial sizes divisible by 16, got {resolution}")
    if (args["hidden_dim"] // 4) % 12:
        raise ValueError("AViT hidden_dim/4 must be divisible by 12 (GroupNorm groups)")
    net = AViT(
        dim_in=n_steps_input * n_fields,
        dim_out=n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    return WellModelAdapter(net, "channels_last", n_steps_input, n_fields)
