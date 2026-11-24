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


def build_cnext_unet(args: dict, resolution, n_steps_input: int, n_fields: int, n_out: int | None = None) -> nn.Module:
    from the_well.benchmark.models import UNetConvNext

    net = UNetConvNext(
        dim_in=n_steps_input * n_fields,
        dim_out=n_out or n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    return WellModelAdapter(net, "channels_first", n_steps_input, n_fields)


def build_tfno(args: dict, resolution, n_steps_input: int, n_fields: int, n_out: int | None = None) -> nn.Module:
    from the_well.benchmark.models import TFNO

    net = TFNO(
        dim_in=n_steps_input * n_fields,
        dim_out=n_out or n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    return WellModelAdapter(net, "channels_first", n_steps_input, n_fields)


# The Well's 2-D AxialAttentionBlock (e63dec8; unchanged on master, checked 2026-11-24) applies its second
# permutation to q, k, v already in the first pass's layout, "b h he w c -> b h w he c", so the second
# scaled_dot_product_attention runs over the heads (`he`) instead of the first spatial axis `h`. Every block then
# mixes tokens only along `w` (y here): each 16-pixel-wide strip in x is predicted from itself
# (docs/report_notes.md section 17). The 3-D path chains its permutations correctly; the fix does the same in 2-D.
UPSTREAM_2D_PERMUTATIONS = [
    ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
    ("b h he w c -> b h w he c", "b h w he c -> b h w (he c)"),
]
FIXED_2D_PERMUTATIONS = [
    ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),   # along w (second spatial axis), as upstream
    ("b h he w c -> b w he h c", "b w he h c -> b h w (he c)"),   # along h (first spatial axis)
]


def fix_axial_attention(net: nn.Module) -> nn.Module:
    """Make every 2-D axial-attention block of an upstream AViT attend along both spatial axes. Parameters and
    state-dict keys are unchanged; only the order of axes in the attention changes."""
    for blk in net.blocks:
        if blk.n_spatial_dims != 2:
            raise ValueError("the axial-attention fix is for 2-D AViT only")
        if [tuple(p) for p in blk.spatial_permutations] != UPSTREAM_2D_PERMUTATIONS:
            raise RuntimeError("upstream AxialAttentionBlock permutations changed; re-check the fix "
                               f"(got {blk.spatial_permutations})")
        blk.spatial_permutations = list(FIXED_2D_PERMUTATIONS)
    return net


def build_avit(args: dict, resolution, n_steps_input: int, n_fields: int, n_out: int | None = None,
               fix_axial: bool = False) -> nn.Module:
    """The Well's AViT; with `fix_axial`, its attention blocks mix along both spatial axes (model `avit_fixed`)."""
    from the_well.benchmark.models import AViT

    if any(r % 16 for r in resolution):
        raise ValueError(f"AViT needs both spatial sizes divisible by 16, got {resolution}")
    if (args["hidden_dim"] // 4) % 12:
        raise ValueError("AViT hidden_dim/4 must be divisible by 12 (GroupNorm groups)")
    net = AViT(
        dim_in=n_steps_input * n_fields,
        dim_out=n_out or n_fields,
        n_spatial_dims=2,
        spatial_resolution=tuple(resolution),
        **args,
    )
    if fix_axial:
        fix_axial_attention(net)
    return WellModelAdapter(net, "channels_last", n_steps_input, n_fields)
