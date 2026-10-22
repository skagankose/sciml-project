"""ConvLSTM next-frame model built on the vendored OpenSTL `ConvLSTMCell`.

Encoder -> stacked ConvLSTM cells unrolled over exactly `n_steps_input` frames -> decoder.
Like OpenSTL's `ConvLSTM_Model`, frames are space-to-depth patched first. Hidden and cell
states are zero-initialised inside every forward call, so a prediction depends only on its
own four-frame window, matching the information horizon of the other three models.

Note: OpenSTL's cell allocates `conv_o` and `conv_last` but never uses them in `forward`;
they count towards `parameters()` but receive no gradient (see docs/upstream_audit.md).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from rbbench.vendor.openstl.convlstm_modules import ConvLSTMCell


class ConvLSTMForecaster(nn.Module):
    def __init__(
        self,
        n_fields: int,
        n_steps_input: int,
        resolution,
        patch_size: int = 4,
        num_hidden=(128, 128, 128, 128),
        filter_size: int = 5,
        encoder_channels: int = 128,
        decoder_channels: int = 128,
    ):
        super().__init__()
        nx, ny = resolution
        if nx % patch_size or ny % patch_size:
            raise ValueError(f"resolution {resolution} not divisible by patch_size {patch_size}")
        self.n_fields = n_fields
        self.n_steps_input = n_steps_input
        self.patch_size = patch_size
        self.num_hidden = list(num_hidden)
        frame_channels = n_fields * patch_size**2
        height, width = nx // patch_size, ny // patch_size

        self.encoder = nn.Sequential(
            nn.Conv2d(frame_channels, encoder_channels, kernel_size=3, padding=1), nn.GELU()
        )
        self.cells = nn.ModuleList(
            ConvLSTMCell(
                encoder_channels if i == 0 else self.num_hidden[i - 1],
                hidden,
                height,
                width,
                filter_size,
                1,
                False,
            )
            for i, hidden in enumerate(self.num_hidden)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, frame_channels, kernel_size=1),
        )
        # Counts how many frames the cells consumed in the last forward (tested).
        self.last_unroll_length = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.n_steps_input or x.shape[-1] != self.n_fields:
            raise ValueError(
                f"expected [B, {self.n_steps_input}, X, Y, {self.n_fields}], got {tuple(x.shape)}"
            )
        p = self.patch_size
        frames = rearrange(x, "b t (h p1) (w p2) f -> b t (p1 p2 f) h w", p1=p, p2=p)
        b, t, _, h, w = frames.shape
        hs = [frames.new_zeros(b, n, h, w) for n in self.num_hidden]
        cs = [frames.new_zeros(b, n, h, w) for n in self.num_hidden]
        for step in range(t):
            inp = self.encoder(frames[:, step])
            hs[0], cs[0] = self.cells[0](inp, hs[0], cs[0])
            for i in range(1, len(self.cells)):
                hs[i], cs[i] = self.cells[i](hs[i - 1], hs[i], cs[i])
        self.last_unroll_length = t
        out = self.decoder(hs[-1])
        return rearrange(out, "b (p1 p2 f) h w -> b (h p1) (w p2) f", p1=p, p2=p)


def build_convlstm(args: dict, resolution, n_steps_input: int, n_fields: int) -> nn.Module:
    return ConvLSTMForecaster(n_fields, n_steps_input, tuple(resolution), **args)
