"""The one autoregressive procedure shared by every model and baseline.

Starting from an observed window of `n_input` frames, predict one frame, drop the oldest
frame, append the prediction and repeat. Ground truth is never passed in, so it cannot
leak into later steps. Everything here is in normalised units.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Predictor = Callable[[torch.Tensor], torch.Tensor]  # [B, T, X, Y, F] -> [B, X, Y, F]


@torch.no_grad()
def rollout(predict: Predictor, context: torch.Tensor, n_steps: int) -> torch.Tensor:
    """`context` [B, T, X, Y, F] -> predictions [B, n_steps, X, Y, F]."""
    window = context
    preds = []
    for _ in range(n_steps):
        nxt = predict(window)
        if nxt.shape != window[:, -1].shape:
            raise ValueError(f"predictor returned {tuple(nxt.shape)}, expected {tuple(window[:, -1].shape)}")
        preds.append(nxt)
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


class Persistence(torch.nn.Module):
    """Diagnostic reference x_{t+1} = x_t (not a learned model)."""

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return window[:, -1]
