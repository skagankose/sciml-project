"""VRMSE through the pinned upstream implementation, plus sample-weighted aggregation.

Upstream `the_well.benchmark.metrics.VRMSE` (pinned commit) computes, per sample, time and
field, sqrt( mean_xy (pred - true)^2 / (std_xy(true)^2 + eps) ), with torch's unbiased std
and eps = 1e-7, on denormalised (physical-unit) fields. The field average is taken after.
That is VRMSE, not NRMSE (which divides by mean(true^2)) and not RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

VRMSE_EPS = 1e-7


@dataclass
class _Meta:
    n_spatial_dims: int = 2


def vrmse(pred: torch.Tensor, true: torch.Tensor, eps: float = VRMSE_EPS) -> torch.Tensor:
    """Per-field VRMSE over the two spatial axes: [..., X, Y, F] -> [..., F]."""
    from the_well.benchmark.metrics import VRMSE

    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(true.shape)}")
    return VRMSE.eval(pred.float(), true.float(), _Meta(), eps=eps)


def vrmse_reference(pred: np.ndarray, true: np.ndarray, eps: float = VRMSE_EPS) -> np.ndarray:
    """Independent NumPy re-implementation used only to cross-check the upstream call."""
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    mse = ((pred - true) ** 2).mean(axis=(-3, -2))
    var = true.var(axis=(-3, -2), ddof=1)
    return np.sqrt(mse / (var + eps))


@dataclass
class MeanAccumulator:
    """Running mean weighted by sample count, so a short last batch is weighted correctly."""

    total: dict = field(default_factory=dict)
    count: dict = field(default_factory=dict)

    def add(self, name: str, values: torch.Tensor | np.ndarray) -> None:
        """`values` holds one entry (or row) per sample along axis 0."""
        arr = values.detach().double().cpu().numpy() if torch.is_tensor(values) else np.asarray(values, np.float64)
        self.total[name] = self.total.get(name, 0.0) + arr.sum(axis=0)
        self.count[name] = self.count.get(name, 0) + arr.shape[0]

    def mean(self, name: str) -> np.ndarray:
        return self.total[name] / self.count[name]

    def means(self) -> dict:
        return {k: self.mean(k) for k in self.total}
