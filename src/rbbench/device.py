"""Device selection: CUDA, then MPS, then CPU. Nothing CUDA-specific runs at import."""

from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_runs_on(model: torch.nn.Module, example: torch.Tensor, device: torch.device) -> bool:
    """Whether one forward/backward pass works on `device` (e.g. TFNO's complex FFTs on MPS)."""
    try:
        m = model.to(device)
        out = m(example.to(device))
        out.float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True)
        return bool(torch.isfinite(out).all())
    except (RuntimeError, NotImplementedError, TypeError):
        return False
    finally:
        model.to("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device: torch.device, precision: str):
    """bf16 autocast on CUDA when requested; full precision otherwise."""
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 autocast is only supported on CUDA in this project")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision != "fp32":
        raise ValueError(f"unknown precision {precision!r} (expected fp32 or bf16)")
    return torch.autocast(device.type, enabled=False)
