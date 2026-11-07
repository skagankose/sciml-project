"""The Well's released RB checkpoints, evaluated as a pipeline check (gate D2, AGENTS.md rule 3).

This is the only module allowed to download pretrained weights (tests/test_backward.py), and
nothing that trains imports it. Each checkpoint is pinned to a Hugging Face revision and the
SHA-256 of its weights file, and is loaded into this project's own full-size model, so the
check exercises exactly the architecture, data path and metric used for our runs.

Checked on the Mac on 2026-11-07: both configs equal our `full` configs (TFNO's extra
`modes3` is unused in 2-D), the weights load strictly, and the loaded models match upstream
`from_pretrained` bitwise on a random input.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch import nn

RELEASED = {
    "cnext_unet": {
        "repo_id": "polymathic-ai/UNetConvNext-rayleigh_benard",
        "revision": "c5fa0f7a50aca21b65fed95e0f7e8d1097eb2b8b",
        "sha256": "4d02a86e6fdb44b2670f25570ea3022223b9783717d5115b7d5b953ad1ceafbd",
    },
    "tfno": {
        "repo_id": "polymathic-ai/TFNO-rayleigh_benard",
        "revision": "7d351350eaf68caba1eac1e545cd6661251dd1fd",
        "sha256": "19ff90469c7b2f7a2aa661e7a47312533cfaa5fdd6dda71ec7aa91f2ed05484a",
    },
}

# One-step VRMSE (field mean) of these checkpoints, from docs/benchmarks.md at the pinned
# The Well commit (best-validation checkpoint after 12 h on one H100).
PUBLISHED_VRMSE = {
    "cnext_unet": {"valid": 0.4807, "test": 0.6699},
    "tfno": {"valid": 0.8568, "test": 0.6566},
}


def fetch(name: str) -> dict:
    """Download config.json and model.safetensors at the pinned revision; verify the weights."""
    from huggingface_hub import hf_hub_download

    from rbbench.manifests import sha256_file

    pin = RELEASED[name]
    config_path = hf_hub_download(pin["repo_id"], "config.json", revision=pin["revision"])
    weights_path = hf_hub_download(pin["repo_id"], "model.safetensors", revision=pin["revision"])
    sha = sha256_file(Path(weights_path))
    if sha != pin["sha256"]:
        raise RuntimeError(f"{pin['repo_id']}: model.safetensors sha256 {sha} != pinned {pin['sha256']} "
                           "(a truncated or empty download, or a different file)")
    return {**pin, "config": json.loads(Path(config_path).read_text()), "weights_path": str(weights_path)}


def config_differences(released_config: dict, cfg: dict) -> list[str]:
    """Settings of our model config that the released config.json does not equal."""
    task = cfg["task"]
    ours = {
        "dim_in": task["n_steps_input"] * len(task["fields"]),
        "dim_out": len(task["fields"]),
        "n_spatial_dims": 2,
        "spatial_resolution": list(task["spatial_resolution"]),
        **cfg["model"]["args"],
    }
    return [f"{key}: released {released_config.get(key)!r}, ours {value!r}"
            for key, value in ours.items() if released_config.get(key) != value]


def load_released(name: str, cfg: dict, fetched: dict) -> nn.Module:
    """Our model for `cfg`, with the released weights loaded strictly into its upstream network."""
    from safetensors.torch import load_model

    from rbbench.models.factory import model_from_config

    diffs = config_differences(fetched["config"], cfg)
    if diffs:
        raise ValueError(f"{fetched['repo_id']} does not match our {name} config: " + "; ".join(diffs))
    model = model_from_config(cfg)
    load_model(model.net, fetched["weights_path"], strict=True)
    return model.eval()
