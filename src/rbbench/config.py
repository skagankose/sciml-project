"""YAML configuration: common.yaml <- experiments/<exp>.yaml <- models/<model>.yaml <- overrides."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from rbbench import MODEL_NAMES
from rbbench.manifests import REPO_ROOT, sha256_json

CONFIG_DIR = REPO_ROOT / "configs"
# Mappings replaced whole when overridden, never merged: merging budgets would leave the
# milestones of both files (e.g. 1h/4h plus a pilot's own labels).
ATOMIC_KEYS = frozenset({"budgets_seconds"})


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats such as 1e9 and 1.0e9 (YAML 1.1 sees strings)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def yaml_load(text: str):
    return yaml.load(text, Loader=_Loader)


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml_load(f.read()) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in ATOMIC_KEYS:
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> dict:
    """`a.b.c=value` -> {"a": {"b": {"c": value}}}, value parsed as YAML."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key.path=value")
    key, raw = text.split("=", 1)
    value: Any = yaml_load(raw)
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


def load_config(
    model: str,
    experiment: str = "main",
    size: str = "full",
    overrides: list[str] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    cfg = load_yaml(config_dir / "common.yaml")
    cfg = deep_merge(cfg, load_yaml(config_dir / "experiments" / f"{experiment}.yaml"))
    model_file = load_yaml(config_dir / "models" / f"{model}.yaml")
    if size not in model_file:
        raise ValueError(f"model config {model}.yaml has no size {size!r}")
    cfg["model"] = {"name": model, "size": size, "args": model_file[size]}
    cfg = deep_merge(cfg, model_file.get("train_overrides", {}) or {})
    cfg["experiment"] = experiment
    for text in overrides or []:
        cfg = deep_merge(cfg, parse_override(text))
    return cfg


def budget_labels(cfg: dict) -> list[str]:
    """Milestone labels (e.g. ["1h", "4h"]) in order of their training-time budget."""
    return [label for _, label in sorted((float(s), label) for label, s in cfg["train"]["budgets_seconds"].items())]


def config_hash(cfg: dict) -> str:
    """Hash of everything that defines a run except where its files go."""
    hashed = {k: v for k, v in cfg.items() if k not in ("paths", "runtime")}
    return sha256_json(hashed)


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=True)
