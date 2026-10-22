"""Hashing, atomic writes and provenance records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

UPSTREAM = {
    "the_well": {
        "repo": "https://github.com/PolymathicAI/the_well",
        "commit": "e63dec8ded515be864755eaf595d39fea409e335",
        "note": "v1.2.0 + fixes #69 (val/test normalization), #73, #92 (VRMSE eps), #93 (long-time metrics)",
    },
    "neuraloperator": {"repo": "https://github.com/neuraloperator/neuraloperator", "version": "0.3.0"},
    "openstl": {
        "repo": "https://github.com/chengtan9907/OpenSTL",
        "commit": "eecf8a3078f0a178dbc7b28723da20f94ce36985",
        "file": "openstl/modules/convlstm_modules.py (vendored verbatim)",
    },
}

REPO_ROOT = Path(__file__).resolve().parents[2]

_PACKAGES = (
    "torch", "the_well", "neuraloperator", "tensorly", "tensorly-torch", "timm",
    "numpy", "h5py", "scipy", "pandas", "einops", "fsspec",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_int(*parts: Any) -> int:
    """63-bit integer from a stable hash of `parts` (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def git_info(root: Path = REPO_ROOT) -> dict:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    # rsync -rlt does not carry file modes, so mode-only differences must not count as dirty
    status = run("-c", "core.fileMode=false", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else status.splitlines()[:50],
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict:
    import torch

    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    info["torch_cuda"] = torch.version.cuda
    info["mps_available"] = torch.backends.mps.is_available()
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            name=torch.cuda.get_device_name(idx),
            capability=list(torch.cuda.get_device_capability(idx)),
            total_memory_gb=round(props.total_memory / 1e9, 2),
            arch_list=torch.cuda.get_arch_list(),
            cudnn=torch.backends.cudnn.version(),
        )
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            )
            info["nvidia_smi"] = out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            info["nvidia_smi"] = None
    return info


def slurm_info() -> dict:
    keys = (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT", "SLURM_JOB_NODELIST", "SLURM_RESTART_COUNT", "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
    )
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def provenance(extra: dict | None = None) -> dict:
    record = {
        "time_utc": now_utc(),
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git": git_info(),
        "packages": package_versions(),
        "upstream": UPSTREAM,
        "gpu": gpu_info(),
        "slurm": slurm_info(),
    }
    if extra:
        record.update(extra)
    return record
