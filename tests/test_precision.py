"""bf16 is refused for FFT-based models before a run directory or any GPU time is used (decisions D8)."""

import pytest
import torch

from rbbench import MODEL_NAMES
from rbbench.device import FP32_ONLY_MODELS, check_precision
from rbbench.train import train_run


def test_tfno_is_fp32_only():
    assert FP32_ONLY_MODELS == {"tfno"}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_check_precision(name):
    check_precision(name, "fp32")
    if name in FP32_ONLY_MODELS:
        with pytest.raises(ValueError, match="bf16"):
            check_precision(name, "bf16")
    else:
        check_precision(name, "bf16")


def test_train_refuses_tfno_bf16_before_writing(dataset_dir, tiny_cfg, tmp_path):
    cfg = tiny_cfg("tfno", "train.precision=bf16")
    with pytest.raises(ValueError, match="fp32"):
        train_run(cfg, tmp_path / "run", dataset_dir, 0, torch.device("cpu"))
    assert not (tmp_path / "run").exists()
