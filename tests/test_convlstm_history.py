import torch

from rbbench.config import load_config
from rbbench.models.convlstm import ConvLSTMForecaster
from rbbench.models.factory import build_model
from rbbench.rollout import rollout
from rbbench.vendor.openstl.convlstm_modules import ConvLSTMCell


def _model():
    args = load_config("convlstm", "smoke", "tiny")["model"]["args"]
    return build_model("convlstm", args, (64, 32), seed=0)


def test_uses_upstream_cell():
    model = _model()
    assert isinstance(model, ConvLSTMForecaster)
    assert all(type(c) is ConvLSTMCell for c in model.cells)


def test_unrolls_exactly_four_frames():
    model = _model()
    model(torch.randn(1, 4, 64, 32, 4))
    assert model.last_unroll_length == 4


def test_state_resets_every_call():
    """Same window -> same output regardless of what the model saw before."""
    model = _model().eval()
    a = torch.randn(1, 4, 64, 32, 4)
    with torch.no_grad():
        y1 = model(a)
        model(torch.randn(1, 4, 64, 32, 4) * 10)
        y2 = model(a)
    assert torch.equal(y1, y2)


def test_output_depends_only_on_current_window():
    """Rollout prediction k equals a fresh call on the k-th four-frame window."""
    model = _model().eval()
    ctx = torch.randn(1, 4, 64, 32, 4)
    preds = rollout(model, ctx, 4)
    frames = torch.cat([ctx, preds], dim=1)
    with torch.no_grad():
        for k in range(4):
            assert torch.allclose(model(frames[:, k:k + 4]), preds[:, k], atol=1e-6)


def test_frame_order_matters():
    model = _model().eval()
    x = torch.randn(1, 4, 64, 32, 4)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x.flip(1)))
