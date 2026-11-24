"""The Well's 2-D AViT attends only along the second spatial axis; `avit_fixed` attends along both.

docs/report_notes.md section 17. The upstream test pins the bug at the pinned commit: if upstream fixes it, the test
fails and `avit_fixed` must be re-checked.
"""

import pytest
import torch

from rbbench.models.factory import build_model, count_parameters
from rbbench.models.well_models import FIXED_2D_PERMUTATIONS, fix_axial_attention

TINY = dict(hidden_dim=48, num_heads=2, processor_blocks=2, drop_path=0.0)


def reached_tokens(blocks, dim, h=8, w=4, at=(5, 3)):
    """Tokens whose output changes when one token's input changes, through a stack of attention blocks.

    The perturbation is a random vector: a constant shift would be removed by the blocks' LayerNorm."""
    for blk in blocks:
        blk.eval()
        blk.gamma.data.fill_(1.0)  # layer scale starts at 1e-6, which would hide the mixing
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, h, w, dim, generator=g)
    x2 = x.clone()
    x2[0, at[0], at[1]] += torch.randn(dim, generator=g)
    with torch.no_grad():
        y, y2 = x, x2
        for blk in blocks:
            y, y2 = blk(y), blk(y2)
        d = (y2 - y).abs().amax(-1)[0]
    return {tuple(t) for t in (d > 1e-4).nonzero().tolist()}


def avit_block():
    from the_well.benchmark.models.avit import AxialAttentionBlock

    torch.manual_seed(0)
    return AxialAttentionBlock(hidden_dim=48, num_heads=2, n_spatial_dims=2)


def test_upstream_2d_block_mixes_only_along_second_axis():
    reached = reached_tokens([avit_block()], 48)
    assert reached == {(5, j) for j in range(4)}  # same h (x) token, every w (y) token, nothing else


def test_fixed_2d_block_mixes_along_both_axes():
    blk = avit_block()
    blk.spatial_permutations = list(FIXED_2D_PERMUTATIONS)
    reached = reached_tokens([blk], 48)
    assert reached == {(5, j) for j in range(4)} | {(i, 3) for i in range(8)}  # its row and its column


def test_avit_fixed_model_same_parameters_and_keys_as_avit():
    a = build_model("avit", TINY, (64, 32), seed=0)
    f = build_model("avit_fixed", TINY, (64, 32), seed=0)
    assert count_parameters(a) == count_parameters(f)
    assert a.state_dict().keys() == f.state_dict().keys()
    for k, v in a.state_dict().items():  # same seed, same initial weights
        assert torch.equal(v, f.state_dict()[k])
    assert [tuple(p) for p in f.net.blocks[0].spatial_permutations] == FIXED_2D_PERMUTATIONS


def test_models_processor_reach():
    """Through the model's own blocks: upstream stays in the perturbed token's x-column, the fix reaches every token."""
    a = build_model("avit", TINY, (128, 64), seed=0).net
    f = build_model("avit_fixed", TINY, (128, 64), seed=0).net
    assert reached_tokens(a.blocks, 48) == {(5, j) for j in range(4)}
    assert reached_tokens(f.blocks, 48) == {(i, j) for i in range(8) for j in range(4)}


def test_fix_refuses_changed_upstream():
    model = build_model("avit_fixed", TINY, (64, 32), seed=0)
    with pytest.raises(RuntimeError, match="permutations changed"):
        fix_axial_attention(model.net)  # already fixed: no longer the upstream permutations


def test_aggregate_collects_several_manifests(tmp_path):
    """The core manifest and the avit_fixed one are aggregated together; a repeated run id is refused."""
    from pathlib import Path

    from rbbench.aggregate import collect

    tasks = Path(__file__).resolve().parents[1] / "tasks"
    rows, loaded, problems, expected = collect([tasks / "train.jsonl", tasks / "train_avit_fixed.jsonl"], tmp_path,
                                               labels=["4h"])
    assert len(rows) == 16 and {r["model"] for r in rows} == {"cnext_unet", "tfno", "avit", "avit_fixed"}
    assert loaded == [] and expected == 16 and problems  # nothing evaluated under tmp_path
    with pytest.raises(ValueError, match="repeat run ids"):
        collect([tasks / "train_avit_fixed.jsonl", tasks / "train_avit_fixed.jsonl"], tmp_path, labels=["4h"])
