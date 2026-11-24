# Draft issue for PolymathicAI/the_well (D32; for the owner to post)

**Title:** AViT (2-D): the second axial-attention pass attends over the heads instead of the first spatial axis

---

In the 2-D path of `AxialAttentionBlock`, the second attention pass never attends along the first spatial axis
(`h`). As a result, tokens exchange information only along the second axis (`w`).

**Where:** `the_well/benchmark/models/avit/__init__.py`, lines 141–143, the same at
[e63dec8](https://github.com/PolymathicAI/the_well/blob/e63dec8ded515be864755eaf595d39fea409e335/the_well/benchmark/models/avit/__init__.py#L141-L144)
and on master
([aac88c0](https://github.com/PolymathicAI/the_well/blob/aac88c0b5eb9cbe4b7b2797eaaa550a43706fb44/the_well/benchmark/models/avit/__init__.py#L141-L144)):

```python
self.spatial_permutations = [
    ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
    ("b h he w c -> b h w he c", "b h w he c -> b h w (he c)"),
]
```

`forward` rearranges `q`, `k` and `v` in place on each pass, so the second `in_permutation` receives the layout the
first one produced (`b h he w c`). It maps that to `b h w he c`, where the second-to-last axis is the heads. So
`F.scaled_dot_product_attention` attends across the `num_heads` heads of each token (batched over `b, h, w`), not
along `h`. The 3-D branch chains its three permutations correctly.

**Reproduction** (the_well 1.2.1, torch 2.8.0, einops 0.8.2):

```python
import torch
from the_well.benchmark.models.avit import AxialAttentionBlock

torch.manual_seed(0)
blk = AxialAttentionBlock(hidden_dim=48, num_heads=2, n_spatial_dims=2).eval()
blk.gamma.data.fill_(1.0)  # layer scale starts at 1e-6; 1 makes the mixing visible

x = torch.randn(1, 8, 4, 48)    # (batch, h tokens, w tokens, channels)
x2 = x.clone()
x2[0, 5, 3] += torch.randn(48)  # perturb one token (a random vector, so the pre-LayerNorm keeps it)
with torch.no_grad():
    changed = ((blk(x2) - blk(x)).abs().amax(-1)[0] > 1e-4).nonzero().tolist()
print(len(changed), changed)
```

This prints `4 [[5, 0], [5, 1], [5, 2], [5, 3]]`: only tokens with the same `h` index change. Axial attention
along both axes should reach the perturbed token's row and column, 8 + 4 − 1 = 11 tokens.

**Suggested fix** (no parameter changes; with it, the script above reaches the 11 tokens):

```diff
             self.spatial_permutations = [
                 ("b h w he c -> b h he w c", "b h he w c -> b h w (he c)"),
-                ("b h he w c -> b h w he c", "b h w he c -> b h w (he c)"),
+                ("b h he w c -> b w he h c", "b w he h c -> b h w (he c)"),
             ]
```

**Effect on 2-D datasets.** The patch embedding (`hMLP_stem`) and decoder (`hMLP_output`) are non-overlapping
strided (transposed) convolutions, so each 16 × 16 output patch depends only on its own token. With the attention
mixing only along `w`, information cannot cross patch boundaries along the first spatial axis. The only coupling
left is GroupNorm's global statistics. In the full model at 512 × 128 (random weights, layer scale set to 1), a
change to one input patch moves the output about 47 times more in the patches sharing its `h` index than in the
others.

We trained AViT from scratch on `rayleigh_benard` (512 × 128, so a 32 × 8 token grid). Each 16-pixel-wide strip in x
is effectively predicted on its own:

- Predictions show seams at the strip boundaries along x. One step ahead, the jump across them is 2.4–3.2 times the
  ordinary pixel-to-pixel jump, growing to 6–13 times after 30 rollout steps. There are no seams along y.
- The one-step error is 1.35–1.87 times higher at the strip edges than at their centres.

The fix keeps parameter names and shapes, so existing checkpoints still load. They were trained without mixing
along `h`, though, and would need retraining to benefit.
