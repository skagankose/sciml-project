# Model architectures and training recipe

For the report. All three models are The Well's own implementations at the pinned commit `e63dec8`
(`the_well.benchmark.models`), built from random initialisation (PyTorch defaults; no pretrained weights).
Each takes the last 4 frames of the 4 fields (buoyancy, pressure, horizontal and vertical velocity), stacked as
16 input channels on the 512 × 128 grid, and predicts the next frame's 4 fields (full state, not the
increment). Sources: `configs/models/*.yaml`, the upstream source read on 2026-11-11, `docs/decisions.md`
(D3, D5–D8, D20) and the released checkpoints' `config.json`.

## Architectures

| | CNextU-Net | TFNO | AViT |
|---|---|---|---|
| Family | convolutional U-Net with ConvNeXt blocks | spectral neural operator (Fourier layers, Tucker-factorised weights) | vision transformer with axial attention |
| Upstream class | `UNetConvNext` | `TFNO` → neuraloperator 0.3.0 `TFNO` (`FNO`, `factorization="Tucker"`) | `AViT` (simplified from MPP) |
| Trainable parameters | 18,575,344 | 19,276,556 | 18,568,512 |
| Encoder / lifting | 3 × 3 conv, 16 → 42 channels | 2-layer pointwise MLP, 16 → 256 → 128 | patch embedding: 4 × 4 then 2 × 2 then 2 × 2 strided convs (patch 16), 16 → 108 → 108 → 432; tokens on a 32 × 8 grid |
| Core | 4 down stages (42 → 84 → 168 → 336 → 672 channels), 2 ConvNeXt blocks each, 2 × 2 strided-conv downsampling; 1 bottleneck block; 4 up stages with 2 × 2 transposed convs, skip connections concatenated and projected by 1 × 1 convs | 4 Fourier layers, width 128: spectral convolution keeping 16 × 16 modes, plus a 1 × 1 linear skip | 8 axial-attention blocks, width 432, 6 heads (72 per head): attention along x and along y, summed, with a parallel MLP (4 × width) |
| Block | ConvNeXt: 7 × 7 depthwise conv → LayerNorm → 1 × 1 expand × 4 → GELU → 1 × 1 → layer scale (init 1e-6) → residual | spectral conv + 1 × 1 skip → GELU (none after the last layer) | pre-LayerNorm → fused q, k, v and MLP projection → QK-LayerNorm → 2 axial attentions + GELU MLP → output projection → layer scale (init 1e-6) → residual |
| Decoder / projection | 3 × 3 conv, 42 → 4 | 2-layer pointwise MLP, 128 → 256 → 4 | mirror of the embedding with transposed convs, 432 → 108 → 108 → 4 |
| Activation | GELU (exact) | GELU (exact) | GELU (exact) |
| Normalisation | LayerNorm (eps 1e-6) inside blocks; before each down/upsampling a "channels-first LayerNorm" that is in fact an L2 normalisation over channels times a learned scale | none | GroupNorm (12 groups) in the embedding and output; LayerNorm without bias before attention and on queries and keys |
| Position information | from convolutions and padding | from the Fourier basis | learned absolute position embedding (32 × 8 × 432, init 0.02 · N(0, 1)) |
| Global mixing | local convolutions; after 4 downsamplings (512 × 128 → 32 × 8) a 7 × 7 kernel spans about a quarter of x and all of y | every Fourier layer is global | every block is global along each axis |
| Dropout / drop path | none / 0 | none | none / 0 |

Notes:

* *Boundary handling.* None of the three encodes the domain's geometry: x is periodic and y has walls, but
  CNextU-Net zero-pads in both directions, AViT's patches see no padding, and TFNO's Fourier basis (no domain
  padding) treats y as periodic too.
* *Parameter matching (D3).* CNextU-Net and TFNO use the published Well configurations. AViT is widened from
  the repository's hidden width 384 (14.69M parameters) to 432 so that all three have 18.6–19.3M parameters.
* *ConvLSTM* (extension, not run): OpenSTL's ConvLSTM cell, 4 layers of width 152, 5 × 5 filters, 4 × 4
  space-to-depth patches; 18.79M trainable parameters.

## Same as the paper and the released code?

| | CNextU-Net | TFNO | AViT |
|---|---|---|---|
| Configuration vs released checkpoint | identical: its `config.json` equals ours, the weights load strictly, and the loaded model matches upstream `from_pretrained` bitwise (D20) | identical configuration (`modes3` is unused in 2-D) | no released `rayleigh_benard` checkpoint; AViT is not among the paper's baselines for this dataset (FNO, TFNO, U-net, CNextU-net) |
| Forward pass vs the paper | same code | **different.** Until the_well commit `e430160` (2025-11-17) the wrapper discarded the Fourier layers' output, so the paper's TFNO (and the released checkpoint) was only lifting → projection, a per-pixel MLP. The pinned code runs the Fourier layers, so our TFNO is a real neural operator. The released checkpoint is scored with the forward it was trained with (D20) | repository config widened 384 → 432 (D3) |

## Training recipe (identical for the three models)

| | This study | The Well paper / code | Same? |
|---|---|---|---|
| Loss | MSE on the next frame, normalised units, equal field weights | MSE, equal weights | yes |
| Normalisation | z-score with the dataset's train-split `stats.yaml` | same | yes |
| Optimizer | AdamW, β = (0.9, 0.999), weight decay 1e-4, no gradient clipping | AdamW; weight decay 1e-4 in `configs/optimizer/adam.yaml` (the text says 0.01); no clipping | yes (as the code) |
| Learning rate | 1e-3 for all three, from our equal-budget search over {1e-4, 5e-4, 1e-3} on validation (D7) | per model from a sweep over {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}: CNextU-Net 5e-4, TFNO 1e-4 | no |
| Schedule | 500 warmup steps, then cosine to 0 over the 4 h training clock (D6) | linear warmup over 5 epochs from 0.1 · lr, then cosine over 200 epochs to 0.1 · lr, stopped at 12 h | no |
| Batch size | 64 | 64 (`rayleigh_benard` data config) | yes |
| Precision | fp32 | fp32 (AMP off; FFTs do not support bf16) | yes |
| Budget | 1 h and 4 h of GPU training time on one A100 80 GB (waiting for data excluded, D24) | 12 h of wall time on one H100; best checkpoint by validation | no |
| Data | 10% of the training trajectories (100% planned) | 100% | no (yet) |
| Epochs reached | 10% data, 4 h: CNextU-Net 28, TFNO 61, AViT 175 | best epoch 12 (CNextU-Net), 31 (TFNO) | |
| Checkpoint used | the 1 h and 4 h checkpoints (no selection) | best validation epoch | no |
