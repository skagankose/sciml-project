# Upstream audit (Phase B)

Audited 2026-10-22 on macOS arm64, Python 3.13.13, torch 2.8.0 (CPU). Parameter counts and shapes
below were measured with `python -m rbbench models` and `tests/`; nothing here was measured on HPC3.

## Pinned sources

| Component | Source | Pin | Licence |
|---|---|---|---|
| The Well (data loader, normalization, VRMSE, CNextU-Net, TFNO wrapper, AViT) | https://github.com/PolymathicAI/the_well | commit `e63dec8ded515be864755eaf595d39fea409e335` (2026-10-12), package version 1.2.1 | BSD-3-Clause |
| NeuralOperator (TFNO implementation used by The Well) | PyPI `neuraloperator==0.3.0` | the version The Well's `[benchmark]` extra pins | MIT |
| OpenSTL `ConvLSTMCell` | https://github.com/chengtan9907/OpenSTL | commit `eecf8a3078f0a178dbc7b28723da20f94ce36985`, file `openstl/modules/convlstm_modules.py` vendored verbatim at `src/rbbench/vendor/openstl/` (sha256 `4a545b9f…40c6`, checked by `tests/test_vendor.py`) | Apache-2.0 |

**Why this The Well commit and not the v1.2.0 tag.** These fixes landed after v1.2.0 and touch code this project uses:
#69 (normalization applied to val/test in the data module), #73 (restricted trajectories), #92 (VRMSE `eps` propagation),
#93 (long-time metrics averaged over all validation batches instead of keeping the last batch). The v1.1.0 fixes
(non-square AViT, test-rollout normalization) are included. The only later commit (`aac88c0`, 2026-10-19) adds
shear_flow v2 warnings and changes nothing for `rayleigh_benard`.

**Why vendor one OpenSTL file.** Importing the `openstl` package pulls video/vision dependencies unrelated to the
cell. The vendored file is byte-identical to upstream.

**Why torch 2.8.0.** neuraloperator 0.3.0 indexes a tensor with a Python list
(`spectral_convolution.py:459`); torch 2.8 warns that the meaning changes in 2.9. Staying on 2.8 keeps TFNO's
behaviour identical to the published benchmark. It is also the cu128 version used in the HPC3 guide.

## Data interface (verified on Well-format synthetic files; to re-verify on real data)

* `WellDataset(path=<dataset dir>, well_split_name=…)`: split folders are `train`, `valid`, `test` (the CLI's
  `--split` accepts exactly these).
* Sample layout `input_fields [T_in, X, Y, F]`, `output_fields [T_out, X, Y, F]`; X = `dimensions/x` (512,
  horizontal, periodic), Y = `dimensions/y` (128, wall-bounded). Field order is t0 fields then t1 components:
  `buoyancy, pressure, velocity_x, velocity_y` (asserted at load time).
* Window index → (file, trajectory, t0) as in `WellDataset._load_one_sample`; `file_index_offsets[0]` is stored
  as `-1` upstream, handled in `rbbench.data.trajectory_window_range` and tested against direct h5py reads.
* Normalization: `ZScoreNormalization` from `<dataset>/stats.yaml` (upstream training statistics), applied
  inside the dataset; `denormalize_flattened(x, "variable")` inverts it.
* Download: `the-well-download --base-path DATA_ROOT --dataset rayleigh_benard --split {train,valid,test}`;
  uses `curl --continue-at -`. The server accepts byte ranges. Remote sizes (HEAD, 2026-10-22):
  train 293.6 GB, valid 36.7 GB, test 36.7 GB (35 files each). No upstream checksums are published (ETags are
  mtime-size strings), so local SHA-256 manifests are computed after verifying sizes.

## Models (random initialisation only)

All wrappers take `[B, 4, X, Y, 4]` and return `[B, X, Y, 4]`. Time-and-field stacking matches upstream
`DefaultChannels{First,Last}Formatter` (channel = t·4 + f). `scripts/compare_upstream.py` checks that our
rollout equals upstream's formatter-based rollout loop (bitwise on CPU for the three Well models).

| Model | Class | Arguments (`full`) | Layout | Params @512×128 | Receiving gradient |
|---|---|---|---|---|---|
| CNextU-Net | `the_well.benchmark.models.UNetConvNext` | `stages=4, blocks_per_stage=2, blocks_at_neck=1, init_features=42` (published RB config; 7×7 depthwise kernels) | channels-first | 18,575,344 | 18,573,454 |
| TFNO | `the_well.benchmark.models.TFNO` → `neuralop.models.TFNO` (FNO with `factorization="Tucker"`) | `modes1=modes2=16, hidden_channels=128`; neuralop defaults: 4 layers, `rank=1.0`, `implementation="factorized"`, lifting/projection 256, no domain padding, `fft_norm="forward"` | channels-first | 19,276,556 | 19,276,556 |
| AViT | `the_well.benchmark.models.AViT` | `hidden_dim=432, num_heads=6, processor_blocks=8` (Well config 384 widened, see below; blocks = class default), patch 16 hard-coded → 32×8 tokens | channels-last | 18,568,512 | 18,568,512 |
| ConvLSTM | `rbbench.models.convlstm.ConvLSTMForecaster` around OpenSTL `ConvLSTMCell` | OpenSTL reference config (4 layers, 5×5 filters, stride 1, 4×4 space-to-depth patches, `layer_norm=False`) with hidden width 152 instead of 128; 3×3 conv encoder (64→152) and 3×3+1×1 decoder | sequence | 23,594,416 | 18,788,784 |

**Parameter matching (deviation from the reference configs, decided before training).** CNextU-Net and
TFNO keep their published RB configs. At reference width AViT has 14.69M parameters (hidden 384) and
ConvLSTM 13.34M trainable (hidden 128), so both are widened within their family to match:
AViT hidden 384 → 432 (head dimension 72; `hidden_dim/4` still divisible by 12) and ConvLSTM hidden,
encoder and decoder width 128 → 152. Trainable parameters are then 18.57M / 19.28M / 18.57M / 18.79M
(CNextU-Net / TFNO / AViT / ConvLSTM), asserted by `tests/test_all_models_rectangular.py`. Matching
parameters does not match FLOPs or throughput; both are measured in Phase G.

Upstream quirks recorded, not changed:

* **CNextU-Net**: the channels-first `LayerNorm` multiplies by `weight` but never adds `bias`, so 1,890 bias
  parameters receive no gradient.
* **OpenSTL `ConvLSTMCell`**: allocates `conv_o` and `conv_last` that `forward` never uses (3.41M parameters,
  no gradient). We report both counts; the effective size is 18.8M (13.3M at the reference width 128).
* **AViT**: needs each spatial size divisible by 16, and `hidden_dim/4` divisible by 12 (GroupNorm groups).
  Its absolute positional embedding is sized from `spatial_resolution`, so 512×128 and 64×32 give different
  parameter shapes. Smoke tests use the `tiny` sizes at 64×32 and 128×32.
* **TFNO**: The Well notes that bf16 autocast is unsupported for FFTs. A bf16 run must be checked on the A100
  (Phase G) before it is used; fp32 is the default.
* `the_well.benchmark.models.common.BaseModel` inherits `PyTorchModelHubMixin` (`from_pretrained`). This
  project never calls it (`tests/test_backward.py` scans the source).

## Metric

`the_well.benchmark.metrics.VRMSE.eval(x, y, meta, eps=1e-7)` = `sqrt(mean_xy (x−y)² / (std_xy(y)² + eps))`
with torch's unbiased `std`, per sample, time and field, on denormalised fields (upstream denormalises before
validation metrics). The field mean is taken after that. `rbbench.metrics.well_metric.vrmse` calls it directly;
`tests/test_vrmse.py` checks it against an independent NumPy version and hand-computed values, and shows it
differs from NRMSE/RMSE. Collection-wide means are sample-weighted (`MeanAccumulator`), avoiding the
last-batch bias.

## Reference numbers (not targets)

From `docs/benchmarks.md` at the pinned commit (12 h on one H100, best validation checkpoint):
`rayleigh_benard` **test** one-step VRMSE — CNextU-Net **0.6699**, TFNO **0.6566**, FNO 0.8395, U-Net 1.4860;
**validation** — CNextU-Net 0.4807, TFNO 0.8568; rollout 6:12 > 10 for all four. The model cards list RB learning
rates 5e-4 (CNextU-Net) and 1e-4 (TFNO). No published Well RB numbers exist for AViT or ConvLSTM.
