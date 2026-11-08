# Design decisions

Each entry: what was decided, why, and what would change it. "PENDING" items must be frozen on
validation/pilot data before the main sweep (Phase I gate) and recorded here with the evidence.

**D1 — Thin runner, not the upstream Hydra trainer.** The upstream `Trainer` is epoch-based, keeps no
scheduler/scaler state on resume (known upstream issue), and has no wall-clock budget or subset manifests.
We reuse upstream data loading, normalization, models and VRMSE, and write only the budgeted loop,
checkpointing, rollout bookkeeping and physics diagnostics.

**D2 — Pins.** The Well `e63dec8`, neuraloperator 0.3.0, OpenSTL cell vendored at `eecf8a3`, torch 2.8.0,
Python 3.13 on both machines. See `docs/upstream_audit.md`.

**D3 — Model sizes: matched parameter counts.** CNextU-Net (18.57M) and TFNO (19.28M) use their published
Well RB configs. AViT is widened from hidden 384 (14.69M) to 432 (18.57M) and ConvLSTM from width 128
(13.34M trainable) to 152 (18.79M trainable, 23.59M allocated because OpenSTL's cell holds unused layers), so all
four families have 18.6–19.3M trainable parameters. Owner's decision (2026-10-22), made before any training,
so that capacity is comparable across families. Since the final abstract (2026-10-30) the core study has three
families; the ConvLSTM widening applies only if that extension runs. Equal parameters do not mean equal compute per update; Phase G
records step time, memory and (where computable) FLOPs, and under a fixed time budget the update counts are
reported.

**D4 — Normalization.** Upstream `stats.yaml` (train-split statistics shipped with the dataset), z-score, the
same frozen statistics for every subset and model. Recomputing statistics per subset would make inputs differ
between the data fractions; using the shipped train statistics never touches validation or test data.
To verify on HPC3: that `stats.yaml` matches the train split (the audit records it).

**D5 — Objective.** One-step, teacher-forced, full-state (not residual) prediction, MSE in normalised units
with equal field weights (upstream uses MSE with equal weights). AdamW (β = 0.9/0.999, weight decay 1e-4 as
upstream) on all parameters, no gradient clipping (as upstream).

**D6 — Learning-rate schedule under a wall-clock budget (PENDING confirmation).** Warmup in optimizer steps,
then cosine decay over *cumulative training seconds* towards the final 4 h milestone. The total step count is
unknown in advance, so a step-based cosine cannot end exactly at 4 h. Decaying on time makes the 4 h
checkpoint the end of the schedule; the 1 h checkpoint is an intermediate state of the same schedule, not a
separately tuned 1 h run. (Budgets were 2 h/6 h before the final abstract.) The rate depends only on
(step, train_seconds), so resume is exact. The pilot compares this with a constant rate on validation only.
The budgets live in `train.budgets_seconds` (`1h: 3600, 4h: 14400`); their labels name the checkpoints and
evaluations everywhere, and an override replaces the whole mapping rather than merging into it.

**D7 — Learning rate and batch size (PENDING).** One shared batch size for all models, chosen from the
Phase G memory/throughput table. The final abstract commits to an equal-budget learning-rate search on
validation data before training: one small grid ({1e-4, 5e-4, 1e-3}), the same budget per (model, rate),
selection on validation only. **Deferred by the owner (2026-10-30):** the search is not implemented, and
whether it picks one rate shared by all models or one per model is open. The Well's own RB rates differ
(5e-4 CNextU-Net, 1e-4 TFNO). Per-model rates need per-model overrides in `make-sweep`. Until then
`optim.lr = 5e-4` for every model. If a model needs gradient accumulation to reach the shared batch size, it is
recorded. No test-set tuning.
* *Replication of the paper (owner request, 2026-11-08):* our trained CNextU-Net and TFNO must be at least as
  good as the paper's (D20). To compare like with like, the two models use the rates The Well's own sweep
  chose for `rayleigh_benard` (model cards): CNextU-Net 5e-4, TFNO 1e-4. They are set in each model's
  `train_overrides`, are not tuned by us, and no test data is involved. AViT keeps 5e-4 until its rate is
  decided (F5).
* *Batch size: 64* for every model, The Well's `rayleigh_benard` data config. It fits every model in fp32
  on an A100 80 GB (profile 2026-11-08, `outputs/pilots/feasibility.csv`):

  | batch 64, fp32 | step | peak memory | steps in 4 h | epochs of S100 in 4 h |
  |---|---|---|---|---|
  | CNextU-Net | 1.42 s | 58.7 GB | 10,121 | 2.4 |
  | TFNO | 0.55 s | 44.9 GB | 26,160 | 6.1 |
  | AViT | 0.38 s | 7.4 GB | 38,246 | 8.9 |
  | ConvLSTM (extension) | 1.49 s | 23.6 GB | 9,669 | 2.3 |

  Time per sample is nearly constant from batch 8 to 64, so a smaller batch would give more updates but not
  more data in 4 h. The paper's best checkpoints came at epoch 12 (CNextU-Net) and 31 (TFNO) of 12 h on an
  H100, so our 4 h runs train for about a fifth as many epochs.
* *Paper's schedule, for reference:* AdamW with a linear warmup over 5 epochs from 0.1·lr, then cosine over
  200 epochs to 0.1·lr. With the 12 h stop (best epoch 12 for CNextU-Net, 31 for TFNO), that is nearly a
  constant rate. Ours decays to 0 over the 4 h budget (D6).

**D8 — Precision.** fp32 for every model in the core study (2026-11-08): the paper's setting (AMP off),
and TFNO cannot run in bf16. bf16 halves CNextU-Net's step time (0.72 s at batch 64) but would give the
models different precisions. Earlier note: fp32 by default. bf16 autocast only for models whose A100 pilot shows finite,
matching validation loss; TFNO stays fp32 unless bf16 FFTs are shown to work. Recorded per model.
* *TFNO is fp32 only (A100 check, 2026-11-07).* Under bf16 autocast, neuralop's `SpectralConv` fails in
  `torch.fft.rfftn` with "Unsupported dtype BFloat16" (torch 2.8). The Well's trainer keeps AMP off for the
  same reason (`training.py:108`, "bfloat not supported in FFT"), so its published TFNO numbers are fp32.
  `train` and `profile` refuse bf16 for TFNO (`device.FP32_ONLY_MODELS`), and
  `tests/test_cuda.py::test_bf16_fft_still_unsupported` flags a torch version that adds bf16 FFTs.
* CNextU-Net, AViT and ConvLSTM pass a full-size bf16 step and 30-step rollout on the A100. Whether to use
  bf16 for them, and so give the models different precisions, is still open.

**D9 — Budget clock.** Per optimizer step, from requesting the batch to the synchronised end of
`optimizer.step()`. Excludes startup, worker spawn, validation and checkpoint writes; those are logged
separately (`overhead_seconds`). DataLoader workers keep prefetching during validation, which slightly
favours data loading after a validation pass; this is small and the same for every model.

**D10 — Milestone checkpoints.** First optimizer step at/after 1:00:00 and 4:00:00 of training time; actual
seconds and overshoot are recorded. `max_steps` (smoke/pilot only) can force milestones, and then
`budget_satisfied=false` is written and aggregation flags the run.

**D11 — Resume.** `last_resume.pt` holds model, optimizer, schedule state, RNG (python/numpy/torch/cuda),
sampler epoch and position, step, train seconds, milestones and validation history. Written atomically every
15 min of wall time, at each milestone, and on SIGUSR1/SIGTERM (Slurm `--signal=B:USR1@300`, `exec` in the
sbatch script). A CPU test shows an interrupted+resumed run equals an uninterrupted one bitwise.

**D12 — Subsets.** Seeded permutation within each (Ra, Pr) group of the train split; each fraction takes a
prefix with largest-remainder counts, which nests S10 ⊂ S30 ⊂ S100 by construction (asserted). With 40
trajectories per regime (expected from file sizes) this gives exactly 4 / 12 / 40 per regime and
140 / 420 / 1400 trajectories in total; to confirm on the real split. The core study uses S10 and S100; S30 is
generated with them, so it nests correctly if the 30% extension runs.

**D13 — ConvLSTM information horizon.** States are zero-initialised in every forward call and the cells unroll
over exactly the 4 input frames, so in rollouts ConvLSTM sees the same 4-frame window as the other models.

**D14 — Evaluation data path.** Each test trajectory is read once through `WellDataset` as a single
4 + 196-frame window, then every one-step window and every rollout start is cut from it in memory. This uses
the upstream parser and normalization while avoiding re-reading ~37 GB per checkpoint.

**D15 — Rollout cohort (PENDING).** All 175 test trajectories × start indices `[20, 80, 140]`, 30 steps.
Very early starts are near the conduction state, where target variance is tiny and VRMSE is ill-conditioned. The
start list must be frozen on the *validation* split before any test evaluation, then `tasks/eval_cohort.json`
is generated once and never changed (its hash is checked).

**D16 — Failure horizon.** First step whose field-mean VRMSE exceeds the threshold; rollouts that never cross
are right-censored, summarised with Kaplan–Meier survival and median. Thresholds 1.0 / 2.0 (primary) with 5.0
as sensitivity; the primary is chosen on validation (PENDING).

**D17 — Noise (extension, evaluation only).** The core evaluation uses σ = 0 only (`eval.noise_sigmas: [0.0]`).
The extension re-evaluates the same checkpoints with σ ∈ {0.01, 0.05, 0.10, 0.20} in normalised units, added
once to the 4 observed frames only. It uses `configs/experiments/noise.yaml` and its own frozen cohort
(`make-cohort --experiment noise --out tasks/eval_cohort_noise.json`), and every run is re-evaluated with it,
because aggregation never mixes cohorts; σ = 0 is part of that pass, so clean results are recomputed alongside. Each (trajectory, start, σ, seed, field) draw comes from a stable hash, so it is identical for all models,
batch sizes and worker counts. Grid points lying exactly on a wall are left unperturbed; with Chebyshev-Gauss
nodes there are none (confirmed 2026-11-07, D18). The files' placeholder y would have put rows 0 and 127 on the
walls, so the noise uses the generated nodes.

**D18 — Physics diagnostics.** Chebyshev interpolation through the y nodes gives the spectral-accuracy
derivatives, wall values and exact quadrature (Dedalus discretisation). Nu (volume) = ⟨u_y b − κ ∂_y b⟩ /
(κ Δb / Ly) with κ = (Ra Pr)^−1/2. The independent check is the wall Nu from −∂_y⟨b⟩ at both walls. Evaluation
computes physics only if the dataset audit passes: wall values of b and u, conductive part ≡ 1, Parseval,
and volume vs wall Nu on late-time reference data (median relative difference ≤ 10%). Spectra use an FFT in x
only; y is never treated as periodic.
* *The y nodes (2026-11-07).* The files' `dimensions/y` is a placeholder `linspace(0, 1, 128)`. The values are
  on Chebyshev-Gauss nodes: with those nodes every audit check passes to ~1e-9, while Lobatto fails, and the
  near-wall rows resolve the boundary layer as only Gauss clustering can (`docs/upstream_audit.md`).
  `physics.y_grid: gauss` generates the nodes on [0, Ly] for the audit, spectra, Nusselt and the noise wall
  mask (`metrics.physics.y_nodes`). The audit still records the stored y. It fails if the stored y is neither
  the configured nodes nor the uniform placeholder. Evaluation computes physics only if the audit checked the
  same family.

**D19 — Concurrency.** Slurm array `%1` (one A100 at a time). 12 core runs × ~4.5 h ≈ 2.3 days sequential on
`free-gpu` if there is no queueing or preemption; the six 10% runs alone take ≈ 27 h. Raising concurrency or
using the paid `gpu` partition (≥ 1,632 SU for the 48 GPU-h of core training) is the owner's call.

**D20 — Pipeline check with released checkpoints.** The final abstract evaluates The Well's released
`polymathic-ai/UNetConvNext-rayleigh_benard` and `polymathic-ai/TFNO-rayleigh_benard` before training, and
compares their one-step VRMSE with the published values (CNextU-Net 0.4807 valid / 0.6699 test, TFNO 0.8568 /
0.6566). AGENTS.md rule 3 was amended on 2026-10-30 to allow this: the checkpoints may be loaded only for
evaluation, never to train from. Checked 2026-10-30:
* Both checkpoints exist on Hugging Face (created 2025-03-28), and their configs match `configs/models/*.yaml`
  `full`.
* The pinned benchmark data config sets `use_normalization: True` with no type, which falls back to
  `ZScoreNormalization` on the shipped `stats.yaml`. That is the same normalization as D4.

A mismatch would point to a problem in our data path or metric. Implemented 2026-11-07:
* `rbbench/released.py` is the only module allowed to download pretrained weights, and only the CLI imports it
  (`tests/test_backward.py`). Each checkpoint is pinned to a revision and the SHA-256 of its weights
  (CNextU-Net `c5fa0f7`, TFNO `7d35135`), which also catches an empty or truncated download.
* The weights load strictly into our own `full` models. On the Mac (2026-11-07) both loaded with no missing or
  unexpected keys, and matched upstream `from_pretrained` bitwise on a random input.
* `rbbench eval-released` scores them with the same per-window code as our runs (`evaluate.one_step_rows`,
  tested equal), on every window of `valid` and `test`, and refuses unverified splits.
* The check passes when every model and split is within 5% of the published value (`--tolerance`). This is a
  judgement: the published runs used their own evaluation loop and an H100, so small differences are expected,
  while a broken normalization, field order or metric would be far off.

*Result (2026-11-07, job 59154129): MISMATCH; the published values cannot be the reference.*

| | valid | test | published valid / test |
|---|---|---|---|
| CNextU-Net | 0.234 | 0.224 | 0.481 / 0.670 |
| TFNO | 10.7 | 10.5 | 0.857 / 0.657 |

* *CNextU-Net works through our pipeline.* From frame 50 on it scores 0.02–0.05, against 0.20 for
  persistence. Of the four conventions, only z-score next-state is good (`scripts/probe_released_conventions.py`:
  late median 0.019; the delta and RMS readings 0.14–0.31). That exercises our loading, normalization, layout,
  orientation, denormalization and metric on the real data. Its mean comes from the first ~20 frames, where the
  nearly still fluid makes velocity VRMSE 3–4.
* *The released TFNO does not work under any convention.* On developed flow it scores ~1.1 on every field
  (the field mean scores 1). Read as a delta, it lands on persistence (0.216 vs 0.204), so its output carries
  almost no information. The two contraction paths (factorized and reconstructed) agree to 2e-6. With the same
  weights, no channel order (t f / f t), x/y transpose, flip or field order gives an output that tracks the
  input: the best worst-field correlation is 0.08 over 40 combinations (`scripts/probe_released_layouts.py`,
  synthetic input). Tracking the input is necessary but not sufficient: CNextU-Net reaches 0.86 with the
  correct layout and up to 0.94 with wrong field orders. So this test can rule layouts out but not confirm
  one. The real-data probe above is the confirmation for CNextU-Net.
* *Upstream status.* The Well issue #49 ("Inconsistent Pre-trained Baseline Performance", open) attributes
  mismatches to a normalization change and to Hub checkpoints trained "before dataset orientations were
  normalized"; retrained checkpoints and corrected paper numbers are promised but not released. A comment of
  2026-11-05 reports the same pattern on `turbulent_radiative_layer_2D`: the Hub UNetConvNext reproduces, while
  the Hub FNO/TFNO score 2.0/3.98 against 0.50 (persistence 0.58), and transposes, field orders, RMS and delta
  do not fix them. `docs/benchmarks.md` on master still lists the paper values for `rayleigh_benard`.
* *Consequence.* The 5% criterion against the published values cannot pass and does not test our pipeline.
  Our own TFNO is trained from scratch in the pinned environment, so the broken checkpoint does not affect
  the study.

*Root causes (2026-11-08), from the paper-era code at tag v1.0.1 and the repository history.*
* *Same inputs as the paper.* The data files on the server are dated 2024-12-05 (HTTP Last-Modified), the
  original release. The splits are the official ones (1400/175/175). The v1.0.1 `stats.yaml` equals the
  current one to the digits shown. The benchmark data config uses z-score normalization and batch 64.
  Evaluating a checkpoint involves no seed.
* *Same code source.* The models, normalization and VRMSE come from `github.com/PolymathicAI/the_well`,
  but at `e63dec8` (2026-10-12), not the paper-era v1.0.x. The model classes differ only in their
  constructor signature, except for the TFNO fix below. The VRMSE code is the same (eps 1e-7).
* *TFNO never used its Fourier layers.* From 2024-06-03 ("Added gradient checkpointing") to commit `e430160`
  (2025-11-17, "actually use the TFNO blocks"), the wrapper called `fno_blocks` without keeping the output.
  The paper's TFNO, and the released checkpoint (2025-03-28), is therefore lifting → projection, a per-pixel
  MLP. Its block weights were never trained, and the fixed code runs them, which gives noise. With the
  paper-era forward, the released TFNO tracks its input (correlation 0.97–1.00 on synthetic data). It is
  evaluated that way (`released.PAPER_FORWARD`).
* *The public v1.0.1 test set is unnormalised, but the published test values were not produced that way.*
  In v1.0.1, `test_dataset` and `rollout_test_dataset` are built without `use_normalization`; v1.1.0 fixed
  this ("Fixed normalization bug in test dataset"). Emulating that bug gives CNextU-Net 9.99 and TFNO 1.18
  against the published 0.67 and 0.66 (rerun, 2026-11-08). So the paper's runs must have normalised the test
  set, and the public v1.0.1 is not exactly the code that produced them. (Corrects the first reading of
  2026-11-08, which blamed the published test values on this bug.)
* *The paper's validation numbers are a short validation in normalised units.* During training, validation
  used 20 shuffled batches of 64 windows (`short_validation_length: 20`), with no conversion to physical
  units. The best checkpoint was chosen on that score.

*Criterion (owner, 2026-11-08): "you don't need to match exactly with the paper, but results should be at
least as good as the paper."* `eval-released` now reports three one-step VRMSEs per window:
* ours (normalized inputs, physical units);
* `paper_valid` (normalized units);
* `paper_test` (raw inputs, as v1.0.1).

It passes when ours ≤ published for every model and split. A paper-protocol value within 10% of the
published one is reported as reproduced; this is evidence, not the criterion.

*Rerun (2026-11-08, job 59224204, commit c8e3065): PASSED.*

| | ours | published | paper-valid protocol | paper-test protocol |
|---|---|---|---|---|
| CNextU-Net valid | 0.234 | 0.481 | 0.701 | |
| CNextU-Net test | 0.224 | 0.670 | | 9.99 |
| TFNO valid | 0.305 | 0.857 | 0.675 | |
| TFNO test | 0.300 | 0.657 | | 1.18 |

* With its paper-era forward, the released TFNO (a per-pixel MLP) scores 0.30, against 10.7 with the pinned
  forward.
* Neither paper protocol reproduces the published values. The valid values come from one 1,280-window
  random draw per epoch on heavy-tailed per-window scores, with the best epoch picked on that draw. The
  test values were not computed with the public test-set bug (above).

*Our trained models (2026-11-08):* comparing them with the published values is not like for like: under the
paper's validation protocol, the paper's own released CNextU-Net scores 0.70 against its published 0.48. So
`compare-paper` compares our run with the released checkpoint scored identically: the same windows, code and
data. It requires ours ≤ released on valid and test, in physical units (our protocol) and in normalised units
(the paper's validation protocol). The published values are reported only.

**D21 — Backup plan (deferred).** If storage, queue time or pilot timings show the plan is too costly, the
data size is reduced for all models equally.
* Fewer trajectories already works: a smaller fraction through `make-subsets`.
* Downsampled fields is **deferred by the owner (2026-10-30)** and not implemented. x can be strided. A y
  stride leaves a non-Chebyshev grid, which invalidates the y operators and Nusselt but not VRMSE or the
  x-only spectra. AViT needs both sizes divisible by 16.

**D22 — GPU-hours accounting.** Every job uses one GPU, so GPU-hours are measured wall hours on CUDA:
* training: each attempt records its wall time from `train_run` start to exit, and how it ended, in
  `provenance.json`. An attempt killed before it could record (e.g. SIGKILL) falls back to its last
  `train_metrics.csv` row, a lower bound, and is counted in `unclosed_attempts`;
* evaluation: each `EVAL_COMPLETE_<label>.json` records the wall time of its pass, counted once per pass.

`rbbench aggregate` writes them per run and in total to `tables/gpu_hours.csv`. Queue time and the ~1 min of
environment setup per job are not included; `sacct` reports Slurm's billed time for comparison. Profiling, the
pipeline check and the learning-rate search are not runs in the task file, so they are read from `sacct`.
