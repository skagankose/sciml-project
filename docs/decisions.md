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
upstream) on all parameters, no gradient clipping (as upstream). The paper's text (Appendix E.1) says "the PyTorch default WD of .01",
but The Well's `configs/optimizer/adam.yaml` has 1e-4 both at v1.0.1 and at `e63dec8` (read 2026-11-08); we
follow the code. Likewise its `rayleigh_benard` data config has batch 64, where the text says the batch was set
to fill GPU memory.

**D6 — Learning-rate schedule under a time budget (fixed 2026-11-09 when the core runs started; the planned comparison with a constant rate was not run).** Warmup in optimizer steps,
then cosine decay over *cumulative training seconds* towards the final 4 h milestone. The total step count is
unknown in advance, so a step-based cosine cannot end exactly at 4 h. Decaying on time makes the 4 h
checkpoint the end of the schedule; the 1 h checkpoint is an intermediate state of the same schedule, not a
separately tuned 1 h run. (Budgets were 2 h/6 h before the final abstract.) The rate depends only on
(step, train_seconds), so resume is exact. The pilot compares this with a constant rate on validation only.
The budgets live in `train.budgets_seconds` (`1h: 3600, 4h: 14400`); their labels name the checkpoints and
evaluations everywhere, and an override replaces the whole mapping rather than merging into it.

**D7 — Learning rate and batch size (fixed 2026-11-09: batch 64, learning rate 1e-3 for every model).** One shared batch size for all models, chosen from the
Phase G memory/throughput table. The final abstract commits to an equal-budget learning-rate search on
validation data before training. Deferred on 2026-10-30; **run, one rate per model (owner, 2026-11-08)**:
* *Search* (`configs/experiments/lr_search.yaml`, `tasks/lr_search.jsonl`): CNextU-Net, TFNO and AViT ×
  {1e-4, 5e-4, 1e-3}, all on the 10% subset, seed 0, each for the same 20 min of GPU training time (D24) with
  the study's recipe. Two differences, both from the 12× shorter budget: 100 warmup steps instead of 500 (500
  would be 60% of CNextU-Net's ~850 steps and end in a jump down the cosine, D6), and 512 validation windows
  instead of 64. About 3 GPU-h of training on `free-gpu`.
* *Selection* (`rbbench select-lr`): per model, the lowest validation VRMSE at 20 min (owner, 2026-11-08: the
  study's metric and the paper's criterion; validation loss is reported too); a diverged rate is the worst. A choice at the edge of the grid is flagged. Results in `outputs/lr_search/`. No test data is read.
* *Where the rates live:* `model_overrides` in each experiment config (merged after the model file, only the
  loaded model's entry, so one model's rate never changes another's hash). `main.yaml` gets the search's
  rates, `replication.yaml` keeps The Well's. `tasks/train.jsonl` is regenerated after the search; until then
  it is out of date and `train-task` refuses it.
* *Known bias:* a 20 min search can favour rates that learn fast early over ones that are best at 4 h. It is
  the same for every model.
* *Result (2026-11-09, all 9 runs; jobs 59374848, 59390860, 59393014; `docs/checkin_2026-11-12.md`):*
  validation VRMSE at 1e-4 / 5e-4 / 1e-3: CNextU-Net 0.841 / 0.469 / **0.408**, TFNO 0.603 / 0.358 / **0.302**,
  AViT 0.646 / 0.332 / **0.257**. Every model takes **1e-3**, set in `main.yaml`'s `model_overrides`;
  `tasks/train.jsonl` was regenerated (10% rows 0, 1, 4, 5, 8, 9). No run diverged. 1e-3 is the edge of the
  grid for all three. **Owner, 2026-11-08: keep the three-rate grid**, so this is reported as a limitation.
  (Had it been widened, the new rates would have gone into a separate manifest: `sweep` is part of the config
  hash, so editing `lr_search.yaml`'s `lrs` changes every search run's id.)
* If a model needs gradient accumulation to reach the shared batch size, it is recorded. No test-set tuning.
* *Replication of the paper (owner request, 2026-11-08):* our trained CNextU-Net and TFNO must be at least as
  good as the paper's (D20). To compare like with like, the two models use the rates The Well's own sweep
  chose for `rayleigh_benard` (model cards): CNextU-Net 5e-4, TFNO 1e-4, set in `replication.yaml`. They are
  not tuned by us, and no test data is involved.
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

**D9 — Budget clock.** Per optimizer step. Excludes startup, worker spawn, validation and checkpoint writes;
those are logged separately (`overhead_seconds`). *Since 2026-11-08 (D24)* the step is counted from the moment
its batch is ready to the synchronised end of `optimizer.step()`, so waiting for the data loader is excluded
too (`data_wait_seconds`). Before, it was counted from requesting the batch (`train.budget_clock: wall`).

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

**D15 — Rollout cohort (decided by the owner 2026-11-10).** All 175 test trajectories × start indices `[20, 80, 140]`, 30 steps.
Very early starts are near the conduction state, where target variance is tiny and VRMSE is ill-conditioned. The
start list must be frozen on the *validation* split before any test evaluation, then `tasks/eval_cohort.json`
is generated once and never changed (its hash is checked).
*2026-11-09:* `eval` in `common.yaml` is part of every run's config hash, so it stays as it is now that the
core runs have started. If validation calls for other starts, they go into the cohort file only (a
`make-cohort` option, still to add), never into `common.yaml`.
*2026-11-10, owner: keep `[20, 80, 140]` and add the paper's protocol (option A of three).* On validation, the
released checkpoints' one-step VRMSE (physical units) by start frame is 1.65 / 1.36 / 0.71 / 0.23 / 0.08 /
≤ 0.06 for CNextU-Net, over frames 0–9 / 10–19 / 20–29 / 30–39 / 40–49 / 50+. So start 20 tests the onset of
convection and 80 and 140 developed convection, with VRMSE well conditioned. Each start is reported separately.
* The alternatives were the paper's frame 0 only, which mostly measures spurious velocity in a nearly still
  fluid (D20), and later starts such as 50/100/150, which need a code change because the starts are hashed.
* Every evaluation also scores the paper's protocol (D20): one-step VRMSE in normalised units on every test
  window, and a 30-step rollout from frame 0 of every trajectory, averaged over steps 6–12 and 13–30.
  `rbbench aggregate` puts these in `tables/paper_protocol.csv`, next to the published values and the
  released checkpoints scored by the same code. The frame-0 rollout reuses the frames already read, so it
  adds no data reading.

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
* *Model cards (read 2026-11-08 at the pinned revisions):* up to 12 h or 500 epochs on one H100 96 GB; the
  released checkpoint is the best on validation (epoch 12 CNextU-Net, 31 TFNO); learning rate the best of 5
  (5e-4, 1e-4); 4 input frames; batch "set to maximize the memory usage" (the data config says 64). No seed
  and no training date are given: one run per model. Summary for the report in `docs/checkin_2026-11-12.md`.
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
  2026-11-08, which blamed the published test values on this bug.) The test values were computed on
  normalised inputs and scored in normalised units, and they are reproduced that way (2026-11-09, below).
* *The paper's validation numbers are a short validation in normalised units.* During training, validation
  used 20 shuffled batches of 64 windows (`short_validation_length: 20`), with no conversion to physical
  units. The best checkpoint was chosen on that score.

*Criterion (owner, 2026-11-08): "you don't need to match exactly with the paper, but results should be at
least as good as the paper."* `eval-released` now reports three one-step VRMSEs per window:
* ours (normalized inputs, physical units);
* `normalised` (normalized inputs, scored in normalized units: the paper's protocol);
* `raw_input` (raw inputs, as the public v1.0.1 test set; a diagnostic).

These two columns were called `paper_valid` and `paper_test` until 2026-11-09; the files of job 59224204
use the old names. It passes when ours ≤ published for every model and split. A normalised-units value
within 10% of the published one is reported as reproduced; this is evidence, not the criterion.

*Rerun (2026-11-08, job 59224204, commit c8e3065): PASSED.* All columns from that job's
`released_check.json`; every window of the 175 trajectories of each split (34,300 windows).

| | ours (physical units) | published | normalised units | raw inputs |
|---|---|---|---|---|
| CNextU-Net valid | 0.234 | 0.481 | 0.701 | 10.15 |
| CNextU-Net test | 0.224 | 0.670 | **0.669** | 9.99 |
| TFNO valid | 0.305 | 0.857 | 0.675 | 1.19 |
| TFNO test | 0.300 | 0.657 | **0.663** | 1.18 |

* With its paper-era forward, the released TFNO (a per-pixel MLP) scores 0.30, against 10.7 with the pinned
  forward.
* *The published test values are reproduced (found 2026-11-09).* In normalised units on the full test set,
  the released checkpoints score CNextU-Net 0.6688 against the published 0.6699 (−0.2%) and TFNO 0.6634
  against 0.6566 (+1.0%; with its paper-era forward). The 2026-11-08 reading ("neither paper protocol
  reproduces the published values") was wrong: it scored valid only in normalised units and test only with
  raw inputs. `eval-released` now judges reproduction in normalised units on both splits
  (`released.PAPER_PROTOCOL`). So our data, splits, normalisation, model loading and metric reproduce the
  paper's test numbers to 1%.
* *The published valid values are not reproduced* (0.70 / 0.68 on every window). They come from one
  1,280-window random draw per epoch on heavy-tailed per-window scores, with the best epoch picked on that
  draw. The Hugging Face model cards list these valid values (0.4807, 0.8568) without naming a split, next
  to the text "We use the model performing best on the validation set to report test set results."
* *Why units matter.* The same predictions score 0.224 and 0.669 (CNextU-Net, test). VRMSE divides by the
  target's spatial variance plus eps = 1e-7. Over the first ~50 frames the fluid is nearly still, the
  velocity variance in physical units is close to eps, and eps damps the score; in normalised units it does
  not. Buoyancy and pressure agree in both units throughout, and from frame 50 on all fields do (CNextU-Net
  0.0382 vs 0.0383). The paper's numbers are therefore dominated by velocity in the first frames.
* *The criterion mixes units.* "Ours ≤ published" compares physical-unit scores with normalised-unit
  published values. In normalised units the released TFNO is 1.0% above its published test value. The
  like-for-like evidence for the pipeline is the reproduction above. The criterion is the owner's and is
  unchanged.

*Our trained models (2026-11-08):* comparing them with the published valid values is not like for like:
under the paper's protocol, the paper's own released CNextU-Net scores 0.70 on every valid window against its
published 0.48. So `compare-paper` compares our run with the released checkpoint scored identically: the same
windows, code and data. It requires ours ≤ released on valid and test, in physical units (our protocol) and in
normalised units (the paper's protocol). The published values are reported only. On test, the released
checkpoint reproduces the published value in normalised units to 1% (2026-11-09), so that check is in effect a
check against the paper's test number.

*Rollouts (Table 3), added 2026-11-09; the paper's ">10" is reproduced on test (job 59407152).* The paper reports test
VRMSE averaged over rollout steps 6–12 and 13–30, with rollouts "initiated from the beginning of the
simulation" (Section 4). For `rayleigh_benard` every model, CNextU-Net and TFNO included, is ">10" in both
windows. Our study's rollouts start at frames 20, 80 and 140 (D15), so they cannot be compared with these.
* `eval-released` now also rolls each released checkpoint out from frame 0 of every trajectory: 30 steps,
  autoregressive on normalised data, as the paper-era trainer did (v1.0.1 neither normalises nor
  denormalises inside a rollout). It reads only the first 34 frames of each trajectory.
* Scores per step: VRMSE in normalised units (the paper's) and in physical units (ours), averaged per window
  over steps and trajectories (`paper.paper_rollout_windows`; the published values and windows are in `src/rbbench/paper.py`). Per-step rows are in
  `<model>/paper_rollout_<split>.parquet`.
* Reported, not judged: on test, "reproduced" means both normalised-unit window means are above 10.
* Job 59407152 (all 175 trajectories per split), window means over steps and trajectories:

  | Split, window | CNextU-Net, normalised | CNextU-Net, physical | TFNO, normalised | TFNO, physical |
  |---|---|---|---|---|
  | test, steps 6–12 | 36.1 | 9.28 | 142 | 37.1 |
  | test, steps 13–30 | 34.7 | 16.5 | 169 | 73.9 |
  | valid, steps 6–12 | 39.0 | 9.98 | 155 | 40.0 |
  | valid, steps 13–30 | 39.0 | 18.5 | 196 | 84.8 |

  * Both checkpoints are above 10 in the paper's units on both splits, as published.
  * The velocity components produce all of it: buoyancy and pressure stay at 0.08–0.49 in both windows for
    both models, while velocity_x and velocity_y score 64–431. From frame 0 the fluid is nearly still, so
    the true velocity has almost no spatial variance and any velocity error is large relative to it.
  * The normalised score peaks near step 15 (CNextU-Net 44, TFNO 297) and falls as convection develops.
  * So ">10" mainly reflects that VRMSE is ill-conditioned at the start of an RB simulation. The models
    still differ (test: CNextU-Net 35–36, TFNO 142–169). The score measures how much spurious velocity a
    model adds to a nearly still fluid, mostly at low Rayleigh numbers, more than how well it tracks
    developed convection.
  * Job 59397615 reached its 2 h limit during the test split (valid took 81 min with startup), and its log
    lost the valid lines because Python buffers stdout. `released_a100.sbatch` now allows 4 h and sets
    `PYTHONUNBUFFERED=1`. The rerun (job 59407152, paid `gpu` partition) gave the same valid values and
    the one-step test reproduction: 0.6688 / 0.6634 against 0.6699 / 0.6566 (−0.2% / +1.0%).
* Window definition: v1.0.1 logs log-spaced windows `np.linspace(0, log T, 5)` of a T-step rollout. With
  T = 30 they are the slices 5:12 and 12:30, which are steps 6–12 and 13–30 (1-based). The v1.0.1 default
  T = 100 would give 3:10 and 10:31, so the paper's runs presumably used 30 steps. We use the paper's stated
  windows, 1-based and inclusive.

*Replication budget (owner, 2026-11-08): "train as much as we need to match the paper"; the study keeps its
limited budgets.* The paper's best checkpoints came at epoch 12 (CNextU-Net) and 31 (TFNO). At batch 64 on
our A100 that is ~20 h of training for either model, so each replication run gets 24 h, with checkpoints at
6, 12, 18 and 24 h (`configs/experiments/replication.yaml`, `tasks/replication.jsonl`). That costs about 50
GPU-h on `free-gpu`, outside the study's ~100 GPU-h, and the runs are never aggregated with the study.

**D23 — Data loading (2026-11-08).** The first replication runs were I/O-bound. CNextU-Net ran at 5.75 s
per step and TFNO at 7.8 s (later 47 s), at batch 64, against 1.42 s and 0.55 s in the profile.
* *Cause:* upstream `WellDataset._load_one_sample` (e63dec8) opens the HDF5 file anew for every sample,
  through fsspec with a fresh 8 MB block cache. On BeeGFS each sample then costs open and metadata round
  trips, and reads many 8 MB blocks for about 5 MB of fields.
* *File layout (checked on HPC3):* every field is stored contiguously and uncompressed (`chunks None`).
  Buoyancy and pressure are `(40, 200, 512, 128)` and velocity `(40, 200, 512, 128, 2)` float32 per file. A
  window of 5 frames is therefore three contiguous reads: 1.3 + 1.3 + 2.6 MB. The native driver reads
  exactly that, so the files need no repacking.
* *Fix:*
  * `data.PersistentWellDataset` overrides only the file opening: one `h5py.File` per path per process,
    native driver, 64 MB chunk cache, reopened after a fork. Reading, normalization and sample assembly stay
    upstream's, and samples are bitwise identical (`tests/test_fast_loader.py`).
  * Loader settings live in `runtime`, excluded from the config hash. `RBBENCH_NUM_WORKERS` sets the
    worker count per job; the Slurm scripts use CPUs − 1.
  * The profiler reads fresh random windows per measurement. Before, every measurement after the first
    re-read the same cached windows, so the D7 step times measure compute with cached I/O.
* *Needed rate at batch 64 so that compute, not I/O, sets the pace:* CNextU-Net about 45 samples/s
  (235 MB/s), TFNO about 116 (610 MB/s), AViT about 170 (890 MB/s). `rbbench bench-data` measures what
  `/pub` delivers, on a CPU node.
* *Measured (`bench-data` on a 16-core `free` node, batch 64, 15 timed batches per row):*

  | loader | 4 workers | 8 workers | 15 workers |
  |---|---|---|---|
  | upstream | 24.5 samples/s (129 MB/s) | 80.7 (423) | 56.8 (298) |
  | persistent | 33.8 (177) | 97.7 (512) | 69.9 (366) |

  * Persistent handles are 1.2–1.4× faster at equal workers. The remaining limit is `/pub` itself: at best
    about 0.5 GB/s into one node, and 15 readers were slower than 8. Each row times only 10–30 s of reads,
    so single rows are noisy, but no setting reaches TFNO's or AViT's rate.
  * The first replication attempt had 4 workers with the upstream loader (about 25 samples/s here). The
    training jobs now have 8 CPUs, so 7 workers: CNextU-Net is compute-bound, and TFNO should run at about
    0.65 s per step against 0.55 s, roughly 85% of its compute rate. That is acceptable for the
    replication, whose TFNO bar (the released checkpoint, D20) never trained its FNO blocks.
  * AViT would get about half its compute rate. With a fixed training time per run that handicaps AViT in
    the core study. **Open before the core sweep:** read the data from node-local disk (depends on the A100
    nodes' local disk size) or measure the shortfall and report it.
    *Measured 2026-11-08 (D24):* on the A100 nodes, the loader keeps up with AViT on the 10% subset between
    storage stalls, and budgets count GPU time, so neither step is needed for the 10% runs. The 100% runs
    (10× more data per epoch) are checked in their first logs.
* The core manifests were regenerated (only the hash changed); the partial runs under the old hashes are
  void.

**D24 — Budgets count GPU training time, not waiting for data (owner, 2026-11-08).** `/pub` delivers about
98 samples/s into one node (D23). CNextU-Net never waits, but AViT could use ~170 and would spend ~40% of a
time budget waiting, so a wall-clock budget would give it about half its steps: the comparison would partly
measure our storage.
* `train.budget_clock: compute` (common.yaml): a step is counted from the moment its batch is ready to the
  synchronised end of the update (host-to-device copy, forward, backward, optimizer). The device is
  synchronised every step, so the time spent in the loader is pure waiting. It is logged as
  `data_wait_seconds` (metrics, milestones, `COMPLETE.json`, `resource_use.csv`). `wall` keeps the old count.
* A budget therefore buys every model the same GPU time whatever the storage does. The learning-rate schedule
  (D6), validation interval and milestones all use this clock.
* Cost: runs that wait take longer in real time (estimate for 4 h: CNextU-Net ~4 h, TFNO ~5 h, AViT ~7 h
  plus validation and checkpoints), and GPU-hours (D22) count the real time. `train_a100.sbatch` allows 10 h.
  The real waits come from the search runs (`rbbench select-lr`, pilot timings).
* The key is part of the config hash, so the manifests were regenerated (replication run ids now
  `…-9dfed09a` and `…-bb6db502`); earlier partial runs are void.
* *Measured in the learning-rate search (2026-11-08, job array 59374848, A100 nodes `hpc3-gpu-l54-05/06`):*
  * GPU seconds per step at batch 64: CNextU-Net 1.19, TFNO 0.55, AViT 0.19, within 1% across runs. These
    replace the D7 profile's 1.42 and 0.38 s, which timed whole steps with cached reads.
  * Between stalls the loader keeps up with every model on the 10% subset. The median wait per logging
    interval is about 0, AViT included at ~330 samples/s (fresh reads in epoch 0 too). So an A100 node reads
    `/pub` well above the ~98 samples/s that `bench-data` measured on a CPU node (D23).
  * All the waiting came from `/pub` stalls of 3–21 min that hit runs on both nodes at the same moments (TFNO
    at ~03:26 UTC, AViT at ~04:28 and ~04:47 UTC). They cost 5–30 min per 20 min run. At 02:55 UTC the same
    trouble killed two runs outright with `OSError: [Errno 70] Communication error on send` (an HDF5 open in a
    loader worker; a Python import). Neither is divergence: no `FAILED.json`, and they resume on resubmission.
  * Consequence: budgets are unaffected. Real time is ~4 h for a 4 h budget without stalls, and up to ~2–3×
    at the stall rate seen that day. `select-lr` reports both estimates: mean wait, and median wait per
    interval.

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
