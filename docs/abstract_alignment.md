# Final abstract vs implementation

This file compares the final project abstract ("Data–Compute Trade-offs in Neural PDE Forecasting") with the
code at commit `41b2162`, checked on 2026-10-30. The final abstract replaces the earlier one (`221Project.pdf`),
and the code and docs still follow the earlier plan. Entries list only what differs; everything else
matches.

## Scope: earlier plan (code) vs final abstract

| | Code and docs now | Final abstract |
|---|---|---|
| Models | 4: CNextU-Net, TFNO, AViT, ConvLSTM | 3 core; ConvLSTM is an extension |
| Fractions | 10 / 30 / 100% | 10 / 100%; 30% is an extension |
| Seeds | 3 | 2; a third is an extension |
| Runs | 36 | 12 |
| Budgets | checkpoints at 2 h and 6 h | checkpoints at 1 h and 4 h |
| Noise robustness | part of every evaluation | extension, evaluation only |
| Nusselt number | computed if the dataset audit passes | not mentioned |
| Released checkpoints | forbidden (AGENTS.md rule 3, test) | evaluated first to validate the pipeline |
| GPU budget | 216 GPU-h training, ≈340 total | 48 GPU-h training, ≈100 total |

## Needs owner approval

**F1 — Evaluating The Well's released checkpoints conflicts with the no-pretrained-weights rule.** **Done
2026-11-07:** `rbbench eval-released` and `cluster/slurm/released_a100.sbatch` (decisions D20).
* *Abstract:* "we first evaluate The Well's released CNextU-Net and TFNO checkpoints and compare with their
  published VRMSE"; this is also the Nov 12 milestone.
* *Code:* AGENTS.md rule 3 says "Never use pretrained weights", and
  `tests/test_backward.py::test_no_pretrained_weight_loading_in_source` fails if any `src/` file calls
  `from_pretrained` or `hf_hub_download`. There is no command to evaluate a released checkpoint.
* *Upstream (checked 2026-10-30):*
  * The checkpoints exist: `polymathic-ai/UNetConvNext-rayleigh_benard` and
    `polymathic-ai/TFNO-rayleigh_benard`, created 2025-03-28.
  * Their configs match `configs/models/*.yaml` `full`: CNextU-Net init 42, 4 stages, 2 blocks per stage,
    1 neck block; TFNO modes 16, hidden 128. Both use 512×128 inputs and dim_in 16.
  * The pinned benchmark data config sets `use_normalization: True` with no type, which falls back to
    `ZScoreNormalization` (the shipped `stats.yaml`). That is the same normalization as D4.
  * Published references: CNextU-Net valid 0.4807 / test 0.6699, TFNO valid 0.8568 / test 0.6566. These are
    one-step, field-averaged VRMSE from the best-validation checkpoint after 12 h on an H100.
* *Fix:*
  * Owner: done 2026-10-30. Rule 3 now reads "Never train from pretrained weights. The Well's released
    checkpoints may be loaded only to evaluate them as a pipeline check."
  * Code: add an `eval-released` command in one module (and allow only that module in the test guard). It
    should reuse the one-step evaluator and write to `outputs/baselines/released/`.

## Code and config changes

**F2 — Core sweep: 3 models × {10, 100}% × 2 seeds = 12 runs.**
* *Code:*
  * `common.yaml` has `sweep.seeds: [0, 1, 2]` and `fractions: [0.1, 0.3, 1.0]`.
  * `make-sweep` defaults to all four `MODEL_NAMES` and is described as "36 rows".
  * `train_a100.sbatch` uses `--array=0-35%1` and `eval_a100.sbatch` uses `0-35%2`.
* *Fix:*
  * Make the core values the defaults: three models, fractions `[0.1, 1.0]`, seeds `[0, 1]`, arrays
    `0-11`.
  * Keep the extensions as overrides: `--models convlstm`, `--fractions 0.3`, `--seeds 2`.
  * Still generate the subset manifests for 10, 30 and 100%. Subsets are prefixes of one seeded permutation,
    so S10 ⊂ S30 ⊂ S100 stays true when 30% is added later.

**F3 — Budgets 1 h and 4 h, not 2 h and 6 h.** **Done 2026-11-07:** labels come from the config everywhere,
and `train_a100.sbatch` allows 5 h.
* *Code:*
  * `common.yaml` has `budgets_seconds: {2h: 7200, 6h: 21600}`. The cosine schedule ends at the last budget,
    so it must become 4 h.
  * The labels `"2h"`/`"6h"` are hard-coded in `aggregate.py` (lines 59, 381) and in `cli.py` (smoke: lines
    339–342; `--labels` defaults: lines 449, 481).
  * `profile.py` reports `est_steps_2h` / `est_steps_6h`.
  * Tests (`test_budget.py`, `test_checkpoint_resume.py`, `test_aggregate.py`) use these labels, and
    `train_a100.sbatch` has `--time=08:00:00`.
* *Fix:*
  * Set `budgets_seconds: {1h: 3600, 4h: 14400}`.
  * Derive the labels from the config instead of hard-coding them.
  * Update the profile columns and the tests.
  * Set the training time limit to about 5 h. That covers 4 h of training plus validation, checkpoints and
    startup, and keeps resume on preemption.

**F4 — Noise robustness is an extension; evaluation runs it every time.** **Done 2026-11-07:** core evaluation
is clean only; `configs/experiments/noise.yaml` holds the extension (decisions D17).
* *Code:* `eval.noise_sigmas: [0.0, 0.01, 0.05, 0.10, 0.20]` makes each rollout run five times (four noisy
  copies plus the clean one).
* *Fix:*
  * Core: set `noise_sigmas: [0.0]`, which saves evaluation GPU time inside the ~20 GPU-h overhead.
  * Extension: re-run `eval-task` with the full list. Its noise draws are fixed by the seeded hash, so the
    re-run evaluates the same noise.

**F5 — Equal-budget learning-rate search is not implemented** (carried over from the earlier check).
**Deferred by the owner (2026-10-30); revisit before the sweep.** Recorded in decisions D7.
* *Abstract:* "an equal-budget learning-rate search on validation data precedes training"; "budgets fixed from
  pilot timings" is a Nov 12 milestone.
* *Code:* one fixed `optim.lr: 5.0e-4` (PENDING, D7), no search command and no search budget. `make-sweep`
  applies the same overrides to every model.
* *Open decision:* one rate shared by all three models, or the best rate per model from the same grid and
  budget. The abstract says the models share the learning-rate *schedule*, which allows either. The Well's
  own RB rates differ (5e-4 CNextU-Net, 1e-4 TFNO).
* *Fix:*
  * Code: add a search config with the same budget per (model, lr) and selection on validation loss only.
    Allow per-model overrides in `make-sweep` if the rates are per model.
  * Budget: keep the whole search to a few GPU-hours, e.g. 3 models × 3 rates × 20 min ≈ 3 GPU-h. It
    shares the ≈20 GPU-h of overhead with profiling, the released-checkpoint check and evaluation.

**F6 — GPU-hours are not computed** (carried over). **Done 2026-11-07:** `tables/gpu_hours.csv` (decisions D22).
* *Abstract:* "We also report GPU-hours, inference latency, and peak GPU memory."
* *Code:* `tables/resource_use.csv` has training seconds, overheads, steps, latency and peak memory, but no
  allocated GPU-hours. Evaluation jobs record no cost.
* *Fix:* record each attempt's wall time, or the Slurm job's elapsed time, in `provenance.json`. Then sum it
  for training and evaluation, which also checks the ≈100 GPU-h estimate.

**F7 — Backup plan: downsampling is not implemented.** **Deferred by the owner (2026-10-30); revisit if the
backup plan is triggered.** Recorded in decisions D21.
* *Abstract:* "reduce the data size (downsampled fields or fewer trajectories) for all models equally."
* *Code:* fewer trajectories already works through the subset manifests. Downsampling the real data does not
  exist in `data.py`.
* *Note:* x can be subsampled with a stride. y is Chebyshev-spaced, so a stride there gives a non-Chebyshev
  grid. The y spectral operators and Nusselt would no longer apply, but x-only spectra and VRMSE would. AViT
  needs both sizes divisible by 16. This only needs building if the backup is triggered.

## Docs and wording

**F8 — Dataset size. Dropped by the owner (2026-10-30).** The abstract says 342 GB. The remote total is
367.0 GB (341.8 GiB). Left as is.

**F9 — Docs describe the earlier plan. Done 2026-10-30.** The docs now describe the final abstract's plan, and
say where the code still differs:
* `README.md`;
* `AGENTS.md`: rule 1, gates G and I, a new gate D2 (pipeline check), and the progress log;
* `docs/methodology.md`;
* `docs/decisions.md`: D3, D4, D6, D7, D10, D12, D17, D19, plus new D20 (pipeline check) and D21 (backup plan);
* `docs/results_schema.md`;
* `cluster/README.md`: a new step 3b, the 12-run sweep commands and the storage and SU figures;
* `tasks/README.md`.

**F11 — The released checkpoints do not reproduce the published VRMSE (found 2026-11-07).** For the report.
* *Abstract:* "we first evaluate The Well's released CNextU-Net and TFNO checkpoints and compare with their
  published VRMSE."
* *Result:* CNextU-Net scores 0.234 valid / 0.224 test, against 0.481 / 0.670 published. It works (0.02–0.05
  on developed flow, against 0.20 for persistence) and is better than the published value. The released TFNO is
  non-functional under every input convention (10.7 / 10.5, against 0.857 / 0.657). Details are in decisions
  D20.
* *Context:* The Well issue #49 (open) acknowledges that the paper numbers and some Hub checkpoints are
  inconsistent with the current code. Another user reported the same FNO/TFNO-only failure on a second dataset.
* *Root causes (2026-11-08, D20):*
  * the paper-era TFNO dropped its Fourier-layer outputs (fixed upstream in `e430160`, 2025-11-17);
  * the paper's validation was a 1,280-window short validation in normalized units;
  * the public v1.0.1 builds the test set without normalization (fixed in v1.1.0), but emulating that does
    not give the published test values, so the paper's runs normalised it.

  The data, splits and normalization statistics are the paper's.
* *Criterion (owner, 2026-11-08):* our results must be at least as good as the paper's.
  * Released checkpoints: ours ≤ published. **PASSED** on the 2026-11-08 rerun: CNextU-Net 0.234 / 0.224,
    TFNO 0.305 / 0.300, against 0.481 / 0.670 and 0.857 / 0.657.
  * Our trained runs: ours ≤ the released checkpoint scored identically (`compare-paper`).
* *Report:* explain the upstream issues, give both the published values and the released checkpoints'
  scores on our pipeline, and say the exact published values could not be reproduced (protocol details are
  not fully recoverable). Note that the paper's TFNO is a per-pixel MLP. Our TFNO, with working Fourier
  layers, is a different and stronger model.

**F10 — Smaller wording points, already consistent with the code.**
* "Each run trains for 4 hours": the clock counts training-loop time and excludes validation and checkpoint
  writes (D9). Say so in the report.
* "Predefined error threshold": thresholds 1.0, 2.0 and 5.0 are computed, and the primary one is frozen on
  validation (D16).
* The persistence baseline is computed but not in the abstract. It is optional to report.

## What already matches

* **Data and task:** 1,750 trajectories, 35 (Ra, Pr) regimes, 200 frames on a 512×128 grid (periodic x,
  Chebyshev y), 4 → 1 frame prediction of 4 fields with autoregressive rollout, the official 80/10/10
  splits, and train-split z-score statistics.
* **Model sizes:** CNextU-Net 18.58M, TFNO 19.28M and AViT 18.57M (hidden 384 → 432), all trained from
  scratch with The Well's classes.
* **Shared settings:** subsets, targets, MSE loss, one batch size and one time-based cosine schedule for all
  models, with the shorter checkpoint as an intermediate state of the longer run.
* **Metrics:** VRMSE (RMSE / std, The Well's implementation, per field) for one-step and rollout,
  threshold-crossing steps (right-censored), seed variation, x-only kinetic-energy spectra, inference latency
  and peak GPU memory.
* **Hardware:** one A100 80 GB per run on `free-gpu`.

## Nov 12 check-in: status on 2026-10-30

* **Released-checkpoint check:** blocked by F1, and needs HPC3 (Phase C) and the test split (36.7 GB).
* **Budgets from pilot timings:** needs Phases C and G, plus F5 and F6.
* **First 10% results:** 6 runs × about 4.5 h ≈ 27 h sequential on `free-gpu`. They are written after F2–F4.
* **Order:** Phase C, download valid and test first (37 GB each, enough for the released-checkpoint check),
  then train. The train split is 294 GB, so start its download early.

## Checklist

- [x] F1 owner approval (rule 3 amended)
- [x] F1 `eval-released` command and test exception (2026-11-07)
- [ ] F2 12-run core sweep defaults (the runbook passes the grid explicitly meanwhile)
- [x] F3 1 h / 4 h budgets (2026-11-07)
- [x] F4 noise off in the core evaluation (2026-11-07)
- [ ] F5 learning-rate search, shared or per model (deferred; revisit before the sweep)
- [x] F6 GPU-hours (2026-11-07; `tables/gpu_hours.csv`, decisions D22)
- [ ] F7 downsampling (deferred; only if the backup is triggered)
- [x] F8 dataset size (dropped)
- [x] F9 docs
- [ ] F10 report wording
- [ ] F11 released checkpoints vs published VRMSE (pass criterion pending the owner; report wording)
