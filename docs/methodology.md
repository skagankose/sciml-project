# Methodology

This is the draft methods section for the report. It follows the final project abstract (2026-10-30). Values
marked *(pending)* are filled in from the pilot and the dataset audit. Where the code still runs the earlier
plan, `docs/abstract_alignment.md` lists the change.

## Task

The data is The Well `rayleigh_benard`: 2-D Boussinesq convection, horizontally periodic and bounded by walls
vertically, on a 512 (x, uniform) × 128 (y, Chebyshev) grid. There are 200 snapshots per trajectory at
Δt = 0.25 and 35 (Ra, Pr) regimes, with Ra ∈ {1e6 … 1e10} and Pr ∈ {0.1 … 10}. Each model sees the previous 4
frames of buoyancy, pressure, u_x and u_y and predicts the next frame. Longer horizons come from autoregressive
rollout of that one-step model.

The values lie on Chebyshev-Gauss nodes in y, but the files store a uniform placeholder for the y coordinates.
We identified the nodes from the data: with Gauss nodes, the boundary conditions hold to ~1e-9 (D18). The
diagnostics that need coordinates (spectra and Nusselt) use those nodes. The models and VRMSE see the
128 rows as an image and use no coordinates.

## Data

* The official train/valid/test splits are unchanged. The 10% and 100% training subsets are nested, stratified
  by (Ra, Pr), and fixed in `tasks/subset_*.json`. Every model and seed uses the same subsets. The 30% subset
  is generated from the same permutation, so it nests between them if the extension runs.
* All 196 windows of each selected trajectory are used.
* Z-score normalization uses the dataset's shipped training statistics, frozen for every run.

## Models

All models are trained from random initialisation; none is trained from pretrained weights.

| Family | Model | Parameters | Role |
|---|---|---|---|
| convolutional | CNextU-Net (The Well) | 18.6M | core |
| spectral | TFNO, Tucker-factorised (The Well → neuraloperator 0.3.0) | 19.3M | core |
| attention | AViT, axial (The Well), widened 384 → 432 | 18.6M | core |
| recurrent | ConvLSTM: OpenSTL cell, 4-frame unroll, states reset per prediction, widened 128 → 152 | 18.8M trainable | extension |

AViT (and ConvLSTM, if it runs) is widened from its reference configuration so that all families have matched
trainable parameter counts (18.6–19.3M). Exact configurations are in `docs/upstream_audit.md`.

## Pipeline check

Before any training, The Well's released checkpoints `polymathic-ai/UNetConvNext-rayleigh_benard` and
`polymathic-ai/TFNO-rayleigh_benard` are evaluated with this project's data path and metric. Their
configurations match ours. The published one-step VRMSE values are CNextU-Net 0.4807 (valid) / 0.6699 (test)
and TFNO 0.8568 / 0.6566. The check passes when our VRMSE (physical units) is at most the published value for
every model and split (owner, 2026-11-08). Scored as the paper did (normalised inputs, VRMSE in normalised
units), the released checkpoints reproduce the published test values to within 1% (0.6688 and 0.6634). The
validation values, which came from one random 1,280-window draw per epoch, are not reproduced. The
checkpoints are pinned to a revision and file hash, scored with the same per-window code as our own runs, and
only evaluated, never trained from (AGENTS.md rule 3; decisions D20).

## Training

* The objective, optimizer, normalization, batch size and schedule are the same for every model; see
  `docs/decisions.md` D5–D8.
* An equal-budget learning-rate search on validation data precedes training: each model tries 1e-4, 5e-4 and
  1e-3 for 20 min of GPU time on the 10% subset and keeps the rate with the lowest validation VRMSE (D7).
* Each (model, fraction, seed) run trains for 4 h of GPU training time on one A100 80 GB: time spent waiting
  for data from storage is excluded and reported (D24). Checkpoints are taken at the first step at/after 1 h
  and at/after 4 h of that same run.
* The core grid is 3 models × 2 fractions × 2 seeds = 12 runs, which gives 24 checkpoints.
* Extensions, if time and compute allow, in priority order:
  1. input-noise robustness (evaluation only);
  2. a third seed;
  3. a 30% data fraction;
  4. ConvLSTM.
* The seed controls initialisation and data order.
* Update counts differ between models at equal time, and they are reported as part of the efficiency
  comparison.
* Backup plan: if storage, queue time or pilot timings show the plan is too costly, the data size is reduced
  for all models equally, with fewer trajectories or downsampled fields (D21).

## Evaluation

The same frozen cohort is used for every checkpoint: all test trajectories, with rollout starts frozen on
validation *(pending)*.

* **One-step VRMSE.** The pinned upstream implementation (RMSE divided by the target field's standard
  deviation), in physical units, per field and field-averaged, averaged over every test window with equal
  weight per window.
* **Rollout VRMSE.** Reported at steps 1, 3, 6, 12, 20 and 30.
* **Failure horizon.** The first step where field-mean VRMSE exceeds the threshold (1.0 and 2.0, with 5.0 as a
  sensitivity check). Rollouts that never cross are right-censored, and the results are reported as
  Kaplan–Meier survival curves and median horizons.
* **Horizontal kinetic-energy spectrum.** A 1-D FFT in x, averaged over y with Chebyshev quadrature
  (Parseval-checked). We report the mean |log10 E_pred − log10 E_true| over non-empty bins with k > 0.
* **Efficiency.** Training steps and samples, train time, peak CUDA memory, batch-1 inference latency per
  frame and per 30-step rollout, and GPU-hours: the measured wall time of every training attempt and
  evaluation pass on one GPU, per run and in total (decisions D22).
* **Persistence.** x_{t+1} = x_t is reported as a diagnostic reference.
* **Robustness (extension).** Gaussian noise σ ∈ {0.01, 0.05, 0.1, 0.2} (normalised units) is added to the 4
  observed frames of clean-trained models. The draws are identical across models.
* **Nusselt number (optional).** Reported only if the dataset audit passes. It uses the volume form
  ⟨u_y b − κ∂_y b⟩/(κΔb/L_y) on predicted and reference frames, with Chebyshev derivatives and quadrature, and
  the reference is checked against the wall-gradient Nusselt number.

## Reporting

* For every model × fraction × budget we report the mean over seeds together with every individual seed value.
* Missing or diverged runs are listed rather than dropped silently.
* Every number traces back to a `run_id` and to that run's `provenance.json`: commit, environment, GPU, Slurm
  job, manifest hashes and seed.
