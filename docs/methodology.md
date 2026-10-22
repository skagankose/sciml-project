# Methodology

This is the draft methods section for the report. Values marked *(pending)* are filled in from the pilot and
the dataset audit.

## Task

The data is The Well `rayleigh_benard`: 2-D Boussinesq convection, horizontally periodic and bounded by walls
vertically, on a 512 (x, uniform) × 128 (y, Chebyshev) grid. There are 200 snapshots per trajectory at
Δt = 0.25 and 35 (Ra, Pr) regimes, with Ra ∈ {1e6 … 1e10} and Pr ∈ {0.1 … 10}. Each model sees the previous 4
frames of buoyancy, pressure, u_x and u_y and predicts the next frame. Longer horizons come from autoregressive
rollout of that one-step model.

## Data

* The official train/valid/test splits are unchanged. The 10%, 30% and 100% training subsets are nested,
  stratified by (Ra, Pr), and fixed in `tasks/subset_*.json`. Every model and seed uses the same subsets.
* All 196 windows of each selected trajectory are used.
* Z-score normalization uses the dataset's shipped training statistics, frozen for every run.

## Models

All four models are trained from random initialisation and none uses pretrained weights.

| Family | Model | Parameters |
|---|---|---|
| convolutional | CNextU-Net (The Well) | 18.6M |
| spectral | TFNO, Tucker-factorised (The Well → neuraloperator 0.3.0) | 19.3M |
| attention | AViT, axial (The Well) | 14.7M |
| recurrent | ConvLSTM: OpenSTL cell, 4-frame unroll, states reset per prediction | 13.3M effective |

Exact configurations are in `docs/upstream_audit.md`.

## Training

* The objective, optimizer, normalization, batch size and schedule are the same for every model; see
  `docs/decisions.md` D5–D8.
* Each (model, fraction, seed) run trains for 6 h of measured training-loop time on one A100 80 GB. Checkpoints
  are taken at the first step at/after 2 h and at/after 6 h of that same run.
* The grid is 4 models × 3 fractions × 3 seeds = 36 runs, which gives 72 checkpoints.
* The seed controls initialisation and data order.
* Update counts differ between models at equal time, and they are reported as part of the efficiency
  comparison.

## Evaluation

The same frozen cohort is used for every checkpoint: all test trajectories, with rollout starts frozen on
validation *(pending)*.

* **One-step VRMSE.** The pinned upstream implementation, in physical units, per field and field-averaged,
  averaged over every test window with equal weight per window.
* **Rollout VRMSE.** Reported at steps 1, 3, 6, 12, 20 and 30.
* **Failure horizon.** The first step where field-mean VRMSE exceeds the threshold (1.0 and 2.0, with 5.0 as a
  sensitivity check). Rollouts that never cross are right-censored, and the results are reported as
  Kaplan–Meier survival curves and median horizons.
* **Robustness.** Gaussian noise σ ∈ {0.01, 0.05, 0.1, 0.2} (normalised units) is added to the 4 observed frames
  only. The draws are identical across models.
* **Nusselt number.** Volume form ⟨u_y b − κ∂_y b⟩/(κΔb/L_y) on predicted and reference frames, using Chebyshev
  derivatives and quadrature. The reference is checked against the wall-gradient Nusselt number before use.
* **Horizontal kinetic-energy spectrum.** A 1-D FFT in x, averaged over y with Chebyshev quadrature
  (Parseval-checked). We report the mean |log10 E_pred − log10 E_true| over non-empty bins with k > 0.
* **Efficiency.** Training steps and samples, train time, peak CUDA memory, batch-1 inference latency per
  frame and per 30-step rollout, and Slurm GPU-hours.
* **Persistence.** x_{t+1} = x_t is reported as a diagnostic reference.

## Reporting

* For every model × fraction × budget we report the mean ± standard deviation over 3 seeds, together with the
  individual seeds.
* Missing or diverged runs are listed rather than dropped silently.
* Every number traces back to a `run_id` and to that run's `provenance.json`: commit, environment, GPU, Slurm
  job, manifest hashes and seed.
