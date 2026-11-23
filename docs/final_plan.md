# Plan for the final submission (report due 2026-12-11)

Written 2026-11-21 after the professors answered the five questions in Section 6 of the mid-project report
(D28), and updated the same day with the owner's answers (D29), including the new uncertainty-quantification
part (WP9, D30). It lists what is done, what is left, the work packages (WP) with cost and order, and what is
still open (section 6).

## 1. The professors' answers and what each changes

| Report question | Answer | What we do |
|---|---|---|
| 1. Numerical-solver baseline | Yes, but bounded: Dedalus on coarser grids for two or three regimes only. Find the grid at which it matches CNextU-Net's accuracy and compare the cost | WP6: 3 regimes instead of D25's 35, more coarse grids, matching rule fixed before any solver result |
| 2. Physics checks | The Nusselt number is the right quantity. Report the time-averaged Nu over the rollout against the true value, per Rayleigh number | WP5: new per-Ra table and figure from outputs we already have. No further diagnostics |
| 3. Full-length TFNO | No: 50–90 GPU-hours for a reference outside the core question | Not run. The bug stays a note; the released CNextU-Net is the full-data reference |
| 4. Onset cut | A cut per trajectory, from the data (e.g. until the kinetic energy reaches a fixed fraction of its developed-state mean). Report the fixed cut as a check | WP4: onset frame for each trajectory from its true kinetic energy, then re-aggregation. No model re-evaluation, since the per-window results already carry trajectory and start frame |
| 5. Metric | Physical-field VRMSE is the main metric; paper protocol only where we compare with the paper. Table 9 is a good justification | Already our practice; the results tables keep paper-protocol columns only where the paper is compared |

## 2. The owner's answers (2026-11-21)

| Item | Decision |
|---|---|
| 100% runs | On the paid `gpu` partition with the class account, as for the 10% runs. The owner submits |
| Copy of the training data | The copy would go to the GPU node's own disk for the duration of the job, never to the Mac (rule 2). Decided once the node's disk size is known (WP1) |
| Onset definition | Check on validation as proposed (WP4) |
| Solver design | As proposed (WP6) |
| 24 h paper replication | Dropped (`tasks/replication.jsonl` is not run) |
| Correlation (D27) | On wherever it costs nothing: the 100% evaluation and the noise/UQ evaluation. It changes no existing number, and `aggregate` ignores the columns. Probably not reported |
| Noise robustness | Included, at evaluation time only, as planned (WP8) |
| Report format | Later. For now, save every result (accuracy, cost, performance) as before; the report follows afterwards, like the mid-project report but more comprehensive (WP7) |
| Uncertainty quantification | Added to the study, following Lecture 17 (WP9) |

## 3. Where we stand

**Push 1 is ready (2026-11-21).** WP0, WP1, WP4's command and WP6's code are committed and tested; the commands to
submit are in `cluster/README.md` section 10. The local copy is on: each A100 node has 1.83 TB of local disk
(`sinfo`, owner), and the owner approved the GPU cost of WP8 and WP9.

* **Done:** pipeline check (D2); recipe and budgets fixed (D6, D7, D8, D24); six 10% runs trained and evaluated
  at 1 h and 4 h; the dedicated 1 h runs at 10% trained and test-evaluated (not reported); the released
  CNextU-Net through the study evaluation; persistence; mid-project report.
* **Left in the core study:** the six 100% runs (rows 2, 3, 6, 7, 10, 11 of `tasks/train.jsonl`), their
  evaluation, and the 10% against 100% comparison at 1 h and 4 h. This is the abstract's main result
  (data-scaling and accuracy–cost curves) and the critical path.
* **New:** per-trajectory onset (WP4), Nusselt per Rayleigh number (WP5), bounded solver baseline (WP6), noise
  robustness (WP8), uncertainty quantification (WP9).
* **Dropped:** full-length TFNO, the 24 h replication, third seed, 30% data, ConvLSTM.
* **Not committed yet:** the D27 opt-in correlation code and tests, the `eval_a100.sbatch` FORCE/CORRELATION
  switches, the last report edits, and this plan (WP0).

## 4. Work packages

### WP0. Housekeeping before the next push (Mac)

* Run `pytest -q` and the smoke run (rule 7), then commit the pending changes, including
  `tests/test_correlation.py`.
* Keep the HPC3 tree clean. `reports/midproject.log` and `reports/project_check_in.pdf` are untracked, and
  `cluster/push.sh` copies them. HPC3's `git status --porcelain` would then list them, and `train-task`
  refuses to start ("production runs need a clean, committed git tree"). Fix: ignore the LaTeX by-products
  (`reports/*.log`, `*.aux`, `*.out`, `*.toc`), and commit or ignore the check-in PDF.

### WP1. Make the 100% runs survive storage trouble, and make them cheaper (Mac; one check on HPC3)

Why: the full training split (~294 GB) no longer fits in a node's memory cache, so TFNO and AViT will wait for
`/pub` much more than at 10% (projected AViT 14–17 h held per 4 h run). Two 10% AViT runs and two search runs
died on `/pub` `Errno 70`, and on the paid partition a crash costs a new queue wait of up to 12–15 h.

1. **Read retries.** `PersistentWellDataset` retries an HDF5 open or read on `OSError` a few times with
   back-off, then fails as now. A fault-injection test checks that the samples are unchanged.
2. **Resume inside the allocation.** If `train-task` exits without `COMPLETE.json` and enough time remains,
   `train_a100.sbatch` resumes in the same job (at most twice) instead of waiting for a resubmission.
3. **Copy of the training split to the GPU node's local disk** (`$TMPDIR`, deleted when the job ends; nothing
   goes to the Mac).
   * The job copies the files at its start, checks their sizes against the download's completion record, and
     reads from the copy.
   * Run ids and samples do not change: `paths` and `runtime` are outside the config hash, and the files are
     identical.
   * The copy takes at least 10 min at `/pub`'s ~0.5 GB/s.
   * Expected effect: TFNO and AViT become compute-bound, at ~4.5–5.5 h held per run instead of 7.5–9 h and
     14–17 h. That saves roughly 900–1,200 SU and about a day of AViT wall time.
   * **Decided (2026-11-21):** `sinfo` shows 1.83 TB of local disk on each A100 node (2 GPUs per node), so two
     jobs' copies fit. `STAGE_DATA=1`; if a copy fails, the job reads `/pub` as before.

### WP2. The six 100% runs (paid partition, class account; the owner submits)

* Rows 2, 3, 6, 7, 10 and 11, with the recipe unchanged (hash-checked), all submitted at once as for 10%.
* Cost: 54–65 allocated GPU-hours and ~2,200–2,600 SU reading from `/pub`, or ~32–38 GPU-hours and
  ~1,300–1,500 SU with the local copy (estimate).
* Watch the first logs: data-wait share per model (the open D23 check for 100%); resubmit failed tasks.
* Latest start that still leaves time: about 2026-11-28.

### WP3. Evaluate the 100% runs and aggregate (paid, ~4.5 GPU-hours, ~180 SU)

* `eval_a100.sbatch` with `CORRELATION=1` for indices 2, 3, 6, 7, 10, 11: 1 h and 4 h checkpoints, same
  cohort and code (hash and cohort checked).
* Then `rbbench aggregate`, `scripts/findings.py` and `scripts/report_figures.py` with both fractions.
* New results:
  * 10% against 100% at 1 h and 4 h per model: one-step (all, onset, developed), rollouts, failure fractions,
    field-wise error, spectra and Nusselt.
  * Accuracy against cost (Fig. 6) at both fractions.
  * The choice table (Table 8) updated.
  * Our 100% CNextU-Net against the released one.

### WP4. Onset per trajectory (CPU job, done on `free` 2026-11-21, and Mac)

Definition, fixed on validation before it touches any test result:
* KE(t): domain mean of ½(u² + w²) of the true fields, with Chebyshev weights in y
  (`physics.mean_kinetic_energy`).
* Developed-state mean: the mean of KE over frames 150–199 of the same trajectory.
* Onset frame t_on: the first frame with KE(t) ≥ 0.1 × the developed-state mean (owner, 2026-11-21; proposed
  0.5, changed after the validation check, `docs/report_notes.md` section 9).
* A one-step window counts as onset if its first input frame is before t_on (the same convention as the
  frame-50 cut). A rollout starts in developed flow if its start frame is at or after t_on.

Steps:
1. A new command, `rbbench onset-frames`, writes one small table per split: trajectory, Ra, Pr, the
   developed-state mean, and the crossing frames for fractions 0.1, 0.25, 0.5 and 0.75. It reads velocity only
   (~18 GB per split). It ran on `free` (job 59905411) before the owner's partition rule (D31).
2. Validation first. Check that every trajectory crosses, look at t_on by Ra, and compare with where the
   released checkpoints' error by window levels off (D20). Then fix the definition and compute the test table.
   It comes from test truth only (no model output) and is committed like `eval_cohort.json`. Done 2026-11-21:
   `tasks/onset_{valid,test}.json`, recomputed on the Mac from the job's KE series (`onset-frames --from-ke`).
3. `aggregate` splits by the per-trajectory onset by default and keeps frame 50 as the check. This runs on the
   Mac from the existing `one_step_*.parquet` (columns `traj_id`, `start`). Done 2026-11-21: metrics
   `one_step_*_{onset,developed}` (per trajectory) and `..._frame50` (check), plus each split's window share.
4. Results:
   * t_on by Rayleigh number;
   * the error by window aligned at each trajectory's onset;
   * whether any conclusion changes between the two cuts;
   * rollouts relabelled per trajectory where they start before t_on.

### WP5. Heat transport per Rayleigh number (Mac, existing outputs)

* `physics_*.parquet` holds the predicted and true Nu at every rollout step. For each rollout from developed
  flow, average both over its 30 steps.
* Then per Ra, over its 70 rollouts (7 Pr × 5 trajectories × 2 starts), report the true and predicted mean Nu
  (median and spread) and the relative error.
* Present it as a table and a figure: predicted against true time-averaged Nu per Ra, for our models at both
  fractions and budgets, the released CNextU-Net and persistence.
* 30 steps are 7.5 time units, so the true average itself fluctuates; the spread is reported.
* Rerun after WP3 to add the 100% runs.

### WP6. Bounded numerical-solver baseline (CPU work; first rollouts on `free`, the rest on the class partition, D31)

* **Regimes:** Ra = 1e6, 1e8 and 1e10 at Pr = 1.
* **Rollouts:** all 5 test trajectories of each regime, starts 80 and 140 (developed flow). That gives 30
  rollouts per grid, a subset of the models' cohort.
* **Grids:** 512 × 128 (the data's grid: reproducibility limit), 384 × 96, 256 × 64, 192 × 48, 128 × 32 and
  64 × 16.
* **Matching rule (fixed now):** per regime, the coarsest grid whose mean physical-field VRMSE on the same 30
  rollouts is at or below CNextU-Net's at steps 1, 10 and 30. CNextU-Net here is our 100% / 4 h model, with
  our 10% / 4 h and the released one also shown. A rollout that diverges counts as a failure.
* **Cost at the matching grid:** core-seconds per rollout against CNextU-Net's 249 ms per 30-step rollout at
  batch 1 on an A100 (6.74 ms per sample at batch 64). Add the training GPU-hours, amortised over the number of
  rollouts, and express both in SU equivalents (1 SU per core-hour, 32 per GPU-hour).
* **Steps:**
  1. Code: a regime and start filter and the extra grids in `scripts/solver_rollouts.py` and
     `rbbench.solver_grid`, with tests.
  2. Build the `rb-dedalus` environment and pass `--selftest`.
  3. Run the `solver_cpu.sbatch` array, then `solver_score_cpu.sbatch`.
  4. Figure: VRMSE against core-seconds per grid, with the models' points.
* **Risk:** 512 × 128 at Ra = 1e10 may take many hours per rollout on one core (capped at 6 h; runs resume
  after preemption). If it does not finish, the coarse grids still answer the question.

### WP7. Results package (Mac; the report is written afterwards)

**Status (2026-11-23):** done. Final figures in `reports/figures/final/` (`scripts/final_figures.py`), the index, cost
tables and the updated "which model when" table in `docs/report_notes.md` section 15. A last `sacct` pull closes the
accounting (the 100% snapshot job ran after the current one).

Every result is saved as for the mid-project report, so that the final report can be written from them:
* `outputs/tables/` and `outputs/findings/`;
* figures via `scripts/report_figures.py`;
* the numbers and their reading in `docs/report_notes.md` (rule 8).

The contents: accuracy (one-step, rollouts, field-wise, by phase and Ra), cost (allocated and active GPU-hours,
SU, data waits, memory, inference time), physical fidelity (spectra, Nu per Ra), the solver comparison, noise
robustness and uncertainty. Costs come from a final `sacct`.

### WP8. Input-noise robustness (evaluation only, paid)

* The noise cohort (`make-cohort --experiment noise`), σ = 0, 0.01, 0.05, 0.10 and 0.20 on the four input
  frames, identical draws for every model.
* 4 h checkpoints of all 12 runs, correlation on.
* It shares one evaluation pass with WP9's seed-pair part. One job per model and fraction evaluates both seeds
  on each data read; the existing evaluator already runs several predictors per read.
* The noise levels are also WP9's input-shift test for interval coverage (lecture slide 20).

### WP9. Uncertainty quantification (new; Lecture 17)

**Status (2026-11-21):** code written and tested on the Mac (`rbbench.uq`, `eval-uq`, the variance output and
β-NLL, `tasks/uq.jsonl`, `tasks/train_1h_f100.jsonl`, `tasks/uq_ensembles.json`), committed, not pushed. It goes
to HPC3 with push 2 (`cluster/README.md` section 11). The analysis (coverage, conformal, figures) is written on
the Mac while the jobs run.

**Question for the study:** how far can each family's forecast be trusted, and how does its uncertainty change
with more data (10% to 100%) and along a rollout? Two parts, following the lecture's "UQ in your project"
recipe.

**UQ-A. Deep ensemble with a variance output (the lecture's recipe, one family).**
* **Model:** CNextU-Net. It is the most accurate family, never waits for data (so its cost is predictable), and
  is the architecture of the released full-data reference. Its last layer outputs a mean and log σ² per field
  (8 channels instead of 4; parameter count +<0.01%).
* **Loss:** β-NLL with β = 1, in normalised units. Everything else follows the core recipe (AdamW, peak lr 1e-3,
  batch 64, fp32, warmup and cosine).
* **Runs:** a dedicated 1 h budget (the cosine ends at 1 h), 5 seeds, at 10% and at 100% data: 10 runs.
  Two MSE runs at 100% with the same 1 h schedule serve as the point-forecast baseline; the 10% ones exist
  (`tasks/train_1h.jsonl`, already test-evaluated).
* **Predictive distribution:** the uniform mixture of the 5 Gaussians. The total variance is the mean of the
  predicted variances (aleatoric) plus the spread of the means (epistemic): the law of total variance.
* **Rollouts:** each member feeds back its own mean, and at each step the members' Gaussians are mixed. Their
  spread then includes the divergence of the rollouts. The aleatoric part is each member's one-step variance
  at that step; it is not propagated by sampling (stated as a limitation).
* **What it shows:** whether the epistemic part shrinks from 10% to 100% while the aleatoric part stays (lecture
  slide 9), which is the uncertainty view of the study's data axis. Also where each part is large: onset,
  boundary layers, Ra = 1e10.

**UQ-B. Uncertainty for all three families without new training.**
* **Seed pairs:** the two seeds of each configuration (3 models × 2 fractions, 4 h) form a 2-member ensemble.
  Their disagreement is an epistemic indicator from the initial weights and data order only, since both seeds
  share one subset. We report the spread-skill ratio with the finite-ensemble factor, and rank histograms
  (ranks 0–2). They are probably under-dispersed, and are reported as such.
* **Split conformal intervals:** any model, no retraining.
  * Calibrate on the validation split and check coverage on test. Validation has 175 trajectories, disjoint
    from test; both splits are random whole-trajectory draws with 5 per regime, so they are exchangeable.
  * Score s = |y − μ(x)| / σ(x) per field, with three choices of σ(x): a constant (the point models CNextU-Net,
    TFNO, AViT and persistence), the seed spread (pairs), and the ensemble's σ (UQ-A).
  * q̂ at 50% and 90%, separately for every rollout step.
* **What it shows across families:** the width of a 90% interval with guaranteed coverage, for each family and
  persistence at each data fraction. Also whether a spread-based σ(x) makes the intervals narrow where the flow
  is easy.

**Measures** (lecture slide 21), on the test split:
* coverage of the 50% and 90% intervals: marginal, per window, by phase (WP4) and by Ra;
* spread-skill ratio;
* CRPS against the absolute error of the MSE point forecast (normalised units, per field);
* predicted std next to the error, along y (boundary layers), along the rollout, and binned (RMSE in bins of
  predicted std);
* PIT histogram (mixture) and rank histogram (members);
* the aleatoric and epistemic parts.

**Shifts** (lecture slide 20):
1. Longer rollout: q̂ fitted at step 1, applied to steps 2–30.
2. Noisy inputs: coverage at σ = 0.01–0.20 (WP8).
3. Regime: calibrate on Ra ≤ 1e9 only, check Ra = 1e10, then recalibrate with one or two Ra = 1e10 validation
   trajectories.

**Implementation** (Mac, tested on synthetic data before any GPU time):
* The variance head and β-NLL appear only in a new experiment config (`uq.yaml`, manifest `tasks/uq.jsonl`), so
  the core runs' config hashes do not change; a test checks the committed manifests' hashes.
* An ensemble evaluation built on the existing evaluator.
* Scores: CRPS of a Gaussian mixture in closed form, checked against numerical integration, plus coverage,
  PIT and spread-skill.
* Compact per-window histograms of the normalised scores, so that calibration, coverage and every grouping are
  computed on the Mac afterwards. No change to the hashed `eval` section (D15).

**Not used:**
* MC dropout: our models have no dropout layers, and the lecture shows it under-covering.
* Laplace: 262k output values per sample make even a last-layer Laplace heavy.
* Ensembles on all three families: 30 trainings, beyond the time left.

**Cost:**
* UQ-A training: ~15–18 allocated GPU-hours, ~600–700 SU.
* UQ-A evaluation (test and validation): ~4–8 GPU-hours, ~150–350 SU.
* UQ-B shares WP8's test pass and adds a validation pass: ~3–5 GPU-hours, ~120–200 SU.

## 5. Order and timeline

The rule is that we never push while array tasks are queued (2026-11-09 incident). So the work goes to HPC3 in
two pushes:

| Dates | Mac | HPC3 (the owner submits) |
|---|---|---|
| 11-21 to 11-23 | WP0; WP1, WP4 (command) and WP6 code; tests and smoke; **push 1** | `sinfo` disk check |
| 11-23 | | the six 100% runs (paid); onset frames, Dedalus env, self-test and 75 of 90 solver tasks ran on `free` (before D31) |
| 11-23 to 11-26 | WP9 code and tests (variance head, β-NLL, ensemble and conformal evaluation, scores); WP4 and WP5 on the 10% results | 100% runs; solver rollouts |
| ~11-26 | **push 2**, once the 100% training array has left the queue | 100% evaluation (WP3); UQ-A training (12 runs of 1 h) |
| 11-27 to 11-30 | | UQ-A evaluation; noise and seed-pair evaluation (6 jobs, WP8 and WP9); validation calibration pass; solver scoring |
| 12-01 to 12-05 | aggregate everything; results package (WP7) | reruns if needed |
| 12-05 to 12-10 | final report (format decided then) | |

Thanksgiving (11-26, 11-27) falls on push 2. If the owner is away then, push 2 and its submissions move to
11-28; that still fits.

## 6. Compute

| Item | Partition | GPU-hours | SU |
|---|---|---|---|
| Used so far (10% study, dedicated 1 h runs, released-checkpoint evaluation) | mixed | ~81 | ~2,500 (rerun `sacct` to confirm) |
| Six 100% runs (WP2) | paid | 54–65 from `/pub`, ~32–38 with the local copy | 2,200–2,600, or 1,300–1,500 |
| Evaluation of the 100% runs (WP3) | paid | ~4.5 | ~180 |
| Noise and seed-pair evaluation, test (WP8, WP9) | paid | ~10–18 | ~400–700 |
| Validation pass for conformal calibration (WP9) | paid | ~3–5 | ~120–200 |
| Deep ensemble: training (WP9, UQ-A) | paid | ~15–18 | ~600–700 |
| Deep ensemble: evaluation (WP9, UQ-A) | paid | ~4–8 | ~150–350 |
| Onset frames, Dedalus setup, first 75 solver tasks (done 2026-11-21) | free CPU, before D31 | 0 | 0 |
| Rest of the solver, its scoring, flow snapshots (CPU work holding one idle GPU, D31) | paid | ~2–3 | ~50–150 |
| **Still to spend** | | | **~3,750–4,850, or ~2,850–3,750 with the local copy** |

The class account had 31,530 of 33,000 SU on 2026-11-09 and is shared by the class. Check the balance before
push 1's submissions.

## 7. Still open

Nothing blocks push 1. Resolved 2026-11-21: the local disk (1.83 TB per A100 node), and the owner's approval of
WP8's and WP9's GPU cost. The 100% runs are approved (paid partition, class account).

## 8. Risks

* **`/pub` stalls and errors:** WP1 (retries, resume in the allocation, local copy).
* **Paid queue waits of up to 12–15 h per job:** submit each batch at once. Everything runs on the class partition
  (D31), so CPU jobs queue with the GPU jobs; they are packed into few, short jobs.
* **The shared class allocation:** check the balance first; the local copy saves ~1,000 SU.
* **The deadline:** the 100% runs must start by about 11-28. If they cannot finish, the backup plan applies
  (50% of the data, equally for all models). WP9 comes after the core study: if time runs short, UQ-B
  (no training) is kept and UQ-A is cut to 10% data only.
* **β-NLL instability** (variance collapse, ignored means; Seitzer et al. 2022): β = 1 is the lecture's remedy.
  log σ² is clamped, and the synthetic smoke run and the first training log are checked before all 12 run.
* **Dedalus on HPC3 is untested:** the self-test is the first step of WP6; it does not block the core study.
