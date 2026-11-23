# Findings log for the final report

Every result, cost, run time and incident, recorded as it happens so that the final report (due 2026-12-11)
can be written from it (owner, 2026-11-10). Numbers come from the job outputs named next to them.

* Tables and figures are regenerated from the pulled outputs with
  `HPC3_USER=skose bash cluster/pull.sh && .venv/bin/python scripts/findings.py`, which writes
  `outputs/findings/` (`runs.csv`, `attempts.csv`, training curves, time breakdown, released rollouts).
* The tables below are copies of those outputs on the date given; rerun the script for current values.
* Decisions and their reasons stay in `docs/decisions.md` (D-numbers); this file records what was observed.

## 1. Pipeline check: The Well's released checkpoints (milestone 1, gate D2)

### Summary for the report

We validated our data and evaluation pipeline by running The Well's own trained models through it, with no
training. The released CNextU-Net and TFNO checkpoints reproduce the paper's test results:
* **One-step test VRMSE** (Table 2): CNextU-Net 0.6688 against the published 0.6699 (−0.2%), TFNO 0.6634
  against 0.6566 (+1.0%).
* **Rollouts from frame 0** (Table 3): above 10 in both windows (steps 6–12 and 13–30), as published:
  CNextU-Net 36 / 35, TFNO 142 / 169.
* **Not reproduced: the validation values** (Table 5). The paper scored validation on a random draw of 20
  batches per epoch, which cannot be recomputed. On every validation window the released checkpoints score
  0.70 (CNextU-Net, published 0.48) and 0.68 (TFNO, published 0.86).

Scope: this validates data loading, normalisation, windowing, the model forward passes and the metric. It
does not validate our training; that is the separate replication (24 h runs, postponed by the owner).

### How the check was done

1. **Checkpoints.** The released `rayleigh_benard` checkpoints of CNextU-Net
   (`polymathic-ai/UNetConvNext-rayleigh_benard`, revision c5fa0f7) and TFNO (`polymathic-ai/TFNO-rayleigh_benard`,
   revision 7d35135) from Hugging Face, pinned by revision and sha256. Loaded into our model classes and
   checked bitwise-equal to upstream's.
2. **Data.** The same dataset as the paper: server files dated 2024-12-05, the official splits, and the
   v1.0.1 normalisation statistics (`stats.yaml` identical to ours). Read through our pipeline
   (`PersistentWellDataset`), whose samples are bitwise-identical to upstream `WellDataset`'s.
3. **Upstream comparison** (`scripts/compare_upstream.py`, real data, 2026-11-08): our rollouts match
   upstream's exactly (max abs difference 0.0 for every model), and our VRMSE equals upstream's exactly and
   NumPy's to 1e-7.
4. **Scoring.** Every window of the valid and test splits (175 trajectories, 34,300 one-step windows each),
   with the same code our trained models are scored with. Two protocols: ours (physical units) and the
   paper's (normalised units). Plus the paper's rollout: 30 steps from frame 0 of every trajectory.
5. **Runs.** Job 59154129 (2026-11-07) found the TFNO mismatch, job 59224204 (2026-11-08) passed, and job
   59407152 (2026-11-09, paid `gpu` partition, A100, commit f42b2e2) is the final run with the rollouts.

What we had to find out on the way (D20):
* The paper's test values are in normalised units. Physical units give lower scores (0.224, 0.300), only
  because VRMSE's eps damps velocity in the first ~50 frames, where the fluid is nearly still. From frame 50
  on, both units agree to four digits.
* The released TFNO needs its paper-era forward pass. The Well's code then skipped TFNO's Fourier blocks (an
  upstream bug, fixed in e430160 on 2025-11-17), so the released TFNO is in effect a per-pixel MLP. With the
  current forward it scores ~10. Upstream issue #49 notes outdated paper numbers and Hub checkpoints.
* The owner's pass criterion (2026-11-08) was "ours ≤ published" in our units; it passed for both models
  and splits.

### One-step VRMSE (Table 2 of the paper)

| Model | Split | Ours (physical units) | Paper's protocol (normalised units) | Published | Difference |
|---|---|---|---|---|---|
| CNextU-Net | test | 0.2238 | **0.6688** | 0.6699 | −0.2% |
| TFNO | test | 0.3003 | **0.6634** | 0.6566 | +1.0% |
| CNextU-Net | valid | 0.2344 | 0.7006 | 0.4807 | +45.7% |
| TFNO | valid | 0.3047 | 0.6754 | 0.8568 | −21.2% |

* **The published test values are reproduced to within 1%.**
* The published valid values are not reproduced. The paper's validation scored one random draw of 20 batches
  per epoch, so it cannot be repeated exactly (D20).
* The units matter on the first ~50 frames only, where the fluid is nearly still. In physical units VRMSE's
  eps damps velocity there, so our scores are lower. From frame 50 on, both units agree to four digits.
* The released TFNO only works with its paper-era forward pass, which skipped the FNO blocks (upstream bug,
  fixed in e430160 on 2025-11-17). It is in effect a per-pixel MLP. With today's forward it scores ~10 (D20).

### Rollouts from frame 0 (Table 3 of the paper)

30 autoregressive steps from the first 4 frames of every trajectory, in normalised units as the paper-era
trainer did. The values are window means over steps and trajectories. The paper reports ">10" for every RB
model in both windows.

| Split | Window | CNextU-Net, normalised | CNextU-Net, physical | TFNO, normalised | TFNO, physical |
|---|---|---|---|---|---|
| test | steps 6–12 | **36.1** | 9.28 | **142** | 37.1 |
| test | steps 13–30 | **34.7** | 16.5 | **169** | 73.9 |
| valid | steps 6–12 | 39.0 | 9.98 | 155 | 40.0 |
| valid | steps 13–30 | 39.0 | 18.5 | 196 | 84.8 |

* **The paper's ">10" is reproduced on test, in both windows, for both models.**
* Velocity produces all of it. On test, buoyancy and pressure stay at 0.01–0.6 over all 30 steps; velocity_x
  and velocity_y reach tens to hundreds. From frame 0 the velocity field has almost no spatial variance, so
  any velocity error is large relative to it.
* It depends strongly on the Rayleigh number. Test, steps 13–30, normalised units, per Rayleigh number
  (35 trajectories each):

  | Ra | CNextU-Net mean | CNextU-Net median | share > 10 | TFNO mean | TFNO median | share > 10 |
  |---|---|---|---|---|---|---|
  | 1e6 | 80.4 | 69.3 | 100% | 371 | 307 | 100% |
  | 1e7 | 45.4 | 41.8 | 100% | 238 | 249 | 100% |
  | 1e8 | 27.5 | 30.5 | 71% | 132 | 165 | 97% |
  | 1e9 | 13.8 | 15.2 | 51% | 74.7 | 70.6 | 74% |
  | 1e10 | 6.7 | 1.7 | 34% | 32.5 | 7.1 | 49% |
  | all | 34.7 | 23.9 | 71% | 169 | 126 | 84% |

  At low Ra convection starts later, so frames 4–34 are still nearly still and VRMSE stays ill-conditioned.
  At Ra = 1e10 most CNextU-Net rollouts are below 10 by steps 13–30.
* So ">10" says more about where the rollouts start than about the models. The models still differ
  (CNextU-Net about 4× lower than TFNO), but mostly in how much spurious velocity they add to a nearly
  still fluid at low Ra. Our study's rollouts start at frames 20, 80 and 140 instead (D15).
* Figures: `outputs/findings/released_paper_rollout_test.{pdf,png}` (and `_valid`): field mean in both
  units with the middle 50% of trajectories, and the four fields separately.

### Cost and run time of the pipeline check

* About 80 min per split (valid and test) on an A100, both models, one-step plus rollouts. Job 59397615
  (free-gpu, 2 h limit) timed out in the test split. `released_a100.sbatch` now allows 4 h.
* Job 59407152 ran on the paid partition with 4 cores (36 SU per hour): 2 h 21 min, **84 SU billed**
  (`sacct`), after 5.4 h in the queue. The earlier free-gpu passes (59154129, 59224204, 59397615) used
  3.2 GPU-hours at no cost.

## 2. Budgets fixed from pilot timings (milestone 2)

### Summary for the report

Done before the 10% runs, and the 10% runs confirmed the predictions.
* **Budgets:** 1 h and 4 h of training on one A100 (from the abstract), counted as GPU training time only
  (owner, 2026-11-08; D24). The data-loading benchmark showed that shared storage (`/pub`) delivers ~98
  samples/s into one node, while AViT can use 170–330 per second of GPU time. A wall-clock budget would have
  given AViT about half its steps, so the comparison would partly measure our storage. Waiting is logged
  and reported separately.
* **Batch 64 and fp32 for all models** (The Well's batch size): the profile showed it fits in A100 80 GB
  memory for every model. TFNO cannot use bf16 (no bf16 FFT in torch; D8).
* **Learning rate 1e-3 for all models**, from an equal-budget search on validation data.
* **Job settings:** 8 cores (7 data-loading workers) per job; job time limit 10 h, then 14 h on the paid
  partition, because real time exceeds the GPU budget.

### Pilots

* **Profile** (2026-11-08, batch 8–64, fp32 and bf16): batch 64 fits every model in fp32 (peak allocated
  memory CNextU-Net 58.7 GB, TFNO 44.9, AViT 7.4, ConvLSTM 23.6).
* **Data-loading benchmark** (`rbbench bench-data`, CPU node, 2026-11-08): our persistent loader is 1.2–1.4×
  faster than upstream's. `/pub` delivers about 0.5 GB/s, 97.7 samples/s with 8 workers, into one node; 15
  workers were slower.
* **The first replication attempt** (2026-11-08) was I/O-bound: 5.75–47 s per step instead of 0.55–1.42,
  because upstream `WellDataset` reopens the HDF5 file for every sample. Fixed with `PersistentWellDataset`
  (D23).
* **Learning-rate search** (below): its 9 runs gave the step times used to plan the budgets.

### Pilot predictions against the 10% runs

| batch 64, fp32, A100 | CNextU-Net | TFNO | AViT |
|---|---|---|---|
| GPU seconds per step: pilot / 10% runs | 1.19 / 1.19 | 0.55 / 0.55 | 0.19 / 0.19 |
| Steps in 4 h: pilot / 10% runs | 12,080 / 12,122–12,194 | 26,170 / 26,339–26,352 | 74,320 / in progress |
| Epochs in 4 h on 10% data: pilot / 10% runs | 28 / 28 | 61 / 61 | 173 / ~172 (projected) |
| Real hours for 4 h: pilot estimate / 10% runs | ~6.0 / 5.2–6.0 | ~8.0 / 6.9–7.1 | ~9.7 / ~13.5 (projected) |

* GPU step times and steps per budget matched the pilots to within 1%. Every run stopped within 1 s of its
  4 h budget (14,400.0–14,400.7 s).
* Real time matched for CNextU-Net and TFNO. AViT took longer than predicted, because the storage stalls
  were longer and more frequent than during the pilots.
* The budgets were unaffected, since they count GPU time; the cost was (SU are charged on wall time).

### Storage stalls (`/pub`)

* Between stalls the data loader keeps up with every model: the median wait per logged interval is 0.
* Nearly all waiting comes from stalls of 5–22 min. AViT seed 0: 13 stalls over 5 min, 3.3 of its 3.6 h
  of waiting. TFNO seed 0: 13 stalls, 2.0 of 2.5 h.
* The stalls hit every running job at the same moments, on different nodes (hpc3-gpu-l54-03, -05 and -06).
  So they are a property of the shared filesystem, not of one node. Examples (Pacific time, 2026-11-09):
  17:34–17:41 (four runs), 22:42–23:04 (four runs), then every 20–40 min until AViT's crash at 02:44.
* Stalls grew longer (15–22 min instead of 5–8) after both AViT runs started at 21:14. Our logs cannot tell
  whether our own reads contributed.
* A copy of the 10% subset on each node's local disk (140 trajectories × 200 frames × 512 × 128 × 4 fields in
  float32, ~29 GB) would remove this dependency. It is not implemented yet.
* Local disk on the GPU nodes (`sinfo -N -p gpu -o "%N %G %d %m"`, owner, 2026-11-21): the four A100 nodes
  (hpc3-gpu-l54-03 to -06, 2 A100s each) have 1.83 TB of local disk and 256 GB of memory; the A30 nodes 1.83 TB,
  the V100 nodes 228 GB. The full training split (~294 GB) fits twice per A100 node, so the 100% runs copy it to
  the job's `$TMPDIR` first (`STAGE_DATA=1`, `rbbench stage-data`; docs/final_plan.md WP1).
* First 100% runs (array 59905410, paid, commit e69e7e9, 2026-11-21):
  * The copy (35 files, 293.6 GB, checked against the download record) took 3.4–9.3 min, at 0.53–1.43 GB/s, on
    hpc3-gpu-l54-03 and -05.
  * CNextU-Net 100% seed 0 (task 2) completed: 12,142 steps in 4 h of GPU training, waiting for data for 9 s in
    all (the 10% runs waited 0.8–1.1 h). The training process took 4.37 h of wall time, plus the 4.4 min copy.
    Short-validation VRMSE: 0.51 at 1 h, 0.168 at 4 h (10%: 1.04 / 0.167 for seed 0).
  * Incident: the other five tasks (3, 6, 7, 10, 11) started after the Dedalus setup job had written
    `env/hpc3-dedalus-conda.yml` into the project folder (16:43). That untracked file made the tree dirty, so
    `train-task` refused to start; the wrapper retried twice and gave up. About 15 min each (copy plus three
    starts), ~50 SU in all, no training lost. Fix: the snapshot is committed. Lesson: no setup job that writes
    into the repository while training tasks can still start.
* Resubmission (array 59907862, rows 3, 6, 7, 10, 11, commit 9e56e06): row 3 (CNextU-Net 100% s1) started on
  hpc3-gpu-l54-03 at 18:10 PST. Its copy took 7.8 min (0.63 GB/s). At 19:20 it had trained 3,000 steps in 3,566 s of GPU
  time (1.19 s per step, as at 10%) and waited for data 4.3 s; wall 3,745 s. Rows 6, 7, 10 and 11 had not started
  (no log) at 19:20.
* Outcome (2026-11-22): rows 3, 6, 7 and 11 completed; row 10 failed at launch in a `/pub` outage. See section 10.

### Learning-rate search (2026-11-08 to 11-09)

Equal-budget search, one rate per model (D7). 20 min of GPU training each, 10% data, seed 0, batch 64, fp32,
on free-gpu. Arrays 59374848, 59390860 and 59393014. Details are in `outputs/lr_search/lr_search.md`.

| Validation VRMSE after 20 min | 1e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| CNextU-Net | 0.841 | 0.469 | **0.408** |
| TFNO | 0.603 | 0.358 | **0.302** |
| AViT | 0.646 | 0.332 | **0.257** |

* Every model chose 1e-3, the top of the grid. The owner kept the three-rate grid (no 2e-3).
* GPU busy share was 24–75% of wall time; the rest was waiting for data from `/pub` and overhead.
* Incidents: tasks 2 and 3 died from a `/pub` error (`Errno 70 Communication error on send`), and tasks 6 and
  7 were preempted. All were resumed and finished.

## 3. Core study: 10% runs (rows 0, 1, 4, 5, 8, 9 of `tasks/train.jsonl`)

10% of the training trajectories (140 trajectories, 4 per regime, 27,440 training windows), batch 64, fp32,
learning rate 1e-3, warmup 500 steps then cosine over the 4 h budget. The budget counts GPU training time
only (D24); checkpoints at 1 h and 4 h. Parameters: CNextU-Net 18.58M, TFNO 19.28M, AViT 18.57M.

### Status and time (2026-11-11: all six complete; `outputs/findings/runs.csv`)

| Run | Status | Attempts | Training h | Data wait h | Other h | Wall h | GPU busy | Steps | Epochs | Val 1 h | Val 4 h | SU est. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNextU-Net s0 | complete | 2 (free-gpu, gpu) | 4.00 | 1.13 | 0.91 | 6.05 | 66% | 12,122 | 28 | 1.043 | **0.167** | 189 |
| CNextU-Net s1 | complete | 2 (free-gpu, gpu) | 4.00 | 0.81 | 0.37 | 5.18 | 77% | 12,194 | 28 | 0.772 | **0.178** | 191 |
| TFNO s0 | complete | 1 (gpu) | 4.00 | 2.54 | 0.32 | 6.86 | 58% | 26,339 | 61 | 0.508 | **0.310** | 274 |
| TFNO s1 | complete | 1 (gpu) | 4.00 | 2.82 | 0.32 | 7.14 | 56% | 26,352 | 61 | 0.524 | **0.325** | 286 |
| AViT s0 | complete | 2 (gpu, gpu) | 4.00 | 6.07 | 0.99 | 11.07 | 36% | 74,978 | 175 | 0.468 | **0.161** | 443 |
| AViT s1 | complete | 2 (gpu, gpu) | 4.00 | 7.32 | 0.89 | 12.20 | 33% | 75,060 | 175 | 0.418 | **0.167** | 488 |

"Val" is the short validation during training (64 fixed validation windows, physical units), not the test
evaluation. "Other" is startup, validation, checkpoint writes and steps replayed after a resume.

* AViT's first attempts (array 59407148) crashed on `/pub` `Errno 70` after 1.65 / 1.68 h of training. The
  second attempts (array 59428460, both on hpc3-gpu-l54-05) resumed from `last_resume.pt` at steps 28,249 /
  29,105, replaying ~2,500 steps, and ran 5.5 / 6.6 h of wall time. Seed 1 started 4 h after seed 0, when a
  second A100 came free.
* The AViT runs span two commits (f42b2e2, then 4c0d3a8). Nothing on the training path changed between them
  (only evaluation, aggregation and provenance code), and the config hash is the same (`f3560de7`).
* After 4 h of GPU time AViT leads on short validation (0.161 / 0.167), just ahead of CNextU-Net (0.167 /
  0.178); TFNO is at 0.310 / 0.325.

**Time per run, by component** (hours; 10% runs measured, range over seeds; `reports/midproject.tex` Table 8):

| Hours per run | CNextU-Net | TFNO | AViT |
|---|---|---|---|
| Queue wait before training | 0.1–0.9 | 0.5–0.7 | 6.7–8.8 |
| GPU training (budget) | 4.0 | 4.0 | 4.0 |
| Waiting for data | 0.8–1.1 | 2.5–2.8 | 6.1–7.3 |
| Start-up (+ replays after preemption/crash) | 0.13–0.65 | 0.14–0.15 | 0.68–0.69 |
| Validation | 0.01–0.02 | 0.02–0.10 | 0.18 |
| Checkpoint writes | 0.23–0.24 | 0.08–0.15 | 0.03–0.13 |
| Allocated time of training | 5.2–6.0 | 6.9–7.1 | 11.1–12.2 |
| Test evaluation job (1 h + 4 h) | 0.87–0.95 | 0.57 | 0.34–0.38 |
| Queue wait before evaluation | 1.0–14.0 | 1.9–2.4 | 3.0–3.3 |
| 100% projected: data wait / allocated / evaluation | 1.0–2.0 / 5.5–6.5 / ~0.9 | 3.0–4.5 / 7.5–9 / ~0.6 | 9.5–12.5 / 14–17 / ~0.4 |

A run that starts once spends ~0.15 h in start-up; SU = 40 × allocated hours.

### Throughput and memory (A100 80 GB, batch 64, fp32)

| Model | Steps per training second | Samples per training second | Peak GPU memory reserved |
|---|---|---|---|
| CNextU-Net | 0.84 | 54 | 64.8 GB |
| TFNO | 1.83 | 117 | 56.9 GB |
| AViT | 5.2 | 333 | 8.8 GB |

* In 4 h of GPU time at 10% data, CNextU-Net sees 28 epochs, TFNO 61 and AViT 175.
* `/pub` delivers at most ~98 samples/s into one node (D23 benchmark). TFNO needs 117/s and AViT 333/s, so
  both wait for data even without stalls. AViT spent 55–60% of its wall time waiting (6.1 / 7.3 h of
  11.1 / 12.2 h); its GPU was busy 33–36% of the time.
* Why AViT needs a seventh of the memory (2026-11-13): parameters are not the cause (~19M each; weights, gradients
  and AdamW's two moments ≈ 0.3 GB). Activations stored for the backward pass dominate, and they scale with the grid
  a layer works on. AViT's patch embedding turns 512 × 128 into 32 × 8 tokens (patch 16), so its 8 blocks see
  256 × 432 numbers per sample. CNextU-Net's outer stages (42 channels, expanded 4× in each ConvNeXt block, plus
  skips) and all 4 of TFNO's Fourier layers (128 channels, plus FFT buffers; lifting to 256) work on all 65,536
  points. A rough count of stored activations, ~30M numbers per sample for AViT against ~200M (TFNO) and ~250M
  (CNextU-Net), gives ~8 / ~50 / ~65 GB at batch 64 in fp32, close to the measured 8.8 / 48.6 / 60.9 GB
  allocated. The same patching explains AViT's speed per step and its patch-sized artifacts in the snapshots.

### Pointwise error and the double penalty (2026-11-13; D27)

* VRMSE rewards smooth predictions (a displaced sharp plume is penalised twice). TFNO is the smoothest model:
  median predicted / true kinetic energy between k_x = 25 and 100 after 30 steps is 0.16 at 1 h and 0.11 at 4 h
  (median over regimes, mean over seeds; CNextU-Net 1.35 / 1.34, AViT 29 / 16). So TFNO's rollout lead at 1 h
  (60% of rollouts below VRMSE 1, against 34%) is partly a reward for smoothness. The mid-project report names this only in the Revised Plan
  (owner, 2026-11-13: report it with the correlation results in the final report).
* Thresholds: 1 = the target's mean; ~1.41 = a realistic field with no correlation (displaced plumes);
  2 and 5 = breakdown. Persistence at step 30 from frames 80 / 140 (0.86 / 0.54) implies r ~ 0.63 / 0.85.
* Correlation and high-correlation time added to the evaluation code as an opt-in (`--correlation`, D27; owner:
  not a default run, to be decided for the final report); not run on HPC3. The report's question on it is
  commented out by the owner.

### Training curves (`outputs/findings/training_curves_train_f010.{pdf,png}`)

* CNextU-Net: training loss falls steadily to ~0.003 at 4 h. Validation VRMSE rises from 0.48–0.55 at 30 min
  to 1.04 / 0.77 at 1 h, then falls to 0.17–0.18 by 3.5 h. The 1 h checkpoint is taken while the learning
  rate is still high, so it is a mid-schedule checkpoint of a 4 h run, not a 1 h-budget run.
* TFNO: training loss plateaus at ~0.013 after ~1.5 h; validation improves slowly to 0.31–0.33.
* AViT: training loss falls from ~0.0062 at 1 h to ~0.0031 at 4 h. Validation falls from 0.47 / 0.42 at
  1 h to 0.161 / 0.167 at 4 h, with bumps at 1.5 h and 2.5 h (seed 1) while the learning rate is still high.
* The two seeds agree closely in training loss for every model. Validation differs more at 1 h than at 4 h.

### Test results, all six 10% runs (2026-11-11; `outputs/tables/`, `rbbench aggregate`)

All 12 checkpoints verified (cohort sha256 0e6dc760…, 34,300 one-step windows, 525 rollouts, physics audit
passed). Evaluation passes: CNextU-Net 50–55 min, TFNO 32 min, AViT 19–21 min (both checkpoints per pass).

One-step VRMSE, mean ± std over seeds (physical fields: all / onset / developed; paper protocol):

| Model | 4 h | 1 h |
|---|---|---|
| CNextU-Net | 0.136±0.006 / 0.387 / **0.050**; 0.349 | 0.728±0.176 / 2.618 / 0.081; 2.345 |
| TFNO | 0.209±0.001 / 0.502 / 0.109; 0.488 | 0.473±0.078 / 1.505 / 0.120; 1.362 |
| AViT | **0.134**±0.001 / **0.308** / 0.075; **0.298** | 0.394±0.036 / 1.299 / 0.085; 1.171 |

References (all / onset / developed; paper protocol): released CNextU-Net 0.224 / 0.766 / 0.038; 0.669;
released TFNO 0.300 / 0.623 / 0.190; 0.663; persistence 0.199 / 0.121 / 0.226; 0.202.

Rollouts (4 h, seed mean): VRMSE after 30 steps from frames 20 / 80 / 140; share below 1 / 2 / 5 after 30 steps;
paper protocol windows 6–12 / 13–30 (normalised):
* CNextU-Net 1.78 / 0.53 / 0.40; 0.89 / 0.94 / 0.98; 15 / 30.
* TFNO 1.79 / 0.59 / 0.45; 0.75 / 0.85 / 0.91; 51 / 77.
* AViT 5.42 / 0.80 / 0.65; 0.63 / 0.80 / 0.87; 75 / 167.
* Persistence 0.68 / 0.86 / 0.54; 0.94 / 1.00 / 1.00; 0.27 / 0.50. Released CNextU-Net 36 / 35, TFNO 142 / 169.

Spectra (4 h, median over regimes of predicted/true by band, steps 1 / 10 / 30):
* CNextU-Net: large 1.00 / 0.98 / 1.02, middle 0.96 / 1.05 / 1.15, small 21 / 26 / 29.
* TFNO: large 1.00 / 0.97 / 0.89, middle 0.40 / 0.24 / 0.12, small 1.0 / 2.8 / 1.2. It is too smooth,
  consistent with its Fourier cutoff. Correction (2026-11-13): the kept modes are n = -8..7 in x (|k_x| <= 12.6), not 16 per direction (k_x ≲ 25); see "Why TFNO trails the other two".
* AViT: large 1.00 / 0.82 / 0.84, middle 1.06 / 3.5 / 5.4, small 24 / 390 / 731. Energy accumulates over the
  rollout. Local peaks at the patch-grid wavenumbers were checked and are not clear, so no cause is claimed.
* Per wavenumber (median over the 35 regimes and both seeds of predicted/true, steps 1, 10, 30; quoted in the
  report because its spectra figure is per k_x, and the band medians above smooth over the extremes):
  * CNextU-Net: 0.80–1.57 below k_x = 100; 190–530 at the highest wavenumbers.
  * TFNO: between k_x ≈ 25 and 100 it falls to 0.33 after one step and 0.09 after 30; 9–50 at the highest.
  * AViT: 0.97–1.34 below k_x = 100 after one step, but 0.55–21 after 30; about 5×10^4 at the highest.
  * The spectrum is computed from velocity (u, w), the snapshots show buoyancy; the two are related but not the
    same measurement. It is a complementary diagnostic, not the paper's binned spectral error.

Speed and memory: steps in 4 h 12.1k / 26.3k / 75.0k; peak training memory (allocated) 60.9 / 48.6 / 8.8 GB;
inference 8.3 / 4.4 / 5.4 ms per step, 249 / 135 / 163 ms per 30-step rollout (A100, batch 1).

Findings:
* After 4 h, CNextU-Net is best in developed convection and in rollouts. AViT matches it over all windows but
  degrades fastest. TFNO is too smooth.
* At 1 h the order flips: AViT and TFNO lead, and CNextU-Net's 1 h checkpoint is far from converged.
* Across seeds, the standard deviation of the all-window VRMSE is at most 0.006 at 4 h and up to 0.18 at 1 h
  (CNextU-Net 0.85 and 0.60).
* Our TFNO, with its Fourier layers, beats the released per-pixel TFNO everywhere.

Error by Rayleigh number and heat transport (2026-11-12; report Section 5 "Additional Analysis", `fig_by_rayleigh`):
* One-step developed VRMSE by Ra = 1e6 → 1e10 (4 h): CNextU-Net 0.016 / 0.020 / 0.036 / 0.066 / 0.111; TFNO 0.042 → 0.191;
  AViT 0.016 → 0.154; persistence 0.098 → 0.351. The released CNextU-Net (100%) is 0.018 / 0.017 / 0.025 / 0.045 /
  0.085. It matches ours at low Ra, and its advantage is confined to high Ra.
* VRMSE after 30 steps (starts 80, 140): CNextU-Net 0.35 → 0.59 and TFNO 0.47 → 0.62, both below persistence
  (0.66 → 0.77) at every Ra. AViT is 0.81 / 0.73 / 0.69 / 0.68 / 0.71: worse than persistence up to Ra 1e8.
* Fraction of rollouts below VRMSE 1 (all starts) rises with Ra: CNextU-Net 0.67 → 0.98, persistence 0.90 → 0.96.
  This fits a later onset at low Ra.
* Nusselt number: Nu = 1 + sqrt(Ra·Pr)·<w b> (κ = (Ra Pr)^-1/2, Ly = Δb = 1), computed per frame by
  `metrics.physics.nusselt`.
  * The instantaneous true Nu one step after frame 80 ranges from −249 to 1,363 across test trajectories, so mean
    relative errors are meaningless. Use medians and the Nu averaged over each 30-step rollout.
  * Developed starts (80, 140), 4 h: step-1 median |rel| CNextU-Net 0.020, AViT 0.037, TFNO 0.072, persistence 0.076.
  * Nu averaged over the 30-step rollout, median |rel| (signed): CNextU-Net 0.485 (−0.015), TFNO 0.196 (+0.064),
    AViT 0.275 (−0.167), persistence 0.267 (+0.023).
  * By Ra, CNextU-Net's averaged-Nu |rel| is 0.19 / 0.33 / 0.62 / 0.87 / 1.00. AViT's signed error is −0.11 / −0.08 /
    −0.13 / −0.18 / −0.43, so it predicts too little heat transport at every Ra.
  * 1 h: CNextU-Net averaged-Nu signed error is +5.45, about 6× the true heat transport.
  * Interpretation (hypothesis): CNextU-Net's excess small-scale energy may create spurious w·b correlation,
    amplified by sqrt(Ra·Pr) ≈ 3e4–3e5 at Ra 1e10.

Field-wise VRMSE (2026-11-12; physical fields, seed mean, from the evaluation parquets; report Table "Field-wise"):
* Developed convection, one step (4 h): CNextU-Net b 0.092, p 0.024, u 0.039, w 0.044. TFNO 0.170 / 0.054 / 0.096 /
  0.116. AViT 0.119 / 0.039 / 0.061 / 0.080. Buoyancy is the hardest field for every model.
* Onset, one step (4 h): velocity dominates (CNextU-Net u 0.863, w 0.639), while b and p are small. On buoyancy
  all three beat persistence even during onset (0.031–0.045 against 0.066); CNextU-Net and AViT also do on
  pressure (persistence 0.014). So the onset deficit against persistence is a velocity effect: near-zero velocity
  variance magnifies relative error. The released CNextU-Net shows the same pattern (onset u 1.47, w 1.56).
* After 30 steps from frames 80 and 140 (4 h): CNextU-Net (0.384 / 0.293 / 0.598 / 0.584) and TFNO stay below
  persistence (0.531 / 0.408 / 0.934 / 0.916) on every field. AViT is worse on u (1.059), about level on p and w,
  and better only on b.
* 1 h: CNextU-Net's all-window error is almost entirely onset velocity (u 6.83). In developed flow it is best or
  tied on b, p and w.

Qualifications from the plausibility review (2026-11-12, in the report):
* Batch-1 latency is overhead-bound. Forward pass per sample in a batch of 64 (training profile, fp32, with
  gradient tracking; `outputs/pilots/feasibility.csv`): AViT 0.89 ms, TFNO 3.15 ms, CNextU-Net 6.74 ms. So AViT
  is the cheapest for batched forecasts (3.5× cheaper than TFNO, 7.6× cheaper than CNextU-Net). TFNO is the fastest
  only at batch 1.
* bf16 vs fp32 at batch 64, GPU time per training step (forward + backward + optimizer):
  * CNextU-Net 1.172 → 0.708 s (1.66×); its memory 58.7 → 34.5 GB allocated.
  * AViT 0.180 → 0.083 s (2.17×).
  * TFNO cannot run in bf16 (D8). Training everything in fp32 therefore favours TFNO.
* The released CNextU-Net (12 h on an H100) probably never annealed. Our estimate: our A100 rate of 54 samples/s,
  with an H100 at most 3× faster, gives ≤ 7.0 M samples in 12 h. At 274,400 windows per epoch that is ≤ ~25 of
  the 200 scheduled epochs. With 5 warmup epochs and a cosine to 0.1·lr over 200, the learning rate at epoch 25
  is still ~98% of peak. So the 1.3× gap to our 10% / 4 h model is not a data-scaling estimate. This is an
  estimate; the model card does not give the epoch count.
* One 10% subset is shared by every seed and model (`src/rbbench/subsets.py`), so the seed spread excludes the
  effect of which trajectories were drawn. Optional: a second subset draw, one run per model, ~25 allocated GPU-h.

With only 1 h of training (the 4 h runs' 1 h checkpoints; report section 4, Tables 4–5):
* One step: AViT is best over all windows (0.394 vs TFNO 0.473, CNextU-Net 0.728) and in onset (1.30). In developed
  convection CNextU-Net (0.081) and AViT (0.085) are close, and TFNO trails (0.120).
* Rollouts: TFNO is most stable.
  * VRMSE after 30 steps from frame 20 / 80 / 140: TFNO 2.67 / 0.90 / 0.78, CNextU-Net 3.60 / 1.09 / 1.04, AViT
    4.67 / 1.11 / 0.98.
  * Fraction staying below 1 / 2 / 5 through step 30: TFNO 0.60 / 0.77 / 0.86, CNextU-Net 0.34 / 0.74 / 0.79, AViT
    0.34 / 0.74 / 0.83.
  * Paper protocol, steps 6–12 / 13–30: TFNO 81 / 116, AViT 129 / 184, CNextU-Net 202 / 148.
* From developed starts (80, 140), mean VRMSE per step stays below persistence's for the first 21 steps (TFNO), 14
  (CNextU-Net) and 12 (AViT). At 4 h, CNextU-Net and TFNO stay below for all 30 steps and AViT through step 25.
  After 30 steps at 1 h no model beats persistence.
* Seed spread is largest for CNextU-Net: all-window 0.85 vs 0.60. One seed has 0% of rollouts below VRMSE 1, the
  other 69% (TFNO 67 / 53%, AViT 41 / 26%).
* With 1 h: AViT for one-step predictions, TFNO for multi-step forecasts.

Accuracy against cost: the project's aim, "useful under limited data and compute" (owner 2026-11-12, report
reframed). Source: `fig_accuracy_cost`, from `scripts/report_figures.py`.
* Returns on training time, developed one-step VRMSE from 1 h to 4 h (seed means):
  * CNextU-Net 0.081 → 0.050 (−38%).
  * AViT 0.085 → 0.075 (−12%).
  * TFNO 0.120 → 0.109 (−9%).
  * At 1 h CNextU-Net and AViT are close. The 1 h points are checkpoints of the 4 h schedule.
* Data efficiency: at 10% data and 4 h on an A100, CNextU-Net's developed error is 1.3× the released full-data model's
  (0.050 vs 0.038; the released model trained 12 h on an H100).
* Cost to train (GPU time held per 4 h run): CNextU-Net 5.2–6.0 h, TFNO 6.9–7.1 h, AViT 11.1–12.2 h. AViT is the
  most GPU-efficient but storage-bound.
* Cost to use (batch 1, fp32, A100, input already on the GPU, 10 warm-ups, 50 timed steps):
  * Per step: 8.3 / 4.4 / 5.4 ms (CNextU-Net / TFNO / AViT).
  * Per 30-step rollout: 249 / 135 / 163 ms.
  * Peak allocated memory: 0.31 / 0.44 / 0.23 GB.
  * The most accurate model is 1.8× slower than the fastest.
* Negative result: averaged over the rollouts from frames 80 and 140, AViT's VRMSE after 30 steps (0.72) is no better
  than persistence's (0.70). CNextU-Net 0.47, TFNO 0.52.
* Context, not a comparison: The Well's dataset card (e63dec8) gives "Approximate time to generate the data: per
  input parameter from ~6,000s to ~50,000s", total ~60 h, on "12 nodes of 64 CPU cores with 8 processes per node".
  It also gives a simulation time-step of 0.25 over 0–50, so a 30-step forecast covers 7.5 time units. The
  parallel layout per trajectory is not stated, so a per-forecast solver cost needs our own solver baseline (D25).

Flow snapshots (`fig_snapshots_buoyancy_*`, `scripts/visualize_rollouts.py`, job 59482062 on the free CPU
partition; seed 0, 4 h, from frame 80):
* At Ra = 1e8 CNextU-Net follows the true plumes for 30 steps.
* TFNO blurs the plumes and builds a spurious warm layer along the bottom wall.
* AViT shows rectangular artifacts at the size of its 16×16-pixel patches by step 12, consistent with its
  middle- and small-scale energy gain.
* At Ra = 1e10 all three miss the finest structures; CNextU-Net keeps the large-scale plume pattern.

### Released CNextU-Net through the study evaluation (2026-11-13; `eval-baseline --released cnext_unet`)

`outputs/baselines/released/cnext_unet/evaluations/`, code 41ec708 (no correlation columns; D27 code not yet
pushed). Complete: 525 rollouts, 34,300 one-step windows, cohort sha256 matches `tasks/eval_cohort.json`, physics
audit passed, eval config identical to persistence's; checkpoint `polymathic-ai/UNetConvNext-rayleigh_benard` at
c5fa0f7a, sha256 4d02a86e…. One evaluation pass: 1,057 s wall (0.29 GPU-h).

* Consistency: one-step all-window VRMSE 0.2238, equal to the pipeline check's; frame-0 rollouts 36.1 / 34.7.
* Study rollouts (physical-field VRMSE, Table 4 of the report): after 30 steps from frames 20 / 80 / 140:
  1.65 / 0.63 / 0.51; fraction of the 525 rollouts staying below VRMSE 1 / 2 / 5 through step 30:
  0.77 / 0.82 / 0.90 (independently recomputed from `rollout_na.parquet`: identical).
* Per field after 30 steps from frames 80 and 140: b 0.505, p 0.366, u 0.713, w 0.697 (mean 0.57).
* Finding: the released model (100% data, 12 h on an H100) is better one step ahead in developed convection
  than our 10% / 4 h CNextU-Net (0.038 vs 0.050) but less stable over rollouts: after 30 steps from frames 80 /
  140 it is at 0.63 / 0.51 against ours 0.53 / 0.40, and 77% of its rollouts stay below VRMSE 1 against 89%. From
  developed flow it stays below persistence (0.86 / 0.54), narrowly from frame 140. The two runs differ in data,
  schedule (the released one was not annealed: learning rate ~98% of peak when it stopped, Appendix B) and
  hardware, so the cause is not identified.

### Why TFNO trails the other two (analysis, 2026-11-13; owner question)

Test one-step VRMSE (physical, 10%): TFNO 0.209 at 4 h vs 0.136 CNextU-Net and 0.134 AViT; developed 0.109 vs
0.050 / 0.075. Gain from 1 h to 4 h in developed convection: TFNO 9%, CNextU-Net 38%, AViT 11%.

* **TFNO hits a ceiling early.** The dedicated 1 h TFNO runs reach short-validation 0.327 / 0.304, the same as the
  4 h runs (0.310 / 0.325), and the 20-minute search run at 1e-3 reached 0.302 (different 512-window sample).
  TFNO also took more optimizer steps than CNextU-Net (61 vs 28 epochs in 4 h). So more training time does not help.
  The limit is the model, not the budget.
* **Its Fourier layers see only coarse scales.** The published Well configuration (`modes1 = modes2 = 16`) keeps,
  per Fourier layer, k_x in [-8, 7] of 512 points and k_y in [0, 8] of 128 (`SpectralConv.n_modes` = [16, 9],
  checked on the pinned neuraloperator 0.3.0; input laid out (B, C, x = 512, y = 128)). The shortest wave
  the layers mix is 64 px in x and 16 rows in y. Finer scales pass only through the pointwise (1 x 1) skip
  path and the pointwise nonlinearity, which cannot move a structure sideways.
* **But the one-step change lives at fine scales.** Data: the two saved snapshots (test, Pr = 1, frame 80,
  Ra = 1e8 and 1e10, 4 h seed-0 models, `outputs/visuals/*.npz`). The split uses exactly the kept modes (k_x index
  -8..7, k_y 0..8) and one-sided rfft energy. Share of the true one-step change (next frame minus last input) above
  that cutoff:
  * buoyancy 88% / 95%, w 87% / 87%, u 50% / 65%, pressure 37% / 51%;
  * the state itself is mostly coarse: 5–18% for p, u and w, and 26% / 45% for b.
* **TFNO's error is at those scales.** Error energy at step 1 relative to persistence's total, below | above the
  cutoff:
  * Buoyancy at Ra = 1e8: persistence 0.119 | 0.881, CNextU-Net 0.002 | 0.034, AViT 0.008 | 0.078, TFNO
    0.032 | 0.384.
  * Buoyancy at Ra = 1e10: persistence 0.055 | 0.945, CNextU-Net 0.002 | 0.243, AViT 0.008 | 0.293, TFNO
    0.016 | 0.472.
  * Of persistence's fine-scale buoyancy error, TFNO removes 56% / 50%, AViT 91% / 69% and CNextU-Net 96% / 74%.
    For w: TFNO 71% / 62%, CNextU-Net 99% / 92%.
  * TFNO's coarse-scale error is also 10–25x CNextU-Net's, so the cutoff is not the whole story.
  * (A first pass with |k_x| <= 8 and no one-sided weighting gave nearly the same numbers.)
* **Walls are not the main one-step problem.** TFNO's spectral path treats y as periodic (no domain padding), so the
  hot bottom and cold top wrap onto each other. At step 1, its buoyancy error near the walls (8 rows) relative to
  the interior is in line with the other models. The warm bottom layer seen in its snapshots grows during
  rollouts, so the walls may matter there.
* **Not the paper-era bug.** Our TFNO runs its Fourier layers (pinned commit after the fix). In developed
  convection it scores 0.109 normalised vs persistence 0.226 and the released, bugged TFNO 0.190.
* Consistent with the smoothing already reported: TFNO's spectral-energy ratio at k_x 25–100 falls from 0.16 at
  1 h to 0.11 at 4 h. Mean-squared-error training favours blurring what the model cannot place.
* Caveats: the scale split uses 2 snapshots (one start, step 1), so it is illustrative. The other candidate causes
  are not separated: the learning rate (1e-3 was the top of the grid and TFNO was still improving) and the
  parameter budget (mostly spent on 16 x 9 modes per layer).
* Possible checks, none run (they need owner approval; any training is an extension):
  1. The scale split over the whole test cohort (evaluation only).
  2. A 1 h TFNO at 10% with more modes (e.g. 64 x 64) and/or y domain padding. One hour is enough, since TFNO
     saturates by then; about 1.1–1.4 h held, roughly 50–60 SU paid, by the dedicated 1 h runs. More modes add
     parameters, so it breaks parameter matching (D3) unless hidden width or Tucker rank is reduced.

### Dedicated 1 h runs at 10% data (array 59487190, 2026-11-11/12; D6)

**Not used in the reports (owner, 2026-11-12):** the 1 h results are the 4 h runs' checkpoints after 1 h (D6).
Their cost stays in the ledger.

Same recipe as the 4 h runs, with the cosine ending at 1 h. All six ran in one attempt on the paid partition and
reached 1.000 h of GPU training, with the learning rate at ~0 at the end. Data waits were small this time
(0.04–0.34 h); wall time per run was 1.09–1.44 h (~305 SU in all).

| Short validation VRMSE (seed 0 / 1) | dedicated 1 h run | 1 h checkpoint of the 4 h run | 4 h run |
|---|---|---|---|
| CNextU-Net | 0.327 / 0.324 | 1.043 / 0.772 | 0.167 / 0.178 |
| TFNO | 0.327 / 0.304 | 0.508 / 0.524 | 0.310 / 0.325 |
| AViT | 0.219 / 0.221 | 0.468 / 0.418 | 0.161 / 0.167 |

* Letting the schedule finish matters. The dedicated 1 h runs beat the 4 h runs' 1 h checkpoints by 1.6–3x. A
  1 h TFNO run is as good as its 4 h run on this sample.
* The seeds agree within 0.02, unlike the 1 h checkpoints.
* Validation only (64 windows); the test evaluation is next.

### Model architectures and training recipe

Full tables in `docs/architectures.md`. In short: all three are The Well's implementations at `e63dec8`, from
random initialisation, 18.6–19.3M parameters, 16 input channels (4 frames × 4 fields) → 4 output channels.
* Activation: GELU in all three. Normalisation: LayerNorm (CNextU-Net; plus an L2 normalisation before each
  resampling), none (TFNO), GroupNorm and LayerNorm with QK-norm (AViT).
* Optimizer: AdamW (0.9, 0.999), weight decay 1e-4, no clipping, MSE in normalised units, batch 64, fp32: as
  The Well's code. Learning rate 1e-3 for all (our search) where the paper used 5e-4 / 1e-4; warmup then cosine
  to 0 over the 4 h clock where the paper decays to 0.1 · lr over 200 epochs and stops at 12 h.
* Same as the released checkpoints: CNextU-Net in every respect; TFNO in configuration but not in forward pass
  (the paper's and the released TFNO skipped its Fourier layers, ours runs them; D20); AViT has no released
  `rayleigh_benard` checkpoint and is widened 384 → 432 (D3).

### Evaluation protocol for the mid-project report (owner, 2026-11-10)

* Each run is scored at its 1 h and 4 h checkpoints on all 175 test trajectories in two ways:
  1. **The paper's protocol** (The Well, Tables 2 and 3): one-step VRMSE in normalised units on every test
     window (34,300), and a 30-step rollout from frame 0 of every trajectory, averaged over steps 6–12 and
     13–30. `tables/paper_protocol.csv` sets our runs (mean ± std over two seeds) beside the published value
     and the released checkpoint scored by our code.
  2. **The study's protocol:** one-step VRMSE in physical units, and rollouts from frames 20, 80 and 140
     (convection onset, developed, late; D15), each 30 steps, with the failure horizon, Nusselt number and
     kinetic-energy spectra.
* The mid-project numbers come from 10% data and 4 h budgets, so they are not a replication of the paper
  (which trained on 100% data for longer, at 5e-4 / 1e-4). The owner will make like-for-like comparisons
  later.
* AViT is not among the paper's `rayleigh_benard` baselines (FNO, TFNO, U-net, CNextU-net), so it has no
  published reference.

### First test results: CNextU-Net 10% seed 0 and persistence (2026-11-11)

Jobs 59428461 (row 0, both checkpoints, 55 min) and 59428462 (persistence, 9.5 min), paid `gpu` partition,
commit 4c0d3a8, cohort sha256 0e6dc760…; 34,300 one-step windows and 525 rollouts (175 × 3 starts) each.
Persistence predicts the last input frame. The released checkpoint was scored on the same windows by
`eval-released` (job 59407152). The other five runs are not evaluated yet.

One-step VRMSE on every test window:

| Model | Physical units (ours) | Normalised units (paper's protocol) |
|---|---|---|
| CNextU-Net 10%, 1 h checkpoint | 0.852 | 2.768 |
| CNextU-Net 10%, 4 h checkpoint | **0.131** | 0.334 |
| Released CNextU-Net (paper, 100% data) | 0.224 | 0.669 (published 0.670) |
| Persistence | 0.199 | **0.202** |

The same, by start frame (physical units; normalised units agree from frame 50 on):

| Start frames | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 | 50–99 | 100–195 | median |
|---|---|---|---|---|---|---|---|---|
| CNextU-Net 10%, 4 h | 0.745 | 0.569 | 0.320 | 0.134 | 0.076 | 0.063 | 0.043 | 0.048 |
| Released CNextU-Net | 1.570 | 1.274 | 0.673 | 0.228 | 0.084 | **0.048** | **0.033** | **0.034** |
| Persistence | **0.022** | **0.086** | **0.133** | 0.158 | 0.204 | 0.270 | 0.203 | 0.174 |

* **The mean over all windows mixes two regimes, so "our 10% model beats the paper's" would be a wrong
  reading.** Our model's lower mean comes from the first 50 frames, where the fluid is nearly still and every
  model's velocity error is large. From frame 50 on, the released checkpoint (100% data, trained longer) is
  ~25% better (0.048 vs 0.063; 0.033 vs 0.043), and its median is lower. That is the expected order, and it
  also argues against any leakage, which would make ours better everywhere.
* Persistence beats both models during convection onset (frames 0–29) and loses by 3–6× from frame 40 on.
* The released TFNO is barely better than persistence from frame 50 on (0.17–0.26 against 0.20–0.30;
  `released_one_step_by_frame_test`). This fits D20: its paper-era forward pass skips the FNO blocks, so it
  acts as a per-pixel map of the last frames.
  Under the paper's protocol (normalised, all windows) persistence (0.202) beats our model (0.334) and the
  paper's released model (0.669).
* From frame 50 on, error grows ~7× with Rayleigh number: ours 0.015 / 0.019 / 0.036 / 0.067 / 0.112 for
  Ra = 1e6 … 1e10 (released 0.018 / 0.017 / 0.025 / 0.045 / 0.085; persistence 0.098 … 0.351).
* The 1 h checkpoint is worse than persistence overall (0.852): 6.6 on frames 0–9, 0.07–0.11 from frame 50.
  It is a mid-schedule checkpoint (high learning rate), as the training curves showed.

Rollouts from frame 0, the paper's Table 3 (mean VRMSE over steps 6–12 / 13–30, 175 trajectories):

| Model | Normalised (paper's protocol) | Physical |
|---|---|---|
| CNextU-Net 10%, 1 h | 244 / 186 | 60.7 / 72.2 |
| CNextU-Net 10%, 4 h | 14.0 / 28.6 | 3.8 / 13.6 |
| Released CNextU-Net | 36.1 / 34.7 | 9.3 / 16.5 |
| Released TFNO | 142 / 169 | 37.1 / 73.9 |
| Persistence | **0.27 / 0.50** | **0.14 / 0.40** |
| Paper (Table 3) | > 10 / > 10 | not reported |

* Our 4 h model also lands in the paper's "> 10" range under its protocol. Persistence stays below 1: the
  frame-0 protocol scores the onset, where copying the input is nearly exact.

Rollouts from the study's starts (D15), 30 steps, VRMSE (physical) at steps 1 / 6 / 12 / 30:

| Start | CNextU-Net 10%, 4 h | Persistence |
|---|---|---|
| 20 (onset) | 0.434 / 0.700 / 0.985 / 1.719 | **0.118 / 0.378 / 0.499 / 0.677** |
| 80 (developed) | **0.059 / 0.167 / 0.298 / 0.530** | 0.270 / 0.641 / 0.777 / 0.856 |
| 140 (late) | **0.041 / 0.124 / 0.222 / 0.399** | 0.199 / 0.440 / 0.522 / 0.539 |

* Starts 80 and 140: the model beats persistence at every step to 30. Start 20: persistence wins throughout.
* Share of rollouts still below the error threshold at step 30 (all starts; the primary threshold, D16, is
  still to be chosen on validation): threshold 1: 4 h 0.89, 1 h 0.002 (Kaplan–Meier median 17 steps),
  persistence 0.94; threshold 2: 0.94 / 0.72 / 1.00; threshold 5: 0.99 / 0.78 / 1.00.
* Log spectral error at steps 1 / 10 / 30: 4 h 0.80 / 1.12 / 1.32; persistence 0.14 / 0.63 / 1.45.
* The model puts too much energy in the smallest scales, not too little
  (`eval_spectra_ratio_4h_test`). Below k_x ≈ 100 (wavenumber index < 64) its spectrum matches the truth
  within a few percent at every step. Above that, the median ratio predicted / true grows to ~50 at step 1
  and ~300–500 by step 30, spread over 35 regimes. At the Nyquist mode the data hold almost no energy, and
  the ratio is ~10^10.
* The log spectral error averages |log10 ratio| over all 255 bins with equal weight, so these near-empty
  high-wavenumber bins dominate it. The figure, or a band-limited version of the metric, is the better
  summary. The metric itself stays unchanged (hashed `eval`; D15).
* Nusselt relative error (mean) at steps 1 / 30: 4 h 0.27 / 2.62, persistence 0.20 / 1.27.
* Latency (A100, batch 1): 8.3 ms per step, 249 ms per 30-step rollout, 2.5 GB reserved.

## 4. Compute and cost

* RCIC rates: paid partitions charge 32 SU per GPU-hour plus 1 SU per core-hour of wall time (40 SU/h for a
  training job with 8 cores); free partitions cost nothing but can be preempted.
* Account: the class allocation `eecs195221_class_gpu` (31,530 of 33,000 SU left on 2026-11-09, shared by the
  class), approved by the owner on 2026-11-09. The personal account `skose` has 1,000 SU.
* **Final for the 10% study (`sacct`, 2026-11-11 12:41):** 2,160 SU billed and 72.6 GPU-hours in all; the evaluations cost 4.2 GPU-hours and 169 SU (7 jobs; queue waits 1–3 h for the last five). The visualisation job ran on the free CPU partition.
  * Paid/free split of the 72.6 GPU-hours: 54.2 on the paid partition (billed 2,160 SU: 40 SU per allocated hour
    for jobs with 8 cores, 36 for the 4-core pipeline check) and 18.3 on free-gpu (not billed). So 72.6 × 40 does
    not equal 2,160. CNextU-Net's first 10% attempts ran on free-gpu, so its SU per run covers 4.8 h of its
    5.2–6.0 h.
  * Memory: 64.8 / 56.9 / 8.9 GB is what PyTorch reserved at peak; 60.9 / 48.6 / 8.8 GB is what tensors
    allocated. Report both with labels.
* **Billed (from `sacct`, 2026-11-11 09:47, `outputs/findings/sacct_jobs.csv`, `sacct_summary.csv`,
  `compute_timeline`):** 2,059 SU on the paid partition, 70 GPU-hours in all (hours with a GPU allocated).
  Billing = `billing` × elapsed hours, `billing` = 40 for 1 GPU + 8 cores (36 for 1 GPU + 4 cores).

  | Purpose | Partition | Jobs | GPU-hours | SU billed | Queue wait, median / max |
  |---|---|---|---|---|---|
  | 10% runs | gpu (paid) | 9 | 47.7 | 1,907 | 2.8 h / 12.4 h |
  | 10% runs (first attempts) | free-gpu | 10 | 2.2 | 0 | 1.7 h / 2.2 h |
  | Evaluation (row 0, persistence; one running) | gpu | 4 | 1.7 | 67 | 14.0 h / 14.9 h |
  | Pipeline check | gpu | 1 | 2.3 | 84 | 5.4 h |
  | Pipeline check | free-gpu | 4 | 3.2 | 0 | 0.2 h |
  | Learning-rate search | free-gpu | 16 | 9.3 | 0 | 2.4 h / 6.8 h |
  | First replication try (stopped, D23) | free-gpu | 2 | 2.7 | 0 | 1.0 h / 1.9 h |
  | Profiling, environment checks | free-gpu | 4 | 0.9 | 0 | 0.2 h |
  | Data download, audit, subsets, setup | free (CPU) | 17 | 0 | 0 | ~0 |

* My earlier estimates from the run logs (~2,020 SU) were within 2% of the billed value.
* Queue waits on the paid partition are long: the AViT resubmission waited 8.5 h and 12.4 h, the evaluations
  ~14 h (8 A100s shared with other users). Waiting in the queue costs no SUs but sets the schedule.
* Waiting for `/pub` is charged as wall time: the 10% runs trained 24 GPU-hours but held 50 GPU-hours.
* Free-gpu work cost no SUs but lost time to preemption (6 tasks), queueing and refused starts.
* To finish the 10% study: four more evaluations, ~40 SU each.

## 5. Incidents and lessons

| Date | What happened | Effect | Fix or lesson |
|---|---|---|---|
| 11-07 | conda downloads into `/pub` came back empty | env build failed | package cache on node-local disk (`setup_env.sh`) |
| 11-07 | TFNO under bf16: `torch.fft.rfftn` has no bf16 | 1 of 99 checks failed | TFNO fp32 only (D8) |
| 11-07 | dataset stores a uniform `dimensions/y`; values are on Chebyshev–Gauss nodes | physics checks failed | `physics.y_grid: gauss` (D18) |
| 11-07/08 | released TFNO scored ~10 | pipeline check mismatch | paper-era forward skipped the FNO blocks (D20) |
| 11-08 | upstream `WellDataset` reopens HDF5 per sample | training I/O-bound (5.75–47 s per step) | `PersistentWellDataset`, bitwise-identical samples (D23) |
| 11-08, 11-10 | `/pub` `Errno 70` on two nodes at once | killed LR-search tasks 2, 3 and AViT 10% runs | resume from `last_resume.pt` (every 15 min); a retry in the data reader would avoid the crash |
| 11-09 | `cluster/push.sh` ran while array tasks were starting | 5 tasks refused a dirty tree | do not push while array tasks are queued |
| 11-09 | free-gpu preemptions | 10% tasks 0, 1 and 4 cancelled (not requeued) | resubmit the indices; state was saved |
| 11-09 | free-gpu A100s planned for higher-priority jobs (`ReqNodeNotAvail`) | jobs waited for hours | moved to the paid `gpu` partition |
| 11-09 | pipeline check hit its 2 h limit; Python buffered stdout | test split not scored, log empty | 4 h limit, `PYTHONUNBUFFERED=1` |
| 11-21 | Dedalus setup job wrote `env/hpc3-dedalus-conda.yml` into the tree while 100% tasks were starting | 5 tasks refused a dirty tree (~50 SU) | snapshot committed; no repo-writing setup job while training can start |
| 11-21 | MPI_Init failed on free nodes with a qedr RoCE card (hpc3-14-*, hpc3-15-*): UCX `rc_verbs/qedr0` | 15 of 90 solver tasks did nothing (no SUs; free partition) | `UCX_TLS=self,sm,tcp` (one rank, one node) in `solver_cpu.sbatch`; rerun on the class partition (D31) |
| 11-22 | `/pub` `Errno 121 Remote I/O error` while saving TFNO 100% s0's checkpoint | attempt 1 died after 1.1 h | in-job restart (`run_with_restarts.sh`) resumed from step 4,322 and completed; ~25 min wall lost |
| 11-22 | `/pub` outage at 00:08 PST: Slurm could not open the jobs' output files | row 10 and the persistence UQ evaluation `FAILED 0:53` in 1 s, no log | a job without a log can have failed at launch: check `sacct` after `/pub` incidents; resubmit |
| 11-22 | same failure at 17:43:09 PST | UQ seed pairs 1-4 (59939381_1-4) `FAILED 0:53` in 0 s, no log; task 5 started 3 s later and completed | second launch failure in a day: check `sacct` for `0:53` after every submission; resubmit `--array=1-4` |
| 11-22 | `/pub` write stalls of 10–22 min on every job at once (01:28, 02:03, 02:54, 04:19 PST) | 100% runs: up to 4,258 s of checkpoint time per run (22% of wall); 10% UQ runs: data waits | node-local data does not protect checkpoint writes; saving locally and copying in the background would (not implemented) |

## 6. Figures for the report

The written reports use `scripts/report_figures.py` (`reports/figures/fig_*.pdf`, rebuilt by
`reports/sync_figures.sh`): each figure drawn at its printed size (6.9 in text width, 7–8 pt text), one message
each, no titles inside, all models on shared axes where they are compared. The table below lists the exploratory
figures in `outputs/findings/`, which show everything.

Report figures: `fig_released_by_window` (pipeline check), `fig_loader` (data supply vs demand), `fig_training`
(loss and validation for all models; the learning-rate row of the exploratory figure is left out, the schedule
is described in the text), `fig_test_by_window` (ours vs released vs persistence), `fig_spectra` (one regime at
step 30, and predicted/true at steps 1 and 30), `fig_rollouts` (VRMSE vs rollout step from frames 80 and 140,
and the Kaplan–Meier fraction staying below VRMSE 1), `fig_accuracy_cost` (developed error against training
time, GPU time held per run, and inference time per 30-step forecast), `fig_snapshots_buoyancy_*` (after `visualize_cpu.sbatch`).

All are rebuilt by `.venv/bin/python scripts/findings.py` after `cluster/pull.sh`, as PDF (for LaTeX) and PNG,
into `outputs/findings/` unless another folder is named. The evaluation figures pick up each run as soon as
its evaluation is pulled.

| Milestone | Figure | What it shows |
|---|---|---|
| 1 Pipeline check | `released_one_step_by_frame_test` (and `_valid`) | released CNextU-Net and TFNO one-step VRMSE by window, both units, against persistence; all-window means beside the paper's |
| 1 | `released_paper_rollout_test` (and `_valid`) | released checkpoints rolled out from frame 0, the paper's Table 3, per field |
| 1 | `released_paper_rollout_by_ra_test` (and `_valid`) | the same, per trajectory, by Rayleigh number: "> 10" comes from low Ra |
| 2 Budgets | `profile_batch_size` | compute throughput and memory vs batch size, fp32 and bf16, four models |
| 2 | `data_loader_benchmark` | samples/s from `/pub` vs loader workers, against each model's demand |
| 2 | `outputs/lr_search/lr_search_val_vrmse`, `lr_search_training_loss` | learning-rate search (from `rbbench select-lr`) |
| 2 | `pilot_vs_actual_wall_time` | pilot predictions of wall time against the 10% runs |
| 2 | `data_wait_timeline_lr_search_f010` | data waits and crashes during the search |
| 2, 3 | `compute_timeline` | every GPU job from `sacct`: queue wait and run, by purpose, paid jobs outlined |
| 3 Training | `training_curves_train_f010` | training loss, validation VRMSE and learning rate vs training hours |
| 3 | `time_breakdown_train_f010` | wall time per run: training, waiting for data, other; SU |
| 3 | `data_wait_timeline_train_f010` | cumulative data wait over UTC time, crashes and preemptions |
| 3 Evaluation | `eval_one_step_by_frame_test` | one-step VRMSE by window: ours 1 h / 4 h, released, persistence |
| 3 | `eval_rollout_by_start_{1h,4h}_test` | rollouts from frames 20 / 80 / 140 against persistence |
| 3 | `eval_paper_rollout_4h_test` | the paper's frame-0 rollouts: ours, released, persistence |
| 3 | `eval_spectra_4h_test` | spectra for one regime (top) and predicted / true over all regimes (bottom); smallest scales shaded |
| 3 | `snapshots_{buoyancy,velocity_y}_<rollout>`, `snapshots_buoyancy_error_<rollout>` | true flow beside each model's rollout at steps 1, 6, 12, 30 (needs `visualize_cpu.sbatch`) |
| Solver (D25) | `solver_vs_models_rollout_4h_test` | Dedalus at 512 × 128, 256 × 64, 128 × 32 against the models and persistence, same 105 rollouts |
| Solver | `solver_cost_accuracy_4h_test` | seconds per rollout against mean VRMSE, per start frame |
| 3 | `rbbench aggregate`: `figures/fig2_rollout_vrmse_vs_step`, `fig3_failure_survival`, `fig5_nusselt_rel_err_by_ra`, `fig6_ke_spectra`, `fig7_resources` | the study's summary figures (`fig1`, data scaling, needs the 100% runs) |

## 7. Open items

* Evaluate rows 1, 4, 5, 8, 9 (row 0 and persistence done 2026-11-11; the five were submitted 2026-11-11),
  then `rbbench aggregate` for the tables (`paper_protocol.csv`, mean ± std over seeds) and add them to
  section 3.
* Rerun the `sacct` command (section 4) at the end of each phase so the billed totals stay current; the
  findings script rebuilds the tables from it.
* Decided by the owner (2026-11-11): all three failure thresholds (1, 2, 5) are reported (D16); the 1 h results
  are reported as the 4 h run's checkpoint after 1 h (learning rate still 0.85e-3; dedicated 1 h runs only if
  they do not make sense next to the 4 h results, not now; D6); one-step results are split into onset
  (frames < 50) and developed convection (D26).
* End of the project (not mid-project): the numerical-solver baseline (section 8): build `rb-dedalus`, pass the
  self-test, run the rollouts at three resolutions, then `eval-solver`.
* Candidate baselines that need no GPU training, named in the mid-project report (owner, 2026-11-12), not yet
  implemented: linear extrapolation in time (2 y_t - y_{t-1}), climatology (the regime's mean training state, a
  vertical profile per field) and dynamic mode decomposition (Schmid 2010). Note for interpretation: a field held
  constant at the target's spatial mean scores VRMSE = sqrt((N-1)/N) ~ 1, so threshold 1 is the "mean only" level.
* Dataset size: 367 GB in decimal units (train 293.6, valid 36.7, test 36.7 GB; 10% subset ~29 GB). The dataset
  card's "342 GB" is 341.8 GiB. The report gives both.
* The revised plan for milestone 3 is not written yet (points so far in `docs/checkin_2026-11-12.md`
  section 3).
* The 100% runs need separate owner approval. At 40 SU/h they pay for the same data waits, likely more
  (the full training set cannot stay in a node's page cache), unless the data moves to node-local disk.

## 8. Numerical-solver baseline (extension, owner request 2026-11-11; D25; end of the project)

* Question: how do the trained models compare with solving the PDE numerically, in accuracy and in run time?
* The data come from Dedalus, and the setup is Dedalus's own Rayleigh–Bénard example (same equations, boundary
  conditions, gauge, initial noise, output every 0.25). The stored 512 × 128 grid is Dedalus's grid at scale 1.
  So the solver can restart from the same frame the models see and predict the same 30 frames.
* Plan: 35 trajectories (one per regime) × starts 20 / 80 / 140 = 105 rollouts, at 512 × 128 (the data's
  resolution), 256 × 64 and 128 × 32; one CPU core each on the free partition (no SUs). Scored with the models'
  code; models compared on the same 105 rollouts.
* Expected: at 512 × 128 the solver reproduces the data up to float32 storage and step-sequence differences
  amplified by chaos (high Ra). The coarse runs show at what cost a solver matches the models' accuracy, the
  comparison McGreivy & Hakim (2024) recommend.
* Cost reference: The Well reports 6,000–50,000 s per 200-frame simulation (~60 h on 12 × 64 cores in all);
  30 frames is 15% of a simulation.
* Status (2026-11-11): code and tests done on the Mac (Dedalus itself is not installed there). The owner moved
  the runs to the end of the project, after the 100% runs; they are not part of the mid-project report. The HPC3
  env, self-test and runs are pending. Results go here.
* 2026-11-12 (owner): the mid-project report names the solver as a planned baseline (Setup, "Method and
  baselines"; Revised Plan) with the emphasis on the coarse grids, since at 512 x 128 it only measures the data's
  reproducibility limit. Still run after the 100% runs.
* Bounded design (D28, D29): Ra 1e6 / 1e8 / 1e10 at Pr 1, 5 test trajectories each (tasks 5-9 / 10-14 / 0-4), starts
  80 and 140, six grids; 30 rollouts per grid. Env and self-test passed (job 59905412).
* First runs (arrays 59907918-23, free partition, one core each, commit 9e56e06, 2026-11-21 17:59-18:52 PST).
  Stepping time per 30-frame rollout, median over the rollouts that finished (build time 1-11 s on top):

  | Grid | Ra 1e6 | Ra 1e8 | Ra 1e10 | Finished / diverged / MPI failure (of 30) |
  |---|---|---|---|---|
  | 512 x 128 | 290 s | 492 s | 1,267 s | 30 / 0 / 0 |
  | 384 x 96 | 149 s | 270 s | 528 s | 30 / 0 / 0 |
  | 256 x 64 | 60 s | 111 s | 175 s | 30 / 0 / 0 |
  | 192 x 48 | 35.6 s* | 56.8 s* (61 s) | 89.7 s* (63 s) | 30 / 0 / 0 (24 in the rerun) |
  | 128 x 32 | 13.5 s | 19.7 s | 26.8 s | 30 / 0 / 0 |
  | 64 x 16 | 3.9 s (2.7 s*) | 4.6 s | 6.5 s | 27 / 3 / 0 (6 in the rerun) |

  \* class-partition rerun (2026-11-22, below); in brackets, the first runs' value where both exist.

  * At 64 x 16 three of ten Ra = 1e10 rollouts diverged (task 0 both starts, task 1 from frame 80) after 28-29
    of 30 outputs: the CFL time step collapsed to ~1e-17 and the fields overflowed. The missing frames are scored
    as VRMSE inf (`solver_eval.score_frames`). The coarsest grid cannot hold Ra = 1e10.
  * Cost grows steeply with Ra at fine grids (512 x 128: 4.4x from Ra 1e6 to 1e10, through the CFL step).
  * The MPI failures are a node problem, not the solver: see the incident table (section 5). They rerun on the
    class partition (D31) with `UCX_TLS=self,sm,tcp`. Accuracy (scoring, grid matching with CNextU-Net) waits for
    the full set. The timings above come from mixed free-partition CPUs; each rollout file records its CPU model.
* Rerun of the 15 MPI failures (job 59920840, class `gpu` partition, `solver_cpu.sbatch` packed 8 side by side,
  commit 620fd64, 2026-11-22 10:27-10:31 PST): all 30 rollouts finished. Finished rollouts, the 3 divergences
  included, were skipped. **All 90 rollouts of the bounded design are done: 87 finished, 3 diverged (64 x 16,
  Ra = 1e10).**
  * 192 x 48 ran on hpc3-gpu-18-02 (Xeon Gold 6148, 2.4 GHz) in 3.8 min; 64 x 16 on hpc3-gpu-l54-09 (Xeon Gold
    6326, 2.9 GHz) in 0.2 min. About 5 SU for the two jobs (a GPU held idle in each).
  * The timings depend on the CPU and on how many rollouts share a node. Ra = 1e10 at 192 x 48 took 78-102 s
    eight side by side on the Gold 6148, against 63 s alone on a free node. Ra = 1e6 at 64 x 16 took 2.7 s on
    the Gold 6326, against 3.9 s. Up to ~1.4x either way, so report solver cost per grid with that spread. Each
    rollout records its CPU and `concurrent_rollouts`. The owner declined a uniform timing rerun (2026-11-21).
  * Next: score with `sbatch cluster/slurm/solver_score_cpu.sbatch` (writes `outputs/baselines/solver_<res>/`),
    then match grids against CNextU-Net (WP6).
* Scored (job 59921126, hpc3-gpu-18-02, commit 620fd64, 2026-11-22 10:46-10:48 PST; pulled 16:50): all six grids,
  15 trajectories and 30 rollouts each, cohort hash 0e6dc760, physics audit passed. Mean VRMSE at step 30: 512 x 128
  0.275, 384 x 96 0.280, 256 x 64 0.324, 192 x 48 0.373, 128 x 32 0.726; 64 x 16 inf (its 3 diverged rollouts).
  The 64 x 16 mean must be reported without them, or as a median. Grid matching against CNextU-Net (WP6) is next.

## 9. Onset per trajectory (D28, D29; final_plan WP4)

* Data: `rbbench onset-frames` (job 59905411, free CPU, commit e69e7e9). KE(t) of every validation and test
  trajectory is in `outputs/onset/ke_{valid,test}.csv`, and the crossing frames in `onset_{valid,test}.json`.
  No trajectory lacks a crossing.
* Proposed definition, fraction 0.5 of the developed mean (frames 150–199). t_on ranges 47–95, median 67, on both
  splits. The medians by Ra are flat (63–72).
* Check on validation, with the released CNextU-Net's per-window one-step errors (physical, from the pipeline
  check; no test data and no model of ours):
  * The released model's error is already at the developed level about 20 frames before t_on(0.5). Median VRMSE
    is 0.055 at 20 frames before, 0.033 at 10 before, 0.030 at t_on, 0.021 from 40 frames after.
  * Median VRMSE by the target frame's KE relative to the developed mean:
    * below 1e-4: 1.34;
    * 1e-4 to 1e-3: 0.22;
    * 1e-3 to 0.01: 0.10;
    * 0.01–0.03: 0.054;
    * 0.03–0.1: 0.039;
    * 0.1–0.2: 0.036;
    * 0.3 and above: 0.016–0.028.
  * The inflation is velocity only (u: 2.50 below 1e-4). Buoyancy is 0.007–0.009 in the nearly still phase,
    lower than in developed flow.
  * So the inflated phase is the nearly still fluid, KE below ~1–3% of the developed mean. 50% falls in the growth
    of convection, where the flow already has structure.
  * A small fraction also shows the expected physics: onset ends later at low Rayleigh numbers. Median t_on by
    fraction:
    * 0.03: 46 (Ra 1e6: 56, 1e10: 42);
    * 0.05: 49 (57 / 44);
    * 0.1: 52 (59 / 49);
    * 0.2: 57 (62 / 55).
  * At 0.5 the medians are flat (63–72).
* Splitting the released model's validation windows (onset mean / developed mean / onset share of windows / onset
  share of the all-window sum):
  * fixed 50: 0.807 / 0.038 / 25.5% / 87.8%;
  * t_on(0.25): 0.700 / 0.038 / 29.7% / 88.6%;
  * t_on(0.5): 0.620 / 0.037 / 33.8% / 89.6%;
  * t_on(0.75): 0.500 / 0.035 / 42.8% / 91.4%.
* **Decision (owner, 2026-11-21): fraction 0.1**, from the validation check above. Frame 50 stays as the check.
  The tables were recomputed on the Mac from the stored KE series (`rbbench onset-frames --from-ke outputs/onset
  --out-dir tasks`; the 0.25 / 0.5 / 0.75 frames came out identical to the HPC3 job's). They are committed as
  `tasks/onset_valid.json` and `tasks/onset_test.json`.
* t_on(0.1), min / median / max:

  | Ra | valid | test |
  |---|---|---|
  | 1e6 | 52 / 59 / 81 | 51 / 60 / 78 |
  | 1e7 | 47 / 54 / 63 | 46 / 54 / 64 |
  | 1e8 | 43 / 52 / 64 | 43 / 48 / 63 |
  | 1e9 | 42 / 47 / 61 | 42 / 49 / 61 |
  | 1e10 | 41 / 49 / 62 | 41 / 49 / 58 |
  | all | 41 / 52 / 81 | 41 / 52 / 78 |

  Every trajectory crosses. On test, t_on(0.1) lies in 41–78 for every cohort trajectory, so the rollouts from
  frame 20 all start in onset and those from 80 and 140 all start in developed flow. The rollout results by start
  frame are therefore already the per-trajectory split; only the one-step split changes.
* The 10% runs, one-step test VRMSE (physical fields), mean ± std over the two seeds, per-trajectory split
  (frame-50 split in brackets). Onset windows are 26.7% of all test windows (25.5% at frame 50):

  | Budget | Model | Onset | Developed |
  |---|---|---|---|
  | 1 h | CNextU-Net | 2.511 ± 0.653 (2.618) | 0.0783 ± 0.0022 (0.0806) |
  | 1 h | TFNO | 1.442 ± 0.292 (1.505) | 0.1203 ± 0.0001 (0.1197) |
  | 1 h | AViT | 1.244 ± 0.135 (1.299) | 0.0846 ± 0.0001 (0.0845) |
  | 4 h | CNextU-Net | 0.370 ± 0.024 (0.387) | 0.0505 ± 0.0002 (0.0498) |
  | 4 h | TFNO | 0.480 ± 0.003 (0.502) | 0.1105 ± 0.0001 (0.1090) |
  | 4 h | AViT | 0.294 ± 0.003 (0.308) | 0.0761 ± 0.0002 (0.0746) |
  | - | Persistence | 0.116 (0.121) | 0.229 (0.226) |
  | - | Released CNextU-Net | 0.733 (0.766) | 0.0382 (0.0382) |

  * Onset means drop 4–5%: the windows between frame 41 and 50 move into developed flow, and the low-Ra windows
    between 50 and 78 move into onset. Developed means change by at most 3%.
  * No ranking changes. Developed: CNextU-Net < AViT < TFNO at both budgets. Onset at 4 h: AViT < CNextU-Net <
    TFNO, all above persistence. The released CNextU-Net still leads in developed flow.
  * Scratch tables: `rbbench aggregate --tables <dir>`. `outputs/tables/` still holds the mid-project version,
    where `..._onset` meant frame 50; the next `aggregate` (with the 100% runs) rewrites it.

## 10. 100% runs and UQ training (2026-11-21/22; class account, `STAGE_DATA=1` for 100%)

### Status (pulled 2026-11-22 09:57 PST, commit 9e56e06)

Every run that started completed its budget in one allocation (`COMPLETE.json`: `training_completed`,
`budget_satisfied`; checkpoints at every milestone). Short-validation VRMSE is the 64-window monitor, not the test
score.

| Run | Job | Attempts | Train s | Data wait s | Checkpoint s | Wall h | Val 1 h / 4 h |
|---|---|---|---|---|---|---|---|
| row 2 CNextU-Net 100% s0 | 59905410_2 | 1 | 14,400 | 9 | 54 | 4.37 | 0.511 / 0.168 |
| row 3 CNextU-Net 100% s1 | 59907862_3 | 1 | 14,400 | 83 | 1,671 | 4.65 | 0.520 / 0.170 |
| row 6 TFNO 100% s0 | 59907862_6 | 2 (in-job restart) | 14,400 | 468 | 3,278 | 5.81 | 0.565 / 0.281 |
| row 7 TFNO 100% s1 | 59907862_7 | 1 | 14,400 | 21 | 4,258 | 5.46 | 0.535 / 0.274 |
| row 10 AViT 100% s0 | 59907862_10 | **failed at launch** (`0:53`, `/pub` outage), no log | | | | | |
| row 11 AViT 100% s1 | 59907862_11 | 1 | 14,400 | 3,098 | 1,279 | 5.66 | 0.486 / 0.152 |
| UQ β-NLL 10% s0–s4 (`tasks/uq.jsonl` 0–4) | 59907938_0–4 | 1 each | 3,600 | 1,399–3,582 | 17–673 | 1.60–2.03 | 0.347–0.387 |
| UQ β-NLL 100% s0–s4 (`tasks/uq.jsonl` 5–9) | 59907939_5–9 | 1 each | 3,600 | 3–4 | 449–1,071 | 1.14–1.54 | 0.341–0.373 |
| MSE 1 h 100% s0, s1 (`tasks/train_1h_f100.jsonl`) | 59907940_0–1 | 1 each | 3,600 | 3–4 | 374 | 1.11 | 0.317 / 0.325 |

* Steps in 4 h at 100%: CNextU-Net 12,142 / 12,129, TFNO 26,272 / 26,340, AViT 74,907. That is 2.8 / 6.1 / 17.5
  passes over the 274,400 training windows; the 10% runs made ~28 / ~61 / ~175 passes over theirs.
* Short validation at 4 h, 100% vs 10%: CNextU-Net 0.168–0.170 vs 0.167–0.178 (no change), TFNO 0.274–0.281 vs
  0.310–0.325, AViT 0.152 vs 0.161–0.167.
* The β-NLL ensemble members score 0.34–0.39 on short validation against 0.32–0.33 for the MSE runs on the same
  1 h schedule (mean prediction; expected cost of fitting the variance). Their calibration is evaluated by
  `eval-uq`.
* The local copy took 2.5–7.8 min per job (0.63–1.99 GB/s). With it, CNextU-Net and TFNO waited for data under
  8 min in 4 h. AViT 100% ran in 5.66 h of wall time, against 11.1–12.2 h at 10% reading `/pub` and the 14–17 h
  projected for 100% without the copy.
* Row 10 and the persistence UQ evaluation (59907864_6) failed at launch. `sacct` (owner, 2026-11-22) shows both
  started at 00:08:36 PST, on hpc3-gpu-l54-05 and -06, and ended within a second with `FAILED 0:53`: Slurm could
  not open the output file in `logs/` on `/pub`. This was 1.5 min before TFNO s0's `Errno 121`, in the same `/pub`
  outage. No log, no output, about 1 s billed. Both are to be resubmitted.

### Status (pulled 2026-11-22 16:50 PST, commit 620fd64)

Every job submitted at 10:40 PST completed. Checked from `COMPLETE.json`, `EVAL_COMPLETE_<label>.json` and the
Slurm logs: no traceback, no non-finite loss, the cohort hash 0e6dc760 and the physics audit passed in every record.

* **Row 10 (AViT 100% s0, job 59920838_10)**: complete in one attempt, 74,916 steps in 14,400 s of training.
  Local copy 3.0 min (1.61 GB/s). Wall 21,460 s (5.96 h): data waits 605 s, checkpoint saves 3,585 s (17%, the
  `/pub` write stalls again), validation 203 s. Short validation at 1 h / 4 h: 0.546 / 0.142 (s1 0.486 / 0.152).
  Peak memory 8.7 GB. Not yet evaluated.
* **Evaluations of rows 3, 6, 7, 11 (59920842)**: both checkpoints, 34,300 windows and 525 rollouts each; 10-27
  min per run. One-step test VRMSE, physical fields, all windows, 1 h / 4 h (from the logs; onset / developed split
  from `aggregate`):
  * CNextU-Net s1: 0.432 / 0.131 (s0 0.481 / 0.129);
  * TFNO s0, s1: 0.525 / 0.202 and 0.458 / 0.197;
  * AViT s1: 0.411 / 0.112.
* **Persistence UQ evaluation (59920839_6)**: valid (128.5 s) and test (624 s, noise 0-0.20) complete
  (`outputs/uq/persistence/{valid,test}/UQ_COMPLETE.json`).
* Status of every manifest row: the 12 core runs, the 6 dedicated 1 h runs at 10%, the 2 MSE 1 h runs at 100% and
  the 10 β-NLL runs all have a complete training record. Still to run: the evaluation of row 10, and the UQ
  evaluations 0-5 (seed pairs at 4 h) and 7-10 (ensembles and their MSE baselines). Those were never submitted,
  because their member runs were not all trained; now they are.

### Status (pulled 2026-11-22 19:40 PST): row 10 evaluation and the UQ evaluations

The owner submitted the row-10 evaluation (59939380_10, correlation on), the UQ seed pairs (59939381, indices 0-5)
and the ensembles with their MSE baselines (59939382, indices 7-10) at commit 620fd64.

* **Row 10 (AViT 100% s0)**: evaluation complete at 1 h and 4 h (644 s, hpc3-gpu-l54-05; 34,300 windows, 525
  rollouts, cohort hash matched, physics audit passed). One-step test VRMSE, physical fields, all windows, 1 h /
  4 h: 0.500 / 0.112 (s1 0.411 / 0.112). **All 12 core runs are now evaluated at both budgets.**
* **UQ evaluations complete** (`UQ_COMPLETE.json` on valid and test; 175 trajectories; cohort hash matched; member
  checkpoints equal to the runs' `COMPLETE.json`; no non-finite value in any `summary.parquet`). Wall time valid /
  test:
  * 0, CNextU-Net 10% seed pair at 4 h: 992 / 4,871 s;
  * 5, AViT 100% seed pair at 4 h: 324 / 1,411 s;
  * 6, persistence: 129 / 624 s (earlier, 59920839_6);
  * 7 / 8, β-NLL ensembles at 10% / 100%: 2,788 / 1,511 s and 2,795 / 1,503 s;
  * 9 / 10, MSE 1 h baselines at 10% / 100%: 955 / 592 s each. They ran side by side on hpc3-gpu-l54-04 and
    finished within 0.1 s of each other. Their outputs and checkpoints differ.
* **Not run: indices 1-4** (CNextU-Net 100%, TFNO 10% and 100%, AViT 10% seed pairs). There is no log and no
  output folder on HPC3. `sacct` (owner, 19:50 PST): all four `FAILED 0:53` at 17:43:09, elapsed 0 s, the
  launch failure seen at 00:08 (Slurm could not open the log on `/pub`). Task 5 started 3 s later and completed.
  No GPU time billed. To be resubmitted (`--array=1-4`).
* **Resubmitted as 59946048 (indices 1-4), all complete** (pulled 2026-11-23 08:30 PST; all on hpc3-gpu-l54-04,
  commit 620fd64; checks as above). Wall time valid / test:
  * 1, CNextU-Net 100% seed pair: 1,171 / 5,481 s;
  * 2, TFNO 10%: 1,217 / 3,360 s;
  * 3, TFNO 100%: 371 / 2,149 s;
  * 4, AViT 10%: 647 / 1,939 s.
* **All 11 UQ entries are complete on both splits (22 passes).** Test carries the noise levels 0-0.20 for the
  seed pairs and persistence (WP8). With this, every GPU job of the final-submission plan has run. What is left
  is Mac analysis (WP3 aggregation, WP5, WP6 matching, the WP8 and WP9 analysis, WP7) and a final `sacct` for
  the costs.

### `/pub` stalls now hit checkpoint writes

The 100% runs read their data from local disk but still write checkpoints and logs to `/pub`. Every running job
lost 10–22 min at the same moments (PST, 11-22): 01:28–01:29, 02:03, 02:54 and 04:19–04:20.

* The 100% runs lost the time in checkpoint saves: single saves of 1,065–1,308 s. Checkpoint time is 54–4,258 s per
  4 h run, up to 22% of TFNO s1's wall time.
* The 10% UQ runs, which read `/pub`, lost it as data waits (523–1,409 s at the same moments).
* AViT 100% recorded most of its stall time as data waits (2,930 of 3,098 s, at the same moments). Why it is
  booked there, and not as checkpoint time, is not established.
* Possible fix, not implemented: save checkpoints to local disk and copy them to `/pub` in the background. Only the
  remaining runs and evaluations would gain.

### Incident: `/pub` I/O error during a checkpoint save (TFNO 100% s0)

* At 00:10 PST, `os.stat` on the run folder inside `atomic_torch_save` raised `OSError: [Errno 121] Remote I/O
  error`.
* `run_with_restarts.sh` restarted the run after 60 s in the same allocation. It resumed from `last_resume.pt` at
  step 4,322 (2,376 s of training) and completed.
* Repeated: ~1,530 steps, ~840 s of GPU training. Cost: ~25 min of wall time, ~17 SU. The budget counts only the
  kept training.
* This was the first real restart, and it worked as designed.

### First 100% test result: CNextU-Net seed 0 (row 2, evaluation 59907942_2)

The evaluation checks passed: 20.4 min, cohort hash matched, physics audit passed, correlation on.

One-step test VRMSE (physical fields, per-trajectory onset split, t_on(0.1)) and VRMSE after 30 rollout steps by
start frame:

| Run | Budget | All | Onset | Developed | Step 30 from 20 / 80 / 140 |
|---|---|---|---|---|---|
| 10% s0 | 1 h | 0.852 | 2.972 | 0.0798 | 4.80 / 1.45 / 1.41 |
| 10% s1 | 1 h | 0.603 | 2.049 | 0.0767 | 2.40 / 0.74 / 0.68 |
| 100% s0 | 1 h | 0.481 | 1.595 | 0.0750 | 2.58 / 0.93 / 0.83 |
| 10% s0 | 4 h | 0.131 | 0.353 | 0.0506 | 1.72 / 0.53 / 0.40 |
| 10% s1 | 4 h | 0.140 | 0.387 | 0.0504 | 1.84 / 0.53 / 0.40 |
| 100% s0 | 4 h | 0.129 | 0.351 | 0.0486 | 1.95 / 0.52 / 0.40 |
| Released (100%, ~12 h H100) | - | 0.224 | 0.733 | 0.0382 | 1.65 / 0.63 / 0.51 |

* At 4 h, ten times more data barely helps CNextU-Net: developed 0.0486 against 0.0505 (−4%), with the same
  rollouts from developed flow. In 4 h it makes only 2.8 passes over the full data, so at this budget it is limited
  by compute, not data.
* At 1 h, the 100% run is best of the three on every one-step measure. The 10% seeds differ by 1.4×, so this
  needs seed 1.
* This is one seed. The comparison waits for row 3 and the TFNO and AViT evaluations.

### Cost (estimated from the run logs; `sacct` pending)

Process wall time plus the local copy, at 40 SU per hour (1 A100 + 8 cores):

| Work | GPU-hours | SU |
|---|---|---|
| Core 100%, rows 3, 6, 7, 11 | 21.9 | ~880 |
| UQ β-NLL, 10% (5 runs) | 9.3 | ~370 |
| UQ β-NLL, 100% (5 runs) | 6.7 | ~270 |
| MSE 1 h, 100% (2 runs) | 2.3 | ~95 |
| Evaluation, row 2 | 0.3 | ~14 |
| **Total** | **40.6** | **~1,620** |

Row 2 (4.4 h, ~180 SU) was logged on 11-21. The five finished 4 h runs held 26.3 GPU-hours; with row 10 the 100%
runs come to ~32 h, at the low end of the plan's 32–38 h with the local copy (docs/final_plan.md WP2). UQ training
came to 18.3 h against the 15–18 h planned.

**Billed (`sacct`, pulled 2026-11-23 into `outputs/sacct_skose.txt`, 228 allocations).** The paid `gpu` partition
bills `billing` × elapsed. Jobs submitted from 2026-11-21 on, which is all of the final-submission work (100% runs,
UQ training, every evaluation, solver reruns and scoring), came to **2,572 SU and 64.3 GPU-hours**, against the
plan's 2,850–3,750 SU with the local copy (`docs/final_plan.md` section 6). Free CPU jobs in that period: no GPU.
Before 11-21: 2,537 SU and 63.7 GPU-hours paid, plus 18.3 GPU-hours on `free-gpu`. **Paid in all: 5,109 SU.** The
per-purpose split comes from `scripts/findings.py` (`sacct_summary.csv`).
