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
* Job 59407152 ran on the paid partition with 4 cores: 36 SU per hour, so ~100 SU for ~2.7 h (2 × ~80 min);
  `sacct` will give the exact value.

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

### Training curves (`outputs/findings/training_curves_train_f010.{pdf,png}`)

* CNextU-Net: training loss falls steadily to ~0.003 at 4 h. Validation VRMSE rises from 0.48–0.55 at 30 min
  to 1.04 / 0.77 at 1 h, then falls to 0.17–0.18 by 3.5 h. The 1 h checkpoint is taken while the learning
  rate is still high, so it is a mid-schedule checkpoint of a 4 h run, not a 1 h-budget run.
* TFNO: training loss plateaus at ~0.013 after ~1.5 h; validation improves slowly to 0.31–0.33.
* AViT: training loss falls from ~0.0062 at 1 h to ~0.0031 at 4 h. Validation falls from 0.47 / 0.42 at
  1 h to 0.161 / 0.167 at 4 h, with bumps at 1.5 h and 2.5 h (seed 1) while the learning rate is still high.
* The two seeds agree closely in training loss for every model. Validation differs more at 1 h than at 4 h.

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
* Paid so far (estimate, 2026-11-11): 8 training attempts, 46.8 h, ~1,870 SU (a lower bound: AViT's crashed
  first attempts are timed to their last log line); pipeline check ~100 SU; evaluation of row 0 and
  persistence ~45 SU. About 2,020 SU in all.
* AViT's second attempts took 5.5 and 6.6 h of wall time (~485 SU), less than the ~8 h per run expected.
* Expected to finish the 10% study: evaluation of the other five runs, ~1 h each (~200 SU). Total ~2,220 SU,
  about 7% of the class allocation.
* Queueing on the paid partition (8 A100s shared with other users): jobs waited with `Resources` /
  `Priority` for hours; this costs no SUs.
* Waiting for `/pub` is charged as wall time: about 40% of the paid training wall time so far.
* Free-gpu work so far: environment checks, data download, three pipeline-check runs, profiling, the
  learning-rate search and the first 10% attempts. It cost no SUs but lost time to preemption and queueing.
* To do: record billed values with
  `sacct -u skose -S 2026-11-07 -X -P --format=JobID%20,JobName,Partition,Account,AllocTRES%60,Elapsed,State,ExitCode > outputs/sacct_skose.txt`
  on HPC3 and pull them.

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

## 6. Figures for the report

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
| 3 Training | `training_curves_train_f010` | training loss, validation VRMSE and learning rate vs training hours |
| 3 | `time_breakdown_train_f010` | wall time per run: training, waiting for data, other; SU |
| 3 | `data_wait_timeline_train_f010` | cumulative data wait over UTC time, crashes and preemptions |
| 3 Evaluation | `eval_one_step_by_frame_test` | one-step VRMSE by window: ours 1 h / 4 h, released, persistence |
| 3 | `eval_rollout_by_start_{1h,4h}_test` | rollouts from frames 20 / 80 / 140 against persistence |
| 3 | `eval_paper_rollout_4h_test` | the paper's frame-0 rollouts: ours, released, persistence |
| 3 | `eval_spectra_ratio_4h_test` | predicted / true kinetic-energy spectrum by wavenumber |
| Solver (D25) | `solver_vs_models_rollout_4h_test` | Dedalus at 512 × 128, 256 × 64, 128 × 32 against the models and persistence, same 105 rollouts |
| Solver | `solver_cost_accuracy_4h_test` | seconds per rollout against mean VRMSE, per start frame |
| 3 | `rbbench aggregate`: `figures/fig2_rollout_vrmse_vs_step`, `fig3_failure_survival`, `fig5_nusselt_rel_err_by_ra`, `fig6_ke_spectra`, `fig7_resources` | the study's summary figures (`fig1`, data scaling, needs the 100% runs) |

## 7. Open items

* Evaluate rows 1, 4, 5, 8, 9 (row 0 and persistence done 2026-11-11; the five were submitted 2026-11-11),
  then `rbbench aggregate` for the tables (`paper_protocol.csv`, mean ± std over seeds) and add them to
  section 3.
* Billed cost and queue times: run
  `sacct -u skose -S 2026-11-07 -X -P --format=JobID%20,JobName,Partition,Account,AllocTRES%60,Submit,Start,End,Elapsed,State,ExitCode > outputs/sacct_skose.txt`
  on HPC3 and pull it. Our logs record run time but not time spent queueing; `Submit` and `Start` give it.
* Decided by the owner (2026-11-11): all three failure thresholds (1, 2, 5) are reported (D16); the 1 h results
  are reported as the 4 h run's checkpoint after 1 h (learning rate still 0.85e-3; dedicated 1 h runs only if
  they do not make sense next to the 4 h results, not now; D6); one-step results are split into onset
  (frames < 50) and developed convection (D26).
* End of the project (not mid-project): the numerical-solver baseline (section 8): build `rb-dedalus`, pass the
  self-test, run the rollouts at three resolutions, then `eval-solver`.
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
