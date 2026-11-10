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

### Status and time (2026-11-10, `outputs/findings/runs.csv`)

| Run | Status | Attempts | Training h | Data wait h | Other h | Wall h | GPU busy | Steps | Epochs | Val 1 h | Val 4 h | SU est. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNextU-Net s0 | complete | 2 (free-gpu, gpu) | 4.00 | 1.13 | 0.91 | 6.05 | 66% | 12,122 | 28 | 1.043 | **0.167** | 189 |
| CNextU-Net s1 | complete | 2 (free-gpu, gpu) | 4.00 | 0.81 | 0.37 | 5.18 | 77% | 12,194 | 28 | 0.772 | **0.178** | 191 |
| TFNO s0 | complete | 1 (gpu) | 4.00 | 2.54 | 0.32 | 6.86 | 58% | 26,339 | 61 | 0.508 | **0.310** | 274 |
| TFNO s1 | complete | 1 (gpu) | 4.00 | 2.82 | 0.32 | 7.14 | 56% | 26,352 | 61 | 0.524 | **0.325** | 286 |
| AViT s0 | crashed at 1.65 h, resubmitted | 1 (gpu) | 1.65 | 3.58 | 0.34 | 5.57 | 30% | 30,800 | 71 | 0.468 | | 223 |
| AViT s1 | crashed at 1.68 h, resubmitted | 1 (gpu) | 1.68 | 3.55 | 0.33 | 5.56 | 30% | 31,650 | 73 | 0.418 | | 222 |

"Val" is the short validation during training (64 fixed validation windows, physical units), not the test
evaluation. "Other" is startup, validation, checkpoint writes and steps replayed after a resume.

### Throughput and memory (A100 80 GB, batch 64, fp32)

| Model | Steps per training second | Samples per training second | Peak GPU memory reserved |
|---|---|---|---|
| CNextU-Net | 0.84 | 54 | 64.8 GB |
| TFNO | 1.83 | 117 | 56.9 GB |
| AViT | 5.2 | 333 | 8.8 GB |

* In 4 h of GPU time at 10% data, CNextU-Net sees 28 epochs, TFNO 61 and AViT ~170 (projected).
* `/pub` delivers at most ~98 samples/s into one node (D23 benchmark). TFNO needs 117/s and AViT 333/s, so
  both wait for data even without stalls. AViT spends ~70% of its wall time waiting.

### Training curves (`outputs/findings/training_curves_train_f010.{pdf,png}`)

* CNextU-Net: training loss falls steadily to ~0.003 at 4 h. Validation VRMSE rises from 0.48–0.55 at 30 min
  to 1.04 / 0.77 at 1 h, then falls to 0.17–0.18 by 3.5 h. The 1 h checkpoint is taken while the learning
  rate is still high, so it is a mid-schedule checkpoint of a 4 h run, not a 1 h-budget run.
* TFNO: training loss plateaus at ~0.013 after ~1.5 h; validation improves slowly to 0.31–0.33.
* AViT (to 1.7 h): training loss still falling (~0.0045), validation 0.37 / 0.48 at 1.5 h.
* The two seeds agree closely in training loss for every model. Validation differs more at 1 h than at 4 h.

## 4. Compute and cost

* RCIC rates: paid partitions charge 32 SU per GPU-hour plus 1 SU per core-hour of wall time (40 SU/h for a
  training job with 8 cores); free partitions cost nothing but can be preempted.
* Account: the class allocation `eecs195221_class_gpu` (31,530 of 33,000 SU left on 2026-11-09, shared by the
  class), approved by the owner on 2026-11-09. The personal account `skose` has 1,000 SU.
* Paid so far (estimate, 2026-11-10): 6 training attempts, 34.6 h, ~1,385 SU; pipeline check ~100 SU.
  About 1,500 SU in all.
* Expected to finish the 10% study: AViT ~8 h of wall time per run (~640 SU for both); evaluation of six
  runs ~1 h each (~240 SU). Total ~2,400 SU, about 7% of the class allocation.
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

## 6. Open items

* AViT 10% runs: resubmitted 2026-11-10; then evaluation of all six 10% runs (needs the D15 cohort decision).
* Evaluation results (one-step and rollouts on test, failure horizon, Nusselt, spectra) go into a new
  section here when they arrive.
* The 100% runs need separate owner approval. At 40 SU/h they pay for the same data waits, likely more
  (the full training set cannot stay in a node's page cache), unless the data moves to node-local disk.
