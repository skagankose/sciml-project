# Instructions for coding agents

The specification is `RB_TheWell_HPC3_Coding_Agent_Implementation_Guide.md`, supplied by the owner. HPC3
conventions follow the owner's "Running Code on UCI HPC3 (Mac for Development)" guide.

Rules:
1. Never start the main sweep (the 12 core runs or any extension runs), request paid Slurm partitions or spend
   lab SUs without explicit owner approval.
2. Never download the dataset to the Mac.
3. Never train from pretrained weights. The Well's released checkpoints may be loaded only to evaluate them as a
   pipeline check.
4. Never tune on the test set.
5. Never edit code on HPC3: change it on the Mac, test, commit, then push.
6. When an upstream API is uncertain, read the pinned source and add a test.
7. Before every push, run `pytest -q` and `python -m rbbench smoke --device cpu --synthetic --grid 64 32 --steps 3 --all-models`.
8. Record every finding (results, cost, run time, training curves, incidents) in `docs/report_notes.md` as it
   happens, for the final report; `scripts/findings.py` regenerates its tables and figures (owner, 2026-11-10).

## Gates

| Phase | Gate | Status |
|---|---|---|
| A Mac bootstrap | tests + synthetic smoke for all 4 models without data | **passed 2026-10-22** (90 passed, 9 CUDA skipped) |
| B Upstream audit | shapes, parameter counts, random init, gradients; `docs/upstream_audit.md` | **passed 2026-10-22** (Mac CPU; 512×128 forward on CPU) |
| C HPC3 env | `check.sbatch` passes on a real A100; no env/caches in `$HOME` | **passed 2026-11-07** (job 59080323, after the TFNO fp32-only fix; first check 98/99, D8) |
| D Data | 3 splits verified (`rbbench_complete_*.json`); audit physics checks pass | download job 59080236 completed 2026-11-07 (all splits verified by the script). **passed 2026-11-07**: 1400/175/175 trajectories; audit physics checks pass with `physics.y_grid: gauss` (the stored y is a uniform placeholder, D18) |
| D2 Pipeline check | Released CNextU-Net and TFNO through our pipeline score at least as well as published (owner, 2026-11-08) | **passed 2026-11-08** (job 59224204): CNextU-Net 0.234/0.224, TFNO 0.305/0.300 vs published 0.481/0.670, 0.857/0.657; released TFNO with its paper-era forward (D20). In normalised units (the paper's protocol) the published **test** values are reproduced: 0.669 / 0.663 vs 0.670 / 0.657 (found 2026-11-09). The paper's rollouts from frame 0 (Table 3, ">10") are reproduced on test: CNextU-Net 36 / 35, TFNO 142 / 169 over steps 6–12 / 13–30 (job 59407152, 2026-11-09) |
| R Replication | our CNextU-Net and TFNO, trained as long as the paper (24 h on an A100), at least as good as the released checkpoints scored identically (`compare-paper`) | `tasks/replication.jsonl` ready 2026-11-08; first attempt I/O-bound and stopped (D23); loader fixed, manifest regenerated; data benchmark done (persistent, 8 workers: ~98 samples/s from `/pub`). **Postponed until after the Nov 12 check-in (owner, 2026-11-08)**; run ids now `…-9dfed09a`/`…-bb6db502` (D24) |
| E Subsets | S10 ⊂ S30 ⊂ S100, no leakage, counts per regime, identical hashes on rerun | real manifests committed 2026-11-07: nested, no leakage, 4/12/40 per regime; rerun-hash check pending |
| F Training engine | real-data mini-run per model on A100; resume and skip; loss decreases | **passed 2026-11-09** on A100 (the learning-rate search): loss decreases for every model, no divergence; runs resumed after `/pub` errors, preemption and a kill during a save |
| G Feasibility | 3 core models (and ConvLSTM if the extension runs): real-data step + 30-step rollout at 512×128 without NaN/OOM; throughput table | **profile 2026-11-08; step times from the search 2026-11-09** (GPU s/step: CNextU-Net 1.19, TFNO 0.55, AViT 0.19; batch 64 fp32, D7). Data waits come from `/pub` stalls, not bandwidth (D24). 30-step rollouts on real data run in the first evaluation |
| H Evaluation | metric tests; deterministic noise; physics self-checks on real data | code + synthetic tests done; real-data checks pending. Rollout starts frozen 2026-11-10 (D15: `[20, 80, 140]` plus the paper's protocol, owner); cohort `tasks/eval_cohort.json` generated on HPC3 2026-11-10 (175 trajectories, 35 regimes × 5, sha256 0e6dc760…) |
| I Sweep | 12-row core manifest (3 models × 10/100% × 2 seeds, 1 h/4 h budgets), A100-only, restart idempotence proven; extensions only with separate approval | **10% runs started 2026-11-09 by the owner** (rows 0, 1, 4, 5, 8, 9, commit f42b2e2; on the paid `gpu` partition, class account, from 2026-11-09 with owner approval); the 100% rows need owner approval. Rates fixed 2026-11-09 (1e-3 for all, D7). `eval` in common.yaml is hashed: do not edit it from now on (D15). Do not push while array tasks are queued (see 2026-11-09 log) |
| J Aggregation | one command → tables and figures, missing runs listed | code done (synthetic) |

The only value still PENDING in `docs/decisions.md` is the primary failure threshold (D16, chosen on validation);
batch size, learning rate, precision, schedule and rollout starts are fixed. The study design follows the final abstract (2026-10-30). Where the code
still differs from it, and which items are deferred, is in `docs/abstract_alignment.md`.

## Progress log

* 2026-10-22: Pinned The Well `e63dec8`, neuraloperator 0.3.0, OpenSTL `eecf8a3` (ConvLSTM cell vendored) and
  torch 2.8.0. Implemented the full package, configs, scripts, Slurm templates and docs. Mac test suite and
  smoke pass on CPU, and all four tiny models also run on MPS. Measured full-size parameter counts:
  18.58M / 19.28M / 14.69M / 13.34M (ConvLSTM effective; 16.74M allocated). Remote dataset sizes confirmed by
  HTTP HEAD. Next: Phase C on HPC3.
* 2026-10-22 (later): Abstract decisions. The 2 h results are checkpoints of the 6 h runs (216 GPU-h of
  training). Normalization uses the shared train-split `stats.yaml`. Models are parameter-matched: AViT
  hidden 384→432 (18.57M), ConvLSTM width 128→152 (18.79M trainable).
* 2026-10-30: Final abstract. The core study is 3 models (CNextU-Net, TFNO, AViT) × 10/100% × 2 seeds = 12
  runs of 4 h each, with checkpoints at 1 h and 4 h, about 100 GPU-h in total. Extensions in priority order:
  noise robustness, a third seed, 30% data, ConvLSTM. Before training, The Well's released CNextU-Net and TFNO
  RB checkpoints are evaluated as a pipeline check (rule 3 amended; checkpoints confirmed on Hugging Face,
  configs match ours). Docs updated to this plan.
  * Code changes still to make: the 12-run sweep defaults, 1 h/4 h budgets, noise off in the core evaluation,
    the released-checkpoint evaluation and GPU-hours accounting.
  * Deferred by the owner: the learning-rate search and downsampling for the backup plan.
* 2026-11-07: Phase C on HPC3. The `well-rb` env built (Python 3.13.18, torch 2.8.0+cu128) after conda's
  downloads into `/pub` kept coming back empty; `setup_env.sh` now keeps conda's package cache on node-local
  disk, and installs run as `setup_cpu.sbatch`. A100 check and the data download submitted. On the Mac:
  * 1 h/4 h budgets, with labels taken from the config everywhere; budget overrides replace the mapping.
  * Core evaluation is clean only; `experiments/noise.yaml` holds the noise extension.
  * `eval-released` (gate D2): released checkpoints pinned by revision and sha256, verified to load into our
    models bitwise-equal to upstream; scored with the same one-step code as our runs.
  * GPU-hours: per-attempt and per-evaluation-pass wall times, summed in `tables/gpu_hours.csv`.
  * `pull.sh --env` no longer also pulls `outputs/`. 100 tests pass; smoke passes.
  * First A100 check (job 59077767): 98 of 99 tests passed, including all four full-size models in fp32 and
    CNextU-Net, AViT and ConvLSTM in bf16. TFNO under bf16 fails in `torch.fft.rfftn` ("Unsupported dtype
    BFloat16"), as The Well's trainer notes. TFNO is now fp32 only: `train` and `profile` refuse bf16 for
    it, and a CUDA test pins the finding (D8). The first download job was preempted after 28 min; it resumes
    on resubmit.
  * After pushing 363bf0a and 78d062f: the second A100 check (job 59080323) passed, so Gate C passed. The
    resubmitted download (job 59080236) completed in 1 h 17 min; the script exits non-zero unless every split
    verifies. Next: dataset audit and subsets (CPU), and the pipeline check (D2) on owner approval.
  * Audit (CPU job): 1400/175/175 trajectories, 40 per regime; subsets 140/420/1400 (4/12/40 per regime)
    written on HPC3. Physics checks failed. The files store a uniform `dimensions/y`, but the README at
    `e63dec8` says "Sampled at Chebyshev nodes along the y axis", and `rayleigh_benard_uniform` differs "only
    by resampling". The wall-value and Nusselt failures follow from applying Chebyshev operators to uniform
    nodes. Added `scripts/probe_y_grid.py` (read-only): stored y, near-wall b rows, and the checks with the
    values on Gauss or Lobatto nodes. On synthetic data with a mislabelled y it identifies Gauss. VRMSE,
    training and the pipeline check do not depend on the y grid; the KE spectrum's y average and Nusselt do.
  * Probe on the real data: the values are on Chebyshev-Gauss nodes. With them, b at the walls is 1/0 within
    6e-10, u is 9e-8 of the rms and the conductive part is 1 within 3e-9; Lobatto fails, and the Ra = 1e10
    boundary layer spans ~5 rows. `physics.y_grid: gauss` now generates the nodes for the audit, spectra,
    Nusselt and the noise wall mask. The placeholder had put rows 0 and 127 on the walls, which would have left
    them unperturbed. The audit records the stored y and fails on anything but the nodes or the placeholder.
    Evaluation requires the audit to have checked the same family.
  * Audit rerun after pushing e87855e: "y grid: stored uniform (placeholder), used gauss; physics checks
    passed: True". Gate D passed. Subset manifests are on HPC3, to be pulled and committed.
  * Pipeline check (job 59154129, commit 78d062f): **MISMATCH**. Released CNextU-Net scores 0.234 valid /
    0.224 test (published 0.481 / 0.670). It reaches 0.03–0.05 from frame 50 on; the mean comes from the first
    ~20 frames, where the nearly still fluid makes velocity VRMSE 3–4. Released TFNO scores 10.7 / 10.5
    (published 0.857 / 0.657) and ~1.1 on every field even on late windows, which looks like a convention
    mismatch. Both delta and RMS stats are in `stats.yaml`; the pinned trainer supports both
    (DeltaWellDataset, RMSNormalization). Added `scripts/probe_released_conventions.py`, which scores both
    checkpoints under z-score/RMS × state/delta, with persistence as the reference. No training until this
    is explained.
  * Probe on real data (5 valid trajectories): CNextU-Net is right only as z-score next-state (late median
    0.019; persistence 0.204). TFNO is never informative: read as a delta it lands on persistence (0.216).
    On the Mac, its factorized and reconstructed contractions agree to 2e-6. No channel order, transpose,
    flip or field order (40 combinations) makes its output track the input. Upstream issue #49 (open)
    acknowledges outdated paper numbers and Hub checkpoints in "the old orientation". A 2026-11-05 comment
    reports the same FNO/TFNO-only failure on turbulent_radiative_layer_2D. Recorded in D20 and alignment F11;
    the pass criterion needs the owner.
* 2026-11-08: Reconciled with the paper.
  * Same data: server files dated 2024-12-05, official splits, and the v1.0.1 `stats.yaml` equals ours.
  * Same code source, at a later commit. Three upstream issues explain the published numbers:
    1. the paper-era TFNO wrapper dropped its FNO-block outputs (2024-06-03 to e430160, 2025-11-17), so the
       released TFNO is a per-pixel MLP whose block weights never trained;
    2. the v1.0.1 test set was built without normalization (later shown not to be how the published test values were made);
    3. the published validation numbers come from 20 shuffled batches, scored in normalized units.
  * With the paper-era forward, the released TFNO tracks its input (0.97–1.00).
  * Owner's criterion: results at least as good as the paper. `eval-released` now scores ours plus the
    paper-valid and paper-test protocols, and passes when ours ≤ published.
  * Replication set up (cluster/README step 3c): our CNextU-Net and TFNO at 100%, seed 0 (two core runs) are
    trained first and compared with the paper by `rbbench compare-paper`. Criterion: on valid, ours ≤ published
    under the paper's protocol; on valid and test, ours ≤ the released checkpoint under our protocol.
    Learning rates are The Well's own swept values (CNextU-Net 5e-4, TFNO 1e-4; D7). The batch size targets
    The Well's 64 and is chosen from profiling.
  * Pipeline-check rerun (job 59224204): **PASSED**. CNextU-Net 0.234 / 0.224, TFNO 0.305 / 0.300 (published
    0.481 / 0.670, 0.857 / 0.657).
    * Neither paper protocol reproduces the published values (valid 0.70 / 0.68; raw-input test 9.99 / 1.18).
      So the published test values did not come from the public test-set bug, and my first reading was wrong.
      (Wrong for test: normalised units reproduce the test values; see 2026-11-09.)
    * `compare-paper` now judges our runs against the released checkpoint scored identically, not against
      the published values.
  * First profiling job: `compare_upstream.py` passed on the real data (rollout max abs diff 0.0 for every Well
    model; VRMSE equals upstream exactly and NumPy to 1e-7). It then failed to write its JSON because
    `outputs/pilots/` did not exist, which stopped the job before the batch-size profiles. The script now
    creates the folder.
  * Profile (batch 8–64, fp32 and bf16): batch 64 fits every model in fp32 (peak CNextU-Net 58.7 GB, TFNO
    44.9, AViT 7.4, ConvLSTM 23.6). Set `train.batch_size: 64` (The Well's) and fp32 for all (D7, D8).
    Core manifest `tasks/train.jsonl` generated (12 rows); the replication runs are rows 2 (CNextU-Net
    100% s0) and 6 (TFNO 100% s0). Starting them starts the core sweep, so they need the owner's go-ahead.
  * Owner (2026-11-08): replicate the paper first, training as long as needed; the study keeps its budgets.
    The replication is now two separate 24 h runs (`replication.yaml`, `tasks/replication.jsonl`), not core
    rows 2 and 6. The core sweep waits until the replication is done.
  * The first replication attempt was I/O-bound (D23).
    * CNextU-Net: preempted after 359 steps, at 5.75 s per step.
    * TFNO: 7.8 s, then 47 s per step, against 1.42 s and 0.55 s profiled. Cancelled.
    * Cause: upstream `WellDataset` opens every file per sample through fsspec with an 8 MB block cache.
    * Fix: `PersistentWellDataset`, bitwise-identical samples, with loader settings moved to the unhashed
      `runtime` section, more workers and a profiler that reads fresh windows.
    * Manifests regenerated. Replication run IDs are now `…-3b7ecf6f` (CNextU-Net) and `…-7d6f032d` (TFNO).
    * Next: `rbbench bench-data` on a CPU node, then restart the replication.
  * `bench-data` result: the persistent loader is 1.2–1.4× faster than upstream. `/pub` delivers about
    0.5 GB/s into one node at best (97.7 samples/s with 8 workers; 15 workers were slower). With 7 workers,
    CNextU-Net is compute-bound and TFNO runs at ~85% of its compute rate, which is enough for the
    replication. AViT would run at ~half its rate, so the core sweep first needs node-local data or a
    measured shortfall (D23).
  * Owner: only the Nov 12 check-in matters now (pipeline validated, budgets fixed from pilot timings, first
    10% results with a revised plan); the replication is postponed until after it. Two owner decisions:
    1. Budgets count GPU training time only; waiting for data is logged, not charged (D24,
       `train.budget_clock: compute`). Replication run ids changed (`…-9dfed09a`, `…-bb6db502`).
    2. Run the equal-budget learning-rate search, one rate per model (D7, F5): `lr_search.yaml`,
       `tasks/lr_search.jsonl` (9 runs, 20 min of GPU time each on 10%), `rbbench select-lr` (rates plus pilot
       timings). Rates now live in each experiment's `model_overrides`: The Well's in `replication.yaml`, the
       search's in `main.yaml`. `tasks/train.jsonl` is regenerated after the search.
    * `train_a100.sbatch` allows 10 h (AViT needs ~7 h of real time for 4 h of GPU time).
* 2026-11-08 (night): learning-rate search (job array 59374848), 4 of 9 runs finished. Validation VRMSE:
  CNextU-Net 0.841 (1e-4), 0.469 (5e-4); TFNO 0.358 (5e-4), 0.302 (1e-3, top of the grid).
  * Tasks 2 and 3 died at 02:55 UTC from a `/pub` error (`Errno 70 Communication error on send`), not
    divergence. Tasks 6 and 7 were preempted. Task 8 is queued. Resubmit 2, 3, 6 and 7.
  * GPU s/step: CNextU-Net 1.19, TFNO 0.55, AViT 0.19. Between stalls the loader keeps up with every model
    on 10%; waits are `/pub` stalls of 3–21 min that hit both nodes at once (D24, D23).
  * `select-lr` now reports unfinished runs (progress, preempted or stopped), takes timings from them too,
    estimates real time with and without stalls, and writes `lr_search.md` and two figures. Results and
    caveats are in `docs/checkin_2026-11-12.md`.
  * The transfer node `hpc3-xfer` accepts rsync without a new login, so `cluster/pull.sh` works from the
    agent's shell.
* 2026-11-09: learning-rate search done (all 9 runs; jobs 59374848, 59390860, 59393014). Validation VRMSE at
  1e-4 / 5e-4 / 1e-3: CNextU-Net 0.841 / 0.469 / 0.408; TFNO 0.603 / 0.358 / 0.302; AViT 0.646 / 0.332 / 0.257.
  * Every model takes 1e-3, the edge of the grid; the owner kept three rates. Set in `main.yaml`;
    `tasks/train.jsonl` regenerated (10% rows 0, 1, 4, 5, 8, 9).
  * Starting the core runs fixes warmup 500 and the time-cosine schedule (D6), and the `eval` section, which
    is part of the run hash (D15).
  * Milestone 2 written up in `docs/checkin_2026-11-12.md`. Next: the six 10% runs, on the owner's go-ahead.
* 2026-11-09 (later): the released checkpoints reproduce the paper's **test** numbers. Job 59224204's output
  already held it: on the full test set, in normalised units, CNextU-Net scores 0.6688 (published 0.6699) and
  TFNO 0.6634 (0.6566). The 2026-11-08 reading scored test only with raw inputs, so it missed this.
  * The published valid values are still not reproduced (0.70 / 0.68): one random draw per epoch. The
    Hugging Face model cards list those valid values (0.4807, 0.8568) without naming a split.
  * Our physical-unit scores (0.224 / 0.300) are lower only because of the units: VRMSE's eps damps velocity
    on the nearly still first ~50 frames; from frame 50 on both units agree to four digits (D20).
  * Code: the diagnostic columns are now `normalised_vrmse_*` and `raw_input_vrmse_*` (were `paper_valid_*`
    and `paper_test_*`), and `eval-released` judges reproduction in normalised units on both splits.
    `compare-paper` refuses to judge against a `released_check.json` with the old names, so
    `eval-released` must be rerun before the replication is judged. The D2 criterion is unchanged.
  * D20, F11, the methodology, README and check-in notes corrected.
  * Paper rollouts (Table 3): `eval-released` now also rolls each released checkpoint out from frame 0 of
    every trajectory (30 steps, normalised data, as v1.0.1) and reports VRMSE over steps 6–12 and 13–30 in
    normalised and physical units. The paper gives ">10" for every RB model in both windows. Reported, not
    judged; result pending the next `eval-released` run (D20).
* 2026-11-09 (afternoon): the owner submitted the six 10% runs.
  * First attempt (array 59395480): task 0 was preempted after 4 min. Tasks 1, 4, 5, 8 and 9 exited at start
    with "production runs need a clean, committed git tree": the agent's `cluster/push.sh` was copying
    `f42b2e2` while they started, and rsync updates top-level files before `.git/`. No training was lost.
    Afterwards the HPC3 tree was checked equal to `f42b2e2` (files fetched and compared; macOS `openrsync -c`
    dry-runs report false differences). Rule: do not push while array tasks are queued.
  * Resubmitted as array 59398100 (rows 0, 1, 4, 5, 8, 9, `%1`) at `f42b2e2`; task 0 started 10:33 on
    hpc3-gpu-l54-04. The pipeline-check rerun with the paper rollouts (job 59397615) started 10:33.
  * Job 59397615 reached its 2 h limit at 12:32 during the test split.
    * Valid finished at 11:54. Its rollouts from frame 0, in normalised units, over steps 6–12 / 13–30:
      CNextU-Net 39 / 39 and TFNO 155 / 196 (paper: >10). Velocity produces all of it; buoyancy and pressure
      stay below 0.5 (D20).
    * `released_a100.sbatch` now allows 4 h and unbuffers stdout. It is not pushed: the training array is
      queued, so rerun with `sbatch --time=04:00:00`.
  * Free-gpu preempted three training tasks. The array does not retry them, but their state is saved.
    * Task 0 (CNextU-Net 10% s0): at 11:52, after 1543 s of training.
    * Task 1 (CNextU-Net 10% s1): at 12:38, after 860 s.
    * Task 4 (TFNO 10% s0): at 12:40, during startup.
    * Resubmit indices 0, 1 and 4 once squeue no longer lists them.
  * The owner resubmitted them as array 59403878 (rows 0, 1, 4, `%1`), alongside 59398100 (rows 5, 8, 9
    pending). The pipeline-check rerun is job 59403844 (`--time=04:00:00`). All waited with `ReqNodeNotAvail`:
    the backfill scheduler had planned the free A100 nodes (sinfo `mix-`) for higher-priority jobs. The only
    reservation is the maintenance window on 2026-12-14 and 12-15, after the report is due.
  * Owner (2026-11-09): cancel the free-gpu jobs and run the six 10% rows and the pipeline check on the paid
    `gpu` partition with the class account `eecs195221_class_gpu` (shared class allocation, 31,530 of 33,000
    SU left). Wall time is charged, so `/pub` waits cost SUs. These cover 1.6–2.7× the training time
    (`data_wait_seconds`), so expect roughly 6–11 h and 240–440 SU per run, about 2,000 SU in all. The
    rows run in parallel with `--time=14:00:00`. The 100% rows still need separate approval.
* 2026-11-10: results of the paid jobs (array 59407148, job 59407152, commit f42b2e2).
  * Pipeline-check rerun passed. On test, both one-step values are reproduced in normalised units and the
    paper's ">10" rollouts are reproduced (D20).
  * Four 10% runs completed their 4 h. Tasks 0 and 1 resumed from their free-gpu attempts. Short-validation
    VRMSE (64 windows) at 1 h / 4 h:
    * CNextU-Net: s0 1.04 / 0.167, s1 0.77 / 0.178.
    * TFNO: s0 0.51 / 0.310, s1 0.52 / 0.325.
    * Wall time per run: CNextU-Net 4.7 h, TFNO 6.9–7.1 h (TFNO waited 9–10 ks for data).
  * Both AViT runs (tasks 8 and 9) crashed at 02:44 on `/pub` `Errno 70`, at the same moment on two nodes.
    * They had trained 5941 s and 6048 s. `checkpoint_1h.pt` exists, and `last_resume.pt` was saved at 02:19.
    * AViT is data-bound: 30% of its wall time is compute, 12.9 ks of 20 ks is waiting for data.
    * The remaining ~2.35 h of training needs ~8 h of wall time per run.
  * Paid wall time so far: ~35 h of training and ~2.5 h of pipeline check, about 1,500 SU. Finishing AViT
    and evaluating all six runs adds roughly 900 SU.
  * Evaluation needs `tasks/eval_cohort.json`, and D15 (rollout starts) is still to be frozen on
    validation. On valid, the released checkpoints' one-step VRMSE (physical) by start frame is:
    * CNextU-Net: 1.65 (frames 0–9), 1.36 (10–19), 0.71 (20–29), 0.23 (30–39), 0.08 (40–49), ≤ 0.06 from 50.
    * TFNO: 1.38, 0.94, 0.49, 0.21, 0.17, then 0.17–0.25.
    * So start 20 falls in convection onset; 80 and 140 fall in developed convection.
  * The owner resubmitted AViT (rows 8, 9) on the paid partition.
  * Findings log (owner): `docs/report_notes.md` collects every result, cost, run time, training curve and
    incident for the final report (rule 8). `scripts/findings.py` rebuilds `outputs/findings/` from the pulled
    outputs: `runs.csv`, `attempts.csv` (SU estimates), training curves, time breakdown and the released
    rollouts. Provenance now records `SLURM_CPUS_PER_TASK`.
  * The released rollouts' ">10" comes from low Rayleigh numbers. On test, steps 13–30, CNextU-Net's median
    is 69 at Ra = 1e6 and 1.7 at Ra = 1e10; only 34% of Ra = 1e10 trajectories exceed 10.
  * Owner (2026-11-10): report the 10% results also exactly as the paper did, and keep the rollout starts
    `[20, 80, 140]` (D15, option A). The mid-project report uses the 10% / 4 h runs; like-for-like
    comparisons with the paper come later.
    * Every evaluation now also scores the paper's protocol: normalised one-step VRMSE on every test window,
      and the 30-step rollout from frame 0 with window means over steps 6–12 and 13–30. It reuses the frames
      already read, so there is no extra data reading.
    * `aggregate` writes `tables/paper_protocol.csv` (ours, mean ± std over seeds, beside the published value
      and the released checkpoint scored by `eval-released`).
    * Tests cover that the in-pass rollout equals `eval-released`'s and the table's references.
    * Not pushed: AViT (rows 8, 9) is queued. Next: `make-cohort --out outputs/eval_cohort.json` on the login
      node (safe while training runs); after AViT, commit the cohort, push, evaluate row 0 and the
      persistence baseline, then the other rows.
  * Cohort generated on the HPC3 login node into `outputs/eval_cohort.json` (175 test trajectories, 35 regimes
    × 5, starts `[20, 80, 140]`, 30 steps, clean only, sha256 0e6dc760…). Hash checked on the Mac and
    committed as `tasks/eval_cohort.json`; it reaches HPC3 with the next push, after AViT.
