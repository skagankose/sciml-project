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
| H Evaluation | metric tests; deterministic noise; physics self-checks on real data | code + synthetic tests done; **first real-data evaluation 2026-11-11** (row 0 and persistence, jobs 59428461/59428462: complete, audit physics check passed, cohort hash matched). Rollout starts frozen 2026-11-10 (D15: `[20, 80, 140]` plus the paper's protocol, owner); cohort `tasks/eval_cohort.json` generated on HPC3 2026-11-10 (175 trajectories, 35 regimes × 5, sha256 0e6dc760…) |
| I Sweep | 12-row core manifest (3 models × 10/100% × 2 seeds, 1 h/4 h budgets), A100-only, restart idempotence proven; extensions only with separate approval | **all six 10% runs complete 2026-11-11** (AViT after a resume, array 59428460). **10% runs started 2026-11-09 by the owner** (rows 0, 1, 4, 5, 8, 9, commit f42b2e2; on the paid `gpu` partition, class account, from 2026-11-09 with owner approval); the 100% rows need owner approval. Rates fixed 2026-11-09 (1e-3 for all, D7). `eval` in common.yaml is hashed: do not edit it from now on (D15). Do not push while array tasks are queued (see 2026-11-09 log) |
| J Aggregation | one command → tables and figures, missing runs listed | code done (synthetic) |

Nothing is PENDING in `docs/decisions.md`: all three failure thresholds are reported (D16, owner 2026-11-11);
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
  * AViT was cancelled while pending, 4c0d3a8 pushed and checked on HPC3, then resubmitted (array 59428460)
    with the row 0 evaluation (59428461) and persistence (59428462), all on the paid partition.
* 2026-11-11: all four jobs completed.
  * AViT s0 / s1 resumed from steps 28,249 / 29,105 and finished 4 h of training. Short-validation VRMSE at
    1 h / 4 h: 0.468 / 0.161 and 0.418 / 0.167. Wall time 11.1 / 12.2 h over two attempts, 55–60% of it
    waiting for data. All six 10% runs are complete.
  * Row 0 (CNextU-Net 10% s0) test one-step VRMSE: 4 h 0.131 physical / 0.334 normalised; 1 h 0.852 / 2.77.
    Persistence 0.199 / 0.202; released CNextU-Net 0.224 / 0.669.
  * Ours beats the released checkpoint only on the onset frames (< 50). From frame 50 on, the released
    checkpoint is ~25% better (0.048 vs 0.063), so the all-window mean must be reported with the split.
  * Frame-0 rollouts (paper's Table 3, normalised): ours 14.0 / 28.6, persistence 0.27 / 0.50.
  * Study rollouts: the model beats persistence at every step from starts 80 and 140; persistence wins from
    start 20.
  * Paid so far ~2,020 SU. Details in `docs/report_notes.md` section 3. Next: the owner evaluates rows 1, 4,
    5, 8, 9; then `aggregate`.
  * Figures for the report: `scripts/findings.py` now also draws the pipeline check by window and by
    Rayleigh number, the batch-size profile, the data-loader benchmark, pilot vs actual wall time, data-wait
    timelines, the learning-rate schedule, and four evaluation figures (`scripts/findings_figures.py`).
    The index by milestone is in `docs/report_notes.md` section 6.
  * Correction: the model puts too much energy in the smallest scales (ratio ~50 to ~500 above k_x ≈ 100),
    not too little. The log spectral error is dominated by those near-empty bins. `aggregate`'s fig6 now
    drops the Nyquist mode. 136 tests pass; smoke passes.
  * Architecture details for the report: `docs/architectures.md` (all GELU; AdamW 1e-4 weight decay as The
    Well's code; ours vs the paper and the released checkpoints).
  * Numerical-solver baseline (owner request; D25). The data are Dedalus's Rayleigh–Bénard example at 512 × 128,
    so the solver restarts from the models' last input frame:
    * Code: `rbbench.solver_grid` (NumPy grids and spectral resampling), `rbbench.solver_dedalus` (driver),
      `scripts/solver_rollouts.py` (env `rb-dedalus`, with `--selftest`), and `rbbench eval-solver` (scored with
      the models' code; parity test with persistence).
    * Jobs: `cluster/setup_dedalus.sh`, `solver_cpu.sbatch` (35 regimes × 3 starts, 3 resolutions, free
      partition), `solver_score_cpu.sbatch`; figures `solver_vs_models_rollout`, `solver_cost_accuracy`.
    * `pull.sh` leaves the raw rollouts on HPC3 (nearly copies of the test data; rule 2).
    * Dedalus could not be installed on the Mac (no conda), so its side is checked only by the HPC3 self-test.
  * Owner (2026-11-11): the numerical-solver runs move to the end of the project, after the 100% runs; they are
    not part of the mid-project report. Code stays as committed; nothing has run on HPC3.
  * Owner (2026-11-11), reporting the 10% results: the 1 h results are the 4 h run's checkpoint after 1 h (D6;
    dedicated 1 h runs only if needed later), all three failure thresholds side by side (D16, no primary), and
    one-step VRMSE also split into onset (< 50) and developed convection (D26). `aggregate` now writes the split
    and draws all three thresholds in fig3. On row 0 (4 h): 0.369 onset, 0.050 developed (physical).
  * `sacct` pulled (`outputs/sacct_skose.txt`): 2,059 SU billed on the paid partition and 70 GPU-hours in all
    as of 09:47; paid queue waits up to 12–15 h. `scripts/findings.py` now writes `sacct_jobs.csv`,
    `sacct_summary.csv` and `compute_timeline`.
  * Revised plan drafted in `docs/checkin_2026-11-12.md` section 3, for the owner. Core study ~140 GPU-h instead
    of ~100, the difference all data waits; 100% runs estimated at ~2,400 SU.
  * Mid-project report drafted as LaTeX: `reports/midproject.tex`, figures copied by `reports/sync_figures.sh`,
    compiles with `tectonic`. The 10% table has "pending" cells until the evaluations finish. `reports/` is not
    in git (the owner's abstract there is untracked too).
  * Report draft extended (owner's review): training recipe against the paper (same activation GELU, optimizer
    AdamW, loss, batch, precision; different rate, schedule, budget, data), time per run (measured 10%,
    projected 100%, validation and test-evaluation time), spectra figure with the smallest scales marked
    (`eval_spectra_4h_test`), and flow snapshots (`scripts/visualize_rollouts.py`, `visualize_cpu.sbatch`;
    pending until it runs on HPC3 after the evaluations, since a push is needed).
  * The owner reviewed `reports/midproject.tex` and marked "verification needed" points; all are now filled
    from code and upstream sources. Sources:
    * VRMSE formula: upstream `NRMSE(norm_mode="std")` at e63dec8.
    * TFNO bug: introduced in the_well `2d8d7eaf` (2024-06-03), fixed in `e4301602` (2025-11-17); issue #49.
    * Timer: per-step `perf_counter` from batch ready to `cuda.synchronize`.
    * Validation sample: 64 evenly spaced windows, all 35 regimes, 28% onset; the search's 512 share 8.
    * GPU-hour reconciliation: ~135–146 h without contingency.
    * Paper schedule: The Well's configs at e63dec8.
    * Wording: "alternative aggregate" became "physical-field VRMSE". The aggregation is identical; only eps acts
      differently on physical vs normalised fields.
  * Report figures redrawn for print (owner: "not good looking and understandable"): `scripts/report_figures.py`
    writes `reports/figures/fig_*.pdf` at their printed size. No titles inside, shared axes, legend on top,
    neutral text. The learning-rate row is gone from the training figure. The spectra figure is now 3 panels,
    plus a new test-error-by-window figure. `reports/sync_figures.sh` rebuilds them.
* 2026-11-11 (evening): all six 10% evaluations done and verified (12 checkpoints). Pushed 5d72d9a.
  * The push also copied the untracked `reports/` to HPC3. Training refuses a dirty tree, so this must be fixed
    before the 100% runs: commit `reports/` or exclude it from push.sh.
  * Results aggregated (`outputs/tables/`) and written into `reports/midproject.tex` (Tables 5–7, figures with
    all three models), `docs/report_notes.md` and the check-in notes. Flow snapshots wait for
    `visualize_cpu.sbatch`.
  * Flow snapshots done (job 59482062, free CPU): CNextU-Net tracks the plumes, TFNO blurs them and grows a warm
    bottom layer, AViT shows patch-sized block artifacts. Added as Figures 6–7 of `reports/midproject.tex`.
  * Final `sacct` for the 10% study: 2,160 SU billed and 72.6 GPU-hours in all.
* 2026-11-11 (night): the owner asked for dedicated 1 h runs at 10%. Until now the 1 h results were checkpoints
  of the 4 h runs, with the learning rate at 85% of its peak.
  * `tasks/train_1h.jsonl`: 6 rows, same recipe, cosine ending at 1 h, new run ids.
  * Estimate: ~12 GPU-h of training wall time plus ~2.5 GPU-h of evaluation, about 600 SU on the paid partition.
  * Commands are in `cluster/README.md` section 9.
* 2026-11-12: the six dedicated 1 h runs finished (array 59487190, paid, one attempt each, ~305 SU).
  * Short validation: CNextU-Net 0.33, TFNO 0.30–0.33, AViT 0.22. The 4 h runs' 1 h checkpoints were 0.77–1.04 /
    0.51–0.52 / 0.42–0.47.
  * Test evaluation next: `eval_a100.sbatch` with `MANIFEST=tasks/train_1h.jsonl`.
* 2026-11-12: mid-project report revised after an independent critique (owner).
  * Corrected:
    * Spectral numbers are now per wavenumber, matching the figure: CNextU-Net 0.80–1.57 below k_x = 100.
    * Seed spread is stated as a standard deviation: ≤ 0.006 at 4 h, up to 0.18 at 1 h.
    * Memory is labelled allocated (reserved).
    * Billing is split paid/free: 54.2 + 18.3 GPU-h, which explains 2,160 SU.
    * Table 9's CNextU-Net SU is explained.
    * The failure table's heading is reworded, and the spectrum calculation is defined.
  * Added `fig_rollouts`: VRMSE vs step, and Kaplan–Meier survival at threshold 1.
  * Moved to appendices: the run table, time table, timer, validation sample, loader figure, TFNO history and the
    Ra = 1e10 snapshots.
  * Figure labels and the colorbar ("Buoyancy b") are clearer, and full commit SHAs are cited. Numbers are in
    `docs/report_notes.md`.
* 2026-11-12: report reframed around the project's aim (owner): which model is useful under limited data and
  compute, with inference speed as a practical comparison.
  * Summary: added the aim, and findings on accuracy, returns on training time, cost to train, cost to use and
    negative results.
  * Setup: the 10% subset, frame spacing (dataset card) and an inference-time definition.
  * New `fig_accuracy_cost` (error against training time, GPU time held per run and inference time). The cost
    table gains GPU time held per run and inference memory.
  * New "which model under which constraint" table (`tab:choice`). The plan names the accuracy–cost curves at both
    data fractions.
  * Numbers are in `docs/report_notes.md`.
* 2026-11-12: the owner dropped the dedicated 1 h runs from the reports (never promised); the 1 h checkpoints of the
  4 h runs stand (D6). Report changes:
  * Table 5 gains 1 h rollout rows, and Tables 4–5 bold the best model at each budget.
  * New paragraph "With only one hour of training": AViT is best for one-step predictions, TFNO for rollouts,
    and CNextU-Net varies most between seeds.
  * The choice table gains a 1 h row, and the summary bullet is updated.
  * Mentions of the dedicated runs are removed from the plan and deliverables.
* 2026-11-12: plausibility review of the 10% results (owner request); the report now carries these
  qualifications:
  * TFNO is fastest only at batch 1. With batched forecasts AViT is cheapest per sample (0.89 / 3.15 / 6.74 ms at
    batch 64, from the profile).
  * fp32 favours TFNO: bf16 speeds up a training step 1.7× for CNextU-Net and 2.2× for AViT.
  * The 1.3× gap to the released CNextU-Net is not a data-scaling estimate: its schedule was probably not
    annealed (≤ ~25 of 200 epochs).
  * Seeds share one fixed 10% subset.
  * Table 6 gains a batch-64 column; the findings section gains a Caveats bullet.
  * Optional work added to the plan: a second subset draw, and batched inference timing.
* 2026-11-12: `reports/midproject_short.tex` written (owner): a shorter mid-project report.
  * The main body is ~7 pages, against ~11, and is framed by the aim (useful under limited data and compute).
  * It adds an inductive-bias column to the model table and a new section, "Why the Families Behave
    Differently", which ties the results to spectral truncation, locality and multi-scale structure, patching,
    one-step training vs rollouts, persistence and the metric, and data vs compute. New citations: FNO, PDEArena,
    spectral bias, PDE-Refiner, Brandstetter et al. 2022 and McCabe et al. 2023, all verified on arXiv.
  * Metric details, the recipe, the TFNO bug, accounting, the learning-rate search, training curves, the
    per-window test figure and the Ra = 1e10 snapshots moved to appendices.
  * `midproject.tex` is kept as the full version.
* 2026-11-12: field-wise VRMSE (promised in the abstract) is added to the long report (`tab:fields`).
  * In developed convection buoyancy is the hardest field.
  * The onset deficit against persistence is a velocity effect: on buoyancy all three models beat persistence.
  * The findings, the choice table and the test-evaluation sentence carry this qualifier.
  * The remaining deliverables now list only the Nusselt error.
  * The grammar slip in the opening paragraph is fixed.
  * The short version is no longer maintained (owner).
* 2026-11-12: new "Additional Analysis" section in the long report (`sec:extra`), with error by Rayleigh number and
  heat transport (`fig_by_rayleigh`, `tab:nusselt`).
  * Nusselt errors are now reported as medians and as the error of the rollout-averaged Nu. The means were
    dominated by near-zero and negative instantaneous true values.
  * A findings bullet was added, and the remaining deliverables were updated.
* 2026-11-12: report checked against the course's check-in items (problem and goal, data, method, progress, issues,
  plan); the owner keeps the long format. Added to `reports/midproject.tex`:
  * Setup: the physical system (heated layer; Ra and Pr defined, with their values; onset and developed
    convection) with `fig_system` (true buoyancy at Ra = 1e8 and 1e10), a fuller data paragraph (367 GB,
    1,400 / 175 / 175 by trajectory, 40 / 5 / 5 per regime, grid, what validation is used for), and "Method and
    baselines": persistence and the released CNextU-Net, the planned Dedalus baseline on coarser grids, and
    candidate baselines (linear extrapolation, climatology, DMD; McGreivy & Hakim 2024 and Schmid 2010 cited).
  * Appendix B: the vertical-grid identification.
  * New section "Issues" (`tab:issues`, `fig_time`: where each run's GPU time went) with questions for feedback.
  * Revised Plan: "Changes since the abstract" (budget clock, estimate, baselines, pipeline check, training
    settings, evaluation, optional work); the solver baseline is step 3 of the next steps.
* 2026-11-12: `reports/midproject_short.tex` re-created (owner): the long report reorganised, with nothing removed.
  * The main body keeps what the check-in asks for and the abstract's three milestones: goal, setup (system, data,
    task, models, training, baselines, metrics), milestones 1-3 (training curves, one-step and rollout tables,
    snapshots), findings, open issues and the revised plan.
  * Moved to new appendices: rollout curves, field-wise error and spectra (F); accuracy against cost and the
    choice table (G); the Rayleigh / Nusselt analysis (H); resolved or handled issues, split from the issues
    table (I); `fig_system` (J).
  * Main body 11 pages instead of 16 (25 in all). A word count of the two sources shows no text lost; only
    references to the moved parts were added. Later edits to the long report are not carried over automatically.
* 2026-11-12: `reports/midproject_short.tex` condensed further (owner): main body 6 pages instead of 11 (25 in all).
  * The main body follows the check-in items: goal, problem and data, method and baselines, milestones 1-3
    (training curves, one-step and rollout tables, example predictions, a short summary), open issues with the
    questions for feedback, and the revised plan with the changes since the abstract.
  * Repetition removed: the milestone list at the top, the long Findings section (now a five-bullet summary) and
    the long plan bullets. The issues table has 3 columns and open issues only.
  * The original detailed paragraphs moved to the matching appendices: system, data and baselines (new App. A),
    training, learning rate and precision (B), VRMSE and failure horizon (C), the onset observation with the
    released-checkpoint figure (E), the budget clock (F), the full results discussion (G) and the full issues
    table with `fig_time` (J). Every number of the previous version is still in the file.

* 2026-11-12: "Where feedback would help" (the four questions for the professor) removed from both reports (owner).
* 2026-11-12: long report, Issues section: new paragraph "Which one-step number to lead with" and `tab:headline` (onset windows give 87% of the released CNextU-Net's all-window mean, 96% under the paper protocol; three comparisons that reverse between all windows and developed convection).
* 2026-11-12: long report, "Which one-step number to lead with" now states that The Well paper reports the all-window mean (normalised fields; reproduced by the pipeline check), so its published values are dominated by onset windows too.
* 2026-11-12: both reports quote the paper's values at its precision (test 0.6699 / 0.6566 against our 0.6688 / 0.6634) and now give the paper's validation values (Table 5: 0.4807 / 0.8568; ours 0.7006 / 0.6754 on the full validation split, not reproduced). The milestone line's "within 1%" was wrong for TFNO (1.04%); now "within 0.2% and 1.0%".
* 2026-11-13: `fig_system` redrawn (owner: "a bit off"). The panels are stacked at full width, since side by side
  they were too small for the Ra = 1e10 structures. The colour scale is a square root, because on a linear scale
  the interior (b ≈ 0.1–0.3) was almost black. Wall, gravity and periodicity labels and two plume pointers were
  added, with Gouraud shading and a 400 dpi raster in the PDF. Caption updated; the long report is now 24 pages,
  the last holding only references.
* 2026-11-13: owner's subsection "Rayleigh–Prandtl Parameter Regimes" (`app:rb_regimes`) added to Appendix B of the
  long report, then moved to its own appendix, Appendix A (owner); the later appendices shift by one letter. Its values (Ra, Pr, δb0 ∈ {0.2,…,1.0}, seeds 40–49, 1,750
  trajectories, 200 steps, 512 × 128) were checked against the dataset card at e63dec8. Report now 25 pages.
* 2026-11-13: long report, Milestone 1 table (`tab:released`) gains the released TFNO row (0.6634 vs 0.6566; rollouts
  142 / 169 vs >10), marked as scored with its paper-era forward pass; details stay in the TFNO appendix (owner).
* 2026-11-13: long report, Setup's "Physical system" simplified (owner): Ra and Pr named in words, value ranges and a
  pointer to Appendix A. The Ra formula and its symbols, Pr for air and water, κ and ν in data units, no-slip walls
  and the initial condition moved to Appendix A, retitled "Physical System and Rayleigh–Prandtl Regimes". 24 pages.
* 2026-11-13: Appendix A of the long report shortened to two paragraphs (owner): Ra and Pr in one equation, value sets
  and the trajectory count inline; every value kept.
* 2026-11-13: Appendix B (training recipe) of the long report rewritten as explanation, without file paths or commit
  hashes (owner): task and loss, optimizer, schedule (ours on the compute clock; The Well's per epoch from 0.1·lr,
  checked in its pinned scheduler), checkpoints and seeds (validation is monitoring only), the learning-rate search,
  the released CNextU-Net's epoch estimate, and the software. Both tables kept. Note: tectonic's console does not
  show undefined references; check the log (`--keep-logs`).
* 2026-11-13: commit numbers removed from the long report's main body (owner): "Models" and "VRMSE" paragraphs. They
  remain only in the TFNO appendix and the reference list.
* 2026-11-13: "Additional Analysis" (`sec:extra`: error by Rayleigh number, heat transport) moved from the long
  report's main body to Appendix G, before "Additional Figures" (now H) (owner). Its two references now read
  "Appendix". Main body ends on page 15; 25 pages in all.
* 2026-11-13: long report, Setup: "Physical system" now describes only the physics (owner). The regime values
  (5 Ra × 7 Pr), the initial condition and the dataset citation moved to "Data", which no longer repeats them
  elsewhere.
* 2026-11-13: long report, Setup's "Training" paragraph rewritten for clarity and split into "Training" and "Seeds"
  (owner); same facts.
* 2026-11-13: long report, "Seeds" paragraph shortened to two sentences (owner).
* 2026-11-13: long report: section "Issues" renamed "Issues and Decisions for the Final Submission" (owner). The
  planned baselines (Dedalus on coarser grids, linear extrapolation, climatology, DMD) moved from "Method and
  baselines" to its new subsection 6.1 (`sec:planned_baselines`); "Method and baselines" keeps persistence and the
  released CNextU-Net and points to 6.1.
* 2026-11-13: long report, section 6 split into 6.1 "Decisions for the Final Submission" (`sec:decisions`: baselines to
  add, which one-step number to lead with) and 6.2 "Issues" (`sec:issues_met`: intro, issues table, GPU-time figure)
  (owner). "Method and baselines" now points to 6.1.
* 2026-11-13: long report, "Method and baselines": persistence is the only reference for now; The Well's released
  CNextU-Net and TFNO (100% data, 12 h H100) are named as references for the final report, with the TFNO-bug caveat
  (owner). The results tables still carry the released CNextU-Net row.
* 2026-11-13: long report: the Setup's VRMSE paragraph now says briefly why we lead with the physical-field VRMSE
  (fixed before training, i.e. in the methodology since 2026-10-22; less onset-dominated; agrees with the paper
  protocol in developed convection). Appendix C explains it in detail (owner): onset vs developed in plain terms
  with the frame-50 cut, why VRMSE blows up at onset, how ε differs between the two units, `tab:units` (0.766 /
  2.510 onset, 0.0382 / 0.0383 developed, 0.224 / 0.669 all, 87% / 96% onset share), the rollout protocols, and
  the reasons. 26 pages.
* 2026-11-13: long report, "Method and baselines": the released CNextU-Net is named as the full-data reference shown
  in the results (not a baseline to beat), and why only CNextU-Net: the released TFNO has the Fourier-layer bug,
  and The Well fixed the code (2025-11-17) but never released a retrained TFNO (Hugging Face
  polymathic-ai/TFNO-rayleigh_benard has one upload, 2025-03-28, checked 2026-11-13; issue #49 still open). A working
  full-data TFNO comes from our 100% runs (owner).
* 2026-11-13: long report, "Method and baselines" rewritten as one consistent account (owner): two references in the
  results, persistence (the baseline to beat) and the released CNextU-Net (full-data reference point, also the
  pipeline check); then why the released TFNO is not a reference, and the planned baselines (6.1).
* 2026-11-13: long report, model table (`tab:models`): the Normalisation column removed (owner).
* 2026-11-13: long report, `tab:models`: columns spread evenly over the line width (`tabular*` with `\extracolsep{\fill}`), so the right side has no empty gap (owner).
* 2026-11-13: long report, the difference between the physical-field VRMSE and the paper protocol rewritten in plain terms (owner): Setup's VRMSE paragraph (now two paragraphs) and Appendix C's "Why VRMSE is large during onset" and "How the two variants differ" explain epsilon as the constant that stops division by zero, which rescaling leaves unchanged; "floor" wording removed.
* 2026-11-13: long report, Setup's VRMSE paragraph simplified to one short paragraph (owner); the training standard deviations moved to Appendix C. Appendix F's (Additional Analysis) dangling reference to the removed `app:pipeline` appendix now points to Milestone 1 (`sec:m1`), where that observation now sits.
* 2026-11-13: long report, `fig_released_by_window` (Fig. 2) redrawn with the paper protocol only (owner): one full-width panel; the physical-field panel and its caption mention removed. The text's claims about it hold under the paper protocol (released TFNO vs persistence: onset 2.05 vs 0.13, developed 0.190 vs 0.226).
* 2026-11-13: long report, Milestone 1 observations: the Rayleigh-number bullet now says how it shapes the reporting (owner): the frame-0 rollouts are named, the low-Ra values are linked to later onset (next bullet), and errors are reported per Rayleigh number (Appendix F).
* 2026-11-13: long report, commit numbers are now links to the commits on GitHub (owner): Appendix D (TFNO) `2d8d7eaf`, `e4301602`, `e63dec8`, and the hashes in references [12] and [13]. All three commits checked through the GitHub API (dates and messages match; `e63dec8` is 29 commits after the fix). The links are clickable but not coloured (`hidelinks`).
* 2026-11-13: long report, Milestone 2: "Resource accounting" rewritten in plain terms as "Budget and cost" (what the budget clock counts, what the job holds and is billed for, queue time), and "Budget clock" (why) moved directly after it, before "Learning rate" (owner).
* 2026-11-13: long report, Milestone 2 keeps only what matters for the study (owner): "Budget and cost" and "Budget clock" merged into "Training budget" (active GPU time, why not elapsed time: data supply ~98 samples/s vs TFNO 116 and AViT 330, and allocated GPU time defined as training plus data waits and overhead). Cluster-specific details moved to Appendix E (Resource Accounting): the file-system stalls (new "Shared storage" paragraph), service units and queue time (added to "Billing"). Learning rate and batch size/precision unchanged.
* 2026-11-13: long report, `tab:time` (Appendix E): the two queue-wait rows removed; the caption now says queue waits are not included and gives their range for the 10% runs (0.1-8.8 h before training, 1-14 h before evaluation). The Billing paragraph no longer points to the table for queue time (owner).
* 2026-11-13: long report: main-body references to appendix tables now read "Appendix Table N" (owner): Tables 11 (learning-rate search), 14 (10% runs), 15 (time per run) and 16 (GPU-hours). The redundant "Appendix B" / "Appendix E" next to Tables 11 and 16 dropped.
* 2026-11-13: long report: `fig:testwindow` (one-step test VRMSE by window) moved from Appendix H to Milestone 3, directly after the Test evaluation paragraph (now Fig. 4); the paragraph cites it without "in Appendix", and the Additional Figures appendix now introduces only the Ra = 1e10 snapshots (owner). Report now 25 pages.
* 2026-11-13: long report, Milestone 3 "Rollouts" paragraph rewritten in shorter sentences, as three paragraphs (what a rollout is and why the ranking changes; model results; persistence and the paper protocol) (owner). Same numbers; the 89% / 63% are now stated as over all three start frames.
* 2026-11-13: `eval-baseline --released cnext_unet` (owner): the released CNextU-Net goes through the study evaluation (same cohort, rollouts from frames 20/80/140, failure horizons, spectra, physics) and writes `outputs/baselines/released/cnext_unet/evaluations/`, so Table 4 can get its released row. Until now the released checkpoints were scored only by `eval-released` (one step and the frame-0 rollout). Job: `sbatch --array=0 --export=ALL,RELEASED=cnext_unet cluster/slurm/eval_a100.sbatch` (about 0.5 GPU-h; cluster/README.md section 6). New test checks the cohort rollouts, the checkpoint provenance, and that its one-step rows equal the pipeline check's. 147 tests pass; smoke passes.
* 2026-11-13: long report, Milestone 3 "With only one hour of training": the dense paragraph replaced by a three-sentence summary and `tab:onehour` (Table 5): one-step VRMSE (all / onset / developed), rollouts (below 1 for all 30 steps, steps ahead of persistence from developed flow, VRMSE at step 30), and seed 0 / seed 1, each with a one-line finding (owner). Numbers checked against the outputs (steps ahead of persistence 14 / 21 / 12 recomputed); CNextU-Net seed 0 keeps 0.2% of rollouts below 1, not "none" as the paragraph said.
* 2026-11-13: long report, "Cost to train" bullet: why AViT needs a seventh of the training memory (patch embedding to 32 x 8 tokens vs full-resolution layers in CNextU-Net and TFNO; activations, not parameters, dominate) and that the patches cost detail (Fig. 8) (owner). The rough activation count is in `docs/report_notes.md` (Throughput and memory).
* 2026-11-13: long report, "Accuracy against cost" bullets shortened (owner): same numbers; dropped the "demand per GPU-second" remark, the 114 ms detail, the 3.5x / 7.6x ratios and the Dedalus data-generation-time comparison (now one sentence pointing to the planned solver baseline, Section 6.1). The AViT-memory explanation added earlier is no longer in the file (removed by the owner); it stays in `docs/report_notes.md`.
* 2026-11-13: double penalty (course Lecture 10, owner). Report: a paragraph in Rollouts and "Pointwise error and the failure thresholds" in Appendix C (1 = the mean, ~1.41 = realistic but displaced, 2 and 5 = breakdown; TFNO flattered by smoothness; persistence implies r ~ 0.6-0.85 at step 30), caveats on TFNO's one-hour rollout lead (Findings, choice table, Caveats), the plan's Evaluation bullet, and Gilleland et al. 2009 (checked via Crossref). Code (D27): `correlation` per field and field mean in every rollout row, `rollout_corr_mean_at` and `high_correlation_time` (0.8 / 0.9 / 0.95, by start frame) in the summaries, same columns in the solver scoring; thresholds in code, not the hashed eval config; `FORCE=1` for `eval_a100.sbatch` re-evaluates finished runs (cluster/README.md section 6). Tests: correlation against NumPy, the displaced-blob case (VRMSE sqrt(2), r ~ 0), the evaluation outputs, solver = models.
* 2026-11-13: double penalty reported only as a plan item for now (owner): the Rollouts paragraph, the Appendix C paragraph "Pointwise error and the failure thresholds" and the TFNO caveats (Findings, choice table, Caveats) removed again; the Revised Plan Evaluation bullet explains it briefly (with Gilleland et al. 2009) and names correlation and high-correlation time for the final report. The evaluation code (D27) stays. Also fixed: the plan's Baselines bullet pointed to Section 1 (Setup) for the candidate baselines, now Section 6.1.
* 2026-11-13: long report, Milestone 3 "Physical fidelity" rewritten shorter (owner): what the spectrum ratio shows and what it cannot, one bullet per model with the same numbers, the MSE/spectral-bias explanation and the two cautions (tiny true energy at the finest scales; persistence's fixed spectrum). Nothing dropped.
* 2026-11-13: long report, Findings section shortened so it no longer repeats the choice table (`tab:choice`, unchanged) (owner): one opening sentence pointing to the table; bullets cut to what the table does not say (Accuracy: 1.3x the released model, not a data-scaling estimate, buoyancy hardest, onset errors from velocity; Training time: 38 / 12 / 9% gains; Cost: storage limits AViT, 11-12 h held, every model forecasts in under 0.25 s); "With only one hour", "Cost to use" and "Negative results" dropped as covered by the table. Rayleigh/heat-transport and Caveats bullets unchanged.
* 2026-11-13: long report, Section 6.1 "Decisions for the Final Submission" rewritten as five numbered questions for the professors, each with context, cost and our default (owner): (1) numerical-solver baseline vs the three cheap baselines; (2) additional physics checks (divergence, buoyancy range, wall conditions, domain-mean buoyancy; now: Nusselt and spectra); (3) a full-length TFNO on all the data (paper's 12 h H100, about 30-40 h on an A100; roughly 50-90 GPU-hours held, 2,000-3,600 SU, estimated from the ~98 samples/s supply vs 117 needed plus stalls); (4) non-pointwise measures (double penalty; correlation and high-correlation time implemented, 10% re-evaluation ~4 GPU-h; scale bands, statistics); (5) which one-step number leads (`tab:headline` kept). Revised Plan bullets (Baselines, Evaluation, Optional) and next step (3) now point to 6.1 instead of repeating it; the optional 24-hour CNextU-Net + TFNO runs are replaced by question 3.
* 2026-11-13: long report, Section 6.1 question 5 reworded (owner): "Is it acceptable to rank the models by their error in developed convection, rather than by the paper's single number over all windows?", with what we did spelled out (physical fields reduce but do not remove the onset effect; onset and developed reported separately; ranking by developed; all-window mean only for the paper comparison) and the reversal example (0.136 vs 0.224 all windows, 0.050 vs 0.038 developed).
* 2026-11-13: long report, Section 6.1 question 5 asks what to do rather than "is it acceptable" (owner): report the one-step error as the paper's all-window number, as separate numbers before and after frame 50, or both as now (the item no longer states a default; owner).
* 2026-11-13: long report, Section 6.1 question 5 rewritten (owner): short question "How should we report the one-step error?"; the paragraph explains the paper's all-window number, why onset dominates it, that we report both (all-window mean and the split at frame 50, ranking by developed), the reversal example, and ends by asking whether to keep both or lead with one.
* 2026-11-13: long report, Section 6 (owner): two questions moved from the issues table to 6.1, now seven questions: (6) multi-step rollout training (pushforward trick, Brandstetter et al. 2022; default: keep one-step training) and (7) extending the learning-rate search to 5e-3 and 1e-2 (~6 GPU-h; a new rate would mean retraining the 10% runs; default: keep 1e-3). Questions carry labels (`q:baselines`, `q:tfno`, `q:physics`, `q:pointwise`, `q:onestep`, `q:rollout`, `q:lr`), so the Revised Plan's "question N" references follow any reordering; the plan's training-settings bullet points to question 7. 6.2: the rows on one-step targets and the learning rate (moved), and on VRMSE during onset and the released TFNO (covered by questions 5 and 2) removed; the intro says these issues need no decision, unlike 6.1.
* 2026-11-13: long report, Revised Plan (owner): "Next steps" removed; "Remaining deliverables" is now "Deliverables remaining for the full report", minimal: the six 100% runs evaluated as the 10% runs were and the 10% vs 100% comparison at 1 and 4 hours; further additions depend on Section 6.1.
* 2026-11-13: released CNextU-Net evaluated through the study evaluation (job run by the owner; pulled; 525 rollouts, cohort hash matched, physics audit passed, one-step 0.2238 = pipeline check). Report Table 4 released row filled: VRMSE after 30 steps from frames 20 / 80 / 140 = 1.65 / 0.63 / 0.51; below VRMSE 1 / 2 / 5 through step 30 = 0.77 / 0.82 / 0.90. Finding (docs/report_notes.md section 3): better one step than our 10% CNextU-Net but less stable over rollouts (0.63 / 0.51 vs 0.53 / 0.40; 77% vs 89% below 1).
* 2026-11-13: correlation and high-correlation time made opt-in (owner: not default runs; to be decided for the final report). `evaluate(..., with_correlation=False)`, `--correlation` on eval-task / eval-baseline / eval-solver, `CORRELATION=1` for eval_a100.sbatch; without it the outputs are exactly as before. Tests: off by default; on, the evaluation and the solver scoring agree. 153 tests pass; smoke passes. Report: the owner commented out questions 4 (pointwise) and 7 (learning rate); the plan's references to them (Evaluation bullet, training-settings bullet) removed, the count is "Five questions", and the now-uncited Gilleland et al. reference is commented out. D27, cluster/README.md and docs/report_notes.md updated.
* 2026-11-13: long report, Section 6.1: the five active questions shortened to 4-6 lines each (owner), keeping for each the reason, the options with their cost and the default: solver vs cheap baselines; full-length TFNO (30-40 h on an A100, 50-90 GPU-hours held, 2,000-3,600 SU); physics checks; one-step reporting (87% onset share, reversal example); rollout training. Intro: "where we have one, the default". The commented-out questions are untouched.
* 2026-11-13: long report, appendices shortened (owner): prose rewritten from about 4,400 to 2,800 words (tables and figures checked byte-identical, 11 blocks); the report is 22 pages instead of 24. Removed as repetitive or beyond what a reader needs: the Optimizer paragraph (in Table 10), The Well's schedule restated, timer implementation details (perf_counter, pinned memory), the validation-index formula and the 8 shared windows, billing per core-hour, the TFNO and AViT ranges by Rayleigh number, and some caveat sentences; onset, the two VRMSE variants and the storage / billing / pilot paragraphs each merged into one. Backup of the previous version: scratchpad midproject_before_appendix_trim.tex.
* 2026-11-13: long report, Fig. 2 (`fig_released_by_window`): x-axis now ends with a 195 tick, and the caption explains it (owner): 200 frames (0-199), four input frames predict the next, so the last window starts at 195; 196 windows per trajectory.
* 2026-11-13: long report, Milestone 2 "Training budget": the reason for counting GPU time shortened to two sentences (owner); the storage rates (98 vs 116 / 330 samples/s) are left to Fig. 10 and Appendix E. The term "allocated GPU time" is kept, since later sections use it. Its two main-body references to Fig. 10 now read "Appendix Fig. 10".
* 2026-11-13: why TFNO trails (owner question; `docs/report_notes.md` section 3, "Why TFNO trails the other two"). It saturates by 1 h: the dedicated 1 h runs equal the 4 h runs. Its Fourier layers keep k_x in [-8, 7] and k_y in [0, 8] (n_modes [16, 9], checked on the pinned neuraloperator), but 87-95% of the true one-step change in b and w lies above that cutoff (2 saved snapshots). Its step-1 error sits there: buoyancy at Ra = 1e8, fine-scale error 0.384 of persistence's total, against 0.034 for CNextU-Net. Not the paper-era bug. Wall-error check inconclusive for one step. Possible checks are listed; none run.
* 2026-11-13: long report, new Appendix E "Why Our TFNO Trails the Other Two Models" (`app:tfno_gap`; owner), after the TFNO-bug appendix; Resource Accounting and Additional Analysis shift to F and G. Contents:
  * TFNO plateaus: training-time evidence only, without the dedicated 1 h runs.
  * The Fourier cutoff (|k_x| <= 12.6, 99% of its parameters in the kept modes) and `tab:tfno_scales`: the scale split of one-step buoyancy error, from 2 snapshots, recomputed with exactly the kept modes.
  * Other causes, not separated.
  * Test evaluation now points to the appendix. The Physical fidelity bullet said TFNO keeps "16 lowest modes in each direction (k_x ≲ 25)"; it keeps n = -8..7 in x, so now |k_x| <= 12.6.
* 2026-11-14: `reports/midproject.tex` contained a second full copy of the report, pasted just after `\label{sec:setup}` (file saved 14:29, two `\end{document}`). Merged into one document that keeps both sets of the owner's edits:
  * from the surrounding copy: the reworded baselines sentence, and the "paper protocol" sentence commented out;
  * from the pasted copy: three `\newpage` lines (before "Data.", "Field-wise error" and "Accuracy against cost").
  * Backup of the duplicated file: scratchpad `midproject_duplicated_20261114.tex`.
* 2026-11-14: long report, contents list before Setup (owner): "Contents" in two columns on page 1. Left: Sections 1–7 with 6.1 and 6.2, and References. Right: Appendices A–G. Every entry links to its section (hyperref). Compact entry spacing, ragged right, no hyphenation (`\compactcontents` in the preamble, `multicol`). 23 pages.
* 2026-11-14: long report, contents list simplified (owner): main Sections 1–7 only, one full-width line each with dotted leaders. Subsections 6.1/6.2, References and the appendices are no longer listed; the appendix writes `\stopcontents` into the .toc. `multicol` removed. Setup starts on page 1 again; 23 pages.
* 2026-11-14: long report, the formal "Contents" list replaced by a paragraph "Structure of the report" (owner): "The rest of the report is organised as follows", then one bullet per main section (1–7), each a linked `\ref` with one line on what it covers. The contents machinery was removed from the preamble and the appendix. Still 23 pages.
* 2026-11-14: long report, "Structure of the report" rewritten as one inline paragraph with the section titles in italics and a few words each (owner); the bullet list is gone.
* 2026-11-14: long report, back to a contents list (owner), now compact: a bold run-in "Contents." with Sections 1–7, one line each (number and title in bold, page number in grey right after the title, no dotted leaders); subsections, references and appendices not listed (`\compactcontents`, `\stopcontents`). It replaces the "Structure of the report" paragraph.
* 2026-11-14: long report, the compact contents list now also lists References and, under an "Appendices" label, Appendices A–G (owner). Same format: grey page number after the title. `\stopcontents` replaced by `\appendixcontents`. It all fits on page 1 under the owner's heading "Structure of the report".
* 2026-11-14: long report, contents entries back to dotted leaders with the page number at the right margin (owner); still compact (1pt between entries), with References and Appendices A–F. The owner commented out the "Why Our TFNO Trails" appendix, so its two main-body pointers (Test evaluation, Physical fidelity) were removed; the corrected cutoff |k_x| <= 12.6 stays. The analysis remains in `docs/report_notes.md`. 22 pages.
* 2026-11-14: long report, 4pt of space between the "Structure of the report" line and the contents list (owner).
* 2026-11-14: long report, Setup "Task" paragraph now states the study task (owner): which of the three families forecasts RB convection most usefully under limited data (10% now, 100% in the final report) and training time (1 or 4 h on one A100), measured by one-step and 30-step accuracy and cost to train and use. The forecasting task (4 frames -> next frame, autoregressive rollouts, 196 windows per trajectory) is kept inside it.
* 2026-11-14: long report, Appendix A (Physical System Details) trimmed to what the main text lacks (owner): the Boussinesq equations with no-slip walls, the Ra and Pr formulas and symbols, Pr for air and water, kappa and nu in data units, the exact Ra and Pr sets, and the 5 amplitudes x 10 seeds per regime. Facts already in Setup (box, walls, 1,750 trajectories, 200 frames, grid, initial condition, meaning of Ra) were removed.
* 2026-11-14: long report, Fig. 2 caption shortened (owner): what is plotted and what the shading means, with a pointer to the metric appendix for the frame-50 cut. Removed as said elsewhere: the window definition (Setup, Task), why the cut is at 50 (appendix), the persistence definition (Setup) and the TFNO bug (the paragraph above).
* 2026-11-14: long report, metric appendix, 'Why we lead with the physical-field VRMSE' rewritten (owner): it now says why the paper protocol's single number is a poor headline. Its smaller effective epsilon lets onset give 96% of the released CNextU-Net's all-window mean, and it reverses comparisons: AViT looks better than CNextU-Net at 4 h (0.298 vs 0.349), though CNextU-Net is 1.5x better in developed convection (0.050 vs 0.075). The 'fixed in our evaluation plan before training' argument was removed.
* 2026-11-14: long report, metric appendix: 'Why VRMSE is large during onset, and how the two variants differ' and 'Scoring with physical-field VRMSE' merged into one paragraph under the owner's title (owner). Mechanism, then the consequence (96% vs 87% onset share; can rank models differently, Table headline), then what we report. The repeated mechanism and the twice-stated 'the two agree in developed convection' removed.
* 2026-11-14: long report, Milestone 2: new paragraph 'Budgets for the final submission' (owner). It covers the 1 h and 4 h budgets (the 1 h result is the 4 h run's checkpoint) and the 12 core runs (48 h of active training; six 10% done, six 100% to go). Allocated time per run: pilot prediction 6.0 / 8.0 / 9.7 h, 10% measured 5.2-6.0 / 6.9-7.1 / 11.1-12.2, 100% projected 5.5-6.5 / 7.5-9 / 14-17. Core study about 131-142 allocated GPU-hours, 72.6 used. Numbers from tab:time and tab:budget; no SU (cluster detail, left to the appendix).
* 2026-11-14: long report, Milestone 2 'Budgets for the final submission' shortened to three sentences (owner): the 12 runs (six 10% done); hours held per run (10% measured, pilots, 100% projected); and the core-study total. The 1-hour-checkpoint explanation was dropped, since Setup's Training paragraph has it.
* 2026-11-14: long report, the GPU-hours table (`tab:budget`) moved from the resource appendix into Milestone 2, right after 'Budgets for the final submission' (owner); now Table 3. References in that paragraph and in the Revised Plan now read 'Table', not 'Appendix Table'. The 100% row points to Appendix Table tab:time for its projection. Note: the merge of the two VRMSE paragraphs in the metric appendix is no longer in the file (they are separate again, probably from an editor save); not redone.
* 2026-11-14: long report, Milestone 2 'Budgets for the final submission' cut to two sentences now that Table 3 shows the breakdown (owner): the 12 runs (six 10% done), then the core-study total with pointers to Table 3 and Appendix Table tab:time. The per-run hours (measured, pilot, projected) remain in Appendix Table tab:time and the appendix's storage/pilots paragraph.
* 2026-11-14: long report, Milestone 2: 'Budgets for the final submission' merged into 'Training budget' (owner). One paragraph covers the 12 runs (evaluated at 1 and 4 h; six 10% done), the active-GPU-time clock and why, allocated GPU time, and the core-study total (Table 3).
* 2026-11-14: long report, Milestone 3 Rollouts: one-hour paragraph added (owner); the existing text now says 'After four hours CNextU-Net is the most stable'. At 1 h TFNO is most stable: 60% of rollouts below VRMSE 1 against 34% (CNextU-Net 0.345, AViT 0.335), and it stays ahead of persistence from developed flow for 21 steps (CNextU-Net 14, AViT 12). After 30 steps none beats persistence (TFNO 0.84, AViT 1.05, CNextU-Net 1.07, persistence 0.70). CNextU-Net gains most from 1 to 4 h (1.07 to 0.47). All recomputed from the 4 h runs' rollout_1h/4h outputs and persistence.
* 2026-11-14: long report, Milestone 3: 'Test evaluation', 'Rollouts' and 'Field-wise error' rewritten with inline-labelled findings (a), (b), ... (owner). Test evaluation (a-e): phases separate the models; seed spread; released vs ours; persistence at onset vs developed; and a new 1-hour finding (AViT lowest all-window 0.394, CNextU-Net lowest developed 0.081, CNextU-Net seed spread ±0.176). Rollouts (a-d): CNextU-Net at 4 h; AViT degrades; TFNO at 1 h; CNextU-Net's gain. Field-wise (a-d). Minor details dropped: AViT's 0.80/0.65, 'also on pressure', 'level on w and p'.
* 2026-11-14: long report, Milestone 3: the (a)/(b) labels were removed again (owner did not like them). 'Test evaluation', 'Rollouts' and 'Field-wise error' were rewritten as plain prose that is easier to follow: a lead sentence, then the findings grouped. Rollouts is in three short paragraphs (why rollouts differ, four hours, one hour). All findings were kept, and three details the labelled version had dropped were restored: AViT's 0.80 / 0.65, 'CNextU-Net and AViT also on pressure', and AViT 'level on vertical velocity and pressure'.
* 2026-11-14: long report, Milestone 3 'Physical fidelity': the three model bullets turned into one paragraph after the definition (owner), opening with 'All three models put too much energy at the finest scales, but they differ at larger scales'. Same numbers; the MSE explanation and the caution close it.
* 2026-11-14: long report, Findings section: the bullet list after the choice table was removed (owner); Accuracy, Training time and Cost repeated Milestone 3. Kept as two short paragraphs at the end, because Milestone 3 lacks them: 'Rayleigh number and heat transport' (pointing to the Additional Analysis appendix) and 'Caveats' (10% data and these configurations; seeds share one subset; fp32 favours TFNO; learning rate at the top of the grid). The owner's stray '\label{tab:choice}' in the removed lead-in went with it.
* 2026-11-14: long report, Milestone 3 trimmed to core findings (owner).
  * Rollouts: one paragraph. Dropped AViT's 0.80 / 0.65, the 5.4 from frame 20, the 21 / 14 / 12 steps ahead of persistence, the 1-hour 30-step values and the 'CNextU-Net gains most' sentence.
  * Field-wise: dropped the onset velocity values, 'also on pressure', AViT 'level on w and p', and the 1-hour sentence.
  * Accuracy against cost: dropped 'every run had 4 h', CNextU-Net's 61 GB note, per-frame times, the 1.2x / 1.8x ratios, the 0.47 vs 0.52 trade, and the solver sentence.
  * Physical fidelity: finest-scale excess for all three in one sentence, with the MSE explanation; then per-model differences at larger scales. AViT's 0.97-1.34 and 0.55-21 ranges dropped.
  * 21 pages.
* 2026-11-14: long report, Section 6.1 baseline question is now solver-only (owner): 'Add a numerical-solver baseline', ending 'Should we add this solver baseline to the final report?'. The cheap baselines (linear extrapolation, climatology, DMD) were removed there and in the Revised Plan's Baselines bullet; the now-uncited Schmid 2010 DMD reference is commented out.
* 2026-11-14: long report, Section 6.1: the 'additional physics checks' question was removed (owner). The intro now reads 'Four questions are open for the final submission, each with the context needed to answer it.' (no remaining question has a default).
* 2026-11-14: long report, Section 6.1 question 1 (one-step reporting) also asks whether reporting onset and developed convection separately is a good approach (owner). Context added: the cut is frame 50 for every regime, although convection starts later at low Ra.
* 2026-11-14: long report, Section 6.1: the one-step question was split in two (owner). Question 1 (q:split) asks whether reporting onset and developed separately is good, with the frame-50 cut. Question 2 (q:onestep) asks whether to keep both the split and the all-window mean or lead with one, with the reversal example. Intro now says five questions.
* 2026-11-14: long report: the Issues subsection (intro, tab:issues, fig:time) moved to a new Appendix F 'Known Issues' (app:issues), after Resource Accounting (owner); Additional Analysis is now G. Not titled 'Accepted issues', since two of the three rows are still Open. Section 6 is now 'Decisions for the Final Submission' (label sec:decisions; no subsections), and its intro points to Appendix F for issues that need no decision. 20 pages.
* 2026-11-14: long report, Revised Plan 'Pipeline check' bullet now ends by pointing to Section 6, question 5 (full-length TFNO): whether a working full-data TFNO reference is trained depends on it (owner).
* 2026-11-14: long report, Revised Plan now opens with 'Plan for the final report', the full plan from the abstract (owner): core study (12 runs, shared recipe, 1 and 4 h; six 10% done), evaluation, expected outcome (accuracy-cost and data-scaling curves, physical fidelity, negative results), the extensions in priority order, and the backup plan. Then 'Changes since the abstract' (the existing bullets). The 'Core resource estimate' bullet was shortened so the section fits on one page. The question list in the Evaluation bullet was put in order (1, 2 and 3), with its grammar fixed. 20 pages.
* 2026-11-14: long report, no page number on page 1 (\thispagestyle{empty} after \begin{document}; owner). Numbering continues from 2 on page 2.
* 2026-11-14: long report, page numbering starts at 1 on the second page (owner): \pagenumbering{gobble} on the title page, \pagenumbering{arabic} after the \newpage before Setup. The title page is unnumbered, Setup is page 1, and the contents list follows (Revised Plan 11, Additional Analysis 18); no duplicate-anchor warnings.
* 2026-11-15: long report, Appendix C (metric details): 'Reported VRMSE' and 'Scoring with physical-field VRMSE' merged into one simpler account, 'Physical-field VRMSE and the paper protocol' (owner): the two variants share the formula and differ only in units; rescaling cancels except for epsilon; then why the paper's all-window mean is onset-dominated (96%), the AViT-vs-CNextU-Net reversal (0.298 vs 0.349; 0.075 vs 0.050 developed), and that the physical-field VRMSE only reduces the effect (87%). `tab:units` gains two velocity rows for the released CNextU-Net (u / w, frames 0-9: 2.84 / 3.42 physical, 12.49 / 15.17 paper; developed: 0.030 / 0.034 both), recomputed from its study evaluation. Still 20 pages.
* 2026-11-15: long report, Appendix C 'Physical-field VRMSE and the paper protocol' shortened from 16 to 11 lines (owner); same facts and numbers.
* 2026-11-15: long report, Setup 'Additional Measures' ends with a sentence on the two further analyses in Appendix G (error by Rayleigh number, heat transport / Nusselt number), being considered for the final report (owner).
* 2026-11-15: long report, Revised Plan: ConvLSTM removed from the extensions (now noise robustness, a third seed, 30% data) and listed under 'Changes since the abstract' as dropped, since there is not expected to be enough time to add another model (owner).
* 2026-11-15: long report, Revised Plan: the third seed and the 30% data fraction also dropped (owner). Plan now keeps one extension, input-noise robustness (evaluation only); the owner's 'Additional model dropped' bullet in 'Changes since the abstract' became 'Extensions dropped' (third seed, 30% data, ConvLSTM; not enough resources).
* 2026-11-15: long report, Appendix F time table (`tab:time`) now spans the full line width like the recipe table (owner): `tabularx` with a flexible label column and three centred 2.3 cm number columns. Still 20 pages.
* 2026-11-15: long report, issues table (`tab:issues`, Encountered Issues appendix): Status column removed (owner); Issue column widened; caption now 'Issues met so far and their effect.'; the intro's 'and how we handle them' dropped to match. The removed status text (node-local staging before the 100% runs, about a day of queue slack, fp32 accepted as a caveat) is not elsewhere in the main text.
* 2026-11-15: long report, Section 6 question 'Way to report the one-step error' now says plainly how the paper's number and ours differ (owner): the paper gives one all-window mean on rescaled fields; we compute the same VRMSE on the stored fields (onset weighs less; Appendix C) and report it over all windows and for onset and developed convection separately, with the paper's number wherever we compare with the paper.
* 2026-11-15: long report, Section 6 question 2 (`q:onestep`) rewritten as 'Physical-field or paper-protocol VRMSE' (owner): the paper scores rescaled fields, ours the stored fields; the two differ only through epsilon (16-45 times smaller relative to rescaled fields), so onset scores explode under the paper protocol (2.51 vs 0.77, 96% vs 87% onset share), developed convection agrees, and the AViT/CNextU-Net order reverses (0.298 vs 0.349; 1.5x in developed). Asks whether the main results should use the physical-field VRMSE and the paper protocol only to compare with the paper. Question 1 (onset/developed split) unchanged. Trimmed to nine lines so Table 9 stays on the Section 6 page; still 20 pages.
* 2026-11-15: long report, Section 6: new question 5 (`q:physics`), 'Checking the physics of the predictions' (owner): the Nusselt number is our physics check besides the spectrum and ranks models differently from VRMSE (TFNO closest over 30 steps, AViT too little heat); further checks (divergence-free velocity, fixed wall temperatures and zero wall velocity) need only a re-evaluation of the 10% runs (about 4 GPU-hours). Asks whether Nu is a good measure and whether to add other physical constraints. With the rollout-training question commented out by the owner, the intro's 'Five questions' is correct again. Table 9 now sits at the top of the Revised Plan page; still 20 pages.
* 2026-11-15: long report, Section 6 question 5 (`q:physics`) reworded to separate the two kinds of physics check (owner): the Nusselt number compares a physical quantity with the truth (accuracy), while divergence-free velocity and the wall conditions would test physical consistency without the truth. Asks whether Nu is a good measure of physical fidelity and whether to add such consistency checks. Still 20 pages.
* 2026-11-15: long report, Section 6 question 5 (`q:physics`) simplified to 'Physics checks' (owner): besides VRMSE, the heat transport (Nusselt number) checks which model is closest to the actual physics and ranks them differently (TFNO closest over 30 steps, CNextU-Net lower VRMSE); asks whether this makes sense and which further physics checks to add. The candidate checks (divergence, wall conditions) and the cost estimate were removed. Still 20 pages.
* 2026-11-15: long report, Section 6 question 5 closing sentence made more formal (owner): 'Is the Nusselt number an appropriate measure of physical fidelity for this comparison, and which further physical diagnostics, if any, should the final report include?'
* 2026-11-15: long report, Section 6 question 3 (`q:split`) closing sentence made more formal (owner): 'Is reporting the two phases separately, with a single cut at frame 50, appropriate for the final report, or should the cut depend on the Rayleigh number?'
* 2026-11-21: professors' answers to the mid-project report's five questions (D28): bounded solver baseline (2–3
  regimes, coarse grids, match CNextU-Net, compare cost); Nusselt per Rayleigh number as time-averaged Nu over the
  rollout against the truth; no full-length TFNO; onset cut per trajectory from the kinetic energy, frame 50 as a
  check; physical-field VRMSE as the main metric. Plan for the final submission (due 2026-12-11) written in
  `docs/final_plan.md`: work packages WP0–WP8, timeline, compute and the owner's open decisions (section 6). Before
  the 100% runs: commit the pending work, keep `reports/` by-products out of the HPC3 tree (dirty-tree check),
  add read retries, and check whether the A100 nodes' local disk can hold the training split.
* 2026-11-21 (later): owner's answers (D29). The 100% runs go to the paid partition with the class account; the
  24 h replication is dropped; the onset and solver designs are confirmed; correlation is on where it costs
  nothing; noise robustness is included (evaluation only); the report format is decided later, and for now
  results are saved. New: uncertainty quantification following Lecture 17 (D30, `docs/final_plan.md` WP9). It
  has two parts: a 5-member CNextU-Net deep ensemble with a variance output (β-NLL, 1 h, 10% and 100%), and
  conformal intervals and seed-pair spread for all three families. Open: the GPU nodes' local disk size
  (`sinfo`, owner), and approval of the UQ and noise GPU cost (~1,300–1,950 SU of the ~3,700–4,700 still to
  spend).
* 2026-11-21 (evening): owner approved the GPU cost of noise robustness and UQ (WP8, WP9) and gave the go-ahead for
  push 1. `sinfo`: the four A100 nodes have 1.83 TB of local disk each, so the 100% runs copy the training split
  there (D29, closes D23's open item). Push 1 code:
  * `PersistentWellDataset` retries storage errors (8 retries, back-off 10 s up to 300 s, about 20 min in all).
  * `rbbench stage-data` copies the split to node-local disk. Other entries are symlinked, sizes are checked
    against the download record, and samples are identical (tested).
  * `cluster/run_with_restarts.sh` restarts a crashed run inside the allocation, at most twice. It forwards
    USR1/TERM and deletes the copy at exit. `train_a100.sbatch` execs into it, and `STAGE_DATA=1` turns the copy on.
  * `rbbench onset-frames` and `onset_cpu.sbatch` (WP4).
  * The solver's bounded design is now the default (`--design bounded`, six grids, 15 tasks per grid; WP6).
  * The LaTeX by-products are ignored.
  * The submitted check-in PDF is committed as `reports/project_check_in.pdf`, replacing `midproject.pdf`.
  * Tests: 172 pass and smoke passes. The core manifest regenerates identically (run ids and hashes). The
    commands to submit are in `cluster/README.md` section 10.
