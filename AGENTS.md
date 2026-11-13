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
