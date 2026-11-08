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

## Gates

| Phase | Gate | Status |
|---|---|---|
| A Mac bootstrap | tests + synthetic smoke for all 4 models without data | **passed 2026-10-22** (90 passed, 9 CUDA skipped) |
| B Upstream audit | shapes, parameter counts, random init, gradients; `docs/upstream_audit.md` | **passed 2026-10-22** (Mac CPU; 512×128 forward on CPU) |
| C HPC3 env | `check.sbatch` passes on a real A100; no env/caches in `$HOME` | **passed 2026-11-07** (job 59080323, after the TFNO fp32-only fix; first check 98/99, D8) |
| D Data | 3 splits verified (`rbbench_complete_*.json`); audit physics checks pass | download job 59080236 completed 2026-11-07 (all splits verified by the script). **passed 2026-11-07**: 1400/175/175 trajectories; audit physics checks pass with `physics.y_grid: gauss` (the stored y is a uniform placeholder, D18) |
| D2 Pipeline check | Released CNextU-Net and TFNO through our pipeline score at least as well as published (owner, 2026-11-08) | **passed 2026-11-08** (job 59224204): CNextU-Net 0.234/0.224, TFNO 0.305/0.300 vs published 0.481/0.670, 0.857/0.657; released TFNO with its paper-era forward (D20) |
| E Subsets | S10 ⊂ S30 ⊂ S100, no leakage, counts per regime, identical hashes on rerun | real manifests committed 2026-11-07: nested, no leakage, 4/12/40 per regime; rerun-hash check pending |
| F Training engine | real-data mini-run per model on A100; resume and skip; loss decreases | code + CPU tests done; A100 pilot not run |
| G Feasibility | 3 core models (and ConvLSTM if the extension runs): real-data step + 30-step rollout at 512×128 without NaN/OOM; throughput table | not run; **owner approval before pilot GPU time** |
| H Evaluation | metric tests; deterministic noise; physics self-checks on real data | code + synthetic tests done; real-data checks pending |
| I Sweep | 12-row core manifest (3 models × 10/100% × 2 seeds, 1 h/4 h budgets), A100-only, restart idempotence proven; extensions only with separate approval | **needs owner approval**; budgets are 1 h/4 h, but `make-sweep` defaults still give 36 rows: pass the core grid explicitly |
| J Aggregation | one command → tables and figures, missing runs listed | code done (synthetic) |

Values still PENDING are listed in `docs/decisions.md`: batch size, learning rate, precision, schedule, rollout
starts and the primary threshold. The study design follows the final abstract (2026-10-30). Where the code
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
    * `compare-paper` now judges our runs against the released checkpoint scored identically, not against
      the published values.
