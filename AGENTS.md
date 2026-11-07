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
| C HPC3 env | `check.sbatch` passes on a real A100; no env/caches in `$HOME` | env built 2026-11-07 (`env/hpc3-*` committed); A100 check submitted, result pending |
| D Data | 3 splits verified (`rbbench_complete_*.json`); audit physics checks pass | download submitted 2026-11-07 (valid, test, then train) |
| D2 Pipeline check | The Well's released CNextU-Net and TFNO checkpoints reproduce their published one-step VRMSE on our data path (within 5%) | `eval-released` + `released_a100.sbatch` done and tested offline; checkpoints verified on the Mac; needs verified valid/test; **owner approval before GPU time** |
| E Subsets | S10 ⊂ S30 ⊂ S100, no leakage, counts per regime, identical hashes on rerun | code + tests done; real manifests not generated |
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
