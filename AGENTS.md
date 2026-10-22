# Instructions for coding agents

The specification is `RB_TheWell_HPC3_Coding_Agent_Implementation_Guide.md`, supplied by the owner. HPC3
conventions follow the owner's "Running Code on UCI HPC3 (Mac for Development)" guide.

Rules:
1. Never start the 36-job sweep, request paid Slurm partitions or spend lab SUs without explicit owner
   approval.
2. Never download the dataset to the Mac.
3. Never use pretrained weights.
4. Never tune on the test set.
5. Never edit code on HPC3: change it on the Mac, test, commit, then push.
6. When an upstream API is uncertain, read the pinned source and add a test.
7. Before every push, run `pytest -q` and `python -m rbbench smoke --device cpu --synthetic --grid 64 32 --steps 3 --all-models`.

## Gates

| Phase | Gate | Status |
|---|---|---|
| A Mac bootstrap | tests + synthetic smoke for all 4 models without data | **passed 2026-10-22** (90 passed, 9 CUDA skipped) |
| B Upstream audit | shapes, parameter counts, random init, gradients; `docs/upstream_audit.md` | **passed 2026-10-22** (Mac CPU; 512×128 forward on CPU) |
| C HPC3 env | `check.sbatch` passes on a real A100; no env/caches in `$HOME` | not started |
| D Data | 3 splits verified (`rbbench_complete_*.json`); audit physics checks pass | not started |
| E Subsets | S10 ⊂ S30 ⊂ S100, no leakage, counts per regime, identical hashes on rerun | code + tests done; real manifests not generated |
| F Training engine | real-data mini-run per model on A100; resume and skip; loss decreases | code + CPU tests done; A100 pilot not run |
| G Feasibility | 4 models: real-data step + 30-step rollout at 512×128 without NaN/OOM; throughput table | not run; **owner approval before pilot GPU time** |
| H Evaluation | metric tests; deterministic noise; physics self-checks on real data | code + synthetic tests done; real-data checks pending |
| I Sweep | 36-row manifest, A100-only, restart idempotence proven | **needs owner approval** |
| J Aggregation | one command → tables and figures, missing runs listed | code done (synthetic) |

Values still PENDING are listed in `docs/decisions.md`: batch size, learning rate, precision, schedule, rollout
starts and the primary threshold.

## Progress log

* 2026-10-22: Pinned The Well `e63dec8`, neuraloperator 0.3.0, OpenSTL `eecf8a3` (ConvLSTM cell vendored) and
  torch 2.8.0. Implemented the full package, configs, scripts, Slurm templates and docs. Mac test suite and
  smoke pass on CPU, and all four tiny models also run on MPS. Measured full-size parameter counts:
  18.58M / 19.28M / 14.69M / 13.34M (ConvLSTM effective; 16.74M allocated). Remote dataset sizes confirmed by
  HTTP HEAD. Next: Phase C on HPC3.
