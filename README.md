# well-rb-four-families

This project compares three neural PDE surrogates, all trained from scratch, on The Well's `rayleigh_benard`
dataset (512×128, 4 fields). The three models are CNextU-Net (convolutional), TFNO (spectral) and AViT
(attention). A fourth, recurrent family (ConvLSTM) is implemented as an extension.

The core study is run under fixed compute and data budgets:
* **Budgets:** one 4 h training run on one A100 per configuration, with checkpoints at 1 h and 4 h.
* **Data:** nested 10% and 100% subsets of the training trajectories.
* **Seeds:** 2 per configuration, which gives 3 × 2 × 2 = 12 runs.
* **Pipeline check:** before training, The Well's released CNextU-Net and TFNO checkpoints are evaluated and
  compared with their published VRMSE.

The evaluation covers:
* one-step and rollout VRMSE;
* failure horizons;
* horizontal kinetic-energy spectra;
* resource use;
* Nusselt numbers, only if the dataset audit passes.

Extensions, if time and compute allow, in priority order:
1. input-noise robustness (evaluation only);
2. a third seed;
3. a 30% data fraction;
4. ConvLSTM.

This is a graduate project for an ML-for-Science class. The check-in is on 2026-11-12 and the final report is
due 2026-12-11.

The study design follows the final project abstract (2026-10-30), and so do the code's budgets (1 h/4 h) and
evaluation (clean only). `make-sweep` still defaults to the earlier 4-model, 10/30/100%, 3-seed grid, so the
runbook passes the core grid explicitly. What remains open is listed in `docs/abstract_alignment.md`.

* Development and smoke tests run on the Mac, and all reported runs run on UCI HPC3. The step-by-step
  cluster workflow is in `cluster/README.md`.
* The protocol and its open decisions are in `docs/methodology.md` and `docs/decisions.md`.
* The pinned upstream code and model configurations are in `docs/upstream_audit.md`.
* The output layout is in `docs/results_schema.md`.
* Where the final abstract differs from the code, and what is deferred, is in `docs/abstract_alignment.md`.
* Status and the gates between phases are in `AGENTS.md`.

## Setup (Mac)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt -r requirements-common.txt
pip install -e . --no-deps
```

Do not download the dataset to the Mac. The tests and the smoke run use a small synthetic dataset in the
same HDF5 format.

## Check before every push

```bash
pytest -q                                   # ~15 s; CUDA tests skip on the Mac
python -m rbbench smoke --device cpu --synthetic --grid 64 32 --steps 3 --all-models
```

The `smoke` command runs the real pipeline end to end on tiny models:
1. synthetic data;
2. dataset audit;
3. split and subset manifests;
4. frozen cohort;
5. task file;
6. training with milestone checkpoints;
7. evaluation of both checkpoints, clean and noisy, with physics;
8. persistence baseline;
9. aggregation into tables and figures.

## Commands

`python -m rbbench --help` lists all commands:

| Command | Purpose |
|---|---|
| `models` | parameter counts (`--size full|tiny`) |
| `inspect-dataset` | dataset audit JSON; gates the physics diagnostics |
| `make-subsets`, `make-cohort`, `make-sweep` | frozen manifests in `tasks/` |
| `train`, `train-task` | one run (explicit or by task index); resumable, budgeted |
| `eval-task`, `eval-baseline` | evaluate the milestone checkpoints / persistence |
| `eval-released` | pipeline check: The Well's released CNextU-Net/TFNO checkpoints vs their published VRMSE |
| `compare-paper` | one of our trained runs vs the paper: at least as good, measured the same way |
| `profile` | A100 throughput, memory and latency (`outputs/pilots/feasibility.csv`) |
| `aggregate` | `tables/` and `figures/` from copied outputs (runs on the Mac) |
| `verify-download`, `check-gpu`, `fixture`, `smoke` | infrastructure |

Paths come from `DATA_ROOT` (the directory containing `datasets/`) and `OUTPUT_ROOT`. Nothing in `src/`
hard-codes a user or machine path.

## Repository

```text
configs/   common.yaml, models/{cnext_unet,tfno,avit,convlstm}.yaml, experiments/{main,pilot,smoke,noise,heldout}.yaml
src/rbbench/   data, models/, train, rollout, evaluate, metrics/{well_metric,physics}, noise, subsets, sweep,
               audit, profile, aggregate, download, manifests, released, cli; vendor/openstl (ConvLSTM cell, verbatim)
scripts/   download_rb.sh, compare_upstream.py, thin CLI wrappers
cluster/   env.sh, setup_env.sh, push.sh, pull.sh,
           slurm/{setup_cpu,check,download_cpu,released_a100,paper_compare_a100,profile_a100,pilot_a100,train_a100,eval_a100}.sbatch
tests/     model, layout, subset, normalization, rollout, VRMSE, physics, noise, resume, budget, aggregate, pipeline, CUDA
tasks/     frozen manifests (generated on HPC3 from the real data, then committed)
docs/      methodology, decisions, upstream audit, results schema
```

## Reference numbers (not targets)

On `rayleigh_benard`, The Well's 12 h H100 benchmark reports a test one-step VRMSE of 0.6699 for CNextU-Net
and 0.6566 for TFNO. There are no published Well RB numbers for AViT or ConvLSTM. This project's budgets are 1 h
and 4 h on an A100, so its trained models are not expected to match. The released checkpoints themselves should
match; that is the pipeline check.
