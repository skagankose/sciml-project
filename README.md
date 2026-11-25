# Data–Compute Trade-offs in Neural PDE Forecasting

This project (package `rbbench`, repository `well-rb-four-families`) compares how accurately and efficiently three
neural-network families forecast Rayleigh–Bénard convection when training data and compute are limited. The models
are CNextU-Net (convolutional), TFNO (spectral) and AViT (attention), all trained from scratch with The Well's
implementations on its `rayleigh_benard` dataset (512 × 128, four fields).

It is a graduate project for an ML-for-Science class (final report due 2026-12-11). The report is
`reports/final_acl.tex` / `reports/final_acl.pdf` (two columns, ACL conference style; the earlier one-column
version is in `reports/deprecated/`); every number in it is recorded, with its source, in
`docs/report_notes.md`.

## The study

* **Core study:** 3 models × 2 data fractions (nested 10% and 100% of the training trajectories) × 2 seeds = 12 runs.
  Each trains for 4 hours of active GPU time on one A100 with one shared recipe. Each run is evaluated at its 1 h and
  4 h checkpoints, the two compute budgets.
* **AViT corrected (D32):** The Well's 2-D AViT attends only along y (reported upstream, PolymathicAI/the_well
  issue #111). The report's AViT is `avit_fixed` (`tasks/train_avit_fixed.jsonl`: same recipe, attention along both
  axes); The Well's AViT as released (the AViT rows of `tasks/train.jsonl`) is compared in the report's Appendix C.
* **Pipeline check:** The Well's released CNextU-Net and TFNO checkpoints, scored by our pipeline, reproduce the
  paper's test VRMSE (0.6688 and 0.6634 against 0.6699 and 0.6566).
* **References:** persistence (the last input frame), the released CNextU-Net, and a numerical solver (Dedalus)
  restarted on coarser grids.
* **Evaluation:** one-step and 30-step rollout VRMSE per field, split at each trajectory's own convection-onset frame;
  failure horizons; kinetic-energy spectra; heat transport (Nusselt number) per Rayleigh number; GPU-hours, inference
  time and memory.
* **Extensions run:** input-noise robustness (evaluation only) and uncertainty quantification (a five-member
  β-NLL deep ensemble, seed pairs, split conformal intervals).
* **Not run:** a third seed, a 30% fraction and the ConvLSTM family (implemented and tested, but dropped for lack of
  compute); a 24-hour replication and a full-length TFNO.
* **Compute used:** 172.5 GPU-hours (154.1 on the paid partition), 6,156 service units for the whole project, from
  Slurm accounting; 26.1 GPU-hours (1,045 SU) of it for the corrected AViT.

Main findings (details and caveats in the report):

* With its attention corrected, AViT is the most accurate and stable forecaster at both fractions and both budgets,
  and the closest in heat transport (median Nusselt error 5–7%); CNextU-Net is second and the most robust to noisy
  inputs. The Well's AViT as released was the least stable model.
* Ten times more data helps little at these budgets (−2.8% to −4.9% one-step error in developed convection); four
  hours instead of one helps more (up to −35%).
* Persistence still wins during onset.
* A coarse solver grid matches CNextU-Net at Ra ≤ 1e8, but matching AViT at Ra = 1e8 takes the full grid; training
  pays off after about 1,500 to 86,000 forecasts.
* Conformal intervals scaled by an ensemble's spread stay calibrated along 30-step rollouts, and their coverage is
  much more even across windows and Rayleigh numbers than that of constant-width intervals.

## Setup (Mac)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt -r requirements-common.txt
pip install -e . --no-deps
```

Do not download the dataset to the Mac (367 GB). The tests and the smoke run use a small synthetic dataset in the
same HDF5 format.

## Checks before every push

```bash
pytest -q                                   # about a minute; CUDA tests skip on the Mac
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

## Reproducing the results

1. **Runs (UCI HPC3).** Training, evaluation and every other job run on HPC3 under Slurm. `cluster/README.md` is the
   step-by-step runbook: environment, data download and audit (sections 1–3), the pipeline check (3b), the
   learning-rate search (4a), the core sweep (5), evaluation (6), the solver baseline (7), flow snapshots (8) and
   the final-submission jobs (10, 11). Code is changed on the Mac, tested, committed and copied with
   `cluster/push.sh`; production runs refuse a dirty tree.
2. **Frozen inputs.** The split and subset manifests, the evaluation cohort, the onset frames and every task manifest
   are committed in `tasks/`. Each run's configuration and subset hash are part of its run id, so a rerun with the
   same commit reproduces the same run ids (`tests/test_committed_subsets.py` checks the subsets).
3. **Results back to the Mac.** `HPC3_USER=<UCInetID> bash cluster/pull.sh` copies the outputs without checkpoints
   or raw data into `outputs/` (not in git).
4. **Tables and figures (Mac, no data, no GPU):**

   ```bash
   python -m rbbench aggregate --tasks tasks/train.jsonl tasks/train_avit_fixed.jsonl \
       --tables outputs/tables --figures outputs/figures                           # accuracy and resource tables
   python scripts/findings.py          # outputs/findings/ and outputs/findings/uq/: comparison, Nusselt, solver,
                                       # uncertainty, noise and cost tables, exploratory figures
   python scripts/final_figures.py     # reports/figures/final/fig_*.pdf at printed size
   ```

   `bash reports/sync_figures.sh` runs the last two, plus the mid-project figures.
5. **Report:** `cd reports && tectonic --keep-logs final_acl.tex` (`acl.sty`, `acl_natbib.bst` and
   `final_acl.bib` are in `reports/`, and tectonic runs BibTeX).

## Commands

`python -m rbbench --help` lists all commands:

| Command | Purpose |
|---|---|
| `models` | parameter counts (`--size full` or `--size tiny`) |
| `inspect-dataset` | dataset audit JSON; gates the physics diagnostics |
| `make-subsets`, `make-cohort`, `make-sweep` | frozen manifests in `tasks/` |
| `train`, `train-task` | one run (explicit or by task index); resumable, budgeted on active GPU time |
| `stage-data` | copy a split to node-local disk for one job |
| `eval-task`, `eval-baseline` | evaluate a run's milestone checkpoints; persistence or a released checkpoint |
| `eval-released` | pipeline check: The Well's released CNextU-Net and TFNO against their published VRMSE |
| `compare-paper` | a trained run against the paper, measured the same way |
| `select-lr` | equal-budget learning-rate search: each model's rate (validation only) and pilot timings |
| `onset-frames` | each trajectory's onset frame from its true kinetic energy |
| `eval-solver` | score the Dedalus solver rollouts with the models' metrics |
| `eval-uq`, `make-uq-ensembles` | uncertainty and noise-robustness evaluation of an ensemble |
| `profile`, `bench-data` | A100 throughput, memory and latency; data-loading throughput |
| `aggregate` | tables and figures from copied outputs (runs on the Mac) |
| `verify-download`, `check-gpu`, `fixture`, `smoke` | infrastructure |

Paths come from `DATA_ROOT` (the directory containing `datasets/`) and `OUTPUT_ROOT` (default `./outputs`).
Nothing in `src/` hard-codes a user or machine path. `make-sweep` still defaults to the earlier four-model,
10/30/100%, three-seed grid; the runbook passes the core grid explicitly.

## Repository

```text
configs/   common.yaml; models/{cnext_unet,tfno,avit,convlstm}.yaml;
           experiments/{main,lr_search,uq,noise,replication,pilot,smoke,heldout}.yaml
src/rbbench/   data, stage, models/, train, lr_search, rollout, evaluate, metrics/{well_metric,physics}, noise,
               onset, uq, uq_analysis, solver_grid, solver_dedalus, solver_eval, subsets, sweep, audit, profile,
               aggregate, visuals, download, manifests, released, cli; vendor/openstl (ConvLSTM cell, verbatim)
scripts/   findings.py, uq_findings.py, final_figures.py, report_figures.py (results and figures);
           solver_rollouts.py, visualize_rollouts.py (HPC3 jobs); probes and thin CLI wrappers
cluster/   env.sh, setup_env.sh, setup_dedalus.sh, push.sh, pull.sh, run_with_restarts.sh, slurm/*.sbatch
tests/     models, data layout, subsets, normalisation, rollout, VRMSE, physics, noise, onset, resume, budget,
           uncertainty, solver, aggregation and figures, pipeline, CUDA
tasks/     frozen manifests (generated on HPC3 from the real data, then committed)
docs/      report_notes (every result), final_plan, decisions, methodology, architectures, upstream audit,
           results schema, abstract alignment
reports/   final_acl.tex / final_acl.pdf (final report, two columns, ACL style), the mid-project report and the
           abstract; figures/ (final/ for the final report); deprecated/ (the one-column final report)
```

Project status, the gates between phases and the full progress log are in `AGENTS.md`; design decisions in
`docs/decisions.md`.

## Reference numbers (not targets)

On `rayleigh_benard`, The Well's 12 h H100 benchmark reports a test one-step VRMSE of 0.6699 for CNextU-Net and
0.6566 for TFNO; there are no published numbers for AViT. Scored in normalised units as the paper did, our
pipeline gives 0.6688 and 0.6634 for the released checkpoints (the released TFNO with the paper-era forward pass
it was trained with, decisions D20). This project's budgets are 1 h and 4 h on an A100, so its own models are not
expected to match the 12-hour reference: our 100% CNextU-Net is 1.28 times the released model's one-step error in
developed convection, but more stable over rollouts.
