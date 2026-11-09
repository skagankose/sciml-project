# HPC3 runbook

This follows the user's HPC3 guide: develop on the Mac, push code through rsync (or git), download data on
HPC3, run every reported job on an A100, and pull compact results back. Every step that spends GPU time is
marked with **[gate]**. Do not run a step past a gate without the owner's approval.

Layout on HPC3:

```text
/pub/$USER/projects/well-rb-four-families/   code (pushed from the Mac), outputs/, logs/
/pub/$USER/data/the_well/datasets/rayleigh_benard/   ~367 GB (train 293.6, valid 36.7, test 36.7 GB)
/pub/$USER/conda/envs/well-rb/               conda env
/pub/$USER/.cache/                           pip/matplotlib/torch caches (never $HOME)
```

Free space needed on `/pub/$USER` (1 TB quota):
* 367 GB of data;
* about 0.25 GB per checkpoint × 3 per run × 12 core runs ≈ 9 GB;
* evaluation outputs;
* the environment, about 6 GB.

Check with `df -h /pub/$USER` before downloading.

## 0. Mac (every change)

```bash
source .venv/bin/activate
pytest -q
python -m rbbench smoke --device cpu --synthetic --grid 64 32 --steps 3 --all-models
git add -A && git commit -m "..."
HPC3_USER=<UCInetID> bash cluster/push.sh        # VPN + DUO; uses the hpc3-xfer host
```

The transfer host `hpc3-xfer` accepts only `rsync`/`scp`. Run commands such as `ls`, `mkdir` or `df` through
`ssh hpc3`. Submit every `sbatch` from the project folder: the job scripts use paths relative to it.

## 1. Environment (once)

```bash
ssh hpc3
cd /pub/$USER/projects/well-rb-four-families
mkdir -p logs && sbatch cluster/slurm/setup_cpu.sbatch   # free CPU node, ~30 min; writes env/hpc3-*
tail -f $(ls -t logs/rb4-setup-*.out | head -1)          # ends with "done; commit env/hpc3-*"
sbatch cluster/slurm/check.sbatch                        # A100: tests incl. CUDA tests at 512x128, smoke on CUDA
```

The install runs as a batch job because interactive `srun` sessions were too slow to use. `archs []` in its
output is normal on a CPU node. The check job must print `CHECK PASSED` on an A100. Look at its log; `sbatch`
accepting the job proves nothing. Then pull the environment snapshot and commit it on the Mac:
`HPC3_USER=<UCInetID> bash cluster/pull.sh --env && git add env/ && git commit`.

If the install fails with `ChecksumMismatchError` and "actual sha256 e3b0c442…" (the hash of an empty file),
conda's downloads into `/pub` came back empty. `setup_env.sh` now keeps conda's package cache on the node's
local disk, which fixed this on 2026-11-07. Each failed or stopped install also leaves `*.partial` files that
break the next attempt. Remove them, and any half-built env, before resubmitting:

```bash
module load miniconda3/24.9.2 && source ~/.mycondainit-24.9.2
rm -fv /pub/$USER/conda/pkgs/*.partial
conda env remove -n well-rb -y
```

## 2. Data (once; CPU only)

```bash
sbatch --export=ALL,SPLITS="valid test train" cluster/slurm/download_cpu.sbatch   # preemptible; resubmit to resume
# or interactively on a free CPU node:  source cluster/env.sh && bash scripts/download_rb.sh valid test train
```

A split counts as done only when `$DATA_ROOT/datasets/rayleigh_benard/rbbench_complete_<split>.json` exists.
That file is written after every registry file matches its remote size, opens in h5py, and is SHA-256 hashed.
Start with `valid test` (37 GB each) if you want to begin the audit early.

## 3. Audit, subsets, cohort (once, on a CPU node)

```bash
source cluster/env.sh
python -m rbbench inspect-dataset --out outputs/audit/dataset_audit.json
python -m rbbench make-subsets --out-dir tasks
```

Read the audit before continuing. It must show:
* splits `train`/`valid`/`test`;
* 1400/175/175 trajectories (expected);
* y on Chebyshev nodes;
* wall values b = 1/0 and u = 0;
* `physics_checks.passed = true`.

If any of these fail, stop and decide; do not loosen tolerances silently.

Then freeze the rollout starts on validation (decisions D15/D16):
1. Edit `eval.rollout_starts` and the thresholds on the Mac and commit.
2. Generate the cohort: `python -m rbbench make-cohort --out tasks/eval_cohort.json`.
3. Pull the generated manifests back with `bash cluster/pull.sh --tasks`, commit them, and push.

## 3b. Pipeline check with released checkpoints **[gate: owner approval of GPU time]**

This needs only verified `valid` and `test` splits; it does not need the audit or the cohort. It evaluates
The Well's released CNextU-Net and TFNO RB checkpoints on our data path and compares their one-step VRMSE with
the published values (decisions D20):

```bash
mkdir -p logs && sbatch cluster/slurm/released_a100.sbatch    # ~30-60 min on one GPU
tail -n 12 $(ls -t logs/rb4-released-*.out | head -1)
```

The log lists, for each model and split:
* our VRMSE;
* the paper-protocol VRMSE next to the published value (decisions D20; the released TFNO runs with the
  forward it was trained with);
* whether ours ≤ published.

It ends with `PIPELINE CHECK PASSED` (ours ≤ published everywhere) or `PIPELINE CHECK: FAILED`. Details are in
`outputs/baselines/released/released_check.json`. Report a failure to the owner before training anything.

## 3c. Replication: our own CNextU-Net and TFNO vs the paper **[gate: owner approval of GPU time]**

Before the study, our training code must produce models at least as good as the paper's (owner, 2026-11-08;
decisions D20). The replication trains each model as long as the paper did, which is not the study's budget:
* 24 h of GPU training time (D24) on one A100, about 14 and 37 epochs. The paper's best checkpoints came at epochs 12
  and 31, which is about 20 h on an A100.
* Checkpoints at 6, 12, 18 and 24 h.
* The paper's learning rates, batch 64 and fp32.
* Its own manifest `tasks/replication.jsonl` (row 0 CNextU-Net, row 1 TFNO), never aggregated with the study.

About 50 GPU-h on `free-gpu` (no SUs), outside the study's ~100 GPU-h.

```bash
sbatch --array=0,1 --time=30:00:00 --export=ALL,MANIFEST=tasks/replication.jsonl cluster/slurm/train_a100.sbatch
```

The two run in parallel when two A100s are free. If one is preempted or reaches its time limit, resubmit
the same command; finished runs skip and the others resume. If `sbatch` rejects `--time=30:00:00`, use the
partition's maximum.

Before training, check that data loading keeps up (D23). This is CPU only, about 10–20 minutes:

```bash
sbatch -p free -c 16 --mem=64G --time=01:00:00 -J rb4-bench -o logs/%x-%j.out --wrap 'source cluster/env.sh && python -m rbbench bench-data --batch-size 64 --batches 15 --workers 4 8 15 --out outputs/pilots/bench_data.json'
```

When both runs have finished, compare each with the released checkpoint scored identically (step 3b):

```bash
sbatch --array=0,1 --export=ALL,MANIFEST=tasks/replication.jsonl,LABEL=24h cluster/slurm/paper_compare_a100.sbatch
tail -n 5 logs/rb4-paper-*_0.out logs/rb4-paper-*_1.out
```

* The run must be at least as good on valid and test, in physical and in normalised units.
* The published values are printed for context.
* Each log ends with `AT LEAST AS GOOD AS THE PAPER'S MODEL` or `WORSE THAN THE PAPER'S MODEL`. The result is
  in `outputs/runs/<run_id>/evaluations/paper_compare_24h.json`.
* The paper chose its checkpoint on validation. `LABEL=12h` or `18h` scores the earlier checkpoints, if
  validation is needed to choose one.

## 4a. Learning-rate search and pilot timings **[approved by the owner, 2026-11-08]**

Nine runs (3 models × 1e-4, 5e-4, 1e-3), 20 min of GPU time each on the 10% subset, seed 0 (decisions D7,
D24). About 3 GPU-h of training on `free-gpu`; with waits for data, AViT's runs take ~35 min of real time.

```bash
sbatch --array=0-8 --time=01:30:00 --export=ALL,MANIFEST=tasks/lr_search.jsonl cluster/slurm/train_a100.sbatch
```

If a task is preempted (exit code 3) or killed by a storage error (exit code 1 with `Errno 70 Communication
error on send` in its log), resubmit only those indices, e.g. `--array=2,3,6,7`; finished tasks exit at once
and partial ones resume. Never include a task that is still queued or running: two copies of one task would
write the same run folder. `sacct -j <jobid> --format=JobID%16,State,ExitCode,Elapsed` shows each task's state.
`select-lr` can run at any time (CPU only, seconds, on the login node); it chooses a model's rate once all
its rates have finished:

```bash
python -m rbbench select-lr --manifest tasks/lr_search.jsonl
```

It prints each run's validation VRMSE (the selection criterion), loss and timing, the pilot timings (GPU seconds per step, share of time
waiting for data, steps and epochs per budget, real hours needed for 4 h), each model's rate, and the
`model_overrides` block for `configs/experiments/main.yaml`. Results are in `outputs/lr_search/`, with a
Markdown table (`lr_search.md`, VRMSE per field) and two figures (validation VRMSE against the rate; training
loss against GPU time per rate). On the Mac, `cluster/pull.sh` then `OUTPUT_ROOT=outputs python -m rbbench
select-lr` gives the same. The rates go into `main.yaml` on the Mac and `tasks/train.jsonl` is regenerated.

## 4. Feasibility on real data **[gate: owner approval of pilot GPU time]**

```bash
sbatch cluster/slurm/profile_a100.sbatch         # batch 1,4,8,16 x fp32/bf16 x 4 models (ConvLSTM only matters for the extension) + upstream comparison
python -m rbbench make-sweep --experiment pilot --fractions 0.1 --seeds 0 --out tasks/pilot.jsonl  # on the Mac
sbatch cluster/slurm/pilot_a100.sbatch           # kill/resume/skip + evaluation per model, short budgets
```

Report to the owner:
* `outputs/pilots/feasibility.csv`: steps/s, peak GB and estimated steps for the first (1 h) and final (4 h)
  budget (`est_steps_first`, `est_steps_final`);
* the pilot logs;
* resume and skip behaviour.

Then freeze batch size, learning rate and precision (decisions D6–D8) and record them in `docs/decisions.md`.

## 5. Main sweep **[gate: explicit owner approval; 12 core jobs, 4 h of GPU time each, ~4–7 h real]**

First put the search's rates into `configs/experiments/main.yaml` (step 4a). `make-sweep` still defaults to
the earlier grid (`docs/abstract_alignment.md` F2), so pass the core grid explicitly:

```bash
# on the Mac, then commit and push:
python -m rbbench make-sweep --experiment main --models cnext_unet tfno avit \
    --fractions 0.1 1.0 --seeds 0 1 --out tasks/train.jsonl              # 12 rows
mkdir -p logs && sbatch --array=0-11%1 cluster/slurm/train_a100.sbatch   # free-gpu, A100
```

Extensions (a third seed, 30% data, ConvLSTM) are separate manifests and need their own approval.

* Watch it with `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS` and `seff <id>`. Do
  not poll the queue in a loop.
* Exit code 3 means the job was preempted or hit its time limit with state saved. Resubmit the same array:
  finished tasks skip and the rest resume.
* The paid `gpu` partition requires `sbatch -p gpu -A <account> …` and the owner's permission. It charges
  32 SU per GPU-hour plus 1 SU per core-hour of wall time, waits for `/pub` included: 40 SU/h for
  `train_a100.sbatch` (8 cores), 36 SU/h for `released_a100.sbatch` (4 cores). The owner approved the class
  account `eecs195221_class_gpu` for the 10% runs and the pipeline check (2026-11-09):

  ```bash
  sbatch -p gpu -A eecs195221_class_gpu --array=0,1,4,5,8,9 --time=14:00:00 cluster/slurm/train_a100.sbatch
  sbatch -p gpu -A eecs195221_class_gpu --time=04:00:00 cluster/slurm/released_a100.sbatch
  ```

## 6. Evaluation and results

```bash
sbatch --array=0-11%2 cluster/slurm/eval_a100.sbatch
sbatch --array=0 --export=ALL,BASELINE=1 cluster/slurm/eval_a100.sbatch   # persistence
# Mac:
HPC3_USER=<UCInetID> bash cluster/pull.sh        # outputs without checkpoints
python -m rbbench aggregate                      # tables/ and figures/
```

`eval-task` evaluates every milestone of a run (1 h and 4 h) in one pass. The core evaluation is clean only
(`eval.noise_sigmas: [0.0]`). `tables/gpu_hours.csv` sums the measured training attempts and evaluation passes
per run and in total (decisions D22); compare its TOTAL row with `sacct` for the billed time.

Keep the checkpoints on HPC3 until the figures are final. Copy anything irreplaceable to CRSP or the Mac,
because `/pub` is not backed up.
