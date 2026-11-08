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

Our trained models must be at least as good as the paper's (decisions D20). The replication runs are two core
runs, CNextU-Net and TFNO at 100% data, seed 0, so they cost no extra GPU time:
1. Profile batch sizes up to The Well's 64 (step 4 below, with `BATCH_SIZES="8 16 32 64"`); the batch size
   is then frozen in `configs/common.yaml` and the 12-row core manifest is generated (step 5).
2. Train the two rows first: `sbatch --array=<cnext row>,<tfno row> cluster/slurm/train_a100.sbatch`.
3. Compare each with the paper (after `eval-released`, step 3b):
   ```bash
   sbatch --array=<cnext row>,<tfno row> --export=ALL,LABEL=4h cluster/slurm/paper_compare_a100.sbatch
   tail -n 4 logs/rb4-paper-*.out
   ```
   It ends with `AT LEAST AS GOOD AS THE PAPER` or `WORSE THAN THE PAPER`. The result is written to
   `outputs/runs/<run_id>/evaluations/paper_compare_4h.json`.

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

## 5. Main sweep **[gate: explicit owner approval; 12 core jobs × ~4.5 h]**

`make-sweep` still defaults to the earlier grid (`docs/abstract_alignment.md` F2), so pass the core grid
explicitly:

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
* The paid `gpu` partition (34 SU/GPU-hour, ≥ 1,632 SU for the core sweep) requires
  `sbatch -p gpu -A <lab>_gpu …` and the owner's permission.

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
