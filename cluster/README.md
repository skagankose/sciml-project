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
* about 0.25 GB per checkpoint × 3 per run × 36 runs ≈ 27 GB;
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

## 1. Environment (once)

```bash
ssh hpc3
srun -c 4 -p free --mem=16G --time=02:00:00 --pty /bin/bash -i
cd /pub/$USER/projects/well-rb-four-families
bash cluster/setup_env.sh                        # torch 2.8.0+cu128 and the shared pins; writes env/hpc3-*
exit
mkdir -p logs && sbatch cluster/slurm/check.sbatch   # A100: tests incl. CUDA tests at 512x128, smoke on CUDA
```

The check job must print `CHECK PASSED` on an A100. Look at its log; `sbatch` accepting the job proves
nothing. Then pull the environment snapshot and commit it on the Mac:
`bash cluster/pull.sh --env && git add env/ && git commit`.

## 2. Data (once; CPU only)

```bash
sbatch cluster/slurm/download_cpu.sbatch         # preemptible; resubmit to resume
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

## 4. Feasibility on real data **[gate: owner approval of pilot GPU time]**

```bash
sbatch cluster/slurm/profile_a100.sbatch         # batch 1,4,8,16 x fp32/bf16 x 4 models + upstream comparison
python -m rbbench make-sweep --experiment pilot --fractions 0.1 --seeds 0 --out tasks/pilot.jsonl  # on the Mac
sbatch cluster/slurm/pilot_a100.sbatch           # kill/resume/skip + evaluation per model, short budgets
```

Report to the owner:
* `outputs/pilots/feasibility.csv`: steps/s, peak GB and estimated steps in 2 h / 6 h;
* the pilot logs;
* resume and skip behaviour.

Then freeze batch size, learning rate and precision (decisions D6–D8) and record them in `docs/decisions.md`.

## 5. Main sweep **[gate: explicit owner approval; 36 jobs × ~7 h]**

```bash
python -m rbbench make-sweep --experiment main --out tasks/train.jsonl    # on the Mac, commit, push
mkdir -p logs && sbatch cluster/slurm/train_a100.sbatch                  # array 0-35%1, free-gpu, A100
```

* Watch it with `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS` and `seff <id>`. Do
  not poll the queue in a loop.
* Exit code 3 means the job was preempted or hit its time limit with state saved. Resubmit the same array:
  finished tasks skip and the rest resume.
* The paid `gpu` partition (34 SU/GPU-hour, ≥ 7,344 SU for the sweep) requires
  `sbatch -p gpu -A <lab>_gpu …` and the owner's permission.

## 6. Evaluation and results

```bash
sbatch --array=0-35%2 cluster/slurm/eval_a100.sbatch
sbatch --array=0 --export=ALL,BASELINE=1 cluster/slurm/eval_a100.sbatch   # persistence
# Mac:
HPC3_USER=<UCInetID> bash cluster/pull.sh        # outputs without checkpoints
python -m rbbench aggregate                      # tables/ and figures/
```

Keep the checkpoints on HPC3 until the figures are final. Copy anything irreplaceable to CRSP or the Mac,
because `/pub` is not backed up.
