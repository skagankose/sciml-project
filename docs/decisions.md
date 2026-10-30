# Design decisions

Each entry: what was decided, why, and what would change it. "PENDING" items must be frozen on
validation/pilot data before the main sweep (Phase I gate) and recorded here with the evidence.

**D1 — Thin runner, not the upstream Hydra trainer.** The upstream `Trainer` is epoch-based, keeps no
scheduler/scaler state on resume (known upstream issue), and has no wall-clock budget or subset manifests.
We reuse upstream data loading, normalization, models and VRMSE, and write only the budgeted loop,
checkpointing, rollout bookkeeping and physics diagnostics.

**D2 — Pins.** The Well `e63dec8`, neuraloperator 0.3.0, OpenSTL cell vendored at `eecf8a3`, torch 2.8.0,
Python 3.13 on both machines. See `docs/upstream_audit.md`.

**D3 — Model sizes: matched parameter counts.** CNextU-Net (18.57M) and TFNO (19.28M) use their published
Well RB configs. AViT is widened from hidden 384 (14.69M) to 432 (18.57M) and ConvLSTM from width 128
(13.34M trainable) to 152 (18.79M trainable, 23.59M allocated because OpenSTL's cell holds unused layers), so all
four families have 18.6–19.3M trainable parameters. Owner's decision (2026-10-22), made before any training,
so that capacity is comparable across families. Since the final abstract (2026-10-30) the core study has three
families; the ConvLSTM widening applies only if that extension runs. Equal parameters do not mean equal compute per update; Phase G
records step time, memory and (where computable) FLOPs, and under a fixed time budget the update counts are
reported.

**D4 — Normalization.** Upstream `stats.yaml` (train-split statistics shipped with the dataset), z-score, the
same frozen statistics for every subset and model. Recomputing statistics per subset would make inputs differ
between the data fractions; using the shipped train statistics never touches validation or test data.
To verify on HPC3: that `stats.yaml` matches the train split (the audit records it).

**D5 — Objective.** One-step, teacher-forced, full-state (not residual) prediction, MSE in normalised units
with equal field weights (upstream uses MSE with equal weights). AdamW (β = 0.9/0.999, weight decay 1e-4 as
upstream) on all parameters, no gradient clipping (as upstream).

**D6 — Learning-rate schedule under a wall-clock budget (PENDING confirmation).** Warmup in optimizer steps,
then cosine decay over *cumulative training seconds* towards the final 4 h milestone. The total step count is
unknown in advance, so a step-based cosine cannot end exactly at 4 h. Decaying on time makes the 4 h
checkpoint the end of the schedule; the 1 h checkpoint is an intermediate state of the same schedule, not a
separately tuned 1 h run. (Budgets were 2 h/6 h before the final abstract; the config still has those
values until the code change in `docs/abstract_alignment.md` F3.) The rate depends only on (step, train_seconds), so resume is exact. The pilot
compares this with a constant rate on validation only.

**D7 — Learning rate and batch size (PENDING).** One shared batch size for all models, chosen from the
Phase G memory/throughput table. The final abstract commits to an equal-budget learning-rate search on
validation data before training: one small grid ({1e-4, 5e-4, 1e-3}), the same budget per (model, rate),
selection on validation only. **Deferred by the owner (2026-10-30):** the search is not implemented, and
whether it picks one rate shared by all models or one per model is open. The Well's own RB rates differ
(5e-4 CNextU-Net, 1e-4 TFNO). Per-model rates need per-model overrides in `make-sweep`. Until then
`optim.lr = 5e-4` for every model. If a model needs gradient accumulation to reach the shared batch size, it is
recorded. No test-set tuning.

**D8 — Precision (PENDING).** fp32 by default. bf16 autocast only for models whose A100 pilot shows finite,
matching validation loss; TFNO stays fp32 unless bf16 FFTs are shown to work. Recorded per model.

**D9 — Budget clock.** Per optimizer step, from requesting the batch to the synchronised end of
`optimizer.step()`. Excludes startup, worker spawn, validation and checkpoint writes; those are logged
separately (`overhead_seconds`). DataLoader workers keep prefetching during validation, which slightly
favours data loading after a validation pass; this is small and the same for every model.

**D10 — Milestone checkpoints.** First optimizer step at/after 1:00:00 and 4:00:00 of training time; actual
seconds and overshoot are recorded. `max_steps` (smoke/pilot only) can force milestones, and then
`budget_satisfied=false` is written and aggregation flags the run.

**D11 — Resume.** `last_resume.pt` holds model, optimizer, schedule state, RNG (python/numpy/torch/cuda),
sampler epoch and position, step, train seconds, milestones and validation history. Written atomically every
15 min of wall time, at each milestone, and on SIGUSR1/SIGTERM (Slurm `--signal=B:USR1@300`, `exec` in the
sbatch script). A CPU test shows an interrupted+resumed run equals an uninterrupted one bitwise.

**D12 — Subsets.** Seeded permutation within each (Ra, Pr) group of the train split; each fraction takes a
prefix with largest-remainder counts, which nests S10 ⊂ S30 ⊂ S100 by construction (asserted). With 40
trajectories per regime (expected from file sizes) this gives exactly 4 / 12 / 40 per regime and
140 / 420 / 1400 trajectories in total; to confirm on the real split. The core study uses S10 and S100; S30 is
generated with them, so it nests correctly if the 30% extension runs.

**D13 — ConvLSTM information horizon.** States are zero-initialised in every forward call and the cells unroll
over exactly the 4 input frames, so in rollouts ConvLSTM sees the same 4-frame window as the other models.

**D14 — Evaluation data path.** Each test trajectory is read once through `WellDataset` as a single
4 + 196-frame window, then every one-step window and every rollout start is cut from it in memory. This uses
the upstream parser and normalization while avoiding re-reading ~37 GB per checkpoint.

**D15 — Rollout cohort (PENDING).** All 175 test trajectories × start indices `[20, 80, 140]`, 30 steps.
Very early starts are near the conduction state, where target variance is tiny and VRMSE is ill-conditioned. The
start list must be frozen on the *validation* split before any test evaluation, then `tasks/eval_cohort.json`
is generated once and never changed (its hash is checked).

**D16 — Failure horizon.** First step whose field-mean VRMSE exceeds the threshold; rollouts that never cross
are right-censored, summarised with Kaplan–Meier survival and median. Thresholds 1.0 / 2.0 (primary) with 5.0
as sensitivity; the primary is chosen on validation (PENDING).

**D17 — Noise (extension, evaluation only).** The core evaluation uses σ = 0 only. The extension re-evaluates
the same checkpoints with σ ∈ {0.01, 0.05, 0.10, 0.20} in normalised units, added once to the 4 observed frames
only. Each (trajectory, start, σ, seed, field) draw comes from a stable hash, so it is identical for all models,
batch sizes and worker counts. Grid points lying exactly on a wall are left unperturbed; with Chebyshev-Gauss
nodes there are none (to confirm from the audit).

**D18 — Physics diagnostics.** Chebyshev interpolation through the y nodes gives the spectral-accuracy
derivatives, wall values and exact quadrature (Dedalus discretisation). Nu (volume) = ⟨u_y b − κ ∂_y b⟩ /
(κ Δb / Ly) with κ = (Ra Pr)^−1/2. The independent check is the wall Nu from −∂_y⟨b⟩ at both walls. Evaluation
computes physics only if the dataset audit passes: wall values of b and u, conductive part ≡ 1, Parseval,
and volume vs wall Nu on late-time reference data (median relative difference ≤ 10%). Spectra use an FFT in x
only; y is never treated as periodic.

**D19 — Concurrency.** Slurm array `%1` (one A100 at a time). 12 core runs × ~4.5 h ≈ 2.3 days sequential on
`free-gpu` if there is no queueing or preemption; the six 10% runs alone take ≈ 27 h. Raising concurrency or
using the paid `gpu` partition (≥ 1,632 SU for the 48 GPU-h of core training) is the owner's call.

**D20 — Pipeline check with released checkpoints.** The final abstract evaluates The Well's released
`polymathic-ai/UNetConvNext-rayleigh_benard` and `polymathic-ai/TFNO-rayleigh_benard` before training, and
compares their one-step VRMSE with the published values (CNextU-Net 0.4807 valid / 0.6699 test, TFNO 0.8568 /
0.6566). AGENTS.md rule 3 was amended on 2026-10-30 to allow this: the checkpoints may be loaded only for
evaluation, never to train from. Checked 2026-10-30:
* Both checkpoints exist on Hugging Face (created 2025-03-28), and their configs match `configs/models/*.yaml`
  `full`.
* The pinned benchmark data config sets `use_normalization: True` with no type, which falls back to
  `ZScoreNormalization` on the shipped `stats.yaml`. That is the same normalization as D4.

A mismatch would point to a problem in our data path or metric. The evaluation command and its narrow exception
to the `from_pretrained` guard in `tests/test_backward.py` are not written yet.

**D21 — Backup plan (deferred).** If storage, queue time or pilot timings show the plan is too costly, the
data size is reduced for all models equally.
* Fewer trajectories already works: a smaller fraction through `make-subsets`.
* Downsampled fields is **deferred by the owner (2026-10-30)** and not implemented. x can be strided. A y
  stride leaves a non-Chebyshev grid, which invalidates the y operators and Nusselt but not VRMSE or the
  x-only spectra. AViT needs both sizes divisible by 16.
