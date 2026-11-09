# Output files

```text
outputs/
  audit/dataset_audit.json              # rbbench inspect-dataset (gates physics diagnostics)
  pilots/feasibility.csv                # rbbench profile (one row per model x precision x batch)
  pilots/compare_upstream.json          # scripts/compare_upstream.py
  lr_search/{lr_search.csv,lr_search.md,pilot_timings.csv,selection.json,
            lr_search_training_loss.{pdf,png},lr_search_val_vrmse.{pdf,png}}   # rbbench select-lr (D7, D24)
  runs/<model>/f<010|030|100>/s<seed>/<subset8>-<config8>/      # run_id
    resolved_config.yaml                # full config; its hash is in the task row
    provenance.json                     # per attempt: git commit/dirty, packages, GPU, Slurm ids, host,
                                        #   wall_seconds and end_status (trained | preempted | failed | error)
    train_metrics.csv                   # event rows: start|resume|train|val|milestone_*|preempted|failed;
                                        #   train_seconds (budget clock, D24) and data_wait_seconds
    checkpoint_<label>.pt               # full state at the milestone step; label = budget name (1h, 4h)
    last_resume.pt                      # latest full state (preemption)
    COMPLETE.json                       # training finished; checkpoint sha256, milestones, overhead
    FAILED.json                         # only if training diverged
    evaluations/
      EVAL_COMPLETE_<label>.json        # label = 1h | 4h; cohort hash, metric version, checkpoint sha,
                                        #   pass {wall_seconds, labels, device} shared by one evaluation pass
      clean_<label>.json                # one-step + rollout + failure + Nusselt + spectra + latency summary
      noise_<sigma>_<label>.json        # noise extension only: the same for each sigma > 0 (no one-step)
      one_step_<label>.parquet          # per test window
      rollout_<label>.parquet           # per trajectory x start x sigma x step
      physics_<label>.parquet           # Nusselt (volume, wall) and log-spectral error per rollout step
      spectra_<label>.parquet           # mean E(k) per regime x sigma x step x {pred, true}
  baselines/persistence/evaluations/    # same files with label "na"
  baselines/released/                   # rbbench eval-released (pipeline check, decisions D20)
    released_check.json                 # pinned checkpoints and forward, ours / paper protocol / published, verdict
    <model>/one_step_<split>.parquet    # per window: the one_step_<label>.parquet columns plus
                                        #   normalised_vrmse_* (the paper's protocol) and raw_input_vrmse_*
                                        #   (v1.0.1 test-set diagnostic); job 59224204 wrote them as
                                        #   paper_valid_vrmse_* and paper_test_vrmse_*
tables/ main_results.csv, per_seed_results.csv, resource_use.csv, gpu_hours.csv, missing_runs.csv
figures/ fig1…fig7 (.pdf, .png)
```

## Key columns

* `one_step_<label>.parquet`: `traj_id, rayleigh, prandtl, start, vrmse_<field>…, vrmse_mean`. Here `start`
  is the first context frame; the target is frame `start + 4`.
* `rollout_<label>.parquet`: `traj_id, rayleigh, prandtl, start, sigma, step (1-based), vrmse_<field>…,
  vrmse_mean`. Step 1 is frame `start + 4`.
* `physics_<label>.parquet`: `nu_pred, nu_true, nu_wall_bottom_true, nu_wall_top_true, nu_wall_bottom_pred,
  nu_abs_err, nu_rel_err` and, at the spectra steps, `log_spectral_error`.
* `main_results.csv`: `model, fraction, budget, metric, sigma, step, mean, std, n_seeds, per_seed`.
* `gpu_hours.csv`: `run_id, model, fraction, seed, attempts, unclosed_attempts, train_wall_hours,
  train_gpu_hours, eval_passes, eval_wall_hours, eval_gpu_hours`, plus a `TOTAL` row (decisions D22).
* `resource_use.csv`: `train_seconds, overshoot_seconds, data_wait_seconds, optimizer_steps, samples_seen, peak_train_mem_gb,
  n_parameters, attempts, val/ckpt overhead, infer_ms_per_frame, rollout_ms, gpus, budget_satisfied`.

## Completion contract

* `COMPLETE.json` is written only after both milestone checkpoints load, match the model's keys and shapes,
  contain finite weights, and are hashed.
* `EVAL_COMPLETE_<label>.json` is written last in its evaluation, so evaluation completion is tracked
  separately from training completion.
* Small files are written to `*.tmp` and then renamed with `os.replace`.
* Aggregation refuses to mix metric versions or cohort hashes.
