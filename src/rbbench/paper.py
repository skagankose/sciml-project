"""The Well paper's reporting protocol for `rayleigh_benard` (decisions D15, D20): its published numbers and
how it scored them, so our runs can be reported exactly as the paper reported. No weights here: the released
checkpoints themselves stay in `released.py` (AGENTS.md rule 3)."""

from __future__ import annotations

# One-step VRMSE (field mean) of these checkpoints, from docs/benchmarks.md at the pinned
# The Well commit (best-validation checkpoint after 12 h on one H100). They were computed with the
# paper-era code (the_well v1.0.x, read at tag v1.0.1; decisions D20):
# * valid: training-time "short validation" (20 shuffled batches of 64 windows), in normalised units;
# * test: a full pass after training, in normalised units. These checkpoints reproduce it on the full test
#   set: CNextU-Net 0.6688, TFNO 0.6634 (job 59224204, D20). The public v1.0.1 builds `test_dataset` without
#   normalisation (fixed in v1.1.0), but emulating that gives 9.99 and 1.18, so the paper's runs did not.
# The valid values are not reproduced (0.70 / 0.68 on every window): one random 1,280-window draw per epoch,
# with the best epoch picked on it. The Hugging Face model cards list these valid values without a split.
PUBLISHED_VRMSE = {
    "cnext_unet": {"valid": 0.4807, "test": 0.6699},
    "tfno": {"valid": 0.8568, "test": 0.6566},
}
# The paper-era protocol, for both published splits: the `normalised_*` columns of evaluate.one_step_rows.
PAPER_PROTOCOL = "normalised"

# The paper's Table 3: VRMSE on the test set averaged over rollout steps 6-12 and 13-30 (1-based, inclusive),
# rollouts "initiated from the beginning of the simulation" (Section 4). For rayleigh_benard every model,
# CNextU-Net and TFNO included, is ">10" in both windows. The paper-era trainer (v1.0.1) logs log-spaced
# windows `np.linspace(0, log T, 5)`; with T = 30 they are slices 5:12 and 12:30, the same steps. Its default
# T = 100 would give 3:10 and 10:31 instead, so the paper's runs presumably rolled out 30 steps.
PAPER_ROLLOUT_STEPS = 30
PAPER_ROLLOUT_WINDOWS = {"6:12": (6, 12), "13:30": (13, 30)}
PUBLISHED_ROLLOUT_ABOVE = 10.0


def paper_rollout_windows(df) -> dict:
    """Table 3's numbers from paper-rollout rows (`evaluate.paper_rollout_rows`): field-mean VRMSE averaged over
    the steps of each window and over trajectories, in normalised units (the paper's) and physical units."""
    windows = {}
    for units, column in (("normalised", "normalised_vrmse_mean"), ("physical", "vrmse_mean")):
        windows[units] = {}
        for name, (lo, hi) in PAPER_ROLLOUT_WINDOWS.items():
            inside = df.step.between(lo, hi)
            windows[units][name] = float(df.loc[inside, column].mean()) if inside.any() else None
    return windows


def paper_rollout_summary(df, split: str) -> dict:
    """Window means of `evaluate.evaluate_paper_rollouts` rows; compared with Table 3 on the test split."""
    windows = paper_rollout_windows(df)
    judged = split == "test"
    above = all(v is not None and v > PUBLISHED_ROLLOUT_ABOVE for v in windows["normalised"].values())
    return {
        "start_frame": 0, "steps": int(df.step.max()), "n_trajectories": int(df.traj_id.nunique()),
        "window_vrmse_mean": windows,
        "published": f"> {PUBLISHED_ROLLOUT_ABOVE:g} in both windows (Table 3)" if judged else None,
        "reproduced": above if judged else None,
    }
