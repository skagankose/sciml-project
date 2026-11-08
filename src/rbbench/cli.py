"""Command-line entry point: `python -m rbbench <command> --help`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")  # neuraloperator imports wandb; we never log to it
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# neuraloperator 0.3.0 indexes with a list; fine on the pinned torch 2.8 (see docs/decisions.md D2).
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing")

from rbbench import MODEL_NAMES
from rbbench.manifests import REPO_ROOT, atomic_write_json, read_json


def _data_root(args) -> Path:
    root = args.data_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set $DATA_ROOT or pass --data-root (the directory that contains datasets/)")
    return Path(root)


def _dataset_dir(args) -> Path:
    return _data_root(args) / "datasets" / "rayleigh_benard"


def _output_root(args) -> Path:
    return Path(args.output_root or os.environ.get("OUTPUT_ROOT") or REPO_ROOT / "outputs")


def _device(name: str):
    from rbbench.device import pick_device

    return pick_device(name)


# --- training / evaluation tasks ----------------------------------------------------------

def run_train_task(row: dict, root: Path, output_root: Path, dataset_dir: Path, device,
                   resume: bool = True, skip_if_complete: bool = True, allow_dirty: bool = False) -> dict:
    from rbbench.config import config_hash
    from rbbench.manifests import git_info, sha256_file
    from rbbench.subsets import load_subset_ids
    from rbbench.sweep import task_config
    from rbbench.train import train_run

    manifest_path = root / row["subset_manifest"]
    if sha256_file(manifest_path) != row["subset_sha256"]:
        raise RuntimeError(f"{manifest_path} changed since the task file was generated")
    cfg = task_config(row["model"], row["experiment"], row["size"], row["subset_manifest"], row.get("overrides"))
    if config_hash(cfg) != row["config_sha256"]:
        raise RuntimeError("resolved config differs from the task manifest; regenerate tasks or revert configs")
    if device.type == "cuda" and not allow_dirty:
        git = git_info()
        if git["dirty"] or not git["commit"]:
            raise RuntimeError("production runs need a clean, committed git tree (use --allow-dirty for pilots)")
    ids = load_subset_ids(read_json(manifest_path))
    run_dir = output_root / "runs" / row["run_id"]
    return train_run(cfg, run_dir, dataset_dir, row["seed"], device, subset_ids=ids, run_meta=row,
                     resume=resume, skip_if_complete=skip_if_complete)


def run_labels(run_dir: Path) -> list[str]:
    """Milestone labels of a run, from its resolved config (e.g. ["1h", "4h"])."""
    from rbbench.config import budget_labels, yaml_load

    return budget_labels(yaml_load((run_dir / "resolved_config.yaml").read_text()))


def load_run_model(run_dir: Path, label: str, device):
    import torch

    from rbbench.config import yaml_load
    from rbbench.models.factory import model_from_config

    cfg = yaml_load((run_dir / "resolved_config.yaml").read_text())
    ckpt = torch.load(run_dir / f"checkpoint_{label}.pt", map_location="cpu", weights_only=False)
    model = model_from_config(cfg)
    model.load_state_dict(ckpt["model"])
    return cfg, model.to(device), ckpt


def run_eval_task(row: dict, output_root: Path, dataset_dir: Path, device, labels: list[str] | None,
                  cohort_path: Path, audit_path: Path | None, max_trajectories: int | None = None) -> dict:
    """Evaluate the milestone checkpoints `labels` of one run (default: all of its milestones)."""
    from rbbench.evaluate import evaluate, model_predictor
    from rbbench.manifests import sha256_file

    run_dir = output_root / "runs" / row["run_id"]
    if not (run_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"{run_dir} has not finished training")
    labels = labels or run_labels(run_dir)
    predictors, cfg, meta = {}, None, {"run_id": row["run_id"], "checkpoints": {}}
    for label in labels:
        cfg, model, ckpt = load_run_model(run_dir, label, device)
        predictors[label] = model_predictor(model, device, cfg["train"]["precision"])
        meta["checkpoints"][label] = {"sha256": sha256_file(run_dir / f"checkpoint_{label}.pt"),
                                      "step": ckpt["step"], "train_seconds": ckpt["train_seconds"]}
    return evaluate(predictors, cfg, dataset_dir, read_json(cohort_path), run_dir / "evaluations", device,
                    audit_path=audit_path, meta=meta, max_trajectories=max_trajectories)


def run_baseline(cfg: dict, output_root: Path, dataset_dir: Path, device, cohort_path: Path,
                 audit_path: Path | None, max_trajectories: int | None = None) -> dict:
    from rbbench.evaluate import evaluate
    from rbbench.rollout import Persistence

    out = output_root / "baselines" / "persistence" / "evaluations"
    return evaluate({"na": Persistence()}, cfg, dataset_dir, read_json(cohort_path), out, device,
                    audit_path=audit_path, meta={"baseline": "persistence"}, max_trajectories=max_trajectories)


# --- commands -----------------------------------------------------------------------------

def cmd_models(args):
    import torch

    from rbbench.config import load_config
    from rbbench.models.factory import count_parameters, count_parameters_used, model_from_config

    for model in args.models:
        cfg = load_config(model, args.experiment, args.size)
        net = model_from_config(cfg, seed=0)
        x = torch.randn(1, cfg["task"]["n_steps_input"], *cfg["task"]["spatial_resolution"], len(cfg["task"]["fields"]))
        print(f"{model:12s} size={args.size:5s} grid={cfg['task']['spatial_resolution']} "
              f"params={count_parameters(net):,} receiving_grad={count_parameters_used(net, x):,}")


def cmd_fixture(args):
    from rbbench.synthetic import make_synthetic_dataset

    d = make_synthetic_dataset(args.out, nx=args.grid[0], ny=args.grid[1], n_steps=args.n_steps, seed=args.seed)
    print(d)


def cmd_inspect(args):
    from rbbench.audit import audit_dataset
    from rbbench.config import load_config

    cfg = load_config("cnext_unet", args.experiment)
    report = audit_dataset(_dataset_dir(args), cfg["physics"], scan_trajectories=args.scan_trajectories,
                           samples_per_regime=args.samples_per_regime)
    atomic_write_json(args.out, report)
    pc = report["physics_checks"]
    print(json.dumps({k: report["splits"][k].get("n_trajectories") for k in report["splits"]}))
    stored = pc["y_grid"]["best"] + (f" ({pc['stored_y_status']})" if "stored_y_status" in pc else "")
    print(f"y grid: stored {stored}, used {pc['y_grid_used']}  physics checks passed: {pc['passed']}  "
          f"failures: {pc['failures']}")
    print(f"wrote {args.out}")


def cmd_make_subsets(args):
    from rbbench.data import list_trajectories
    from rbbench.subsets import build_split_manifest, make_nested_subsets, subset_filename

    dataset_dir = _dataset_dir(args)
    out = Path(args.out_dir)
    split = build_split_manifest(dataset_dir, fingerprints=not args.no_fingerprints)
    leaks = split["cross_split_duplicates"] or {}
    if any(v["ids"] or v["fingerprints"] for v in leaks.values()):
        sys.exit(f"cross-split duplicates found: {leaks}")
    atomic_write_json(out / "split_manifest.json", split)
    holdout = [tuple(r) for r in json.loads(args.holdout)] if args.holdout else None
    subsets = make_nested_subsets(list_trajectories(dataset_dir, "train"), args.fractions, args.seed, holdout)
    previous = None
    for frac, manifest in sorted(subsets.items()):
        from rbbench.manifests import sha256_file

        manifest["split_manifest_sha256"] = sha256_file(out / "split_manifest.json")
        ids = set(manifest["trajectory_ids"])
        if previous is not None and not previous <= ids:
            sys.exit("subsets are not nested")
        previous = ids
        atomic_write_json(out / subset_filename(frac, bool(holdout)), manifest)
        print(f"{frac:>5}: {manifest['n_trajectories']:4d} trajectories  {manifest['counts_by_regime']}")
    print(f"wrote {out}/split_manifest.json and subset manifests")


def cmd_make_cohort(args):
    from rbbench.config import load_config
    from rbbench.data import full_trajectory_dataset
    from rbbench.evaluate import make_cohort

    cfg = load_config("cnext_unet", args.experiment)
    _, total = full_trajectory_dataset(_dataset_dir(args), "test", cfg["task"]["n_steps_input"])
    cohort = make_cohort(_dataset_dir(args), cfg["eval"], cfg["task"]["n_steps_input"], total)
    atomic_write_json(args.out, cohort)
    print(f"{len(cohort['trajectories'])} trajectories x starts {cohort['starts']} -> {args.out}")


def cmd_make_sweep(args):
    from rbbench.sweep import make_sweep, write_jsonl

    rows = make_sweep(args.experiment, Path(args.tasks_dir), args.size, tuple(args.models),
                      args.fractions, args.seeds, args.override, repo_root=REPO_ROOT)
    write_jsonl(rows, Path(args.out))
    print(f"{len(rows)} tasks -> {args.out}")


def cmd_train(args):
    from rbbench.config import load_config
    from rbbench.subsets import load_subset_ids
    from rbbench.train import train_run

    cfg = load_config(args.model, args.experiment, args.size, args.override)
    ids = load_subset_ids(read_json(args.subset)) if args.subset else None
    res = train_run(cfg, args.run_dir, _dataset_dir(args), args.seed, _device(args.device), subset_ids=ids,
                    resume=not args.no_resume)
    print(json.dumps(res, indent=2))


def cmd_train_task(args):
    from rbbench.sweep import get_task

    row = get_task(Path(args.manifest), args.task_index)
    res = run_train_task(row, REPO_ROOT, _output_root(args), _dataset_dir(args), _device(args.device),
                         resume=args.resume_if_present, skip_if_complete=args.skip_if_complete,
                         allow_dirty=args.allow_dirty)
    print(json.dumps(res, indent=2))
    # 3 = preempted with state saved: the sbatch wrapper may requeue the same task.
    return 3 if res["status"] == "preempted" else 0


def cmd_eval_task(args):
    from rbbench.sweep import get_task

    row = get_task(Path(args.manifest), args.task_index)
    run_dir = _output_root(args) / "runs" / row["run_id"]
    labels = args.labels or (run_labels(run_dir) if (run_dir / "resolved_config.yaml").exists() else [])
    if args.skip_if_complete and labels and all(
        (run_dir / "evaluations" / f"EVAL_COMPLETE_{lab}.json").exists() for lab in labels
    ):
        print(f"{row['run_id']}: evaluation already complete")
        return 0
    res = run_eval_task(row, _output_root(args), _dataset_dir(args), _device(args.device), labels or None,
                        Path(args.cohort), Path(args.audit) if args.audit else None, args.max_trajectories)
    for lab, s in res.items():
        print(lab, json.dumps(s["by_sigma"]["0"].get("one_step"), indent=None))


def cmd_eval_baseline(args):
    from rbbench.config import load_config

    cfg = load_config("cnext_unet", args.experiment)
    res = run_baseline(cfg, _output_root(args), _dataset_dir(args), _device(args.device), Path(args.cohort),
                       Path(args.audit) if args.audit else None, args.max_trajectories)
    print(json.dumps(res["na"]["by_sigma"]["0"].get("one_step")))


def cmd_eval_released(args):
    """Pipeline check (gate D2): The Well's released checkpoints, one-step VRMSE on our data path."""
    from rbbench import METRIC_VERSION
    from rbbench.config import load_config
    from rbbench.data import FIELDS
    from rbbench.evaluate import evaluate_one_step, model_predictor
    from rbbench.manifests import provenance
    from rbbench.released import PAPER_PROTOCOL, PUBLISHED_VRMSE, fetch, load_released

    device = _device(args.device)
    dataset_dir = _dataset_dir(args)
    unverified = [s for s in args.splits if not (dataset_dir / f"rbbench_complete_{s}.json").exists()]
    if unverified and not args.allow_unverified_data:
        sys.exit(f"splits {unverified} are not verified downloads (no rbbench_complete_<split>.json)")
    out = _output_root(args) / "baselines" / "released"
    predictors, checkpoints, cfg = {}, {}, None
    for name in args.models:
        cfg = load_config(name, args.experiment, args.size, args.override)
        fetched = fetch(name)
        model = load_released(name, cfg, fetched)
        predictors[name] = model_predictor(model.to(device), device, "fp32")
        checkpoints[name] = {**{k: fetched[k] for k in ("repo_id", "revision", "sha256")},
                             "forward": model.released_forward or "pinned the_well"}
    complete = args.max_trajectories is None
    results, passed = {}, True
    for split in args.splits:
        tables = evaluate_one_step(predictors, dataset_dir, split, cfg["task"]["n_steps_input"], cfg["eval"],
                                   device, args.max_trajectories, paper_protocols=True)
        for name, df in tables.items():
            (out / name).mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / name / f"one_step_{split}.parquet", index=False)
            published = PUBLISHED_VRMSE[name][split]
            ours = float(df["vrmse_mean"].mean())
            protocol = PAPER_PROTOCOL[split]
            paper = float(df[f"{protocol}_vrmse_mean"].mean())
            rel = (paper - published) / published
            as_good = ours <= published
            passed &= as_good
            results.setdefault(name, {})[split] = {
                "vrmse_mean": ours, **{f"vrmse_{f}": float(df[f"vrmse_{f}"].mean()) for f in FIELDS},
                "paper_valid_vrmse_mean": float(df["paper_valid_vrmse_mean"].mean()),
                "paper_test_vrmse_mean": float(df["paper_test_vrmse_mean"].mean()),
                "paper_protocol": protocol, "n_windows": len(df), "n_trajectories": int(df.traj_id.nunique()),
                "published_vrmse_mean": published, "at_least_as_good": as_good,
                "reproduction_relative_difference": rel, "reproduced": abs(rel) <= args.tolerance,
            }
            print(f"{name:11s} {split:5s} ours {ours:.4f} | paper protocol ({protocol}) {paper:.4f} vs published "
                  f"{published:.4f} ({100 * rel:+.1f}%{', reproduced' if abs(rel) <= args.tolerance else ''}) | "
                  f"ours <= published: {'yes' if as_good else 'NO'}  ({len(df)} windows)")
    verdict = passed if complete else None
    atomic_write_json(out / "released_check.json", {
        "passed": verdict, "criterion": "our VRMSE <= published for every model and split (owner, 2026-11-08)",
        "reproduction_tolerance": args.tolerance, "complete_splits": complete,
        "max_trajectories": args.max_trajectories, "checkpoints": checkpoints, "results": results,
        "metric_version": METRIC_VERSION, "provenance": provenance({"device": str(device)}),
    })
    if not complete:
        print(f"partial run (--max-trajectories {args.max_trajectories}): not a pipeline check")
    else:
        print("PIPELINE CHECK PASSED" if passed else "PIPELINE CHECK: FAILED (ours worse than published)")
    print(f"wrote {out / 'released_check.json'}")
    return 1 if verdict is False else 0


def cmd_compare_paper(args):
    """One of our trained runs against the paper: at least as good, measured the same way (decisions D20).

    * valid, paper protocol (normalised units, as the published validation values): ours <= published;
    * valid and test, our protocol: ours <= the released checkpoint (the paper's model, eval-released).
    The paper's raw-input test protocol is reported but not a criterion: it scores the v1.0.x bug.
    """
    from rbbench.evaluate import evaluate_one_step, model_predictor
    from rbbench.manifests import provenance, sha256_file
    from rbbench.released import PAPER_PROTOCOL, PUBLISHED_VRMSE
    from rbbench.sweep import get_task

    device = _device(args.device)
    dataset_dir = _dataset_dir(args)
    unverified = [s for s in args.splits if not (dataset_dir / f"rbbench_complete_{s}.json").exists()]
    if unverified and not args.allow_unverified_data:
        sys.exit(f"splits {unverified} are not verified downloads (no rbbench_complete_<split>.json)")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.manifest is not None and args.task_index is not None:
        run_dir = _output_root(args) / "runs" / get_task(Path(args.manifest), args.task_index)["run_id"]
    else:
        sys.exit("give --run-dir, or --manifest with --task-index")
    if not (run_dir / "COMPLETE.json").exists():
        sys.exit(f"{run_dir} has not finished training")
    cfg, model, ckpt = load_run_model(run_dir, args.label, device)
    name = cfg["model"]["name"]
    if name not in PUBLISHED_VRMSE:
        sys.exit(f"the paper reports no rayleigh_benard VRMSE for {name}")
    released_path = _output_root(args) / "baselines" / "released" / "released_check.json"
    released = read_json(released_path)["results"].get(name, {}) if released_path.exists() else {}
    predict = model_predictor(model, device, cfg["train"]["precision"])
    out_dir = run_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, passed = {}, True
    for split in args.splits:
        df = evaluate_one_step({"run": predict}, dataset_dir, split, cfg["task"]["n_steps_input"], cfg["eval"],
                               device, args.max_trajectories, paper_protocols=True)["run"]
        df.to_parquet(out_dir / f"paper_compare_{args.label}_{split}.parquet", index=False)
        r = {key: float(df[f"{key}_mean"].mean()) for key in ("vrmse", "paper_valid_vrmse", "paper_test_vrmse")}
        published = PUBLISHED_VRMSE[name][split]
        released_ours = released.get(split, {}).get("vrmse_mean")
        checks = {}
        if split == "valid":
            checks["paper protocol <= published"] = r["paper_valid_vrmse"] <= published
        if released_ours is not None:
            checks["ours <= released checkpoint"] = r["vrmse"] <= released_ours
        passed &= all(checks.values())
        results[split] = {**r, "published_vrmse_mean": published, "published_protocol": PAPER_PROTOCOL[split],
                          "released_checkpoint_vrmse_mean": released_ours, "checks": checks, "n_windows": len(df)}
        ref = f"released {released_ours:.4f}" if released_ours is not None else "released: not evaluated yet"
        print(f"{name:11s} {args.label} {split:5s} ours {r['vrmse']:.4f} ({ref}) | paper protocol "
              f"{r[PAPER_PROTOCOL[split] + '_vrmse']:.4f} vs published {published:.4f} | "
              + ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items()))
    complete = args.max_trajectories is None and bool(released)
    verdict = passed if complete else None
    atomic_write_json(out_dir / f"paper_compare_{args.label}.json", {
        "passed": verdict, "run_dir": str(run_dir), "label": args.label, "model": name,
        "checkpoint": {"sha256": sha256_file(run_dir / f"checkpoint_{args.label}.pt"), "step": ckpt["step"],
                       "train_seconds": ckpt["train_seconds"]},
        "released_reference": str(released_path) if released else None, "results": results,
        "provenance": provenance({"device": str(device)}),
    })
    if verdict is None:
        print("not a complete comparison (partial data, or eval-released has not been run)")
    else:
        print("AT LEAST AS GOOD AS THE PAPER" if verdict else "WORSE THAN THE PAPER")
    return 1 if verdict is False else 0


def cmd_profile(args):
    from rbbench.config import load_config
    from rbbench.profile import profile_model, write_rows

    device = _device(args.device)
    rows = []
    for model in args.models:
        overrides = list(args.override)
        if args.grid:
            overrides.append(f"task.spatial_resolution=[{args.grid[0]},{args.grid[1]}]")
        cfg = load_config(model, args.experiment, args.size, overrides)
        for precision in args.precision:
            row = profile_model(cfg, device, _dataset_dir(args), args.batch_size, args.steps, args.warmup, precision)
            rows.append(row)
            print(f"{model:12s} {precision}: {row['status']} step={row.get('step_s', float('nan')):.4f}s "
                  f"peak={row.get('peak_alloc_gb')} GB est_steps_final={row.get('est_steps_final')} {row['error']}")
    write_rows(rows, Path(args.out))
    print(f"appended {len(rows)} rows to {args.out}")


def cmd_aggregate(args):
    from rbbench.aggregate import aggregate

    out = aggregate(Path(args.tasks), _output_root(args), Path(args.tables), Path(args.figures), args.labels,
                    baseline_dir=_output_root(args) / "baselines" / "persistence" / "evaluations")
    print(json.dumps({k: v for k, v in out.items() if k != "figures"}, indent=2))
    print(f"{len(out['figures'])} figure files")


def cmd_verify_download(args):
    from rbbench.download import verify_split

    rep = verify_split(_data_root(args), args.split, compute_sha=not args.no_sha, check_remote=not args.offline)
    print(json.dumps({k: rep[k] for k in ("split", "n_registry_files", "total_bytes", "n_trajectories",
                                          "complete", "problems")}, indent=2))
    return 0 if rep["complete"] else 1


def cmd_check_gpu(args):
    import torch

    from rbbench.manifests import gpu_info

    info = gpu_info()
    print(json.dumps(info, indent=2))
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    cap = "sm_%d%d" % torch.cuda.get_device_capability()
    if cap not in torch.cuda.get_arch_list():
        sys.exit(f"this torch build lacks {cap}")
    if args.require and args.require not in torch.cuda.get_device_name():
        sys.exit(f"allocated GPU {torch.cuda.get_device_name()} is not {args.require}")
    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(x @ x).all()
    print("GPU OK")


def cmd_smoke(args):
    """Whole pipeline on synthetic data: audit, subsets, cohort, tasks, train, eval, aggregate."""
    import torch

    from rbbench.audit import audit_dataset
    from rbbench.config import budget_labels, load_config
    from rbbench.data import full_trajectory_dataset, list_trajectories
    from rbbench.evaluate import make_cohort
    from rbbench.models.factory import count_parameters
    from rbbench.subsets import build_split_manifest, make_nested_subsets, subset_filename
    from rbbench.sweep import make_sweep, write_jsonl
    from rbbench.synthetic import make_synthetic_dataset

    if not args.synthetic:
        sys.exit("smoke currently runs on --synthetic data only")
    device = _device(args.device)
    work = Path(args.workdir or tempfile.mkdtemp(prefix="rbbench_smoke_"))
    nx, ny = args.grid
    data_root = work / "data"
    dataset_dir = make_synthetic_dataset(data_root, nx=nx, ny=ny, n_steps=40, seed=0)
    grid_override = f"task.spatial_resolution=[{nx},{ny}]"
    cfg0 = load_config("cnext_unet", "smoke", "tiny", [grid_override])
    audit = audit_dataset(dataset_dir, cfg0["physics"], scan_trajectories=1, samples_per_regime=1)
    atomic_write_json(work / "audit.json", audit)
    tasks = work / "tasks"
    atomic_write_json(tasks / "split_manifest.json", build_split_manifest(dataset_dir))
    for frac, m in make_nested_subsets(list_trajectories(dataset_dir, "train"), [0.3, 1.0], seed=0).items():
        atomic_write_json(tasks / subset_filename(frac), m)
    _, total = full_trajectory_dataset(dataset_dir, "test")
    atomic_write_json(tasks / "eval_cohort.json", make_cohort(dataset_dir, cfg0["eval"], 4, total))
    models = list(MODEL_NAMES) if args.all_models else args.models
    rows = make_sweep("smoke", tasks, "tiny", tuple(models), [0.3], [args.seed],
                      [f"train.max_steps={args.steps}", grid_override], repo_root=work)
    write_jsonl(rows, tasks / "train.jsonl")
    out_root = work / "outputs"
    labels = budget_labels(cfg0)
    final = labels[-1]
    print(f"workdir {work}  device {device}  torch {torch.__version__}  physics audit passed: "
          f"{audit['physics_checks']['passed']}")
    for row in rows:
        res = run_train_task(row, work, out_root, dataset_dir, device, allow_dirty=True)
        ev = run_eval_task(row, out_root, dataset_dir, device, labels, tasks / "eval_cohort.json",
                           work / "audit.json", max_trajectories=2)
        cfg, model, _ = load_run_model(out_root / "runs" / row["run_id"], final, torch.device("cpu"))
        one = ev[final]["by_sigma"]["0"]["one_step"]
        print(f"  {row['model']:11s} seed={row['seed']} params={count_parameters(model):>9,} "
              f"status={res['status']} steps={res.get('final_step')} one-step VRMSE({final} ckpt)={one['vrmse_mean']:.3f}")
    run_baseline(cfg0, out_root, dataset_dir, device, tasks / "eval_cohort.json", work / "audit.json", 2)
    from rbbench.aggregate import aggregate

    agg = aggregate(tasks / "train.jsonl", out_root, work / "tables", work / "figures",
                    baseline_dir=out_root / "baselines" / "persistence" / "evaluations")
    print(f"aggregate: {agg['found_evaluations']}/{agg['expected_evaluations']} evaluations, "
          f"{len(agg['figures'])} figure files, problems: {len(agg['problems'])} (forced milestones expected)")
    print(f"smoke OK -> {work}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rbbench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, data=True, out=False, device=False, exp=True):
        if data:
            sp.add_argument("--data-root", help="directory containing datasets/ (default $DATA_ROOT)")
        if out:
            sp.add_argument("--output-root", help="default $OUTPUT_ROOT or ./outputs")
        if device:
            sp.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
        if exp:
            sp.add_argument("--experiment", default="main")

    sp = sub.add_parser("smoke", help="end-to-end pipeline on synthetic data (tiny models)")
    common(sp, data=False, device=True, exp=False)
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--grid", type=int, nargs=2, default=[64, 32], metavar=("NX", "NY"))
    sp.add_argument("--steps", type=int, default=3)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--all-models", action="store_true")
    sp.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    sp.add_argument("--workdir")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("models", help="parameter counts")
    common(sp, data=False)
    sp.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    sp.add_argument("--size", default="full", choices=["full", "tiny"])
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("fixture", help="write the synthetic Well-format dataset")
    sp.add_argument("--out", required=True)
    sp.add_argument("--grid", type=int, nargs=2, default=[64, 32])
    sp.add_argument("--n-steps", type=int, default=40)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_fixture)

    sp = sub.add_parser("inspect-dataset", help="audit splits, grids, BCs, regimes, physics conventions")
    common(sp)
    sp.add_argument("--out", default=str(REPO_ROOT / "outputs" / "audit" / "dataset_audit.json"))
    sp.add_argument("--scan-trajectories", type=int, default=2, help="per file NaN scan; -1 = all")
    sp.add_argument("--samples-per-regime", type=int, default=2)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("make-subsets", help="split manifest + nested stratified train subsets")
    common(sp, exp=False)
    sp.add_argument("--out-dir", default=str(REPO_ROOT / "tasks"))
    sp.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--holdout", help='JSON list of [Ra, Pr] regimes to exclude (stretch experiment)')
    sp.add_argument("--no-fingerprints", action="store_true")
    sp.set_defaults(func=cmd_make_subsets)

    sp = sub.add_parser("make-cohort", help="freeze the rollout evaluation cohort")
    common(sp)
    sp.add_argument("--out", default=str(REPO_ROOT / "tasks" / "eval_cohort.json"))
    sp.set_defaults(func=cmd_make_cohort)

    sp = sub.add_parser("make-sweep", help="write the task manifest (36 rows for the main sweep)")
    common(sp, data=False)
    sp.add_argument("--tasks-dir", default=str(REPO_ROOT / "tasks"))
    sp.add_argument("--out", default=str(REPO_ROOT / "tasks" / "train.jsonl"))
    sp.add_argument("--size", default="full", choices=["full", "tiny"])
    sp.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    sp.add_argument("--fractions", type=float, nargs="+")
    sp.add_argument("--seeds", type=int, nargs="+")
    sp.add_argument("--override", action="append", default=[])
    sp.set_defaults(func=cmd_make_sweep)

    sp = sub.add_parser("train", help="one run with explicit arguments")
    common(sp, device=True)
    sp.add_argument("--model", required=True, choices=MODEL_NAMES)
    sp.add_argument("--size", default="full", choices=["full", "tiny"])
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--subset", help="subset manifest JSON (default: all train trajectories)")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--override", action="append", default=[])
    sp.add_argument("--no-resume", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("train-task", help="one row of a task manifest (Slurm array entry point)")
    common(sp, out=True, device=True, exp=False)
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--task-index", type=int, required=True)
    sp.add_argument("--resume-if-present", action="store_true")
    sp.add_argument("--skip-if-complete", action="store_true")
    sp.add_argument("--allow-dirty", action="store_true", help="pilots only")
    sp.set_defaults(func=cmd_train_task)

    sp = sub.add_parser("eval-task", help="evaluate the milestone checkpoints of one task")
    common(sp, out=True, device=True, exp=False)
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--task-index", type=int, required=True)
    sp.add_argument("--labels", nargs="+", help="milestones to evaluate (default: all in the run's config)")
    sp.add_argument("--cohort", default=str(REPO_ROOT / "tasks" / "eval_cohort.json"))
    sp.add_argument("--audit", default=str(REPO_ROOT / "outputs" / "audit" / "dataset_audit.json"))
    sp.add_argument("--max-trajectories", type=int)
    sp.add_argument("--skip-if-complete", action="store_true")
    sp.set_defaults(func=cmd_eval_task)

    sp = sub.add_parser("eval-baseline", help="persistence baseline on the same cohort")
    common(sp, out=True, device=True)
    sp.add_argument("--cohort", default=str(REPO_ROOT / "tasks" / "eval_cohort.json"))
    sp.add_argument("--audit", default=str(REPO_ROOT / "outputs" / "audit" / "dataset_audit.json"))
    sp.add_argument("--max-trajectories", type=int)
    sp.set_defaults(func=cmd_eval_baseline)

    sp = sub.add_parser("eval-released",
                        help="pipeline check: The Well's released CNextU-Net/TFNO checkpoints on our data path")
    common(sp, out=True, device=True)
    sp.add_argument("--models", nargs="+", default=["cnext_unet", "tfno"], choices=["cnext_unet", "tfno"])
    sp.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test"])
    sp.add_argument("--size", default="full", choices=["full", "tiny"], help="tiny is for tests only")
    sp.add_argument("--override", action="append", default=[])
    sp.add_argument("--tolerance", type=float, default=0.10,
                    help="paper-protocol VRMSE within this relative difference of the published value counts as "
                         "reproduced (reported, not the pass criterion)")
    sp.add_argument("--max-trajectories", type=int, help="per split; a partial run is not a pipeline check")
    sp.add_argument("--allow-unverified-data", action="store_true")
    sp.set_defaults(func=cmd_eval_released)

    sp = sub.add_parser("compare-paper", help="a trained run vs the paper: at least as good, measured the same way")
    common(sp, out=True, device=True)
    sp.add_argument("--run-dir", help="a run directory (or use --manifest and --task-index)")
    sp.add_argument("--manifest")
    sp.add_argument("--task-index", type=int)
    sp.add_argument("--label", default="4h", help="milestone checkpoint to compare")
    sp.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["valid", "test"])
    sp.add_argument("--max-trajectories", type=int, help="per split; a partial run is not a comparison")
    sp.add_argument("--allow-unverified-data", action="store_true")
    sp.set_defaults(func=cmd_compare_paper)

    sp = sub.add_parser("profile", help="throughput/memory/latency per model (A100 feasibility)")
    common(sp, device=True)
    sp.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    sp.add_argument("--size", default="full", choices=["full", "tiny"])
    sp.add_argument("--grid", type=int, nargs=2)
    sp.add_argument("--batch-size", type=int, default=1)
    sp.add_argument("--steps", type=int, default=20)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--precision", nargs="+", default=["fp32"], choices=["fp32", "bf16"])
    sp.add_argument("--override", action="append", default=[])
    sp.add_argument("--out", default=str(REPO_ROOT / "outputs" / "pilots" / "feasibility.csv"))
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("aggregate", help="tables and figures from copied outputs")
    common(sp, data=False, out=True, exp=False)
    sp.add_argument("--tasks", default=str(REPO_ROOT / "tasks" / "train.jsonl"))
    sp.add_argument("--tables", default=str(REPO_ROOT / "tables"))
    sp.add_argument("--figures", default=str(REPO_ROOT / "figures"))
    sp.add_argument("--labels", nargs="+", help="milestones to aggregate (default: all in each task's config)")
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("verify-download", help="check a downloaded split against the registry")
    common(sp, exp=False)
    sp.add_argument("--split", required=True, choices=["train", "valid", "test"])
    sp.add_argument("--no-sha", action="store_true")
    sp.add_argument("--offline", action="store_true", help="skip remote size checks")
    sp.set_defaults(func=cmd_verify_download)

    sp = sub.add_parser("check-gpu", help="print GPU/torch info; fail on unsupported GPUs")
    sp.add_argument("--require", help="substring the GPU name must contain, e.g. A100")
    sp.set_defaults(func=cmd_check_gpu)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    code = args.func(args)
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
