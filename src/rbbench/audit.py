"""Dataset audit (Phase D): what is actually on disk, and do the physics conventions hold?

Physics diagnostics (Nusselt, spectra) are enabled in evaluation only if
`physics_checks.passed` is true in the audit JSON written here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml

from rbbench.data import FIELDS, SPLITS, list_trajectories, make_well_dataset, split_files
from rbbench.manifests import now_utc, provenance
from rbbench.metrics import physics


def _attrs(obj) -> dict:
    out = {}
    for k, v in obj.attrs.items():
        v = np.asarray(v)
        out[k] = v.tolist() if v.dtype.kind != "S" and v.dtype.kind != "O" else [str(e) for e in np.atleast_1d(v)]
    return out


def describe_file(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        fields = {}
        for group in ("t0_fields", "t1_fields", "t2_fields"):
            for name in h[group].attrs["field_names"]:
                ds = h[group][name]
                fields[f"{group}/{name}"] = {"shape": list(ds.shape), "dtype": str(ds.dtype), **_attrs(ds)}
        bcs = {name: {**_attrs(g), "mask_true_idx": np.nonzero(g["mask"][...])[0].tolist()[:8]}
               for name, g in h["boundary_conditions"].items()}
        time = np.asarray(h["dimensions/time"][...], np.float64)
        return {
            "name": path.name,
            "bytes": path.stat().st_size,
            "root_attrs": _attrs(h),
            "n_trajectories": int(h.attrs["n_trajectories"]),
            "n_steps": int(time.shape[-1]),
            "time_first_last": [float(time.reshape(-1)[0]), float(time.reshape(-1)[-1])],
            "dt_unique": sorted({round(float(d), 8) for d in np.diff(time.reshape(-1, time.shape[-1])[0])}),
            "spatial_dims": [str(d) for d in h["dimensions"].attrs["spatial_dims"]],
            "x": np.asarray(h["dimensions/x"][...], np.float64),
            "y": np.asarray(h["dimensions/y"][...], np.float64),
            "fields": fields,
            "scalars": {k: _attrs(v) | {"value": np.asarray(v[...]).reshape(-1)[:4].tolist()}
                        for k, v in h["scalars"].items()},
            "boundary_conditions": bcs,
        }


def scan_finite(path: Path, max_trajectories: int | None) -> dict:
    out = {}
    with h5py.File(path, "r") as h:
        n = int(h.attrs["n_trajectories"])
        sel = range(n) if max_trajectories is None else range(min(n, max_trajectories))
        for group, name in (("t0_fields", "buoyancy"), ("t0_fields", "pressure"), ("t1_fields", "velocity")):
            bad = 0
            for i in sel:
                arr = h[group][name][i]
                bad += int(arr.size - np.isfinite(arr).sum())
            out[name] = bad
    return {"trajectories_scanned": len(sel), "nonfinite_counts": out}


def physics_checks(dataset_dir: Path, cfg_physics: dict, x: np.ndarray, y: np.ndarray,
                   samples_per_regime: int, tol: dict) -> dict:
    """Checks on reference (ground-truth) train trajectories."""
    failures, results = [], {}
    family = cfg_physics.get("y_grid", "auto")
    grid = physics.classify_y_grid(y)  # the file's stored coordinates
    best = grid["best"]
    results["y_grid"] = grid
    if family == "auto":
        results["y_grid_used"] = best
        lo, hi = grid["candidates"][best]["interval"]
        if best not in ("gauss", "lobatto") or grid["candidates"][best]["max_abs_dev"] > tol["grid"]:
            failures.append(f"y grid is not Chebyshev within {tol['grid']} (best fit {best})")
        if abs(lo - 0.0) > tol["grid"] or abs(hi - cfg_physics["Ly"]) > tol["grid"]:
            failures.append(f"y interval [{lo}, {hi}] != [0, {cfg_physics['Ly']}]")
    else:
        # The family is set in the config (D18); the checks below test it on the values themselves.
        results["y_grid_used"] = family
        results["stored_y_status"] = physics.stored_y_status(y, cfg_physics, tol["grid"])
        if results["stored_y_status"] == "unexpected":
            failures.append(f"stored y is neither the {family} nodes nor the uniform placeholder on [0, Ly]")
    y = physics.y_nodes(y, cfg_physics)
    dx = np.diff(x)
    results["x"] = {"n": len(x), "dx_min": float(dx.min()), "dx_max": float(dx.max()),
                    "Lx_from_spacing": float(dx.mean() * len(x)), "x0": float(x[0])}
    if np.ptp(dx) > tol["grid"] or abs(dx.mean() * len(x) - cfg_physics["Lx"]) > 1e3 * tol["grid"]:
        failures.append("x grid is not uniform/periodic with the configured Lx")

    ops = physics.ChebyshevY(y, 0.0, cfg_physics["Ly"])
    delta_b = cfg_physics["b_bottom"] - cfg_physics["b_top"]
    per_regime = []
    refs = list_trajectories(dataset_dir, "train")
    by_regime: dict = {}
    for r in refs:
        by_regime.setdefault(r.regime, []).append(r)
    for regime, rs in sorted(by_regime.items()):
        for ref in rs[:samples_per_regime]:
            with h5py.File(Path(dataset_dir) / ref.relpath, "r") as h:
                n_steps = h["t0_fields/buoyancy"].shape[1]
                late = slice(n_steps // 2, n_steps, max(1, n_steps // 20))
                b = np.asarray(h["t0_fields/buoyancy"][ref.index, late], np.float64)
                u = np.asarray(h["t1_fields/velocity"][ref.index, late], np.float64)
            kappa = float(physics.kappa_from(*regime))
            nu = physics.nusselt(b, u[..., 1], ops, kappa, delta_b)
            u_wall = max(np.abs(ops.wall_value(u[..., c], w)).max() for c in (0, 1) for w in ("bottom", "top"))
            u_rms = float(np.sqrt((u**2).mean())) + 1e-12
            k, e = physics.kinetic_energy_spectrum(u[..., 0], u[..., 1], ops, cfg_physics["Lx"])
            ke = physics.mean_kinetic_energy(u[..., 0], u[..., 1], ops)
            row = {
                "id": ref.id, "rayleigh": regime[0], "prandtl": regime[1],
                "nu_volume_late_mean": float(nu["nu_volume"].mean()),
                "nu_wall_bottom_late_mean": float(nu["nu_wall_bottom"].mean()),
                "nu_wall_top_late_mean": float(nu["nu_wall_top"].mean()),
                "conductive_part_max_dev": float(np.abs(nu["nu_conductive_part"] - 1).max()),
                "b_wall_bottom": float(nu["b_wall_bottom"].mean()), "b_wall_top": float(nu["b_wall_top"].mean()),
                "u_wall_over_rms": float(u_wall / u_rms),
                "parseval_rel_err": float(np.abs(e.sum(-1) - ke).max() / (np.abs(ke).max() + 1e-30)),
            }
            wall_mean = 0.5 * (row["nu_wall_bottom_late_mean"] + row["nu_wall_top_late_mean"])
            row["nu_volume_vs_wall_rel"] = abs(row["nu_volume_late_mean"] - wall_mean) / max(abs(wall_mean), 1e-12)
            per_regime.append(row)
    results["reference_trajectories"] = per_regime

    def worst(key):
        return max(r[key] for r in per_regime)

    summary = {
        "b_wall_bottom_max_err": max(abs(r["b_wall_bottom"] - cfg_physics["b_bottom"]) for r in per_regime),
        "b_wall_top_max_err": max(abs(r["b_wall_top"] - cfg_physics["b_top"]) for r in per_regime),
        "u_wall_over_rms_max": worst("u_wall_over_rms"),
        "conductive_part_max_dev": worst("conductive_part_max_dev"),
        "parseval_rel_err_max": worst("parseval_rel_err"),
        "nu_volume_vs_wall_rel_median": float(np.median([r["nu_volume_vs_wall_rel"] for r in per_regime])),
        "nu_volume_vs_wall_rel_max": worst("nu_volume_vs_wall_rel"),
    }
    results["summary"] = summary
    if max(summary["b_wall_bottom_max_err"], summary["b_wall_top_max_err"]) > tol["wall_b"] * delta_b:
        failures.append(f"buoyancy wall values differ from ({cfg_physics['b_bottom']}, {cfg_physics['b_top']})")
    if summary["u_wall_over_rms_max"] > tol["wall_u"]:
        failures.append("velocity does not vanish at the walls")
    if summary["conductive_part_max_dev"] > tol["conductive"]:
        failures.append("conductive Nusselt part deviates from 1 (derivative/quadrature/BC problem)")
    if summary["parseval_rel_err_max"] > 1e-8:
        failures.append("spectrum fails Parseval")
    if summary["nu_volume_vs_wall_rel_median"] > tol["nu_agreement"]:
        failures.append(f"volume and wall Nusselt disagree (median rel {summary['nu_volume_vs_wall_rel_median']:.3f})")
    results["tolerances"] = tol
    results["failures"] = failures
    results["passed"] = not failures
    return results


DEFAULT_TOL = {"grid": 1e-4, "wall_b": 2e-2, "wall_u": 2e-2, "conductive": 1e-2, "nu_agreement": 0.1}


def audit_dataset(dataset_dir: Path, cfg_physics: dict, scan_trajectories: int | None = 2,
                  samples_per_regime: int = 2, tol: dict | None = None) -> dict:
    dataset_dir = Path(dataset_dir)
    tol = {**DEFAULT_TOL, **(tol or {})}
    report: dict = {"dataset_dir": str(dataset_dir), "created_utc": now_utc(), "provenance": provenance()}
    found = sorted(p.name for p in (dataset_dir / "data").iterdir() if p.is_dir())
    report["split_folders"] = found
    stats_path = dataset_dir / "stats.yaml"
    report["stats_yaml"] = yaml.safe_load(stats_path.read_text()) if stats_path.exists() else None

    grids = []
    report["splits"] = {}
    for split in SPLITS:
        if split not in found:
            report["splits"][split] = {"missing": True}
            continue
        files = [describe_file(p) for p in split_files(dataset_dir, split)]
        names = {str(f["root_attrs"].get("dataset_name")) for f in files}
        for f in files:
            grids.append((f.pop("x"), f.pop("y")))
        refs = list_trajectories(dataset_dir, split)
        regimes = Counter(f"Ra={r.rayleigh:.6g},Pr={r.prandtl:.6g}" for r in refs)
        ds = make_well_dataset(dataset_dir, split, 4, 1)
        report["splits"][split] = {
            "n_files": len(files),
            "n_trajectories": len(refs),
            "bytes": sum(f["bytes"] for f in files),
            "dataset_names": sorted(names),
            "n_steps": sorted({f["n_steps"] for f in files}),
            "dt_unique": sorted({d for f in files for d in f["dt_unique"]}),
            "counts_by_regime": dict(sorted(regimes.items())),
            "well_dataset": {
                "field_names": list(FIELDS), "size_tuple": list(ds.size_tuple),
                "n_windows": len(ds), "windows_per_trajectory": sorted(set(ds.n_windows_per_trajectory)),
                "metadata_spatial_resolution": list(ds.metadata.spatial_resolution),
                "boundary_condition_types": ds.metadata.boundary_condition_types,
            },
            "finite_scan": {f.name: scan_finite(f, scan_trajectories) for f in split_files(dataset_dir, split)},
            "files": files,
        }
    if "uniform" in str(dataset_dir.name) or any(
        "rayleigh_benard_uniform" in n for s in report["splits"].values() for n in s.get("dataset_names", [])
    ):
        raise RuntimeError("this is rayleigh_benard_uniform, not rayleigh_benard")

    x0, y0 = grids[0]
    report["grids_identical_across_files"] = all(
        np.array_equal(x, x0) and np.array_equal(y, y0) for x, y in grids
    )
    report["x_coords"] = {"n": len(x0), "first": x0[:3].tolist(), "last": x0[-3:].tolist()}
    report["y_coords"] = {"n": len(y0), "values": y0.tolist()}
    nonfinite = sum(v for s in report["splits"].values() for f in s.get("finite_scan", {}).values()
                    for v in f["nonfinite_counts"].values())
    report["nonfinite_total_in_scan"] = nonfinite
    report["physics_checks"] = physics_checks(dataset_dir, cfg_physics, x0, y0, samples_per_regime, tol)
    if not report["grids_identical_across_files"]:
        report["physics_checks"]["failures"].append("grids differ between files")
        report["physics_checks"]["passed"] = False
    if nonfinite:
        report["physics_checks"]["failures"].append(f"{nonfinite} non-finite values found")
        report["physics_checks"]["passed"] = False
    return report
