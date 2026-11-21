"""Copy dataset splits to node-local disk for one job (`rbbench stage-data`; docs/final_plan.md WP1).

`/pub` delivers at most ~98 training samples/s into one node and stalls for 5-22 min (D23, D24). At 100% data the
training split (~294 GB) no longer fits in a node's page cache, so TFNO and AViT would spend most of their
allocated time waiting. The A100 nodes have 1.83 TB of local disk (`sinfo`, 2026-11-21), so a job can copy the
split there once and train from the copy.

The staged tree mirrors `<data_root>/datasets/rayleigh_benard`: the staged splits' HDF5 files are copied, and
every other entry (other splits' files, stats.yaml, the download records) is a symlink to the original. Samples
are identical because the files are (sizes checked; trajectories are addressed by their path relative to the
dataset directory, and `data._file_index` resolves symlinks). Paths are outside the config hash, so run ids do not
change. Nothing here is ever written to the Mac's disk in practice: the destination is the job's `$TMPDIR`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rbbench.data import DATASET_NAME

COPY_RETRIES = 6
COPY_RETRY_SECONDS = 30.0
SPACE_MARGIN = 1.05  # free space needed, relative to the bytes to copy


class StagingError(RuntimeError):
    pass


def _copy_one(src: Path, dst: Path, retries: int, base: float) -> int:
    """Copy one file to `dst` (via a temporary name), retrying on storage errors; return its size."""
    size = src.stat().st_size
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == size:
        return size  # already staged (e.g. a restart within the same job)
    tmp = dst.with_name(dst.name + ".partial")
    for attempt in range(retries + 1):
        try:
            shutil.copyfile(src, tmp)
            break
        except OSError as err:
            if attempt == retries:
                raise
            wait = base * 2 ** attempt
            print(f"[stage] copy error on {src.name} ({err}); retry {attempt + 1}/{retries} in {wait:.0f} s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    copied = tmp.stat().st_size
    if copied != size:
        tmp.unlink(missing_ok=True)
        raise StagingError(f"{src.name}: copied {copied} bytes, source has {size}")
    os.replace(tmp, dst)
    return size


def stage_dataset(data_root: Path, dest_root: Path, splits=("train",), workers: int = 4,
                  retries: int = COPY_RETRIES, retry_seconds: float = COPY_RETRY_SECONDS) -> dict:
    """Mirror `<data_root>/datasets/rayleigh_benard` under `<dest_root>`, copying the files of `splits`.

    Returns a report (also written to the staged dataset directory). Raises StagingError if the destination
    lacks space, a copy is incomplete, or the copied bytes disagree with the download's completion record.
    """
    src_dir = Path(data_root) / "datasets" / DATASET_NAME
    dst_dir = Path(dest_root) / "datasets" / DATASET_NAME
    if not (src_dir / "data").is_dir():
        raise StagingError(f"{src_dir}/data does not exist")
    splits = tuple(splits)
    to_copy = [(p, dst_dir / "data" / s / p.name) for s in splits for p in sorted((src_dir / "data" / s).glob("*.h*5"))]
    if not to_copy:
        raise StagingError(f"no HDF5 files to stage for splits {splits}")
    need = sum(p.stat().st_size for p, _ in to_copy)
    dst_dir.mkdir(parents=True, exist_ok=True)
    already = sum(d.stat().st_size for _, d in to_copy if d.exists() and not d.is_symlink())
    free = shutil.disk_usage(dst_dir).free
    if free < (need - already) * SPACE_MARGIN:
        raise StagingError(f"{dst_dir}: {free / 1e9:.0f} GB free, staging needs {need / 1e9:.0f} GB")

    # Everything that is not copied is linked, so the staged directory looks exactly like the original.
    for entry in sorted(src_dir.iterdir()):
        target = dst_dir / entry.name
        if entry.name == "data":
            continue
        if not target.exists() and not target.is_symlink():
            target.symlink_to(entry.resolve())
    for split_dir in sorted(p for p in (src_dir / "data").iterdir() if p.is_dir()):
        out = dst_dir / "data" / split_dir.name
        out.mkdir(parents=True, exist_ok=True)
        if split_dir.name in splits:
            continue
        for f in sorted(split_dir.iterdir()):
            link = out / f.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(f.resolve())

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        sizes = list(pool.map(lambda pair: _copy_one(*pair, retries, retry_seconds), to_copy))
    seconds = time.perf_counter() - t0

    checked = {}
    for s in splits:
        record = src_dir / f"rbbench_complete_{s}.json"
        names = sorted(p.name for p, d in to_copy if d.parent.name == s)
        copied = sum(sz for (p, d), sz in zip(to_copy, sizes) if d.parent.name == s)
        if record.exists():
            rec = json.loads(record.read_text())
            if int(rec["total_bytes"]) != copied or sorted(rec.get("sha256", {})) != names:
                raise StagingError(f"split {s}: staged {len(names)} files / {copied} bytes, but the download record "
                                   f"lists {len(rec.get('sha256', {}))} files / {rec['total_bytes']} bytes")
            checked[s] = "matches rbbench_complete record"
        else:
            checked[s] = "no download record; sizes checked against the source only"
    report = {"source": str(src_dir), "staged": str(dst_dir), "splits": list(splits), "files": len(to_copy),
              "bytes": need, "seconds": round(seconds, 1),
              "gb_per_s": round(need / 1e9 / seconds, 3) if seconds > 0 else None, "checked": checked}
    (dst_dir / "rbbench_stage.json").write_text(json.dumps(report, indent=2))
    return report
