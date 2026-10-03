"""Run a manifest of STTP --eval-only jobs and append rows to a master CSV.

Each manifest row declares (label, design, flow_source, k, config, resume).
For every row the driver:
  1. Hashes the config YAML and the resume checkpoint (sha256 captures the
     exact files that produced the metric, so the row is reproducible even if
     the YAML or ckpt later change).
  2. Snapshots the source YAML + the trainer-resolved config (after _base_
     inheritance) into ``outputs/sttp_results/snapshots/<ts>_<label>/``.
  3. Runs ``python -m smcfnet.core.train --config <yaml> --resume <ckpt>
     --eval-only`` as a subprocess, capturing the full log.
  4. Parses ADE@5 / mAP / mAP50 from the eval log.
  5. Appends one CSV row to the master CSV (single writer, single job, no
     locking needed because the SLURM driver runs the manifest serially).

Missing configs or checkpoints are recorded as ``MISSING_CONFIG`` /
``MISSING_RESUME`` rows so the manifest can be re-run as in-flight training
checkpoints land. The CSV is append-only; pick the latest row per label by
``timestamp`` if you need to deduplicate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

CSV_FIELDS = [
    "timestamp",
    "label",
    "design",
    "flow_source",
    "k",
    "config_path",
    "config_sha256",
    "resume_path",
    "resume_sha256",
    "experiment_name",
    "ade5_px",
    "fde_px",
    "ade_per_step",
    "ade_per_step_curve",
    "per_class_ade",
    "map50",
    "map_full",
    "snapshot_dir",
    "exit_code",
    "slurm_job_id",
    "git_sha",
]

# FDE is optional so logs from before the per-class/FDE change still parse.
_VALIDATION_RE = re.compile(
    r"(?:Validation|Eval)\s*-\s*mAP:\s*([\d.]+),\s*mAP50:\s*([\d.]+),\s*ADE:\s*([\d.]+)px"
    r"(?:,\s*FDE:\s*([\d.]+)px)?"
)
_PERCLASS_RE = re.compile(r"Trajectory per-class ADE:\s*(.+)")
_PERSTEP_RE = re.compile(r"Trajectory per-step ADE:\s*(.+)")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_eval_metrics(log_path: Path) -> Optional[dict]:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = _VALIDATION_RE.findall(text)
    if not matches:
        return None
    map_full, map50, ade, fde = matches[-1]
    out = {
        "map_full": float(map_full),
        "map50": float(map50),
        "ade5_px": float(ade),
    }
    # fde is the optional 4th group: "" when an older log lacks the FDE field.
    # With the multi-step head FDE (final step t+k) is distinct from ADE; older
    # single-box logs lack the field, so fall back to ADE for those.
    out["fde_px"] = float(fde) if fde else float(ade)
    pc = _PERCLASS_RE.findall(text)
    if pc:
        out["per_class_ade"] = pc[-1].strip()
    ps = _PERSTEP_RE.findall(text)
    if ps:
        # e.g. "ADE_step1=3.9000 ADE_step2=5.1000 ..." (genuine per-step curve)
        out["ade_per_step_curve"] = ps[-1].strip()
    return out


def git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def resolve_experiment_name(config_path: Path) -> str:
    from smcfnet.config.config import (
        Config,
        load_config_from_yaml,
        update_config_from_dict,
    )

    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(str(config_path)))
    return cfg.log.experiment_name


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            label = (r.get("label") or "").strip()
            if not label or label.startswith("#"):
                continue
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def run_eval_subprocess(config: Path, resume: Path, log_path: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "smcfnet.core.train",
        "--config",
        str(config),
        "--resume",
        str(resume),
        "--eval-only",
    ]
    with log_path.open("wb") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/sttp_results/results.csv"),
        help="append-only master CSV (default: outputs/sttp_results/results.csv)",
    )
    ap.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("outputs/sttp_results/snapshots"),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="do not run evals; emit MISSING / DRY_RUN rows only",
    )
    args = ap.parse_args()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.snapshots_root.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    if not rows:
        print(f"manifest {args.manifest} has no usable rows", file=sys.stderr)
        return 2

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    git_sha = git_head_sha()

    write_header = not args.csv.exists()
    csv_handle = args.csv.open("a", newline="")
    writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_handle.flush()

    print(f"manifest:         {args.manifest}")
    print(f"output CSV:       {args.csv}")
    print(f"snapshots root:   {args.snapshots_root}")
    print(f"git sha:          {git_sha or '(no git)'}")
    print(f"slurm job:        {slurm_job_id}")
    print(f"rows in manifest: {len(rows)}")
    print("=" * 78)

    for row in rows:
        label = row["label"]
        config = Path(row["config"])
        resume = Path(row["resume"])

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = args.snapshots_root / f"{ts}_{label}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        eval_log = snap_dir / "eval.log"
        record = {
            "timestamp": ts,
            "label": label,
            "design": row.get("design", ""),
            "flow_source": row.get("flow_source", ""),
            "k": row.get("k", ""),
            "config_path": str(config),
            "config_sha256": "",
            "resume_path": str(resume),
            "resume_sha256": "",
            "experiment_name": "",
            "ade5_px": "",
            "map50": "",
            "map_full": "",
            "snapshot_dir": str(snap_dir),
            "exit_code": "",
            "slurm_job_id": slurm_job_id,
            "git_sha": git_sha,
        }

        if not config.exists():
            record["exit_code"] = "MISSING_CONFIG"
            print(f"[{label}] SKIP — config missing: {config}")
            writer.writerow(record)
            csv_handle.flush()
            continue
        record["config_sha256"] = sha256_of_file(config)
        shutil.copy2(config, snap_dir / "config_source.yaml")

        if not resume.exists():
            record["exit_code"] = "MISSING_RESUME"
            print(f"[{label}] SKIP — resume missing: {resume}")
            writer.writerow(record)
            csv_handle.flush()
            continue
        record["resume_sha256"] = sha256_of_file(resume)

        try:
            record["experiment_name"] = resolve_experiment_name(config)
        except Exception as exc:
            record["exit_code"] = f"CONFIG_PARSE_ERROR:{type(exc).__name__}"
            print(f"[{label}] SKIP — config parse error: {exc}")
            writer.writerow(record)
            csv_handle.flush()
            continue

        if args.dry_run:
            record["exit_code"] = "DRY_RUN"
            print(f"[{label}] dry-run, skipping eval")
            writer.writerow(record)
            csv_handle.flush()
            continue

        print(f"[{label}] running eval — config={config.name}  resume={resume}")
        rc = run_eval_subprocess(config, resume, eval_log)
        record["exit_code"] = str(rc)

        # Trainer writes resolved config to the run's output dir; copy it for
        # the per-eval snapshot so we capture _base_ inheritance.
        try:
            from smcfnet.config.config import (
                Config,
                load_config_from_yaml,
                update_config_from_dict,
            )

            cfg = Config()
            update_config_from_dict(cfg, load_config_from_yaml(str(config)))
            resolved = (
                Path(cfg.log.output_dir)
                / cfg.log.experiment_name
                / "config.yaml"
            )
            if resolved.exists():
                shutil.copy2(resolved, snap_dir / "config_resolved.yaml")
        except Exception:
            pass

        metrics = parse_eval_metrics(eval_log)
        if metrics:
            for mk, mv in metrics.items():
                record[mk] = mv if isinstance(mv, str) else f"{mv:.6f}"
            # per-step displacement = ADE@k / k, the horizon-normalized reading
            # the thesis Exp 3.2 calls "per-step displacement error".
            try:
                k_val = int(row.get("k", "") or 0)
                if k_val > 0:
                    record["ade_per_step"] = f"{metrics['ade5_px'] / k_val:.6f}"
            except ValueError:
                pass
            print(
                f"[{label}] OK — ADE={metrics['ade5_px']:.4f}px "
                f"FDE={metrics['fde_px']:.4f}px mAP50={metrics['map50']:.4f}"
            )
        else:
            print(f"[{label}] WARN — could not parse metrics from {eval_log}")

        # Per-row JSON sidecar mirrors the CSV row, useful for grep / programmatic
        # post-processing without re-parsing the CSV.
        (snap_dir / "record.json").write_text(json.dumps(record, indent=2))

        writer.writerow(record)
        csv_handle.flush()

    csv_handle.close()
    print("=" * 78)
    print(f"wrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
