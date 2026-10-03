#!/usr/bin/env python3
"""Upload finished SMCM-MCFNet text logs to Weights & Biases.

This is intended for runs where WandB was disabled during training. It parses
the project train logs and replays the scalar metrics into a new WandB run.

Example:
  python tools/upload_wandb_from_logs.py \
    --run-dir outputs/baseline_v3/rgb_m_noamp_noaug \
    --project smcfnet_dsec \
    --run-name rgb_m_noamp_noaug \
    --dry-run
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smcfnet.utils.wandb import _try_import_wandb  # noqa: E402
from smcfnet.utils.wandb_log_importer import (  # noqa: E402
    choose_log_file,
    iter_artifact_files,
    load_run_config,
    parse_log_file,
    summarize_records,
)


logger = logging.getLogger("upload_wandb_from_logs")


def _config_lookup(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _infer_run_name(run_dir: Path, config: Dict[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return str(_config_lookup(config, "wandb.run_name") or _config_lookup(config, "log.experiment_name") or run_dir.name)


def _wandb_config(run_dir: Path, config: Dict[str, Any], log_file: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "imported_from_log": True,
        "source_run_dir": str(run_dir),
        "source_log_file": str(log_file),
        "import_summary": summary,
        "training_config": config,
    }


def upload_one_run(args: argparse.Namespace, run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    config = load_run_config(run_dir)
    log_file = choose_log_file(run_dir, args.log_file)
    parsed = parse_log_file(log_file, max_records=args.max_records)
    summary = summarize_records(parsed.records)

    run_name = _infer_run_name(run_dir, config, args.run_name if len(args.run_dir) == 1 else None)
    project = args.project or _config_lookup(config, "wandb.project_name", "smcfnet_dsec")
    entity = args.entity if args.entity is not None else _config_lookup(config, "wandb.entity")
    group = args.group if args.group is not None else _config_lookup(config, "wandb.group")
    tags = list(_config_lookup(config, "wandb.tags", []) or [])
    tags.extend(args.tags or [])
    tags.extend(["posthoc-log-import", "smcm-mcfnet"])

    logger.info(
        "Parsed %d records from %s: steps=%s..%s sources=%s",
        summary["records"],
        log_file,
        summary["first_step"],
        summary["last_step"],
        summary["by_source"],
    )

    if args.dry_run:
        print(yaml.safe_dump({"run_dir": str(run_dir), "run_name": run_name, "project": project, **summary}, sort_keys=True))
        return 0

    if not parsed.records:
        logger.warning("No metrics parsed from %s; skipping upload", log_file)
        return 0

    wandb = _try_import_wandb()
    if wandb is None:
        logger.error("wandb is not importable. Install wandb or use --dry-run to validate parsing.")
        return 2

    init_kwargs: Dict[str, Any] = {
        "project": project,
        "entity": entity,
        "name": run_name,
        "group": group,
        "tags": tags,
        "dir": str(run_dir),
        "mode": args.mode,
        "notes": args.notes,
        "config": _wandb_config(run_dir, config, log_file, summary),
    }
    if args.wandb_id:
        init_kwargs["id"] = args.wandb_id if len(args.run_dir) == 1 else f"{args.wandb_id}-{run_dir.name}"
        init_kwargs["resume"] = args.resume

    run = wandb.init(**init_kwargs)
    try:
        wandb.define_metric("train/step")
        for prefix in ("train", "epoch", "eval", "mon_eval", "mcm_train", "mcm_val", "mcm_epoch", "gpu", "profile"):
            wandb.define_metric(f"{prefix}/*", step_metric="train/step")

        for record in parsed.records:
            wandb.log(dict(record.metrics), step=int(record.step))

        if args.include_artifacts:
            artifact = wandb.Artifact(name=f"{run_name}-logs", type="training-run-log")
            for file_path in iter_artifact_files(run_dir, checkpoint_policy=args.checkpoint_policy, log_file=log_file):
                artifact.add_file(str(file_path), name=file_path.name)
            run.log_artifact(artifact)

        logger.info("Uploaded %d records to WandB run %s", len(parsed.records), run.name)
    finally:
        run.finish()

    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True, help="Run output directory/directories to import")
    parser.add_argument("--log-file", default="auto", help="Log filename/path relative to run dir, or 'auto' (default)")
    parser.add_argument("--project", default=None, help="WandB project. Defaults to config wandb.project_name or smcfnet_dsec")
    parser.add_argument("--entity", default=None, help="WandB entity. Defaults to config wandb.entity")
    parser.add_argument("--run-name", default=None, help="WandB run name. Only valid with a single --run-dir")
    parser.add_argument("--group", default=None, help="WandB group. Defaults to config wandb.group")
    parser.add_argument("--tags", nargs="*", default=[], help="Extra WandB tags")
    parser.add_argument("--mode", choices=["online", "offline"], default="online", help="WandB mode")
    parser.add_argument("--notes", default=None, help="Optional WandB notes")
    parser.add_argument("--wandb-id", default=None, help="Optional WandB run id for deterministic resume/replay")
    parser.add_argument("--resume", choices=["allow", "must", "never"], default="never", help="WandB resume mode when --wandb-id is set")
    parser.add_argument("--include-artifacts", action="store_true", help="Upload config/log files as a WandB artifact")
    parser.add_argument(
        "--checkpoint-policy",
        choices=["none", "best-latest", "all"],
        default="none",
        help="Which .pth checkpoints to add when --include-artifacts is set. Defaults to none to avoid large uploads.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Debug limit for parsed/uploaded records")
    parser.add_argument("--dry-run", action="store_true", help="Parse and summarize without uploading")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.run_name and len(args.run_dir) != 1:
        parser.error("--run-name can only be used with a single --run-dir")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    rc = 0
    for run_dir_str in args.run_dir:
        try:
            rc = max(rc, upload_one_run(args, Path(run_dir_str)))
        except Exception:
            logger.exception("Failed to import %s", run_dir_str)
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
