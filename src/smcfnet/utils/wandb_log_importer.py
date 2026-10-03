from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_DETECTOR_TRAIN_RE = re.compile(
    rf"Epoch \[(?P<epoch>\d+)/(?P<num_epochs>\d+)\] Iter (?P<step>\d+) "
    rf"LR: (?P<lr>{_FLOAT}) Loss: (?P<loss>{_FLOAT})(?P<tail>.*)$"
)
_DETECTOR_EPOCH_RE = re.compile(
    rf"Epoch \[(?P<epoch>\d+)/(?P<num_epochs>\d+)\] completed in "
    rf"(?P<seconds>{_FLOAT})s Avg Loss: (?P<loss>{_FLOAT})"
)
_DETECTOR_FINAL_METRICS_RE = re.compile(r"Final metrics:\s*(?P<metrics>.+)$")
_DETECTOR_EVAL_RE = re.compile(
    rf"Eval - mAP: (?P<mAP>{_FLOAT}), mAP50: (?P<mAP50>{_FLOAT})(?:, ADE: (?P<ADE>{_FLOAT})px)?"
)
# The Trainer logs monitor eval with display_name "Monitor Validation"
# (core/trainer.py evaluate_mon); "Mon Eval" is kept as a legacy alias. This is
# matched before _DETECTOR_EVAL_RE so the "Eval - mAP" substring cannot shadow it.
_DETECTOR_MON_EVAL_RE = re.compile(
    rf"(?:Monitor Validation|Mon Eval) - mAP: (?P<mAP>{_FLOAT}), mAP50: (?P<mAP50>{_FLOAT})(?:, ADE: (?P<ADE>{_FLOAT})px)?"
)
_KEY_VALUE_RE = re.compile(rf"(?P<key>[A-Za-z_][A-Za-z0-9_./-]*)=(?P<value>{_FLOAT})")
_LABEL_VALUE_RE = re.compile(rf"(?P<label>[A-Za-z][A-Za-z0-9_]*):\s*(?P<value>{_FLOAT})")
_GPU_RE = re.compile(r"GPU:\s*(?P<alloc>\d+(?:\.\d+)?)/(?P<reserved>\d+(?:\.\d+)?)MiB")
_DATA_RE = re.compile(rf"Data:\s*(?P<seconds>{_FLOAT})s")
_PROF_RE = re.compile(rf"(?P<name>[A-Za-z_]+)=(?P<ms>{_FLOAT})")

_MCM_TRAIN_RE = re.compile(r"\[MCM\]\s+iter=(?P<step>\d+)\s+(?P<metrics>.+)$")
_MCM_VAL_RE = re.compile(r"\[MCM\]\[VAL\]\s+(?P<metrics>.+)$")
_MCM_EPOCH_RE = re.compile(
    rf"\[MCM\]\s+Epoch (?P<epoch>\d+) done in (?P<seconds>{_FLOAT})s\s*(?P<metrics>.*)$"
)

_DETECTOR_LABEL_TO_KEY = {
    "Obj": "loss_obj",
    "Cls": "loss_cls",
    "Box": "loss_box",
    "FG": "fg_ratio",
    "IoU": "loss_iou",
    "CE": "loss_ce",
    "BBox": "loss_bbox",
    "GIoU": "loss_giou",
    "MCM": "loss_mcm",
    "ECM": "loss_ecm",
    "SMCM": "loss_smcm",
    "Fut": "loss_fut",
    "TC": "loss_tc",
}


@dataclass(frozen=True)
class ParsedWandbRecord:
    """One post-hoc WandB log payload recovered from a text log."""

    step: int
    metrics: Dict[str, float]
    source: str
    line_number: int


@dataclass(frozen=True)
class ParsedLogSummary:
    records: List[ParsedWandbRecord]
    lines_read: int
    source_path: Path


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load the saved run config if it exists."""

    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def choose_log_file(run_dir: Path, preferred: str = "auto") -> Path:
    """Choose the text log to import from a run directory."""

    run_dir = Path(run_dir)
    if preferred != "auto":
        path = Path(preferred)
        return path if path.is_absolute() else run_dir / path

    candidates = [run_dir / "train.log"]
    candidates.extend(sorted(run_dir.glob("stderr*.log")))
    candidates.extend(sorted(run_dir.glob("stdout*.log")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No train/stderr/stdout log found in {run_dir}")


def parse_log_file(path: Path, *, max_records: Optional[int] = None) -> ParsedLogSummary:
    """Parse SMCM-MCFNet text logs into WandB-ready metric payloads.

    The parser handles both detector training logs and MCM-only pretraining logs.
    It only emits records for the canonical training/eval message forms produced
    by the project loggers, so arbitrary log noise is ignored.

    Rank safety (issue #112): there is intentionally no rank filter here, and the
    log lines carry no rank tag to filter on. It is not needed: eval and
    "Final metrics:" lines are emitted on rank 0 only (the eval routine returns
    early for non-main processes, see Trainer eval guards in core/trainer.py), so
    non-rank-0 workers do not write duplicate metric lines. Within a single step,
    a "Final metrics:" line records the step in ``final_metric_steps`` and the
    subsequent "Eval - ..." line at that step is skipped, so each eval step yields
    a single record.
    """

    path = Path(path)
    records: List[ParsedWandbRecord] = []
    last_step = 0
    last_epoch_step: Dict[int, int] = {}
    final_metric_steps: set[int] = set()
    lines_read = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            lines_read = line_number
            parsed = _parse_line(line.rstrip("\n"), last_step, last_epoch_step, final_metric_steps)
            if parsed is None:
                continue

            step, metrics, source = parsed
            if step < 0:
                step = last_step
            if source in {"train", "mcm_train"}:
                last_step = max(last_step, step)
                epoch_value = metrics.get("train/epoch")
                if epoch_value is not None:
                    last_epoch_step[int(epoch_value)] = step
            elif step > 0:
                last_step = max(last_step, step)

            records.append(
                ParsedWandbRecord(
                    step=int(step),
                    metrics=metrics,
                    source=source,
                    line_number=line_number,
                )
            )
            if max_records is not None and len(records) >= int(max_records):
                break

    return ParsedLogSummary(records=records, lines_read=lines_read, source_path=path)


def iter_artifact_files(
    run_dir: Path,
    *,
    checkpoint_policy: str = "none",
    log_file: Optional[Path] = None,
) -> Iterator[Path]:
    """Yield small run files and optional checkpoints for a WandB artifact."""

    run_dir = Path(run_dir)
    seen: set[Path] = set()
    for candidate in [
        run_dir / "config.yaml",
        log_file,
        run_dir / "stdout.log",
        run_dir / "stderr.log",
    ]:
        if candidate is None:
            continue
        candidate = Path(candidate)
        if candidate.exists() and candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            yield candidate

    for pattern in ("stdout_*.log", "stderr_*.log"):
        for candidate in sorted(run_dir.glob(pattern)):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    if checkpoint_policy == "none":
        return
    if checkpoint_policy == "best-latest":
        candidates = [run_dir / "best.pth", run_dir / "latest.pth"]
    elif checkpoint_policy == "all":
        candidates = sorted(run_dir.glob("*.pth"))
    else:
        raise ValueError(f"Unsupported checkpoint policy: {checkpoint_policy}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _parse_line(
    line: str,
    last_step: int,
    last_epoch_step: Dict[int, int],
    final_metric_steps: set[int],
) -> Optional[tuple[int, Dict[str, float], str]]:
    detector_train = _DETECTOR_TRAIN_RE.search(line)
    if detector_train is not None:
        epoch = int(detector_train.group("epoch"))
        step = int(detector_train.group("step"))
        metrics: Dict[str, float] = {
            "train/step": float(step),
            "train/epoch": float(epoch),
            "train/lr": float(detector_train.group("lr")),
            "train/loss": float(detector_train.group("loss")),
        }
        tail = detector_train.group("tail")
        for label_match in _LABEL_VALUE_RE.finditer(tail):
            label = label_match.group("label")
            key = _DETECTOR_LABEL_TO_KEY.get(label)
            if key is not None:
                metrics[f"train/{key}"] = float(label_match.group("value"))

        data_match = _DATA_RE.search(tail)
        if data_match is not None:
            metrics["train/data_time_s"] = float(data_match.group("seconds"))

        gpu_match = _GPU_RE.search(tail)
        if gpu_match is not None:
            metrics["gpu/mem_alloc_mib"] = float(gpu_match.group("alloc"))
            metrics["gpu/mem_reserved_mib"] = float(gpu_match.group("reserved"))

        if "Prof(ms):" in tail:
            prof_tail = tail.split("Prof(ms):", maxsplit=1)[1]
            for prof_match in _PROF_RE.finditer(prof_tail):
                metrics[f"profile/{prof_match.group('name')}_ms"] = float(prof_match.group("ms"))

        return step, metrics, "train"

    detector_epoch = _DETECTOR_EPOCH_RE.search(line)
    if detector_epoch is not None:
        epoch = int(detector_epoch.group("epoch"))
        step = last_epoch_step.get(epoch, last_step)
        return (
            step,
            {
                "train/step": float(step),
                "train/epoch": float(epoch),
                "epoch/total": float(detector_epoch.group("loss")),
                "epoch/time_sec": float(detector_epoch.group("seconds")),
            },
            "epoch",
        )

    final_metrics = _DETECTOR_FINAL_METRICS_RE.search(line)
    if final_metrics is not None:
        metrics = {
            f"eval/{key}": value
            for key, value in _parse_key_value_metrics(final_metrics.group("metrics")).items()
        }
        if metrics:
            metrics["train/step"] = float(last_step)
            final_metric_steps.add(last_step)
            return last_step, metrics, "eval"

    # Check mon-eval before eval: "Monitor Validation"/"Mon Eval" must be claimed
    # first so the "Eval - mAP" substring in the eval regex cannot shadow it.
    mon_eval = _DETECTOR_MON_EVAL_RE.search(line)
    if mon_eval is not None:
        metrics = _eval_match_to_metrics(mon_eval, "mon_eval", last_step)
        return last_step, metrics, "mon_eval"

    detector_eval = _DETECTOR_EVAL_RE.search(line)
    if detector_eval is not None and last_step not in final_metric_steps:
        metrics = _eval_match_to_metrics(detector_eval, "eval", last_step)
        return last_step, metrics, "eval"

    mcm_train = _MCM_TRAIN_RE.search(line)
    if mcm_train is not None:
        step = int(mcm_train.group("step"))
        parsed = _parse_key_value_metrics(mcm_train.group("metrics"))
        metrics = {"train/step": float(step)}
        for key, value in parsed.items():
            out_key = "mcm_train/lr" if key == "lr" else f"mcm_train/{key}"
            metrics[out_key] = value
        return step, metrics, "mcm_train"

    mcm_val = _MCM_VAL_RE.search(line)
    if mcm_val is not None:
        parsed = _parse_key_value_metrics(mcm_val.group("metrics"))
        step = int(parsed.pop("iter", last_step))
        epoch = parsed.pop("epoch", None)
        metrics = {"train/step": float(step)}
        if epoch is not None:
            metrics["train/epoch"] = float(epoch)
        for key, value in parsed.items():
            metrics[f"mcm_val/{key}"] = value
        return step, metrics, "mcm_val"

    mcm_epoch = _MCM_EPOCH_RE.search(line)
    if mcm_epoch is not None:
        epoch = int(mcm_epoch.group("epoch"))
        parsed = _parse_key_value_metrics(mcm_epoch.group("metrics"))
        metrics = {
            "train/step": float(last_step),
            "train/epoch": float(epoch),
            "mcm_epoch/time_sec": float(mcm_epoch.group("seconds")),
        }
        for key, value in parsed.items():
            metrics[f"mcm_epoch/{key}"] = value
        return last_step, metrics, "mcm_epoch"

    return None


def _parse_key_value_metrics(text: str) -> Dict[str, float]:
    return {match.group("key"): float(match.group("value")) for match in _KEY_VALUE_RE.finditer(text)}


def _eval_match_to_metrics(match: re.Match[str], prefix: str, step: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {"train/step": float(step)}
    for key in ("mAP", "mAP50", "ADE"):
        value = match.groupdict().get(key)
        if value is not None:
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def summarize_records(records: Iterable[ParsedWandbRecord]) -> Dict[str, Any]:
    records = list(records)
    by_source: Dict[str, int] = {}
    for record in records:
        by_source[record.source] = by_source.get(record.source, 0) + 1
    return {
        "records": len(records),
        "first_step": min((r.step for r in records), default=None),
        "last_step": max((r.step for r in records), default=None),
        "by_source": by_source,
    }
