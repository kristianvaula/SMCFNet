"""Structured archive and plotting helpers for EA runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from smcfnet.evolution.framework import Candidate


OBJECTIVE_NAMES = ("f_quality", "f_flow_structure", "f_snn_stability", "f_compute_cost")


def candidate_record(
    candidate: Candidate,
    *,
    budget_iters: int,
    seed: int,
) -> Dict[str, Any]:
    """Convert an evaluated candidate to one JSON-safe archive row."""

    fitness = candidate.fitness
    metrics = dict(fitness.metrics if fitness is not None else {})
    notes = dict(fitness.notes if fitness is not None else {})
    objectives_tuple = tuple(float(v) for v in (fitness.objectives if fitness is not None else ()))
    objectives = {
        name: float(objectives_tuple[index])
        for index, name in enumerate(OBJECTIVE_NAMES)
        if index < len(objectives_tuple) and math.isfinite(float(objectives_tuple[index]))
    }

    valid = bool(fitness.valid) if fitness is not None else False
    failure_reason = notes.get("reason")
    status = "completed" if valid else "failed"

    rung = dict(candidate.metadata.get("rung") or {})
    candidate_budget = int(rung.get("budget_iters", budget_iters))
    return {
        "candidate_id": candidate.identifier,
        "eval_id": candidate.metadata.get("eval_id"),
        "generation": int(candidate.generation),
        "parent_ids": list(candidate.parents),
        "operator": _operator_label(candidate),
        "mutation_paths": _mutation_paths(candidate),
        "status": status,
        "failure_reason": failure_reason,
        "budget_iters": int(candidate_budget),
        "rung_index": int(rung["index"]) if "index" in rung else None,
        "seed": int(seed),
        "genome": candidate.genome,
        "objectives": objectives,
        "metrics_late": _late_metrics(metrics),
        "metrics": metrics,
        "notes": notes,
        "metadata": candidate.metadata,
    }


def generation_summary_record(
    generation: int,
    evaluated: Sequence[Candidate],
    survivors: Sequence[Candidate],
    best: Optional[Candidate],
) -> Dict[str, Any]:
    """Build one generation-level summary row."""

    records = [candidate_record(candidate, budget_iters=0, seed=0) for candidate in evaluated]
    completed = [record for record in records if record["status"] == "completed"]
    failed = [record for record in records if record["status"] != "completed"]
    summary: Dict[str, Any] = {
        "generation": int(generation),
        "evaluated": int(len(evaluated)),
        "completed": int(len(completed)),
        "failed": int(len(failed)),
        "failure_reasons": _count_by(record.get("failure_reason") or "unknown" for record in failed),
        "survivor_ids": [candidate.identifier for candidate in survivors],
        "best_candidate_id": best.identifier if best is not None else None,
    }
    for name in OBJECTIVE_NAMES:
        values = [float(record["objectives"][name]) for record in completed if name in record["objectives"]]
        if values:
            summary[f"{name}_best"] = min(values)
            summary[f"{name}_median"] = _median(values)
            summary[f"{name}_worst"] = max(values)
    return summary


class SearchArchive:
    """Append-only candidate/generation archive with optional W&B logging."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        budget_iters: int,
        seed: int,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_mode: str = "disabled",
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_jsonl = self.root_dir / "candidates.jsonl"
        self.generations_jsonl = self.root_dir / "generation_summary.jsonl"
        self.candidates_csv = self.root_dir / "candidates.csv"
        self.generations_csv = self.root_dir / "generation_summary.csv"
        self.generations_md = self.root_dir / "generation_summary.md"
        self.budget_iters = int(budget_iters)
        self.seed = int(seed)
        self.records: List[Dict[str, Any]] = _load_jsonl(self.candidates_jsonl)
        self.generation_records: List[Dict[str, Any]] = _load_jsonl(self.generations_jsonl)
        self._wandb = None
        if use_wandb and str(wandb_mode).lower() != "disabled":
            self._wandb = _init_wandb(
                project=wandb_project,
                run_name=wandb_run_name,
                group=wandb_group,
                mode=wandb_mode,
            )

    def on_candidate_evaluated(self, candidate: Candidate) -> None:
        record = candidate_record(candidate, budget_iters=self.budget_iters, seed=self.seed)
        self.records.append(record)
        _append_jsonl(self.candidates_jsonl, record)
        _write_records_csv(self.candidates_csv, self.records)
        if self._wandb is not None:
            self._wandb.log(_wandb_candidate_payload(record))

    def on_generation_end(
        self,
        generation: int,
        evaluated: Sequence[Candidate],
        survivors: Sequence[Candidate],
        best: Optional[Candidate],
    ) -> None:
        record = generation_summary_record(generation, evaluated, survivors, best)
        self.generation_records.append(record)
        _append_jsonl(self.generations_jsonl, record)
        _write_records_csv(self.generations_csv, self.generation_records)
        _append_generation_markdown(self.generations_md, record, evaluated)
        self.write_plots()
        if self._wandb is not None:
            self._wandb.log(_flatten_for_wandb(record, prefix="ea/generation"))

    def write_plots(self) -> None:
        try:
            from smcfnet.evolution.plotting import plot_ea_progress
        except Exception:
            return
        plot_dir = self.root_dir / "plots"
        try:
            plot_ea_progress(self.candidates_jsonl, plot_dir)
        except Exception as exc:
            (plot_dir / "plot_error.txt").parent.mkdir(parents=True, exist_ok=True)
            (plot_dir / "plot_error.txt").write_text(str(exc))


def _operator_label(candidate: Candidate) -> str:
    has_crossover = bool(candidate.metadata.get("crossover"))
    has_mutation = bool(candidate.metadata.get("mutation"))
    if has_crossover and has_mutation:
        return "crossover+mutation"
    if has_crossover:
        return "crossover"
    if has_mutation:
        return "mutation"
    return "seed"


def _mutation_paths(candidate: Candidate) -> List[str]:
    mutation = dict(candidate.metadata.get("mutation") or {})
    return [str(record.get("path")) for record in mutation.get("mutations", []) if record.get("path")]


def _late_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    mapping = {
        "mcm_epe_median": "fitness/epe_late",
        "mcm_epe_iqr": "fitness/epe_iqr_late",
        "flow_mag_mean_median": "fitness/flow_mag_late",
        "flow_centered_mag_mean_median": "fitness/flow_centered_late",
        "centered_ratio": "fitness/centered_ratio_late",
        "flow_mean_flow_ratio_median": "fitness/flow_mean_flow_ratio_late",
        "spatial_std_median": "fitness/spatial_std_late",
        "gpu_peak_mb": "fitness/mem_peak_mb_late",
        "num_steps_median": "fitness/num_steps_late",
        "num_readouts_median": "fitness/num_readouts_late",
        "spike_proxy": "fitness/spike_proxy_late",
        "wall_ms_per_iter": "fitness/wall_ms_per_iter",
        "iwe_contrast_median": "fitness/iwe_contrast_late",
        "mcm_contrast_median": "fitness/mcm_contrast_late",
        "smooth_raw_median": "fitness/smooth_raw_late",
        "warp_structure_proxy": "fitness/warp_structure_proxy_late",
        "validation_mcm_epe_median": "fitness/validation_mcm_epe_late",
        "validation_quality": "fitness/validation_quality",
    }
    out: Dict[str, float] = {}
    for name, key in mapping.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _write_records_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [_flat_record(record) for record in records]
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _append_generation_markdown(path: Path, record: Mapping[str, Any], evaluated: Sequence[Candidate]) -> None:
    best_by_quality = _best_candidate(evaluated, "f_quality")
    best_by_structure = _best_candidate(evaluated, "f_flow_structure")
    lines = [
        f"## Generation {record.get('generation')}",
        "",
        f"- completed: {record.get('completed')} / {record.get('evaluated')}",
        f"- failed: {record.get('failed')} ({record.get('failure_reasons')})",
        f"- survivors: {', '.join(str(item) for item in record.get('survivor_ids', []))}",
        "",
    ]
    if best_by_quality is not None:
        lines.extend(_candidate_markdown("Best quality", best_by_quality))
    if best_by_structure is not None and best_by_structure is not best_by_quality:
        lines.extend(_candidate_markdown("Best structure", best_by_structure))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def _best_candidate(candidates: Sequence[Candidate], objective: str) -> Optional[Candidate]:
    metric_key = {
        "f_quality": "fitness/objective_quality",
        "f_flow_structure": "fitness/objective_flow_structure",
        "f_snn_stability": "fitness/objective_snn_stability",
        "f_compute_cost": "fitness/objective_compute_cost",
    }[objective]
    completed = [
        candidate
        for candidate in candidates
        if candidate.fitness is not None and candidate.fitness.valid and metric_key in candidate.fitness.metrics
    ]
    if not completed:
        return None
    return min(completed, key=lambda candidate: float(candidate.fitness.metrics.get(metric_key, float("inf"))))


def _candidate_markdown(title: str, candidate: Candidate) -> List[str]:
    metrics = candidate.fitness.metrics if candidate.fitness is not None else {}
    return [
        f"### {title}",
        "",
        f"- candidate: {candidate.identifier}",
        f"- EPE late: {metrics.get('fitness/epe_late')}",
        f"- flow mag: {metrics.get('fitness/flow_mag_late')}",
        f"- centered ratio: {metrics.get('fitness/centered_ratio_late')}",
        f"- mean-flow ratio: {metrics.get('fitness/flow_mean_flow_ratio_late')}",
        f"- cost: {metrics.get('fitness/objective_compute_cost')}",
        "",
    ]


def _flat_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[str(key)] = value
        else:
            flat[str(key)] = json.dumps(value, sort_keys=True)
    return flat


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _count_by(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _init_wandb(*, project: Optional[str], run_name: Optional[str], group: Optional[str], mode: str):
    try:
        import wandb
    except Exception:
        return None
    return wandb.init(project=project, name=run_name, group=group, mode=mode, reinit=True)


def _wandb_candidate_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "ea/candidate/generation": record.get("generation"),
        "ea/candidate/completed": 1 if record.get("status") == "completed" else 0,
        "ea/candidate/failed": 1 if record.get("status") != "completed" else 0,
    }
    for key, value in dict(record.get("objectives") or {}).items():
        payload[f"ea/objectives/{key}"] = value
    for key, value in dict(record.get("metrics_late") or {}).items():
        payload[f"ea/late/{key}"] = value
    return payload


def _flatten_for_wandb(record: Mapping[str, Any], *, prefix: str) -> Dict[str, Any]:
    flat = _flat_record(record)
    return {f"{prefix}/{key}": value for key, value in flat.items() if isinstance(value, (int, float))}
