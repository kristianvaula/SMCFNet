"""Plotting utilities for EA candidate archives."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from smcfnet.evolution.archive import OBJECTIVE_NAMES


def load_candidate_records(path: str | Path) -> List[Dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in in_path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def plot_ea_progress(results: str | Path, out_dir: str | Path) -> None:
    """Generate the standard EA progress figure set."""

    records = load_candidate_records(results)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [record for record in records if record.get("status") == "completed"]
    _plot_objectives_by_generation(completed, out / "01_objectives_by_generation.png", plt)
    _plot_pareto(completed, out / "02_pareto_epe_vs_cost.png", plt, y_objective="f_compute_cost")
    _plot_pareto(completed, out / "02b_pareto_epe_vs_structure.png", plt, y_objective="f_flow_structure")
    _plot_flow_structure(completed, out / "03_flow_structure_by_generation.png", plt)
    _plot_parallel_coordinates(completed, out / "04_parallel_coordinates_elites.png", plt)
    _plot_gene_distributions(completed, out / "05_gene_distributions.png", plt)
    _plot_failure_modes(records, out / "06_failure_modes.png", plt)
    _plot_qualitative_warp_iwe(completed, out / "07_qualitative_warp_iwe_metrics.png", plt)
    _plot_mutation_impact(completed, out / "08_mutation_impact.png", plt)
    _plot_lineage_quality(completed, out / "09_lineage_quality.png", plt)


def _plot_objectives_by_generation(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    generations = sorted({int(record["generation"]) for record in records})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, objective in zip(axes.ravel(), OBJECTIVE_NAMES):
        medians = []
        lows = []
        highs = []
        best = []
        for generation in generations:
            values = [
                float(record["objectives"][objective])
                for record in records
                if int(record["generation"]) == generation and objective in record.get("objectives", {})
            ]
            medians.append(_quantile(values, 0.50))
            lows.append(_quantile(values, 0.25))
            highs.append(_quantile(values, 0.75))
            best.append(min(values) if values else float("nan"))
        ax.plot(generations, medians, label="median")
        ax.fill_between(generations, lows, highs, alpha=0.25, label="q25-q75")
        ax.scatter(generations, best, s=24, label="best")
        ax.set_title(objective)
        ax.set_xlabel("generation")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pareto(records: Sequence[Mapping[str, Any]], path: Path, plt: Any, *, y_objective: str) -> None:
    if not records:
        return
    xs = [_metric(record, "metrics_late.mcm_epe_median", default=_objective(record, "f_quality")) for record in records]
    ys = [_objective(record, y_objective) for record in records]
    generations = [int(record["generation"]) for record in records]
    sizes = [20.0 + 12.0 * max(0.0, _metric(record, "metrics_late.flow_mag_mean_median", default=0.0)) for record in records]
    pareto = _pareto_indices([(x, y) for x, y in zip(xs, ys)])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(xs, ys, c=generations, s=sizes, alpha=0.65, cmap="viridis")
    if pareto:
        ax.scatter([xs[i] for i in pareto], [ys[i] for i in pareto], marker="*", s=180, facecolors="none", edgecolors="black", label="Pareto")
    ax.set_xlabel("late EPE median")
    ax.set_ylabel(y_objective)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="generation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_flow_structure(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    keys = [
        ("flow_mag_mean_median", "flow mag"),
        ("flow_centered_mag_mean_median", "centered mag"),
        ("centered_ratio", "centered/total"),
        ("flow_mean_flow_ratio_median", "mean-flow ratio"),
        ("spatial_std_median", "spatial std"),
    ]
    generations = sorted({int(record["generation"]) for record in records})
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for key, label in keys:
        values = []
        lows = []
        highs = []
        for generation in generations:
            group = [
                _metric(record, f"metrics_late.{key}", default=float("nan"))
                for record in records
                if int(record["generation"]) == generation
            ]
            values.append(_quantile(group, 0.50))
            lows.append(_quantile(group, 0.25))
            highs.append(_quantile(group, 0.75))
        ax.plot(generations, values, label=label)
        ax.fill_between(generations, lows, highs, alpha=0.12)
    ax.set_xlabel("generation")
    ax.set_ylabel("late-window metric")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_parallel_coordinates(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    selected = sorted(records, key=lambda record: _objective(record, "f_quality"))[: max(5, min(50, len(records) // 5 or 1))]
    axes = [
        ("genome.mcm.events_per_step_target", "events"),
        ("genome.mcm.bptt_steps", "bptt"),
        ("genome.mcm.flow_scaling", "flow_scale"),
        ("genome.mcm.smooth_weight", "smooth"),
        ("genome.sofe_architecture.count_scale", "count"),
        ("genome.sofe_architecture.firenet_leak.0", "leak"),
        ("genome.sofe_architecture.firenet_thresh.0", "thresh"),
        ("genome.sofe_architecture.surrogate_alpha", "surrogate"),
        ("objectives.f_quality", "quality"),
        ("objectives.f_flow_structure", "structure"),
        ("objectives.f_compute_cost", "cost"),
    ]
    matrix: List[List[float]] = []
    for record in selected:
        row = [_metric(record, key, default=float("nan")) for key, _ in axes]
        if all(math.isfinite(value) for value in row):
            matrix.append(row)
    if not matrix:
        return
    cols = list(zip(*matrix))
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = list(range(len(axes)))
    for row in matrix:
        norm = [
            0.5 if max_v == min_v else (value - min_v) / (max_v - min_v)
            for value, min_v, max_v in zip(row, mins, maxs)
        ]
        ax.plot(x, norm, alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in axes], rotation=35, ha="right")
    ax.set_ylabel("normalized value")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gene_distributions(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    genes = [
        ("genome.mcm.events_per_step_target", "events"),
        ("genome.mcm.bptt_steps", "bptt"),
        ("genome.mcm.flow_scaling", "flow scaling"),
        ("genome.mcm.smooth_weight", "smooth"),
        ("genome.sofe_architecture.count_scale", "count scale"),
        ("genome.sofe_architecture.firenet_leak.0", "leak"),
        ("genome.sofe_architecture.firenet_thresh.0", "threshold"),
    ]
    fig, axes = plt.subplots(len(genes), 1, figsize=(10, 14), constrained_layout=True)
    for ax, (key, label) in zip(axes, genes):
        xs = [int(record["generation"]) for record in records]
        ys = [_metric(record, key, default=float("nan")) for record in records]
        color = [_objective(record, "f_quality") for record in records]
        ax.scatter(xs, ys, c=color, cmap="magma_r", alpha=0.75, s=24)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("generation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_failure_modes(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    generations = sorted({int(record["generation"]) for record in records})
    reasons = sorted({str(record.get("failure_reason") or record.get("status") or "unknown") for record in records})
    counts = {reason: [] for reason in reasons}
    for generation in generations:
        group = [record for record in records if int(record["generation"]) == generation]
        for reason in reasons:
            counts[reason].append(sum(1 for record in group if str(record.get("failure_reason") or record.get("status") or "unknown") == reason))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bottom = [0 for _ in generations]
    for reason in reasons:
        ax.bar(generations, counts[reason], bottom=bottom, label=reason)
        bottom = [a + b for a, b in zip(bottom, counts[reason])]
    ax.set_xlabel("generation")
    ax.set_ylabel("candidates")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_qualitative_warp_iwe(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    keys = [
        ("iwe_contrast_median", "IWE contrast"),
        ("smooth_raw_median", "smooth raw"),
        ("warp_structure_proxy", "warp structure proxy"),
        ("flow_mean_flow_ratio_median", "mean-flow ratio"),
    ]
    generations = sorted({int(record["generation"]) for record in records})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plotted = False
    for ax, (key, label) in zip(axes.ravel(), keys):
        medians = []
        lows = []
        highs = []
        for generation in generations:
            group = [
                _metric(record, f"metrics_late.{key}", default=float("nan"))
                for record in records
                if int(record["generation"]) == generation
            ]
            medians.append(_quantile(group, 0.50))
            lows.append(_quantile(group, 0.25))
            highs.append(_quantile(group, 0.75))
        if any(math.isfinite(value) for value in medians):
            plotted = True
            ax.plot(generations, medians)
            ax.fill_between(generations, lows, highs, alpha=0.20)
        ax.set_title(label)
        ax.set_xlabel("generation")
        ax.grid(True, alpha=0.25)
    if plotted:
        fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mutation_impact(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    impacts = _mutation_impacts(records)
    if not impacts:
        return
    grouped: Dict[str, List[Mapping[str, float]]] = {}
    for row in impacts:
        grouped.setdefault(str(row["path"]), []).append(row)
    ordered = sorted(
        grouped,
        key=lambda key: abs(_quantile([float(item["delta_quality"]) for item in grouped[key]], 0.50)),
        reverse=True,
    )[:14]
    if not ordered:
        return

    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.8 * len(ordered)), 10), constrained_layout=True)
    specs = [
        ("delta_quality", "delta f_quality vs best parent"),
        ("delta_structure", "delta f_flow_structure vs best parent"),
        ("delta_cost", "delta f_compute_cost vs best parent"),
    ]
    for ax, (field, ylabel) in zip(axes, specs):
        for x, mutation_path in enumerate(ordered):
            values = [float(item[field]) for item in grouped[mutation_path] if math.isfinite(float(item[field]))]
            if not values:
                continue
            q25 = _quantile(values, 0.25)
            q50 = _quantile(values, 0.50)
            q75 = _quantile(values, 0.75)
            ax.vlines(x, q25, q75, color="tab:blue", linewidth=4, alpha=0.45)
            ax.scatter([x for _ in values], values, s=18, alpha=0.45)
            ax.scatter([x], [q50], color="black", s=28, zorder=3)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(list(range(len(ordered))))
    axes[-1].set_xticklabels([_short_path(mutation_path) for mutation_path in ordered], rotation=35, ha="right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_lineage_quality(records: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    if not records:
        return
    record_map = _record_index(records)
    selected = sorted(records, key=lambda record: _objective(record, "f_quality"))[: min(8, len(records))]
    traces: List[List[Mapping[str, Any]]] = []
    for record in selected:
        trace = _lineage_trace(record, record_map)
        if len(trace) >= 2:
            traces.append(trace)
    if not traces:
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for trace in traces:
        ordered = sorted(trace, key=lambda item: int(item.get("generation", 0)))
        xs = [int(item.get("generation", 0)) for item in ordered]
        ys = [_objective(item, "f_quality") for item in ordered]
        label = str(ordered[-1].get("candidate_id") or "candidate")
        ax.plot(xs, ys, marker="o", alpha=0.65, label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("f_quality")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _metric(record: Mapping[str, Any], dotted: str, *, default: float = float("nan")) -> float:
    value: Any = record
    for part in dotted.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and part.isdigit():
            idx = int(part)
            value = value[idx] if idx < len(value) else default
        else:
            return float(default)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


def _objective(record: Mapping[str, Any], name: str) -> float:
    return _metric(record, f"objectives.{name}", default=float("inf"))


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("candidate_id") or "")


def _base_record_id(candidate_id: str) -> str:
    text = str(candidate_id)
    if "_r" in text:
        head, tail = text.rsplit("_r", 1)
        if tail.isdigit():
            return head
    return text


def _record_index(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        candidate_id = _record_id(record)
        if not candidate_id:
            continue
        out[candidate_id] = record
        out.setdefault(_base_record_id(candidate_id), record)
    return out


def _find_record(record_map: Mapping[str, Mapping[str, Any]], candidate_id: str) -> Mapping[str, Any] | None:
    return record_map.get(str(candidate_id)) or record_map.get(_base_record_id(str(candidate_id)))


def _mutation_impacts(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, float | str]]:
    record_map = _record_index(records)
    rows: List[Dict[str, float | str]] = []
    for record in records:
        parents = [
            _find_record(record_map, str(parent_id))
            for parent_id in list(record.get("parent_ids") or [])
            if str(parent_id)
        ]
        parents = [parent for parent in parents if parent is not None]
        mutation_paths = [str(path) for path in list(record.get("mutation_paths") or []) if str(path)]
        if not parents or not mutation_paths:
            continue
        parent_quality = min(_objective(parent, "f_quality") for parent in parents)
        parent_structure = min(_objective(parent, "f_flow_structure") for parent in parents)
        parent_cost = min(_objective(parent, "f_compute_cost") for parent in parents)
        child_quality = _objective(record, "f_quality")
        child_structure = _objective(record, "f_flow_structure")
        child_cost = _objective(record, "f_compute_cost")
        if not all(math.isfinite(value) for value in [parent_quality, parent_structure, parent_cost, child_quality, child_structure, child_cost]):
            continue
        for mutation_path in mutation_paths:
            rows.append(
                {
                    "path": mutation_path,
                    "delta_quality": float(child_quality - parent_quality),
                    "delta_structure": float(child_structure - parent_structure),
                    "delta_cost": float(child_cost - parent_cost),
                }
            )
    return rows


def _lineage_trace(record: Mapping[str, Any], record_map: Mapping[str, Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    trace = [record]
    seen = {_record_id(record)}
    current = record
    while True:
        parents = [
            _find_record(record_map, str(parent_id))
            for parent_id in list(current.get("parent_ids") or [])
            if str(parent_id)
        ]
        parents = [parent for parent in parents if parent is not None and _record_id(parent) not in seen]
        if not parents:
            break
        parent = min(parents, key=lambda item: _objective(item, "f_quality"))
        trace.append(parent)
        seen.add(_record_id(parent))
        current = parent
    return trace


def _short_path(path: str) -> str:
    parts = [part for part in str(path).split(".") if part]
    return ".".join(parts[-3:]) if len(parts) > 3 else str(path)


def _quantile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _pareto_indices(points: Sequence[tuple[float, float]]) -> List[int]:
    indices: List[int] = []
    for i, (x_i, y_i) in enumerate(points):
        if not math.isfinite(x_i) or not math.isfinite(y_i):
            continue
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j or not math.isfinite(x_j) or not math.isfinite(y_j):
                continue
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return indices
