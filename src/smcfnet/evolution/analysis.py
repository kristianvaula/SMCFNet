"""Offline analysis helpers for completed EA candidate archives."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from smcfnet.evolution.archive import OBJECTIVE_NAMES
from smcfnet.evolution.plotting import load_candidate_records


DEFAULT_TARGET_KEYS = (
    "objectives.f_quality",
    "objectives.f_flow_structure",
    "objectives.f_snn_stability",
    "objectives.f_compute_cost",
    "metrics_late.mcm_epe_median",
    "metrics_late.validation_mcm_epe_median",
    "metrics_late.validation_quality",
    "metrics_late.flow_mag_mean_median",
    "metrics_late.centered_ratio",
    "metrics_late.flow_mean_flow_ratio_median",
    "metrics_late.spatial_std_median",
    "metrics_late.iwe_contrast_median",
    "metrics_late.smooth_raw_median",
    "metrics_late.warp_structure_proxy",
    "metrics_late.gpu_peak_mb",
    "metrics_late.wall_ms_per_iter",
)


def analyze_ea_archive(results: str | Path, out_dir: str | Path) -> Dict[str, Any]:
    """Write correlation and hyperparameter-importance artifacts for an archive.

    The analysis intentionally avoids heavy optional dependencies. Importance is
    a model-free screening score based on absolute Spearman correlation between
    each flattened genome leaf and each objective/late metric.
    """

    records = [record for record in load_candidate_records(results) if record.get("status") == "completed"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not records:
        summary = {"completed_records": 0, "targets": [], "genes": []}
        (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    target_keys = _available_numeric_keys(records, DEFAULT_TARGET_KEYS)
    gene_keys = sorted({key for record in records for key, _ in _flatten_leaves(record.get("genome", {}), prefix="genome")})

    correlations = objective_correlations(records, target_keys)
    importance = hyperparameter_importance(records, gene_keys, target_keys)

    _write_csv(out / "objective_correlations.csv", correlations)
    _write_csv(out / "hyperparameter_importance.csv", importance)
    _write_markdown_summary(out / "analysis_summary.md", records, correlations, importance)
    summary = {
        "completed_records": int(len(records)),
        "targets": target_keys,
        "genes": gene_keys,
        "objective_correlation_rows": int(len(correlations)),
        "importance_rows": int(len(importance)),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_analysis_plots(out, correlations, importance)
    return summary


def objective_correlations(records: Sequence[Mapping[str, Any]], target_keys: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, left_key in enumerate(target_keys):
        for right_key in target_keys[i + 1 :]:
            pairs = _paired_numeric(records, left_key, right_key)
            if len(pairs) < 3:
                continue
            xs, ys = zip(*pairs)
            pearson = _pearson(xs, ys)
            spearman = _spearman(xs, ys)
            if not math.isfinite(pearson) and not math.isfinite(spearman):
                continue
            rows.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "n": int(len(pairs)),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    rows.sort(key=lambda row: abs(float(row.get("spearman", 0.0))) if math.isfinite(float(row.get("spearman", 0.0))) else 0.0, reverse=True)
    return rows


def hyperparameter_importance(
    records: Sequence[Mapping[str, Any]],
    gene_keys: Sequence[str],
    target_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gene_key in gene_keys:
        encoded = _encoded_series(records, gene_key)
        values = [value for value in encoded if value is not None and math.isfinite(float(value))]
        if len(values) < 3 or len(set(values)) < 2:
            continue
        target_scores: Dict[str, float] = {}
        n_values: List[int] = []
        for target_key in target_keys:
            pairs = []
            for record, gene_value in zip(records, encoded):
                target_value = _numeric_at(record, target_key)
                if gene_value is None or target_value is None:
                    continue
                pairs.append((float(gene_value), float(target_value)))
            if len(pairs) < 3:
                continue
            xs, ys = zip(*pairs)
            score = _spearman(xs, ys)
            if math.isfinite(score):
                target_scores[target_key] = float(score)
                n_values.append(len(pairs))
        if not target_scores:
            continue
        abs_scores = [abs(score) for score in target_scores.values()]
        row: Dict[str, Any] = {
            "gene": gene_key,
            "n_min": int(min(n_values)),
            "n_max": int(max(n_values)),
            "distinct_values": int(len(set(values))),
            "importance_mean_abs_spearman": float(sum(abs_scores) / len(abs_scores)),
            "importance_max_abs_spearman": float(max(abs_scores)),
            "strongest_target": max(target_scores, key=lambda key: abs(target_scores[key])),
            "strongest_spearman": float(target_scores[max(target_scores, key=lambda key: abs(target_scores[key]))]),
        }
        for target_key in target_keys:
            if target_key in target_scores:
                row[f"spearman::{target_key}"] = target_scores[target_key]
        rows.append(row)
    rows.sort(key=lambda row: (float(row["importance_max_abs_spearman"]), float(row["importance_mean_abs_spearman"])), reverse=True)
    return rows


def _available_numeric_keys(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[str]:
    return [key for key in keys if sum(1 for record in records if _numeric_at(record, key) is not None) >= 3]


def _paired_numeric(records: Sequence[Mapping[str, Any]], left_key: str, right_key: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for record in records:
        left = _numeric_at(record, left_key)
        right = _numeric_at(record, right_key)
        if left is not None and right is not None:
            pairs.append((left, right))
    return pairs


def _encoded_series(records: Sequence[Mapping[str, Any]], key: str) -> List[Optional[float]]:
    raw = [_value_at(record, key) for record in records]
    numeric: List[Optional[float]] = []
    all_numeric = True
    for value in raw:
        value_f = _to_float(value)
        numeric.append(value_f)
        if value is not None and value_f is None:
            all_numeric = False
    if all_numeric:
        return numeric

    categories = sorted({json.dumps(value, sort_keys=True) for value in raw if value is not None})
    mapping = {category: float(index) for index, category in enumerate(categories)}
    return [mapping.get(json.dumps(value, sort_keys=True)) if value is not None else None for value in raw]


def _numeric_at(record: Mapping[str, Any], key: str) -> Optional[float]:
    return _to_float(_value_at(record, key))


def _value_at(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and part.isdigit():
            idx = int(part)
            if idx >= len(value):
                return None
            value = value[idx]
        else:
            return None
    return value


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _flatten_leaves(value: Any, *, prefix: str) -> List[Tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: List[Tuple[str, Any]] = []
        for key in sorted(value):
            rows.extend(_flatten_leaves(value[key], prefix=f"{prefix}.{key}"))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_leaves(item, prefix=f"{prefix}.{index}"))
        return rows
    return [(prefix, value)]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float(num / (den_x * den_y))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0 for _ in ordered]
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for _, index in ordered[i:j]:
            ranks[index] = rank
        i = j
    return ranks


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    importance: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# EA Archive Analysis",
        "",
        f"- completed candidates: {len(records)}",
        f"- objective-correlation rows: {len(correlations)}",
        f"- hyperparameter-importance rows: {len(importance)}",
        "",
        "## Strongest Objective Correlations",
        "",
    ]
    for row in correlations[:10]:
        lines.append(
            f"- `{row['left']}` vs `{row['right']}`: "
            f"spearman={float(row['spearman']):.4f}, pearson={float(row['pearson']):.4f}, n={row['n']}"
        )
    lines.extend(["", "## Strongest Hyperparameter Signals", ""])
    for row in importance[:15]:
        lines.append(
            f"- `{row['gene']}` -> `{row['strongest_target']}`: "
            f"spearman={float(row['strongest_spearman']):.4f}, "
            f"max_abs={float(row['importance_max_abs_spearman']):.4f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_analysis_plots(
    out: Path,
    correlations: Sequence[Mapping[str, Any]],
    importance: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out / "plot_error.txt").write_text(str(exc))
        return

    _plot_correlation_heatmap(correlations, out / "07_objective_correlation_heatmap.png", plt)
    _plot_importance(importance, out / "08_hyperparameter_importance.png", plt)


def _plot_correlation_heatmap(correlations: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    labels = sorted({str(row["left"]) for row in correlations} | {str(row["right"]) for row in correlations})
    if not labels:
        return
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0.0 for _ in labels] for _ in labels]
    for i in range(len(labels)):
        matrix[i][i] = 1.0
    for row in correlations:
        i = index[str(row["left"])]
        j = index[str(row["right"])]
        value = float(row.get("spearman", 0.0))
        matrix[i][j] = value
        matrix[j][i] = value

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([_short_label(label) for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([_short_label(label) for label in labels])
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_importance(importance: Sequence[Mapping[str, Any]], path: Path, plt: Any) -> None:
    rows = list(importance[:20])
    if not rows:
        return
    labels = [_short_label(str(row["gene"]).replace("genome.", "")) for row in rows]
    values = [float(row["importance_max_abs_spearman"]) for row in rows]
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(rows))), constrained_layout=True)
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("max |Spearman| across objectives/late metrics")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _short_label(label: str) -> str:
    replacements = {
        "objectives.": "",
        "metrics_late.": "",
        "mcm_epe_median": "EPE",
        "flow_mag_mean_median": "flow_mag",
        "flow_mean_flow_ratio_median": "mean_flow",
        "spatial_std_median": "spatial_std",
        "wall_ms_per_iter": "wall_ms",
        "sofe_architecture.": "sofe.",
    }
    out = label
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out
