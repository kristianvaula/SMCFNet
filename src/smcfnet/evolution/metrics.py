"""Generic metric parsing and objective helpers for EA evaluation."""

from __future__ import annotations

import math
import statistics
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from smcfnet.evolution.framework import Fitness


_METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")

QUALITY_FAILURE_VALUE = 1.0e6
EPE_MISSING_VALUE = 1.0e5
FLOW_MAG_MIN = 0.10
FLOW_CENTERED_RATIO_MIN = 0.25
FLOW_MEAN_RATIO_MAX = 0.995
FLOW_SPATIAL_STD_MIN = 0.002
SPIKE_RATE_MIN = 0.001
SPIKE_RATE_MAX = 0.50
VOLTAGE_P99_MAX = 20.0
MEM_PEAK_SOFT_LIMIT_MB = 40000.0
WALL_MS_SOFT_LIMIT = 1000.0


def parse_mcm_log_metrics(
    path: str | Path,
    *,
    max_iter: Optional[int] = None,
    include_validation: bool = False,
) -> List[Dict[str, float]]:
    """Parse scalar MCM metrics from train.log lines."""

    log_path = Path(path)
    if not log_path.exists():
        return []

    rows: List[Dict[str, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "[MCM]" not in line:
            continue
        if "[MCM][VAL]" in line and not include_validation:
            continue
        if "[MCM] Epoch" in line:
            continue
        row: Dict[str, float] = {}
        for key, value in _METRIC_RE.findall(line):
            try:
                row[str(key)] = float(value)
            except ValueError:
                continue
        if "iter" in row and max_iter is not None and int(row["iter"]) > int(max_iter):
            continue
        if row:
            rows.append(row)
    return rows


def parse_mcm_validation_log_metrics(
    path: str | Path,
    *,
    max_iter: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Parse ``[MCM][VAL]`` rows and normalize ``val_*`` keys.

    Validation rows are intentionally mapped onto the same key names as train
    rows where possible, so late-window quality helpers can reuse the same
    objective code. For example ``val_mcm_epe`` becomes ``mcm_epe``.
    """

    log_path = Path(path)
    if not log_path.exists():
        return []

    rows: List[Dict[str, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "[MCM][VAL]" not in line:
            continue
        row: Dict[str, float] = {}
        for key, value in _METRIC_RE.findall(line):
            normalized_key = str(key)
            if normalized_key.startswith("val_"):
                normalized_key = normalized_key[4:]
            try:
                row[normalized_key] = float(value)
            except ValueError:
                continue
        if "iter" in row and max_iter is not None and int(row["iter"]) > int(max_iter):
            continue
        if row:
            rows.append(row)
    return rows


def summarize_metric_rows(rows: Sequence[Mapping[str, float]], *, prefix: str = "train") -> Dict[str, float]:
    """Summarize scalar metrics from parsed rows."""

    if not rows:
        return {}

    keys = sorted({key for row in rows for key in row})
    summary: Dict[str, float] = {}
    latest = rows[-1]
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if not values:
            continue
        summary[f"{prefix}/{key}_last"] = float(latest[key]) if key in latest else float(values[-1])
        summary[f"{prefix}/{key}_mean"] = float(sum(values) / float(len(values)))
        summary[f"{prefix}/{key}_median"] = _median(values)
        summary[f"{prefix}/{key}_iqr"] = _iqr(values)
        summary[f"{prefix}/{key}_min"] = float(min(values))
        summary[f"{prefix}/{key}_max"] = float(max(values))
    return summary


def _metric(metrics: Mapping[str, float], name: str, *, default: float = float("inf")) -> float:
    value = metrics.get(name, default)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return value_f if math.isfinite(value_f) else default


def _finite_values(rows: Sequence[Mapping[str, float]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _median(values: Sequence[float], *, default: float = float("inf")) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float(default)
    return float(statistics.median(finite))


def _iqr(values: Sequence[float], *, default: float = 0.0) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if len(finite) < 2:
        return float(default)
    mid = len(finite) // 2
    lower = finite[:mid]
    upper = finite[mid:] if len(finite) % 2 == 0 else finite[mid + 1 :]
    if not lower or not upper:
        return float(default)
    return float(statistics.median(upper) - statistics.median(lower))


def _last_fraction(rows: Sequence[Mapping[str, float]], frac: float = 0.30) -> Sequence[Mapping[str, float]]:
    if not rows:
        return []
    count = max(1, int(math.ceil(float(len(rows)) * float(frac))))
    return rows[-count:]


def _first_fraction(rows: Sequence[Mapping[str, float]], frac: float = 0.30) -> Sequence[Mapping[str, float]]:
    if not rows:
        return []
    count = max(1, int(math.ceil(float(len(rows)) * float(frac))))
    return rows[:count]


def _relu(value: float) -> float:
    return max(float(value), 0.0)


def _finite_or(value: float, default: float) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_f if math.isfinite(value_f) else float(default)


def _median_from_rows_or_summary(
    rows: Sequence[Mapping[str, float]],
    metrics: Mapping[str, float],
    key: str,
    *,
    prefix: str,
    default: float,
) -> float:
    values = _finite_values(rows, key)
    if values:
        return _median(values, default=default)
    return _metric(metrics, f"{prefix}/{key}_median", default=_metric(metrics, f"{prefix}/{key}_mean", default=default))


def _iqr_from_rows_or_summary(
    rows: Sequence[Mapping[str, float]],
    metrics: Mapping[str, float],
    key: str,
    *,
    prefix: str,
    default: float = 0.0,
) -> float:
    values = _finite_values(rows, key)
    if values:
        return _iqr(values, default=default)
    return _metric(metrics, f"{prefix}/{key}_iqr", default=default)


def _has_finite_metric(
    rows: Sequence[Mapping[str, float]],
    metrics: Mapping[str, float],
    key: str,
    *,
    prefix: str,
) -> bool:
    if _finite_values(rows, key):
        return True
    return math.isfinite(
        _metric(
            metrics,
            f"{prefix}/{key}_median",
            default=_metric(metrics, f"{prefix}/{key}_mean", default=float("nan")),
        )
    )


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float(default)
    return float(sum(finite) / float(len(finite)))


def _spike_rate_keys(rows: Sequence[Mapping[str, float]]) -> List[str]:
    preferred = [
        "G1_spike_rate",
        "R1a_spike_rate",
        "R1b_spike_rate",
        "G2_spike_rate",
        "R2a_spike_rate",
        "R2b_spike_rate",
    ]
    present = {key for row in rows for key in row}
    keys = [key for key in preferred if key in present]
    if keys:
        return keys
    fallback = [
        "head_spike_rate_active_mean",
        "head_spike_rate_readout_mean",
        "head_spike_rate_final",
        "head_spike_rate",
    ]
    return [key for key in fallback if key in present]


def _build_row_fitness(
    metrics: Mapping[str, float],
    rows: Sequence[Mapping[str, float]],
    *,
    prefix: str,
    budget_iters: Optional[int],
    wall_ms_soft_limit: float = WALL_MS_SOFT_LIMIT,
    val_quality_key: Optional[str] = None,
    flow_mag_min: float = FLOW_MAG_MIN,
    flow_centered_ratio_min: float = FLOW_CENTERED_RATIO_MIN,
    flow_mean_ratio_max: float = FLOW_MEAN_RATIO_MAX,
    flow_spatial_std_min: float = FLOW_SPATIAL_STD_MIN,
    constraint_violation_weight: float = 0.05,
) -> Fitness:
    eps = 1e-8
    del budget_iters
    late_rows = list(_last_fraction(rows, frac=0.50))
    early_rows = list(_first_fraction(rows, frac=0.50))

    quality_key = (str(val_quality_key) if val_quality_key else "mcm_epe") if prefix == "val" else "total"
    quality_source = 1.0 if quality_key == "mcm_epe" else 2.0
    quality_available = _has_finite_metric(rows, metrics, quality_key, prefix=prefix)
    quality_late = _median_from_rows_or_summary(
        late_rows,
        metrics,
        quality_key,
        prefix=prefix,
        default=QUALITY_FAILURE_VALUE,
    )
    quality_iqr = _iqr_from_rows_or_summary(late_rows, metrics, quality_key, prefix=prefix)
    quality_early = _median_from_rows_or_summary(early_rows, metrics, quality_key, prefix=prefix, default=quality_late)
    quality_late = _finite_or(quality_late, QUALITY_FAILURE_VALUE)
    quality_iqr = _finite_or(quality_iqr, 0.0)
    quality_early = _finite_or(quality_early, quality_late)
    quality_improvement = float(quality_early - quality_late)

    epe_available = _has_finite_metric(rows, metrics, "mcm_epe", prefix=prefix)
    epe_late = _median_from_rows_or_summary(
        late_rows,
        metrics,
        "mcm_epe",
        prefix=prefix,
        default=EPE_MISSING_VALUE,
    )
    epe_iqr = _iqr_from_rows_or_summary(late_rows, metrics, "mcm_epe", prefix=prefix)
    epe_early = _median_from_rows_or_summary(early_rows, metrics, "mcm_epe", prefix=prefix, default=epe_late)
    epe_late = _finite_or(epe_late, EPE_MISSING_VALUE)
    epe_iqr = _finite_or(epe_iqr, 0.0)
    epe_early = _finite_or(epe_early, epe_late)
    epe_improvement = float(epe_early - epe_late) if epe_available else 0.0

    flow_mag = _median_from_rows_or_summary(late_rows, metrics, "flow_mag_mean", prefix=prefix, default=0.0)
    centered_mag = _median_from_rows_or_summary(
        late_rows, metrics, "flow_centered_mag_mean", prefix=prefix, default=0.0
    )
    centered_ratio = float(centered_mag / (flow_mag + eps))
    mean_ratio = _median_from_rows_or_summary(
        late_rows, metrics, "flow_mean_flow_ratio", prefix=prefix, default=1.0
    )
    u_std = _median_from_rows_or_summary(
        late_rows, metrics, "flow_u_spatial_std_mean", prefix=prefix, default=0.0
    )
    v_std = _median_from_rows_or_summary(
        late_rows, metrics, "flow_v_spatial_std_mean", prefix=prefix, default=0.0
    )
    spatial_std = 0.5 * (u_std + v_std)
    if spatial_std <= 0.0:
        dx = _median_from_rows_or_summary(late_rows, metrics, "flow_dx_mean", prefix=prefix, default=0.0)
        dy = _median_from_rows_or_summary(late_rows, metrics, "flow_dy_mean", prefix=prefix, default=0.0)
        spatial_std = 0.5 * (dx + dy)

    p_under = _relu(flow_mag_min - flow_mag) / max(flow_mag_min, eps)
    p_centered = _relu(flow_centered_ratio_min - centered_ratio) / max(flow_centered_ratio_min, eps)
    p_global = _relu(mean_ratio - flow_mean_ratio_max) / max(1.0 - flow_mean_ratio_max, eps)
    p_flat = _relu(flow_spatial_std_min - spatial_std) / max(flow_spatial_std_min, eps)
    f_flow_structure = float(p_under + p_centered + p_global + p_flat)

    spike_keys = _spike_rate_keys(late_rows)
    layer_rates = [
        _median_from_rows_or_summary(late_rows, metrics, key, prefix=prefix, default=0.0)
        for key in spike_keys
    ]
    p_silent = (
        _mean([_relu(SPIKE_RATE_MIN - rate) / max(SPIKE_RATE_MIN, eps) for rate in layer_rates])
        if layer_rates
        else 0.0
    )
    p_saturated = _mean([_relu(rate - SPIKE_RATE_MAX) for rate in layer_rates]) if layer_rates else 0.0
    voltage = _median_from_rows_or_summary(
        late_rows,
        metrics,
        "head_v_minus_thresh_active_p99",
        prefix=prefix,
        default=0.0,
    )
    p_voltage = _relu(voltage - VOLTAGE_P99_MAX) / VOLTAGE_P99_MAX
    p_noisy_epe = epe_iqr / (epe_late + eps) if math.isfinite(epe_late) else 0.0
    f_snn_stability = float(p_silent + p_saturated + 0.25 * p_voltage)

    mem_peak = max(
        _median_from_rows_or_summary(
            late_rows,
            metrics,
            "cuda_peak_memory_allocated_after_chunks_mb",
            prefix=prefix,
            default=0.0,
        ),
        _metric(metrics, f"{prefix}/gpu_mem_peak_allocated_mb_max", default=0.0),
    )
    num_steps = _median_from_rows_or_summary(late_rows, metrics, "num_steps", prefix=prefix, default=0.0)
    num_readouts = _median_from_rows_or_summary(late_rows, metrics, "num_readouts", prefix=prefix, default=0.0)
    spike_proxy = _mean(layer_rates, default=0.0)
    wall_ms_per_iter = 1000.0 * _metric(metrics, "candidate/wall_time_per_iter_sec", default=0.0)
    contrast = _median_from_rows_or_summary(late_rows, metrics, "mcm_contrast", prefix=prefix, default=0.0)
    iwe_contrast = _median_from_rows_or_summary(late_rows, metrics, "iwe_contrast", prefix=prefix, default=contrast)
    smooth_raw = _median_from_rows_or_summary(late_rows, metrics, "mcm_smooth_raw", prefix=prefix, default=0.0)
    warp_structure_proxy = float(flow_mag * centered_ratio * max(0.0, 1.0 - min(mean_ratio, 1.0)))
    f_compute_cost = float(
        0.25 * math.log1p(max(mem_peak, 0.0) / 2000.0)
        + 0.25 * math.log1p(max(num_steps, 0.0) / 100.0)
        + 0.20 * math.log1p(max(num_readouts, 0.0) / 20.0)
        + 0.20 * math.log1p(max(wall_ms_per_iter, 0.0) / 1000.0)
        + 0.10 * max(spike_proxy, 0.0)
    )
    compute_constraint = float(
        _relu(mem_peak - MEM_PEAK_SOFT_LIMIT_MB) / MEM_PEAK_SOFT_LIMIT_MB
        + _relu(wall_ms_per_iter - wall_ms_soft_limit) / wall_ms_soft_limit
    )
    if prefix == "val":
        f_flow_structure = 0.0
        f_snn_stability = 0.0
        f_compute_cost = 0.0
        compute_constraint = 0.0
    constraint_violation = float(f_flow_structure + f_snn_stability + compute_constraint)
    f_quality = float(
        quality_late
        + 0.10 * quality_iqr
        - 0.02 * max(quality_improvement, 0.0)
        + float(constraint_violation_weight) * min(constraint_violation, 10.0)
        + 0.005 * min(p_noisy_epe, 10.0)
        + 0.01 * min(f_compute_cost, 10.0)
    )
    if not quality_available:
        f_quality = QUALITY_FAILURE_VALUE + constraint_violation

    enriched = dict(metrics)
    enriched.update(
        {
            "fitness/epe_late": float(epe_late),
            "fitness/epe_iqr_late": float(epe_iqr),
            "fitness/epe_improvement": float(epe_improvement),
            "fitness/quality_late": float(quality_late),
            "fitness/quality_iqr_late": float(quality_iqr),
            "fitness/quality_improvement": float(quality_improvement),
            "fitness/quality_source": float(quality_source),
            "fitness/epe_available": float(epe_available),
            "fitness/epe_tiebreaker": float(epe_late + 0.10 * epe_iqr),
            "fitness/flow_mag_late": float(flow_mag),
            "fitness/flow_centered_late": float(centered_mag),
            "fitness/centered_ratio_late": float(centered_ratio),
            "fitness/flow_mean_flow_ratio_late": float(mean_ratio),
            "fitness/spatial_std_late": float(spatial_std),
            "fitness/mag_min_threshold": float(flow_mag_min),
            "fitness/centered_ratio_min_threshold": float(flow_centered_ratio_min),
            "fitness/mean_ratio_max_threshold": float(flow_mean_ratio_max),
            "fitness/std_min_threshold": float(flow_spatial_std_min),
            "fitness/constraint_violation_weight": float(constraint_violation_weight),
            "fitness/p_under_scaled": float(p_under),
            "fitness/p_centered": float(p_centered),
            "fitness/p_global_collapse": float(p_global),
            "fitness/p_flat_spatial": float(p_flat),
            "fitness/p_silent": float(p_silent),
            "fitness/p_saturated": float(p_saturated),
            "fitness/p_voltage": float(p_voltage),
            "fitness/p_noisy_epe": float(p_noisy_epe),
            "fitness/mem_peak_mb_late": float(mem_peak),
            "fitness/num_steps_late": float(num_steps),
            "fitness/num_readouts_late": float(num_readouts),
            "fitness/spike_proxy_late": float(spike_proxy),
            "fitness/wall_ms_per_iter": float(wall_ms_per_iter),
            "fitness/iwe_contrast_late": float(iwe_contrast),
            "fitness/mcm_contrast_late": float(contrast),
            "fitness/smooth_raw_late": float(smooth_raw),
            "fitness/warp_structure_proxy_late": float(warp_structure_proxy),
            "fitness/compute_constraint": float(compute_constraint),
            "fitness/constraint_violation": float(constraint_violation),
            "fitness/objective_quality": float(f_quality),
            "fitness/objective_flow_structure": float(f_flow_structure),
            "fitness/objective_snn_stability": float(f_snn_stability),
            "fitness/objective_compute_cost": float(f_compute_cost),
        }
    )
    objectives = (f_quality, f_flow_structure, f_snn_stability, f_compute_cost)
    valid = (
        bool(quality_available)
        and all(math.isfinite(v) for v in objectives)
        and f_flow_structure <= 2.0
        and f_snn_stability <= 1.0
        and compute_constraint <= 0.0
    )
    violations = []
    if not quality_available:
        violations.append(f"missing_{quality_key}")
    if f_flow_structure > 2.0:
        violations.append("degenerate_flow")
    if f_snn_stability > 1.0:
        violations.append("unstable_snn")
    if compute_constraint > 0.0:
        violations.append("compute_budget")
    notes = {
        "quality_source": quality_key,
        "constraints": ",".join(violations),
        "epe_available": bool(epe_available),
    }
    return Fitness(metrics=enriched, objectives=objectives, maximize=(False, False, False, False), valid=valid, notes=notes)


def build_convergence_fitness(
    metrics: Mapping[str, float],
    *,
    prefix: str = "train",
    memory_weight: float = 0.05,
    rows: Optional[Sequence[Mapping[str, float]]] = None,
    budget_iters: Optional[int] = None,
    wall_ms_soft_limit: float = WALL_MS_SOFT_LIMIT,
    val_quality_key: Optional[str] = None,
    flow_mag_min: float = FLOW_MAG_MIN,
    flow_centered_ratio_min: float = FLOW_CENTERED_RATIO_MIN,
    flow_mean_ratio_max: float = FLOW_MEAN_RATIO_MAX,
    flow_spatial_std_min: float = FLOW_SPATIAL_STD_MIN,
    constraint_violation_weight: float = 0.05,
) -> Fitness:
    """Build a convergence-oriented multi-objective fitness.

    Training quality is always based on the self-supervised total loss so
    candidates stay commensurable even when EPE is missing. Validation fitness
    defaults to EPE (``val_quality_key=None``) because final-rung validation
    historically asks for that signal; pass ``val_quality_key="total"`` for
    datasets without GT flow (e.g. PEOD) so validation fitness falls back to
    the held-out self-supervised contrast+smoothness loss instead of being
    unconditionally invalid. Degenerate flow, unstable SNN dynamics, and
    compute over-budget are logged as finite penalties and expressed through
    ``Fitness.valid``; the flow-structure thresholds and the weight applied to
    constraint_violation inside f_quality are configurable so callers can
    tighten the DC/global-collapse guardrail for data where it is a known
    cheap local minimum.
    """
    del memory_weight
    threshold_kwargs = dict(
        val_quality_key=val_quality_key,
        flow_mag_min=flow_mag_min,
        flow_centered_ratio_min=flow_centered_ratio_min,
        flow_mean_ratio_max=flow_mean_ratio_max,
        flow_spatial_std_min=flow_spatial_std_min,
        constraint_violation_weight=constraint_violation_weight,
    )

    if rows:
        return _build_row_fitness(
            metrics, list(rows), prefix=prefix, budget_iters=budget_iters,
            wall_ms_soft_limit=wall_ms_soft_limit, **threshold_kwargs,
        )

    synthetic_rows = []
    if metrics:
        synthetic_rows.append({})
    return _build_row_fitness(
        metrics, synthetic_rows, prefix=prefix, budget_iters=budget_iters,
        wall_ms_soft_limit=wall_ms_soft_limit, **threshold_kwargs,
    )
