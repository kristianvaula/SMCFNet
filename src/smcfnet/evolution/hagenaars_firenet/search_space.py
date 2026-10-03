"""Hagenaars FireNet-LIF EA search-space presets and guardrails."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from smcfnet.evolution.framework import GeneSpec


HAGENAARS_FIRENET_LIF_SEARCH_SPACE: Dict[str, Any] = {
    "mcm": {
        "step_time_us": [250, 500, 1000, 1250, 2500, 5000, 10000],
        "events_per_step_target": [5000, 10000, 15000, 25000, 50000],
        "bptt_steps": [3, 5, 8, 10],
        "decay_tau_us": [2500.0, 5000.0, 10000.0, 20000.0, 40000.0],
        "flow_scaling": [1.0, 4.0, 8.0, 12.0, 16.0, 24.0],
        "smooth_weight": [0.03, 0.1, 0.3, 1.0, 2.0],
        "contrast_weight": [0.5, 1.0, 2.0],
        "loss_weight": [0.05, 0.1, 0.2],
    },
    "sofe_architecture": {
        "count_scale": [0.25, 0.5, 1.0, 2.0, 4.0],
        "base_channels": [16, 24, 32, 48],
        "kernel_size": [3, 5],
        "spiking_v_threshold": [0.5, 0.8, 1.0, 1.2],
        "surrogate_function": ["atan", "sigmoid", "softsign"],
        "surrogate_alpha": [0.5, 1.0, 2.0, 5.0, 10.0],
        "detach_reset": [True, False],
        "firenet_spike_decay_mode": ["hold", "decay", "zero"],
        "flow_activation": ["softsign", "tanh", "identity"],
        "flow_head_init_std": [0.005, 0.01, 0.03, 0.05, 0.08],
        "hagenaars_use_bias": [True, False],
        "hagenaars_pred_kernel_size": [1, 3, 5],
    },
}

HAGENAARS_FIRENET_LIF_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("mcm", "step_time_us"): GeneSpec(
        path=("mcm", "step_time_us"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["step_time_us"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "events_per_step_target"): GeneSpec(
        path=("mcm", "events_per_step_target"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["events_per_step_target"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "bptt_steps"): GeneSpec(
        path=("mcm", "bptt_steps"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["bptt_steps"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "decay_tau_us"): GeneSpec(
        path=("mcm", "decay_tau_us"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["decay_tau_us"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "flow_scaling"): GeneSpec(
        path=("mcm", "flow_scaling"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["flow_scaling"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "smooth_weight"): GeneSpec(
        path=("mcm", "smooth_weight"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["smooth_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "contrast_weight"): GeneSpec(
        path=("mcm", "contrast_weight"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["contrast_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "loss_weight"): GeneSpec(
        path=("mcm", "loss_weight"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["mcm"]["loss_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "count_scale"): GeneSpec(
        path=("sofe_architecture", "count_scale"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["count_scale"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "base_channels"): GeneSpec(
        path=("sofe_architecture", "base_channels"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["base_channels"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "kernel_size"): GeneSpec(
        path=("sofe_architecture", "kernel_size"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "spiking_v_threshold"): GeneSpec(
        path=("sofe_architecture", "spiking_v_threshold"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["spiking_v_threshold"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "surrogate_function"): GeneSpec(
        path=("sofe_architecture", "surrogate_function"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["surrogate_function"],
        sensitivity="light",
        ordered=False,
    ),
    ("sofe_architecture", "surrogate_alpha"): GeneSpec(
        path=("sofe_architecture", "surrogate_alpha"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["surrogate_alpha"],
        sensitivity="light",
        ordered=True,
    ),
    ("sofe_architecture", "detach_reset"): GeneSpec(
        path=("sofe_architecture", "detach_reset"),
        kind="bool",
        sensitivity="light",
    ),
    ("sofe_architecture", "firenet_spike_decay_mode"): GeneSpec(
        path=("sofe_architecture", "firenet_spike_decay_mode"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["firenet_spike_decay_mode"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "flow_activation"): GeneSpec(
        path=("sofe_architecture", "flow_activation"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["flow_activation"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "flow_head_init_std"): GeneSpec(
        path=("sofe_architecture", "flow_head_init_std"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["flow_head_init_std"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "hagenaars_use_bias"): GeneSpec(
        path=("sofe_architecture", "hagenaars_use_bias"),
        kind="bool",
        sensitivity="light",
    ),
    ("sofe_architecture", "hagenaars_pred_kernel_size"): GeneSpec(
        path=("sofe_architecture", "hagenaars_pred_kernel_size"),
        kind="categorical",
        values=HAGENAARS_FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["hagenaars_pred_kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
}

HAGENAARS_FIRENET_LIF_CROSSOVER_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "temporal_partitioning": (
        ("mcm", "step_time_us"),
        ("mcm", "events_per_step_target"),
        ("mcm", "decay_tau_us"),
        ("mcm", "bptt_steps"),
    ),
    "training_bptt": (
        ("mcm", "smooth_weight"),
        ("mcm", "contrast_weight"),
        ("mcm", "loss_weight"),
    ),
    "lif_dynamics": (
        ("sofe_architecture", "spiking_v_threshold"),
        ("sofe_architecture", "surrogate_function"),
        ("sofe_architecture", "surrogate_alpha"),
        ("sofe_architecture", "detach_reset"),
        ("sofe_architecture", "firenet_spike_decay_mode"),
    ),
    "architecture_capacity": (
        ("sofe_architecture", "base_channels"),
        ("sofe_architecture", "kernel_size"),
        ("sofe_architecture", "count_scale"),
        ("sofe_architecture", "hagenaars_use_bias"),
    ),
    "flow_readout": (
        ("mcm", "flow_scaling"),
        ("sofe_architecture", "flow_activation"),
        ("sofe_architecture", "flow_head_init_std"),
        ("sofe_architecture", "hagenaars_pred_kernel_size"),
    ),
}

HAGENAARS_FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("mcm", "decay_tau_us"),
    ("mcm", "flow_scaling"),
    ("mcm", "smooth_weight"),
    ("mcm", "contrast_weight"),
    ("mcm", "loss_weight"),
    ("sofe_architecture", "flow_head_init_std"),
    ("sofe_architecture", "surrogate_alpha"),
)

HAGENAARS_FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("sofe_architecture", "spiking_v_threshold"),
)


@dataclass
class HagenaarsFireNetLIFConstraintResult:
    """Result of applying Hagenaars FireNet-LIF-specific guardrails."""

    genome: Dict[str, Any]
    valid: bool = True
    repairs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def apply_hagenaars_firenet_lif_guardrails(genome: Mapping[str, Any]) -> HagenaarsFireNetLIFConstraintResult:
    """Validate/repair Hagenaars FireNet-LIF genomes before evaluation."""

    repaired = copy.deepcopy(dict(genome))
    repairs: List[str] = []
    warnings: List[str] = []

    mcm = dict(repaired.setdefault("mcm", {}))
    sofe = dict(repaired.setdefault("sofe_architecture", {}))

    events = int(mcm.get("events_per_step_target", 15000))
    bptt = int(mcm.get("bptt_steps", 5))
    flow_scaling = float(mcm.get("flow_scaling", 4.0))
    has_step_time_us = "step_time_us" in mcm
    step_time_us = mcm.get("step_time_us", None)
    step_time_us = None if step_time_us is None else int(step_time_us)
    decay_tau_us = mcm.get("decay_tau_us", 10000.0)
    decay_tau_us = None if decay_tau_us is None else float(decay_tau_us)
    if step_time_us is not None:
        if step_time_us <= 0:
            raise ValueError(f"step_time_us must be positive or null, got {step_time_us}")
        if 50000 % step_time_us != 0:
            raise ValueError("fixed-time step_time_us must evenly divide the 50 ms event window")
    if events <= 0:
        raise ValueError(f"events_per_step_target must be positive, got {events}")
    if bptt < 1:
        raise ValueError(f"bptt_steps must be >= 1, got {bptt}")
    if flow_scaling <= 0.0:
        raise ValueError(f"flow_scaling must be positive, got {flow_scaling}")
    if decay_tau_us is not None and decay_tau_us <= 0.0:
        raise ValueError(f"decay_tau_us must be positive or null, got {decay_tau_us}")

    count_scale = float(sofe.get("count_scale", 1.0))
    base_channels = int(sofe.get("base_channels", 32))
    kernel_size = int(sofe.get("kernel_size", 3))
    threshold = float(sofe.get("spiking_v_threshold", 1.0))
    pred_kernel_size = int(sofe.get("hagenaars_pred_kernel_size", 3))

    if kernel_size <= 0 or kernel_size % 2 == 0:
        old_kernel = kernel_size
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sofe["kernel_size"] = int(kernel_size)
        repairs.append(f"repaired kernel_size from {old_kernel} to {kernel_size}")
    if pred_kernel_size <= 0 or pred_kernel_size % 2 == 0:
        old_pred_kernel = pred_kernel_size
        pred_kernel_size = max(1, pred_kernel_size)
        if pred_kernel_size % 2 == 0:
            pred_kernel_size += 1
        sofe["hagenaars_pred_kernel_size"] = int(pred_kernel_size)
        repairs.append(f"repaired hagenaars_pred_kernel_size from {old_pred_kernel} to {pred_kernel_size}")

    if threshold <= 0.0:
        sofe["spiking_v_threshold"] = 0.5
        repairs.append(f"raised spiking_v_threshold from {threshold} to 0.5")

    if events * bptt > 350000:
        old_bptt = bptt
        bptt = max(1, 350000 // max(events, 1))
        mcm["bptt_steps"] = int(bptt)
        repairs.append(f"capped bptt_steps from {old_bptt} to {bptt} for full-resolution events_per_step_target={events}")

    if base_channels >= 48 and events * bptt > 200000:
        old_bptt = bptt
        bptt = max(1, 200000 // max(events, 1))
        mcm["bptt_steps"] = int(bptt)
        repairs.append(f"capped bptt_steps from {old_bptt} to {bptt} for base_channels={base_channels}")

    if count_scale >= 4.0 and threshold <= 0.5:
        sofe["spiking_v_threshold"] = 0.8
        repairs.append("raised spiking_v_threshold to 0.8 for high count_scale")

    if base_channels >= 48 and events >= 50000:
        warnings.append("full-resolution Hagenaars FireNet with large chunks and 48 channels may OOM")

    if has_step_time_us:
        mcm["step_time_us"] = step_time_us
    else:
        mcm.pop("step_time_us", None)
    mcm["decay_tau_us"] = decay_tau_us
    repaired["mcm"] = mcm
    repaired["sofe_architecture"] = sofe
    return HagenaarsFireNetLIFConstraintResult(genome=repaired, valid=True, repairs=repairs, warnings=warnings)


def make_hagenaars_firenet_lif_target(search_space: Mapping[str, Any] | None = None):
    """Create a Hagenaars FireNet-LIF target with the default safe search space."""

    from smcfnet.evolution.targets import HagenaarsFireNetLIFTarget

    return HagenaarsFireNetLIFTarget(
        search_space=copy.deepcopy(dict(search_space or HAGENAARS_FIRENET_LIF_SEARCH_SPACE))
    )
