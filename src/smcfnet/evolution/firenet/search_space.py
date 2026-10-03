"""FireNet-LIF EA search-space presets and guardrails."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from smcfnet.evolution.framework import GeneSpec

FIRENET_LIF_SEARCH_SPACE: Dict[str, Any] = {
    "mcm": {
        "step_time_us": [250, 500, 1000, 1250, 2500, 5000, 10000],
        "events_per_step_target": [5000, 10000, 15000, 25000, 50000],
        "bptt_steps": [5, 10, 15],
        "flow_scaling": [1.0, 4.0, 8.0, 12.0, 16.0, 24.0],
        "smooth_weight": [0.03, 0.1, 0.3, 1.0, 2.0],
        "contrast_weight": [0.5, 1.0, 2.0],
        "loss_weight": [0.05, 0.1, 0.2],
    },
    "sofe_architecture": {
        "count_scale": [0.25, 0.5, 1.0, 2.0, 4.0],
        "firenet_downsample_mode": ["sum"],
        "firenet_spike_decay_mode": ["hold", "decay", "zero"],
        "flow_head_init_std": [0.005, 0.01, 0.03, 0.05, 0.08],
        "surrogate_alpha": [5.0, 10.0, 20.0],
        "detach_reset": [True, False],
        "spiking_neuron": {
            "leak": [[-4.0, 0.1], [-2.0, 0.1], [-1.0, 0.1]],
            "thresh": [[1.0, 0.1], [0.8, 0.1], [0.6, 0.1], [0.5, 0.1]],
            "learn_leak": [True],
            "learn_thresh": [True],
            "hard_reset": [True, False],
        },
    },
}

FIRENET_LIF_HEAVY_MUTATION_FIELDS: Dict[Tuple[str, ...], str] = {
    ("mcm", "step_time_us"): "heavy",
    ("mcm", "events_per_step_target"): "heavy",
    ("mcm", "bptt_steps"): "heavy",
    ("mcm", "flow_scaling"): "medium",
    ("mcm", "smooth_weight"): "medium",
    ("sofe_architecture", "count_scale"): "heavy",
    ("sofe_architecture", "firenet_spike_decay_mode"): "medium",
    ("sofe_architecture", "flow_head_init_std"): "medium",
    ("sofe_architecture", "spiking_neuron", "leak"): "heavy",
    ("sofe_architecture", "spiking_neuron", "thresh"): "heavy",
    ("sofe_architecture", "surrogate_alpha"): "light",
    ("sofe_architecture", "detach_reset"): "light",
    ("sofe_architecture", "spiking_neuron", "hard_reset"): "light",
}

FIRENET_LIF_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("mcm", "step_time_us"): GeneSpec(
        path=("mcm", "step_time_us"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["step_time_us"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "events_per_step_target"): GeneSpec(
        path=("mcm", "events_per_step_target"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["events_per_step_target"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "bptt_steps"): GeneSpec(
        path=("mcm", "bptt_steps"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["bptt_steps"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "flow_scaling"): GeneSpec(
        path=("mcm", "flow_scaling"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["flow_scaling"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "smooth_weight"): GeneSpec(
        path=("mcm", "smooth_weight"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["smooth_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "contrast_weight"): GeneSpec(
        path=("mcm", "contrast_weight"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["contrast_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "loss_weight"): GeneSpec(
        path=("mcm", "loss_weight"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["mcm"]["loss_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "count_scale"): GeneSpec(
        path=("sofe_architecture", "count_scale"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["count_scale"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "firenet_spike_decay_mode"): GeneSpec(
        path=("sofe_architecture", "firenet_spike_decay_mode"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["firenet_spike_decay_mode"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "flow_head_init_std"): GeneSpec(
        path=("sofe_architecture", "flow_head_init_std"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["flow_head_init_std"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "surrogate_alpha"): GeneSpec(
        path=("sofe_architecture", "surrogate_alpha"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["surrogate_alpha"],
        sensitivity="light",
        ordered=True,
    ),
    ("sofe_architecture", "detach_reset"): GeneSpec(
        path=("sofe_architecture", "detach_reset"),
        kind="bool",
        sensitivity="light",
    ),
    ("sofe_architecture", "spiking_neuron", "leak"): GeneSpec(
        path=("sofe_architecture", "spiking_neuron", "leak"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["spiking_neuron"]["leak"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "spiking_neuron", "thresh"): GeneSpec(
        path=("sofe_architecture", "spiking_neuron", "thresh"),
        kind="categorical",
        values=FIRENET_LIF_SEARCH_SPACE["sofe_architecture"]["spiking_neuron"]["thresh"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "spiking_neuron", "hard_reset"): GeneSpec(
        path=("sofe_architecture", "spiking_neuron", "hard_reset"),
        kind="bool",
        sensitivity="light",
    ),
}

FIRENET_LIF_CROSSOVER_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "temporal_partitioning": (
        ("mcm", "step_time_us"),
        ("mcm", "events_per_step_target"),
    ),
    "training_bptt": (
        ("mcm", "bptt_steps"),
        ("mcm", "smooth_weight"),
        ("mcm", "contrast_weight"),
        ("mcm", "loss_weight"),
    ),
    "lif_dynamics": (
        ("sofe_architecture", "spiking_neuron", "leak"),
        ("sofe_architecture", "spiking_neuron", "thresh"),
        ("sofe_architecture", "spiking_neuron", "hard_reset"),
        ("sofe_architecture", "surrogate_alpha"),
        ("sofe_architecture", "detach_reset"),
        ("sofe_architecture", "firenet_spike_decay_mode"),
    ),
    "architecture_capacity": (
        ("sofe_architecture", "count_scale"),
        ("sofe_architecture", "firenet_downsample_mode"),
    ),
    "flow_readout": (
        ("mcm", "flow_scaling"),
        ("sofe_architecture", "flow_head_init_std"),
    ),
}

FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("mcm", "flow_scaling"),
    ("mcm", "smooth_weight"),
    ("mcm", "contrast_weight"),
    ("mcm", "loss_weight"),
    ("sofe_architecture", "flow_head_init_std"),
    ("sofe_architecture", "surrogate_alpha"),
)

FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("sofe_architecture", "spiking_neuron", "leak"),
    ("sofe_architecture", "spiking_neuron", "thresh"),
)


@dataclass
class FireNetLIFConstraintResult:
    """Result of applying FireNet-LIF-specific guardrails."""

    genome: Dict[str, Any]
    valid: bool = True
    repairs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _as_float_pair(value: Any, *, name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list/tuple, got {value!r}")
    return float(value[0]), float(value[1])


def apply_firenet_lif_guardrails(genome: Mapping[str, Any]) -> FireNetLIFConstraintResult:
    """Validate/repair FireNet-LIF genomes before evaluation.

    Runtime OOM is still possible and should be handled by the evaluator as
    an invalid candidate. These guardrails only catch predictable issues.
    """

    repaired = copy.deepcopy(dict(genome))
    repairs: List[str] = []
    warnings: List[str] = []

    mcm = dict(repaired.setdefault("mcm", {}))
    sofe = dict(repaired.setdefault("sofe_architecture", {}))
    spiking_value = sofe.get("spiking_neuron", {})
    spiking = dict(spiking_value) if isinstance(spiking_value, Mapping) else {}

    events = int(mcm.get("events_per_step_target", 15000))
    bptt = int(mcm.get("bptt_steps", 5))
    flow_scaling = float(mcm.get("flow_scaling", 4.0))
    has_step_time_us = "step_time_us" in mcm
    step_time_us = mcm.get("step_time_us", None)
    step_time_us = None if step_time_us is None else int(step_time_us)
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

    count_scale = float(sofe.get("count_scale", 1.0))
    thresh_value = spiking.get("thresh", sofe.get("firenet_thresh", [0.8, 0.1]))
    thresh_mu, thresh_sigma = _as_float_pair(thresh_value, name="spiking_neuron.thresh")
    _as_float_pair(spiking.get("leak", sofe.get("firenet_leak", [-2.0, 0.1])), name="spiking_neuron.leak")

    if events * bptt > 500000:
        old_bptt = bptt
        bptt = max(1, 500000 // max(events, 1))
        mcm["bptt_steps"] = int(bptt)
        repairs.append(f"capped bptt_steps from {old_bptt} to {bptt} for events_per_step_target={events}")

    if count_scale >= 4.0 and thresh_mu <= 0.5:
        spiking["thresh"] = [0.6, thresh_sigma]
        sofe["firenet_thresh"] = [0.6, thresh_sigma]
        repairs.append("raised threshold mean to 0.6 for high count_scale")

    if count_scale >= 4.0 and events >= 50000:
        warnings.append("high count_scale with large events_per_step_target may OOM or saturate")

    if isinstance(spiking_value, Mapping):
        sofe["spiking_neuron"] = spiking
    if has_step_time_us:
        mcm["step_time_us"] = step_time_us
    else:
        mcm.pop("step_time_us", None)
    repaired["mcm"] = mcm
    repaired["sofe_architecture"] = sofe
    return FireNetLIFConstraintResult(genome=repaired, valid=True, repairs=repairs, warnings=warnings)


def make_firenet_lif_target(search_space: Mapping[str, Any] | None = None) -> FireNetLIFTarget:
    """Create a FireNet-LIF target with the default broad safe search space."""

    from smcfnet.evolution.targets import FireNetLIFTarget

    return FireNetLIFTarget(search_space=copy.deepcopy(dict(search_space or FIRENET_LIF_SEARCH_SPACE)))
