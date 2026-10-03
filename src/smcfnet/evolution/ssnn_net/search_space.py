"""SSNN Net EA search-space presets and guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from smcfnet.evolution.framework import GeneSpec


SSNN_NET_SEARCH_SPACE: Dict[str, Any] = {
    "mcm": {
        "step_time_us": [250, 500, 1000, 1250, 2000, 2500, 5000, 10000, 25000],
        "events_per_step_target": [2500, 5000, 10000, 15000, 25000, 50000, 75000],
        "bptt_steps": [2, 3, 5, 8, 10, 12],
        "decay_tau_us": [1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0],
        "flow_scaling": [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0],
        "smooth_weight": [0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 4.0],
        "contrast_weight": [0.25, 0.5, 1.0, 2.0, 4.0],
        "loss_weight": [0.02, 0.05, 0.1, 0.2, 0.4],
    },
    "sofe_architecture": {
        "count_scale": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "base_channels": [12, 16, 24, 32, 48],
        "kernel_size": [3, 5, 7],
        "ssnn_multiply_factor": [2.5, 5.0, 10.0, 15.0, 20.0],
        "spiking_v_threshold": [0.4, 0.5, 0.8, 1.0, 1.2, 1.6],
        "surrogate_alpha": [1.0, 2.0, 4.0, 8.0, 12.0],
        "detach_reset": [True, False],
        "ssnn_hard_reset": [False, True],
        "ssnn_spike_decay_mode": ["hold", "decay", "zero"],
        "flow_activation": ["softsign", "tanh", "identity"],
        "flow_head_init_std": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
        "ssnn_pred_kernel_size": [1, 3, 5, 7],
    },
}

# v2-specific search space.  Keep "mixed" encoder_signal out until hand
# ablations confirm it is useful.  "pyramid_residual" readout is included
# now that Phase 2 is implemented.
SSNN_NET_V2_SEARCH_SPACE: Dict[str, Any] = {
    "mcm": SSNN_NET_SEARCH_SPACE["mcm"],
    "sofe_architecture": {
        "ssnn_version": ["v2"],
        "ssnn_stage_channels": [
            [12, 24, 48, 96],
            [16, 32, 64, 128],
            [24, 48, 96, 192],
            [32, 64, 128, 256],
            [48, 96, 192, 384],
        ],
        "ssnn_bottleneck_dilations": [
            [1],
            [1, 2],
            [1, 2, 4],
            [1, 3, 6],
            [1, 2, 4, 8],
        ],
        "ssnn_encoder_signal": ["membrane", "spike"],
        "ssnn_edge_tap": ["hist", "none"],
        "ssnn_refine_channels": [8, 16, 32, 48, 64],
        "ssnn_refine_layers": [1, 2, 3],
        "ssnn_readout": ["single", "pyramid_residual"],
        # Shared v1/v2 keys
        "kernel_size": [3, 5, 7],
        "ssnn_multiply_factor": [2.5, 5.0, 10.0, 15.0, 20.0],
        "spiking_v_threshold": [0.4, 0.5, 0.8, 1.0, 1.2, 1.6],
        "surrogate_alpha": [1.0, 2.0, 4.0, 8.0, 12.0],
        "detach_reset": [True, False],
        "ssnn_hard_reset": [False, True],
        "ssnn_spike_decay_mode": ["hold", "decay", "zero"],
        "flow_activation": ["softsign", "tanh", "identity"],
        "flow_head_init_std": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
        "ssnn_pred_kernel_size": [1, 3, 5, 7],
        "count_scale": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    },
}

SSNN_NET_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("mcm", "step_time_us"): GeneSpec(
        path=("mcm", "step_time_us"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["step_time_us"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "events_per_step_target"): GeneSpec(
        path=("mcm", "events_per_step_target"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["events_per_step_target"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "bptt_steps"): GeneSpec(
        path=("mcm", "bptt_steps"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["bptt_steps"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "decay_tau_us"): GeneSpec(
        path=("mcm", "decay_tau_us"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["decay_tau_us"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mcm", "flow_scaling"): GeneSpec(
        path=("mcm", "flow_scaling"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["flow_scaling"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "smooth_weight"): GeneSpec(
        path=("mcm", "smooth_weight"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["smooth_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "contrast_weight"): GeneSpec(
        path=("mcm", "contrast_weight"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["contrast_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("mcm", "loss_weight"): GeneSpec(
        path=("mcm", "loss_weight"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["mcm"]["loss_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "count_scale"): GeneSpec(
        path=("sofe_architecture", "count_scale"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["count_scale"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "base_channels"): GeneSpec(
        path=("sofe_architecture", "base_channels"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["base_channels"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "kernel_size"): GeneSpec(
        path=("sofe_architecture", "kernel_size"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "ssnn_multiply_factor"): GeneSpec(
        path=("sofe_architecture", "ssnn_multiply_factor"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["ssnn_multiply_factor"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "spiking_v_threshold"): GeneSpec(
        path=("sofe_architecture", "spiking_v_threshold"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["spiking_v_threshold"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("sofe_architecture", "surrogate_alpha"): GeneSpec(
        path=("sofe_architecture", "surrogate_alpha"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["surrogate_alpha"],
        sensitivity="light",
        ordered=True,
    ),
    ("sofe_architecture", "detach_reset"): GeneSpec(
        path=("sofe_architecture", "detach_reset"),
        kind="bool",
        sensitivity="light",
    ),
    ("sofe_architecture", "ssnn_hard_reset"): GeneSpec(
        path=("sofe_architecture", "ssnn_hard_reset"),
        kind="bool",
        sensitivity="light",
    ),
    ("sofe_architecture", "ssnn_spike_decay_mode"): GeneSpec(
        path=("sofe_architecture", "ssnn_spike_decay_mode"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["ssnn_spike_decay_mode"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "flow_activation"): GeneSpec(
        path=("sofe_architecture", "flow_activation"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["flow_activation"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "flow_head_init_std"): GeneSpec(
        path=("sofe_architecture", "flow_head_init_std"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["flow_head_init_std"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "ssnn_pred_kernel_size"): GeneSpec(
        path=("sofe_architecture", "ssnn_pred_kernel_size"),
        kind="categorical",
        values=SSNN_NET_SEARCH_SPACE["sofe_architecture"]["ssnn_pred_kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
}

# v2-specific gene specs.  Only stable v2 knobs are exposed here; "mixed"
# encoder_signal is excluded until hand ablations confirm it is useful.
SSNN_NET_V2_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("sofe_architecture", "ssnn_stage_channels"): GeneSpec(
        path=("sofe_architecture", "ssnn_stage_channels"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_stage_channels"],
        sensitivity="heavy",
        ordered=False,
    ),
    ("sofe_architecture", "ssnn_bottleneck_dilations"): GeneSpec(
        path=("sofe_architecture", "ssnn_bottleneck_dilations"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_bottleneck_dilations"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "ssnn_encoder_signal"): GeneSpec(
        path=("sofe_architecture", "ssnn_encoder_signal"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_encoder_signal"],
        sensitivity="heavy",
        ordered=False,
    ),
    ("sofe_architecture", "ssnn_edge_tap"): GeneSpec(
        path=("sofe_architecture", "ssnn_edge_tap"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_edge_tap"],
        sensitivity="medium",
        ordered=False,
    ),
    ("sofe_architecture", "ssnn_refine_channels"): GeneSpec(
        path=("sofe_architecture", "ssnn_refine_channels"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_refine_channels"],
        sensitivity="medium",
        ordered=True,
    ),
    ("sofe_architecture", "ssnn_refine_layers"): GeneSpec(
        path=("sofe_architecture", "ssnn_refine_layers"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_refine_layers"],
        sensitivity="light",
        ordered=True,
    ),
    ("sofe_architecture", "ssnn_readout"): GeneSpec(
        path=("sofe_architecture", "ssnn_readout"),
        kind="categorical",
        values=SSNN_NET_V2_SEARCH_SPACE["sofe_architecture"]["ssnn_readout"],
        sensitivity="heavy",
        ordered=False,
    ),
}

SSNN_NET_CROSSOVER_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "temporal_partitioning": (
        ("mcm", "step_time_us"),
        ("mcm", "events_per_step_target"),
        ("mcm", "decay_tau_us"),
        ("mcm", "bptt_steps"),
    ),
    "loss_scaling": (
        ("mcm", "flow_scaling"),
        ("mcm", "smooth_weight"),
        ("mcm", "contrast_weight"),
        ("mcm", "loss_weight"),
    ),
    "lif_dynamics": (
        ("sofe_architecture", "spiking_v_threshold"),
        ("sofe_architecture", "surrogate_alpha"),
        ("sofe_architecture", "detach_reset"),
        ("sofe_architecture", "ssnn_hard_reset"),
        ("sofe_architecture", "ssnn_spike_decay_mode"),
    ),
    "architecture_capacity": (
        ("sofe_architecture", "base_channels"),
        ("sofe_architecture", "kernel_size"),
        ("sofe_architecture", "ssnn_multiply_factor"),
        ("sofe_architecture", "count_scale"),
    ),
    "flow_readout": (
        ("sofe_architecture", "flow_activation"),
        ("sofe_architecture", "flow_head_init_std"),
        ("sofe_architecture", "ssnn_pred_kernel_size"),
    ),
    # v2-only group: keep stage capacity and inter-stage signal together since
    # they jointly determine the bottleneck gradient dynamics.
    "v2_encoder_design": (
        ("sofe_architecture", "ssnn_stage_channels"),
        ("sofe_architecture", "ssnn_bottleneck_dilations"),
        ("sofe_architecture", "ssnn_encoder_signal"),
    ),
    "v2_refinement": (
        ("sofe_architecture", "ssnn_edge_tap"),
        ("sofe_architecture", "ssnn_refine_channels"),
        ("sofe_architecture", "ssnn_refine_layers"),
        ("sofe_architecture", "ssnn_readout"),
    ),
}


@dataclass
class SSNNNetConstraintResult:
    """Result of applying SSNN-specific guardrails."""

    genome: Dict[str, Any]
    valid: bool = True
    repairs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def apply_ssnn_net_guardrails(genome: Mapping[str, Any]) -> SSNNNetConstraintResult:
    """Validate and lightly repair a streaming SSNN SOFE genome."""

    normalized = {
        "mcm": dict((genome or {}).get("mcm") or {}),
        "sofe_architecture": dict((genome or {}).get("sofe_architecture") or {}),
    }
    repairs: List[str] = []
    warnings: List[str] = []

    mcm = normalized["mcm"]
    arch = normalized["sofe_architecture"]
    arch["family"] = "ssnn_net"

    if "step_time_us" in mcm and mcm.get("step_time_us") is not None:
        mcm["step_time_us"] = int(mcm["step_time_us"])
        if mcm["step_time_us"] <= 0:
            raise ValueError("SSNN step_time_us must be > 0 or null")
        if 50000 % int(mcm["step_time_us"]) != 0:
            raise ValueError(
                "SSNN fixed-time step_time_us must evenly divide the 50 ms event window"
            )

    if "events_per_step_target" in mcm and mcm.get("events_per_step_target") is not None:
        mcm["events_per_step_target"] = int(mcm.get("events_per_step_target", 15000))
        if mcm["events_per_step_target"] <= 0:
            raise ValueError("SSNN events_per_step_target must be > 0 or null")
    mcm["bptt_steps"] = int(mcm.get("bptt_steps", 5))
    if mcm["bptt_steps"] <= 0:
        raise ValueError("SSNN bptt_steps must be > 0")
    decay_tau_us = mcm.get("decay_tau_us", 10000.0)
    mcm["decay_tau_us"] = None if decay_tau_us is None else float(decay_tau_us)
    if mcm["decay_tau_us"] is not None and mcm["decay_tau_us"] <= 0.0:
        raise ValueError("SSNN decay_tau_us must be > 0 or null")
    mcm["flow_scaling"] = float(mcm.get("flow_scaling", 4.0))
    if mcm["flow_scaling"] <= 0.0:
        raise ValueError("SSNN flow_scaling must be > 0")
    mcm["smooth_weight"] = float(mcm.get("smooth_weight", 1.0))
    if mcm["smooth_weight"] < 0.0:
        raise ValueError("SSNN smooth_weight must be >= 0")
    mcm["contrast_weight"] = float(mcm.get("contrast_weight", 1.0))
    if mcm["contrast_weight"] <= 0.0:
        raise ValueError("SSNN contrast_weight must be > 0")
    mcm["loss_weight"] = float(mcm.get("loss_weight", 0.1))
    if mcm["loss_weight"] < 0.0:
        raise ValueError("SSNN loss_weight must be >= 0")

    arch["count_scale"] = float(arch.get("count_scale", 1.0))
    if arch["count_scale"] <= 0.0:
        raise ValueError("SSNN count_scale must be > 0")
    base_channels = int(arch.get("base_channels", 24))
    if base_channels <= 0:
        raise ValueError("SSNN base_channels must be > 0")
    arch["base_channels"] = base_channels

    kernel_size = int(arch.get("kernel_size", 5))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("SSNN kernel_size must be a positive odd integer")
    arch["kernel_size"] = kernel_size

    multiply_factor = float(arch.get("ssnn_multiply_factor", 10.0))
    if multiply_factor <= 0.0:
        raise ValueError("SSNN ssnn_multiply_factor must be > 0")
    arch["ssnn_multiply_factor"] = multiply_factor
    arch["spiking_v_threshold"] = float(arch.get("spiking_v_threshold", 1.0))
    if arch["spiking_v_threshold"] <= 0.0:
        raise ValueError("SSNN spiking_v_threshold must be > 0")
    arch["surrogate_alpha"] = float(arch.get("surrogate_alpha", 4.0))
    if arch["surrogate_alpha"] <= 0.0:
        raise ValueError("SSNN surrogate_alpha must be > 0")
    arch["detach_reset"] = bool(arch.get("detach_reset", True))
    arch["ssnn_hard_reset"] = bool(arch.get("ssnn_hard_reset", False))
    spike_decay_mode = str(arch.get("ssnn_spike_decay_mode", "decay")).lower()
    if spike_decay_mode not in {"hold", "decay", "zero"}:
        raise ValueError(f"SSNN ssnn_spike_decay_mode is invalid: {spike_decay_mode!r}")
    arch["ssnn_spike_decay_mode"] = spike_decay_mode
    flow_activation = str(arch.get("flow_activation", "softsign")).lower()
    if flow_activation not in {"softsign", "tanh", "identity", "none", "linear"}:
        raise ValueError(f"SSNN flow_activation is invalid: {flow_activation!r}")
    arch["flow_activation"] = flow_activation
    arch["flow_head_init_std"] = float(arch.get("flow_head_init_std", 0.01))
    if arch["flow_head_init_std"] <= 0.0:
        raise ValueError("SSNN flow_head_init_std must be > 0")
    pred_kernel = int(arch.get("ssnn_pred_kernel_size", 3))
    if pred_kernel <= 0 or pred_kernel % 2 == 0:
        raise ValueError("SSNN ssnn_pred_kernel_size must be a positive odd integer")
    arch["ssnn_pred_kernel_size"] = pred_kernel

    return SSNNNetConstraintResult(
        genome=normalized,
        valid=True,
        repairs=repairs,
        warnings=warnings,
    )


def make_ssnn_net_target(search_space: Mapping[str, Any] | None = None):
    from smcfnet.evolution.targets import SSNNNetTarget

    return SSNNNetTarget(search_space=dict(search_space or SSNN_NET_SEARCH_SPACE))


def make_ssnn_net_v2_target(search_space: Mapping[str, Any] | None = None):
    from smcfnet.evolution.targets import SSNNNetV2Target

    return SSNNNetV2Target(search_space=dict(search_space or SSNN_NET_V2_SEARCH_SPACE))
