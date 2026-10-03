"""SOFE architecture-spec utilities and modular builders.

This module separates three concerns:

1. Architecture spec normalization
2. Search-space expansion into concrete specs
3. Construction of modular SOFE core/readout modules from a concrete spec

The goal is to keep the streaming scheduler in ``sofe.py`` stable while
allowing both YAML-driven experiments and external search code to generate
different network variants from the same declarative parameter surface.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.network_blocks import BaseConv, DWConv
from smcfnet.model.spiking_blocks import ConvGRUCell, clip_floor, surrogate_heaviside


DEFAULT_SOFE_ARCHITECTURE: Dict[str, Any] = {
    "family": "fire",
    "step_encoding": "hist2",
    "normalize_counts": True,
    "count_encoding": "normalized",
    "count_scale": 1.0,
    "mini_voxel_bins": 4,
    "base_channels": 32,
    "stage_channels": None,
    "stage_depths": 1,
    "num_stages": None,
    "num_res_blocks": 1,
    "recurrent_backend": "native",
    "recurrent_cell": "convgru",
    "recurrent_placement": "every_stage",
    "spiking_neuron": "lif",
    "surrogate_function": "sigmoid",
    "surrogate_alpha": 4.0,
    "detach_reset": True,
    "spiking_backend": "torch",
    "spiking_v_threshold": 1.0,
    "spiking_v_reset": 0.0,
    "downsample_mode": "stride_conv",
    "upsample_mode": "nearest_conv",
    "skip_mode": "concat",
    "separable_conv": False,
    "readout_type": "single_scale",
    "readout_scales": None,
    "readout_update_rule": "residual",
    "readout_alpha_init": 1.0,
    "readout_learn_alpha": False,
    "readout_state_channels": None,
    "stflow_alignment_enabled": False,
    "stflow_feedback_enabled": False,
    "flow_activation": "tanh",
    "readout_mid_channels": 32,
    "flow_head_init": "zero",
    "flow_head_init_std": 1e-3,
    "kernel_size": 3,
    "firenet_act_width": 10.0,
    "firenet_leak": [-4.0, 0.1],
    "firenet_thresh": [0.8, 0.0],
    "firenet_learn_leak": True,
    "firenet_learn_thresh": True,
    "firenet_hard_reset": True,
    "firenet_spike_decay_mode": "hold",
    "firenet_input_downsample": 1,
    "firenet_downsample_mode": "sum",
    "hagenaars_use_bias": True,
    "hagenaars_pred_kernel_size": 3,
    "ssnn_multiply_factor": 10.0,
    "ssnn_hard_reset": False,
    "ssnn_spike_decay_mode": "hold",
    "ssnn_pred_kernel_size": 3,
    # v2-specific keys
    "ssnn_version": "v1",
    "ssnn_num_stages": 4,
    "ssnn_stage_channels": None,
    "ssnn_bottleneck_dilations": [1, 2, 4],
    "ssnn_encoder_signal": "membrane",
    "ssnn_edge_tap": "hist",
    "ssnn_refine_channels": 32,
    "ssnn_refine_layers": 2,
    "ssnn_flow_scale_x": 1.0,
    "ssnn_flow_scale_y": 1.0,
    "ssnn_readout": "single",
}


SUPPORTED_V1_FAMILIES = {
    "fire",
    "unet",
    "stflow_lite",
    "firenet_lif",
    "hagenaars_firenet_lif",
    "ssnn_net",
}
SUPPORTED_V1_STEP_ENCODINGS = {"hist2", "hist4", "hist4_legacy"}
SUPPORTED_V1_COUNT_ENCODINGS = {"normalized", "raw", "log", "sqrt"}
SUPPORTED_V1_RECURRENT_BACKENDS = {"native", "spikingjelly"}
SUPPORTED_V1_RECURRENT_CELLS = {"convgru", "convrnn", "convlif"}
SUPPORTED_V1_RECURRENT_PLACEMENTS = {"every_stage", "encoder_only", "bottleneck_only", "none"}
SUPPORTED_V1_SPIKING_NEURONS = {"lif", "if", "plif"}
SUPPORTED_V1_SURROGATE_FUNCTIONS = {"sigmoid", "atan", "softsign"}
SUPPORTED_V1_SPIKING_BACKENDS = {"torch"}
SUPPORTED_V1_DOWNSAMPLE_MODES = {"stride_conv", "maxpool"}
SUPPORTED_V1_UPSAMPLE_MODES = {"nearest_conv", "bilinear_conv", "transposed_conv"}
SUPPORTED_V1_SKIP_MODES = {"concat", "sum", "none"}
SUPPORTED_V1_READOUT_TYPES = {"single_scale", "multi_scale", "persistent_delta"}
SUPPORTED_V1_READOUT_UPDATE_RULES = {"residual", "ema"}
SUPPORTED_V1_FLOW_ACTIVATIONS = {"tanh", "softsign", "identity", "none", "linear"}
SUPPORTED_V1_FLOW_HEAD_INITS = {
    "zero",
    "small_normal",
    "kaiming_normal",
    "xavier_uniform",
    "firenet_small_uniform",
}
SUPPORTED_V1_FIRENET_SPIKE_DECAY_MODES = {"hold", "decay", "zero"}
SUPPORTED_V1_FIRENET_DOWNSAMPLE_MODES = {"sum", "avg", "max"}
SUPPORTED_V1_SSNN_SPIKE_DECAY_MODES = {"hold", "decay", "zero"}
SUPPORTED_V1_SSNN_VERSIONS = {"v1", "v2"}
SUPPORTED_V1_SSNN_ENCODER_SIGNALS = {"spike", "membrane", "mixed"}
SUPPORTED_V1_SSNN_EDGE_TAPS = {"hist", "none"}
SUPPORTED_V1_SSNN_READOUTS = {"single", "pyramid_residual"}


def _import_spikingjelly_modules():
    try:
        from spikingjelly.activation_based import neuron as sj_neuron
        from spikingjelly.activation_based import surrogate as sj_surrogate
    except ImportError as exc:
        raise ImportError(
            "SOFE recurrent_backend='spikingjelly' requires the "
            "'spikingjelly' package in the active environment."
        ) from exc
    return sj_neuron, sj_surrogate


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _family_default_channels(family: str, base_channels: int) -> List[int]:
    family = str(family).lower()
    if family == "fire":
        return [base_channels, base_channels * 2, base_channels]
    if family == "stflow_lite":
        return [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
    return [base_channels, base_channels * 2, base_channels * 4]


def _normalize_stage_depths(stage_depths: Any, *, num_stages: int) -> List[int]:
    if isinstance(stage_depths, int):
        return [int(stage_depths)] * num_stages
    if isinstance(stage_depths, Sequence) and not isinstance(stage_depths, (str, bytes)):
        depths = [int(v) for v in stage_depths]
        if not depths:
            depths = [1] * num_stages
        if len(depths) < num_stages:
            depths = depths + [depths[-1]] * (num_stages - len(depths))
        return depths[:num_stages]
    return [1] * num_stages


def _validate_supported_value(*, field: str, value: str, supported: Iterable[str]) -> None:
    supported_values = {str(v).lower() for v in supported}
    value_norm = str(value).lower()
    if value_norm not in supported_values:
        supported_str = ", ".join(sorted(repr(v) for v in supported_values))
        raise NotImplementedError(
            f"SOFE {field}={value!r} is outside the frozen v1 support matrix. "
            f"Currently supported: {supported_str}."
        )


def normalize_sofe_architecture_spec(
    spec: Optional[Dict[str, Any]] = None,
    *,
    encoder_channels: Optional[Sequence[int]] = None,
    num_res_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize a possibly-partial SOFE architecture spec into a concrete dict."""
    merged = _deep_merge(DEFAULT_SOFE_ARCHITECTURE, spec)
    spec = dict(spec or {})

    family = str(merged.get("family", "fire")).lower()
    base_channels = int(merged.get("base_channels", 32))
    recurrent_backend_default = "spikingjelly" if family == "hagenaars_firenet_lif" else "native"
    recurrent_cell_default = "convlif" if family == "hagenaars_firenet_lif" else "convgru"
    raw_spiking_neuron = merged.get("spiking_neuron", "lif")
    spiking_neuron_params: Dict[str, Any] = {}
    if isinstance(raw_spiking_neuron, dict):
        spiking_neuron_params = copy.deepcopy(raw_spiking_neuron)
        spiking_neuron = str(spiking_neuron_params.pop("type", "lif")).lower()
    else:
        spiking_neuron = str(raw_spiking_neuron).lower()

    firenet_leak = copy.deepcopy(
        spiking_neuron_params.get("leak", merged.get("firenet_leak", [-4.0, 0.1]))
    )
    firenet_thresh = copy.deepcopy(
        spiking_neuron_params.get("thresh", merged.get("firenet_thresh", [0.8, 0.0]))
    )

    raw_stage_channels = merged.get("stage_channels")
    if raw_stage_channels is None:
        if encoder_channels:
            raw_stage_channels = list(encoder_channels)
        else:
            raw_stage_channels = _family_default_channels(family, base_channels)
    stage_channels = [int(v) for v in raw_stage_channels]
    if not stage_channels:
        stage_channels = _family_default_channels(family, base_channels)

    num_stages_val = merged.get("num_stages")
    num_stages = int(num_stages_val) if num_stages_val is not None else len(stage_channels)
    if num_stages <= 0:
        raise ValueError("SOFE num_stages must be >= 1")
    if len(stage_channels) < num_stages:
        stage_channels = stage_channels + [stage_channels[-1]] * (num_stages - len(stage_channels))
    stage_channels = stage_channels[:num_stages]

    stage_depths = _normalize_stage_depths(merged.get("stage_depths", 1), num_stages=num_stages)

    if "num_res_blocks" in spec:
        raw_num_res_blocks = spec.get("num_res_blocks")
    elif num_res_blocks is not None:
        raw_num_res_blocks = num_res_blocks
    else:
        raw_num_res_blocks = merged.get("num_res_blocks", DEFAULT_SOFE_ARCHITECTURE["num_res_blocks"])
    resolved_num_res_blocks = int(raw_num_res_blocks)
    if resolved_num_res_blocks < 0:
        raise ValueError("SOFE num_res_blocks must be >= 0")

    if "stflow_alignment_enabled" in spec:
        stflow_alignment_enabled = bool(merged.get("stflow_alignment_enabled", False))
    else:
        stflow_alignment_enabled = bool(family == "stflow_lite")

    normalized = {
        "family": family,
        "step_encoding": str(merged.get("step_encoding", "hist2")).lower(),
        "normalize_counts": bool(merged.get("normalize_counts", True)),
        "count_encoding": str(
            merged.get(
                "count_encoding",
                "normalized" if bool(merged.get("normalize_counts", True)) else "raw",
            )
        ).lower(),
        "count_scale": float(merged.get("count_scale", 1.0)),
        "mini_voxel_bins": int(merged.get("mini_voxel_bins", 4)),
        "base_channels": base_channels,
        "stage_channels": stage_channels,
        "stage_depths": stage_depths,
        "num_stages": num_stages,
        "num_res_blocks": resolved_num_res_blocks,
        "recurrent_backend": str(
            recurrent_backend_default
            if family == "hagenaars_firenet_lif"
            else (
                merged.get("recurrent_backend")
                if "recurrent_backend" in spec
                else recurrent_backend_default
            )
        ).lower(),
        "recurrent_cell": str(
            recurrent_cell_default
            if family == "hagenaars_firenet_lif"
            else (
                merged.get("recurrent_cell")
                if "recurrent_cell" in spec
                else recurrent_cell_default
            )
        ).lower(),
        "recurrent_placement": str(merged.get("recurrent_placement", "every_stage")).lower(),
        "spiking_neuron": spiking_neuron,
        "spiking_neuron_params": spiking_neuron_params,
        "surrogate_function": str(merged.get("surrogate_function", "sigmoid")).lower(),
        "surrogate_alpha": float(merged.get("surrogate_alpha", 4.0)),
        "detach_reset": bool(merged.get("detach_reset", True)),
        "spiking_backend": str(merged.get("spiking_backend", "torch")).lower(),
        "spiking_v_threshold": float(merged.get("spiking_v_threshold", 1.0)),
        "spiking_v_reset": (
            None if merged.get("spiking_v_reset", 0.0) is None else float(merged.get("spiking_v_reset", 0.0))
        ),
        "downsample_mode": str(merged.get("downsample_mode", "stride_conv")).lower(),
        "upsample_mode": str(merged.get("upsample_mode", "nearest_conv")).lower(),
        "skip_mode": str(merged.get("skip_mode", "concat")).lower(),
        "separable_conv": bool(merged.get("separable_conv", False)),
        "readout_type": str(merged.get("readout_type", "single_scale")).lower(),
        "readout_scales": copy.deepcopy(merged.get("readout_scales")),
        "readout_update_rule": str(merged.get("readout_update_rule", "residual")).lower(),
        "readout_alpha_init": float(merged.get("readout_alpha_init", 1.0)),
        "readout_learn_alpha": bool(merged.get("readout_learn_alpha", False)),
        "readout_state_channels": (
            None
            if merged.get("readout_state_channels", None) is None
            else int(merged.get("readout_state_channels"))
        ),
        "stflow_alignment_enabled": bool(stflow_alignment_enabled),
        "stflow_feedback_enabled": bool(merged.get("stflow_feedback_enabled", False)),
        "flow_activation": str(merged.get("flow_activation", "tanh")).lower(),
        "readout_mid_channels": int(merged.get("readout_mid_channels", base_channels)),
        "flow_head_init": str(merged.get("flow_head_init", "zero")).lower(),
        "flow_head_init_std": float(merged.get("flow_head_init_std", 1e-3)),
        "kernel_size": int(merged.get("kernel_size", 3)),
        "firenet_act_width": float(
            spiking_neuron_params.get("act_width", merged.get("firenet_act_width", 10.0))
        ),
        "firenet_leak": [float(v) for v in firenet_leak],
        "firenet_thresh": [float(v) for v in firenet_thresh],
        "firenet_learn_leak": bool(
            spiking_neuron_params.get("learn_leak", merged.get("firenet_learn_leak", True))
        ),
        "firenet_learn_thresh": bool(
            spiking_neuron_params.get("learn_thresh", merged.get("firenet_learn_thresh", True))
        ),
        "firenet_hard_reset": bool(
            spiking_neuron_params.get("hard_reset", merged.get("firenet_hard_reset", True))
        ),
        "firenet_spike_decay_mode": str(
            spiking_neuron_params.get(
                "spike_decay_mode",
                merged.get("firenet_spike_decay_mode", "hold"),
            )
        ).lower(),
        "firenet_input_downsample": int(merged.get("firenet_input_downsample", 1) or 1),
        "firenet_downsample_mode": str(merged.get("firenet_downsample_mode", "sum")).lower(),
        "hagenaars_use_bias": bool(merged.get("hagenaars_use_bias", True)),
        "hagenaars_pred_kernel_size": int(merged.get("hagenaars_pred_kernel_size", 3)),
        "ssnn_multiply_factor": float(merged.get("ssnn_multiply_factor", 10.0)),
        "ssnn_hard_reset": bool(merged.get("ssnn_hard_reset", False)),
        "ssnn_spike_decay_mode": str(merged.get("ssnn_spike_decay_mode", "hold")).lower(),
        "ssnn_pred_kernel_size": int(merged.get("ssnn_pred_kernel_size", 3)),
        # v2-specific keys
        "ssnn_version": str(merged.get("ssnn_version", "v1")).lower(),
        "ssnn_num_stages": int(merged.get("ssnn_num_stages", 4)),
        "ssnn_stage_channels": (
            [int(c) for c in merged["ssnn_stage_channels"]]
            if merged.get("ssnn_stage_channels") is not None
            else None
        ),
        "ssnn_bottleneck_dilations": [
            int(d)
            for d in (
                merged.get("ssnn_bottleneck_dilations")
                if isinstance(merged.get("ssnn_bottleneck_dilations"), (list, tuple))
                else [1, 2, 4]
            )
        ],
        "ssnn_encoder_signal": str(merged.get("ssnn_encoder_signal", "membrane")).lower(),
        "ssnn_edge_tap": str(merged.get("ssnn_edge_tap", "hist")).lower(),
        "ssnn_refine_channels": int(merged.get("ssnn_refine_channels", 32)),
        "ssnn_refine_layers": int(merged.get("ssnn_refine_layers", 2)),
        "ssnn_flow_scale_x": float(merged.get("ssnn_flow_scale_x", 1.0)),
        "ssnn_flow_scale_y": float(merged.get("ssnn_flow_scale_y", 1.0)),
        "ssnn_readout": str(merged.get("ssnn_readout", "single")).lower(),
    }
    _validate_supported_value(field="family", value=normalized["family"], supported=SUPPORTED_V1_FAMILIES)
    _validate_supported_value(
        field="step_encoding",
        value=normalized["step_encoding"],
        supported=SUPPORTED_V1_STEP_ENCODINGS,
    )
    _validate_supported_value(
        field="count_encoding",
        value=normalized["count_encoding"],
        supported=SUPPORTED_V1_COUNT_ENCODINGS,
    )
    _validate_supported_value(
        field="recurrent_backend",
        value=normalized["recurrent_backend"],
        supported=SUPPORTED_V1_RECURRENT_BACKENDS,
    )
    _validate_supported_value(
        field="recurrent_cell",
        value=normalized["recurrent_cell"],
        supported=SUPPORTED_V1_RECURRENT_CELLS,
    )
    _validate_supported_value(
        field="recurrent_placement",
        value=normalized["recurrent_placement"],
        supported=SUPPORTED_V1_RECURRENT_PLACEMENTS,
    )
    _validate_supported_value(
        field="spiking_neuron",
        value=normalized["spiking_neuron"],
        supported=SUPPORTED_V1_SPIKING_NEURONS,
    )
    _validate_supported_value(
        field="surrogate_function",
        value=normalized["surrogate_function"],
        supported=SUPPORTED_V1_SURROGATE_FUNCTIONS,
    )
    _validate_supported_value(
        field="spiking_backend",
        value=normalized["spiking_backend"],
        supported=SUPPORTED_V1_SPIKING_BACKENDS,
    )
    _validate_supported_value(
        field="downsample_mode",
        value=normalized["downsample_mode"],
        supported=SUPPORTED_V1_DOWNSAMPLE_MODES,
    )
    _validate_supported_value(
        field="upsample_mode",
        value=normalized["upsample_mode"],
        supported=SUPPORTED_V1_UPSAMPLE_MODES,
    )
    _validate_supported_value(
        field="skip_mode",
        value=normalized["skip_mode"],
        supported=SUPPORTED_V1_SKIP_MODES,
    )
    _validate_supported_value(
        field="readout_type",
        value=normalized["readout_type"],
        supported=SUPPORTED_V1_READOUT_TYPES,
    )
    _validate_supported_value(
        field="flow_activation",
        value=normalized["flow_activation"],
        supported=SUPPORTED_V1_FLOW_ACTIVATIONS,
    )
    _validate_supported_value(
        field="readout_update_rule",
        value=normalized["readout_update_rule"],
        supported=SUPPORTED_V1_READOUT_UPDATE_RULES,
    )
    _validate_supported_value(
        field="flow_head_init",
        value=normalized["flow_head_init"],
        supported=SUPPORTED_V1_FLOW_HEAD_INITS,
    )
    _validate_supported_value(
        field="firenet_spike_decay_mode",
        value=normalized["firenet_spike_decay_mode"],
        supported=SUPPORTED_V1_FIRENET_SPIKE_DECAY_MODES,
    )
    _validate_supported_value(
        field="firenet_downsample_mode",
        value=normalized["firenet_downsample_mode"],
        supported=SUPPORTED_V1_FIRENET_DOWNSAMPLE_MODES,
    )
    _validate_supported_value(
        field="ssnn_spike_decay_mode",
        value=normalized["ssnn_spike_decay_mode"],
        supported=SUPPORTED_V1_SSNN_SPIKE_DECAY_MODES,
    )
    _validate_supported_value(
        field="ssnn_version",
        value=normalized["ssnn_version"],
        supported=SUPPORTED_V1_SSNN_VERSIONS,
    )
    _validate_supported_value(
        field="ssnn_encoder_signal",
        value=normalized["ssnn_encoder_signal"],
        supported=SUPPORTED_V1_SSNN_ENCODER_SIGNALS,
    )
    _validate_supported_value(
        field="ssnn_edge_tap",
        value=normalized["ssnn_edge_tap"],
        supported=SUPPORTED_V1_SSNN_EDGE_TAPS,
    )
    _validate_supported_value(
        field="ssnn_readout",
        value=normalized["ssnn_readout"],
        supported=SUPPORTED_V1_SSNN_READOUTS,
    )
    if normalized["ssnn_refine_channels"] <= 0:
        raise ValueError("SOFE ssnn_refine_channels must be > 0.")
    if normalized["ssnn_refine_layers"] <= 0:
        raise ValueError("SOFE ssnn_refine_layers must be > 0.")
    if normalized["ssnn_flow_scale_x"] <= 0.0:
        raise ValueError("SOFE ssnn_flow_scale_x must be > 0.")
    if normalized["ssnn_flow_scale_y"] <= 0.0:
        raise ValueError("SOFE ssnn_flow_scale_y must be > 0.")
    if normalized["surrogate_alpha"] <= 0.0:
        raise ValueError("SOFE surrogate_alpha must be > 0")
    if not (0.0 < normalized["readout_alpha_init"] <= 1.0):
        raise ValueError("SOFE readout_alpha_init must be in (0, 1].")
    if normalized["stflow_feedback_enabled"] and not normalized["stflow_alignment_enabled"]:
        raise ValueError(
            "SOFE stflow_feedback_enabled requires stflow_alignment_enabled."
        )
    if normalized["hagenaars_pred_kernel_size"] <= 0 or normalized["hagenaars_pred_kernel_size"] % 2 == 0:
        raise ValueError("SOFE hagenaars_pred_kernel_size must be a positive odd integer.")
    if normalized["ssnn_multiply_factor"] <= 0.0:
        raise ValueError("SOFE ssnn_multiply_factor must be > 0.")
    if normalized["ssnn_pred_kernel_size"] <= 0 or normalized["ssnn_pred_kernel_size"] % 2 == 0:
        raise ValueError("SOFE ssnn_pred_kernel_size must be a positive odd integer.")
    if normalized["recurrent_backend"] == "spikingjelly" and normalized["recurrent_cell"] != "convlif":
        raise NotImplementedError(
            "SOFE recurrent_backend='spikingjelly' currently only supports recurrent_cell='convlif'."
        )
    return normalized


def _expand_search_overrides(search_space: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not search_space:
        yield {}
        return

    items: List[Tuple[str, List[Any]]] = []
    for key, value in search_space.items():
        if isinstance(value, dict):
            items.append((key, list(_expand_search_overrides(value))))
        elif isinstance(value, (list, tuple)):
            items.append((key, [copy.deepcopy(v) for v in value]))
        else:
            items.append((key, [copy.deepcopy(value)]))

    keys = [key for key, _ in items]
    value_lists = [values for _, values in items]
    for combo in itertools.product(*value_lists):
        override: Dict[str, Any] = {}
        for key, option in zip(keys, combo):
            override[key] = copy.deepcopy(option)
        yield override


def expand_sofe_search_space(
    *,
    base_spec: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
    encoder_channels: Optional[Sequence[int]] = None,
    num_res_blocks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Expand an array-valued search space into normalized concrete specs."""
    search_space = dict(search_space or {})
    if not search_space:
        return [
            normalize_sofe_architecture_spec(
                base_spec,
                encoder_channels=encoder_channels,
                num_res_blocks=num_res_blocks,
            )
        ]

    concrete_specs: List[Dict[str, Any]] = []
    for override in _expand_search_overrides(search_space):
        merged = _deep_merge(base_spec or {}, override)
        concrete_specs.append(
            normalize_sofe_architecture_spec(
                merged,
                encoder_channels=encoder_channels,
                num_res_blocks=num_res_blocks,
            )
        )
    return concrete_specs


class ConvRNNCell(nn.Module):
    """Minimal convolutional RNN cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        pad = kernel_size // 2
        self.conv_x = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        if h_prev is None or h_prev.shape[0] != x.shape[0] or h_prev.shape[2:] != x.shape[2:]:
            h_prev = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
        return torch.tanh(self.conv_x(x) + self.conv_h(h_prev))


class ConvLIFCell(nn.Module):
    """Explicit-state convolutional LIF recurrent cell.

    The carried state is the membrane tensor. The emitted feature map is the
    current spike/rate output, which allows SOFE to keep runtime state in
    ``SOFEState.hidden`` without relying on module-local membrane buffers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        kernel_size: int = 3,
        alpha: float = 0.9,
        theta: float = 1.0,
        ann_levels: int = 4,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.ann_levels = int(ann_levels)
        self._mode = "ann"
        pad = kernel_size // 2
        self.conv_x = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad, bias=False)

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"ann", "snn"}:
            raise ValueError(f"Invalid ConvLIF mode: {mode!r}")
        self._mode = mode

    def reset_state(self) -> None:
        return None

    def _activate(self, membrane: torch.Tensor) -> torch.Tensor:
        if self._mode == "snn":
            return surrogate_heaviside(membrane - self.theta)
        return clip_floor(membrane, self.ann_levels, self.theta)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None or h_prev.shape[0] != x.shape[0] or h_prev.shape[2:] != x.shape[2:]:
            h_prev = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
        prev_spike = self._activate(h_prev)
        membrane = self.alpha * h_prev + self.conv_x(x) + self.conv_h(prev_spike)
        spike = self._activate(membrane)
        return spike, membrane


def _alpha_to_tau(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0.0:
        return 1.0
    if alpha >= 0.999:
        return 1e3
    return max(1.0 / max(1e-3, 1.0 - alpha), 1.0)


def _make_spikingjelly_surrogate(name: str, alpha: float):
    _sj_neuron, sj_surrogate = _import_spikingjelly_modules()
    name = str(name).lower()
    alpha = float(alpha)
    if name == "sigmoid":
        return sj_surrogate.Sigmoid(alpha=alpha, spiking=True)
    if name == "atan":
        return sj_surrogate.ATan(alpha=alpha, spiking=True)
    if name == "softsign":
        return sj_surrogate.SoftSign(alpha=alpha, spiking=True)
    raise NotImplementedError(
        f"SOFE surrogate_function={name!r} is not implemented yet. "
        "Currently supported: 'sigmoid', 'atan', 'softsign'."
    )


class SpikingJellyConvLIFCell(nn.Module):
    """Explicit-state recurrent conv cell backed by SpikingJelly neurons.

    Persistent runtime state still lives outside the module in ``SOFEState``.
    The wrapped SpikingJelly node only acts as a one-step membrane/spike update
    primitive and is reset after each call so no hidden buffers leak across
    samples or optimizer steps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        kernel_size: int = 3,
        alpha: float = 0.9,
        theta: float = 1.0,
        spiking_neuron: str = "lif",
        surrogate_function: str = "sigmoid",
        surrogate_alpha: float = 4.0,
        detach_reset: bool = True,
        spiking_backend: str = "torch",
        v_reset: Optional[float] = 0.0,
    ):
        super().__init__()
        sj_neuron, _sj_surrogate = _import_spikingjelly_modules()

        self.hidden_channels = int(hidden_channels)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.spiking_neuron = str(spiking_neuron).lower()
        self._mode = "snn"

        pad = kernel_size // 2
        self.conv_x = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad, bias=True)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad, bias=False)

        surrogate_fn = _make_spikingjelly_surrogate(surrogate_function, surrogate_alpha)
        node_kwargs = dict(
            v_threshold=float(theta),
            v_reset=v_reset,
            surrogate_function=surrogate_fn,
            detach_reset=bool(detach_reset),
            step_mode="s",
            backend=str(spiking_backend).lower(),
            store_v_seq=False,
        )

        if self.spiking_neuron == "lif":
            self.node = sj_neuron.LIFNode(
                tau=_alpha_to_tau(alpha),
                decay_input=False,
                **node_kwargs,
            )
        elif self.spiking_neuron == "if":
            self.node = sj_neuron.IFNode(**node_kwargs)
        elif self.spiking_neuron == "plif":
            self.node = sj_neuron.ParametricLIFNode(
                init_tau=_alpha_to_tau(alpha),
                decay_input=False,
                **node_kwargs,
            )
        else:
            raise NotImplementedError(
                f"SOFE spiking_neuron={self.spiking_neuron!r} is not implemented yet. "
                "Currently supported: 'lif', 'if', 'plif'."
            )

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"ann", "snn"}:
            raise ValueError(f"Invalid SpikingJelly ConvLIF mode: {mode!r}")
        # The SpikingJelly backend is intentionally always spiking; the external
        # mode flag is stored only so the wrapper contract remains consistent.
        self._mode = mode

    def reset_state(self) -> None:
        reset = getattr(self.node, "reset", None)
        if callable(reset):
            reset()

    def _zero_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros(
            x.shape[0],
            self.hidden_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        return {
            "mem": zeros,
            "spike": torch.zeros_like(zeros),
        }

    def _normalize_state(self, x: torch.Tensor, prev_state: Optional[Any]) -> Dict[str, torch.Tensor]:
        state = prev_state
        if isinstance(state, dict):
            mem = state.get("mem")
            spike = state.get("spike")
        elif torch.is_tensor(state):
            mem = state
            spike = None
        else:
            mem = None
            spike = None

        if (
            mem is None
            or not torch.is_tensor(mem)
            or mem.shape[0] != x.shape[0]
            or mem.shape[1] != self.hidden_channels
            or mem.shape[2:] != x.shape[2:]
        ):
            return self._zero_state(x)

        mem = mem.to(device=x.device, dtype=x.dtype)
        if spike is None or not torch.is_tensor(spike) or spike.shape != mem.shape:
            spike = torch.zeros_like(mem)
        else:
            spike = spike.to(device=x.device, dtype=x.dtype)
        return {
            "mem": mem,
            "spike": spike,
        }

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self._normalize_state(x, prev_state)
        drive = self.conv_x(x) + self.conv_h(state["spike"])
        self.node.v = state["mem"]
        spike = self.node(drive)
        next_mem = self.node.v
        self.reset_state()
        next_state = {
            "mem": next_mem,
            "spike": spike,
        }
        return spike, next_state


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    separable: bool = False,
    act: str = "silu",
) -> nn.Module:
    if separable:
        return DWConv(in_channels, out_channels, ksize=kernel_size, stride=stride, act=act)
    return BaseConv(in_channels, out_channels, ksize=kernel_size, stride=stride, act=act)


def _apply_decay_to_hidden(hidden: Optional[Any], decay: Optional[torch.Tensor]) -> Optional[Any]:
    if hidden is None or decay is None:
        return hidden
    if torch.is_tensor(hidden):
        decay_view = decay.view(-1, *([1] * max(hidden.ndim - 1, 0))).to(device=hidden.device, dtype=hidden.dtype)
        return hidden * decay_view
    if isinstance(hidden, tuple):
        return tuple(_apply_decay_to_hidden(v, decay) for v in hidden)
    if isinstance(hidden, list):
        return [_apply_decay_to_hidden(v, decay) for v in hidden]
    if isinstance(hidden, dict):
        decayed: Dict[str, Any] = {}
        for key, value in hidden.items():
            if key == "spike":
                decayed[key] = value
            else:
                decayed[key] = _apply_decay_to_hidden(value, decay)
        return decayed
    return hidden


def _run_recurrent_cell(
    cell: nn.Module,
    x: torch.Tensor,
    prev_state: Optional[Any],
) -> Tuple[torch.Tensor, Optional[Any]]:
    cell_out = cell(x, prev_state)
    if isinstance(cell_out, tuple):
        if len(cell_out) != 2:
            raise ValueError("SOFE recurrent cells returning tuples must return (output, next_state)")
        return cell_out
    return cell_out, cell_out


def _make_recurrent_cell(
    cell_type: str,
    channels: int,
    *,
    alpha: float = 0.9,
    theta: float = 1.0,
    recurrent_backend: str = "native",
    spiking_neuron: str = "lif",
    surrogate_function: str = "sigmoid",
    surrogate_alpha: float = 4.0,
    detach_reset: bool = True,
    spiking_backend: str = "torch",
    spiking_v_threshold: float = 1.0,
    spiking_v_reset: Optional[float] = 0.0,
) -> nn.Module:
    cell_type = str(cell_type).lower()
    recurrent_backend = str(recurrent_backend).lower()
    if recurrent_backend == "spikingjelly":
        if cell_type != "convlif":
            raise NotImplementedError(
                f"SOFE recurrent_backend={recurrent_backend!r} currently only supports recurrent_cell='convlif'."
            )
        return SpikingJellyConvLIFCell(
            channels,
            channels,
            kernel_size=3,
            alpha=alpha,
            theta=float(spiking_v_threshold),
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            surrogate_alpha=surrogate_alpha,
            detach_reset=detach_reset,
            spiking_backend=spiking_backend,
            v_reset=spiking_v_reset,
        )
    if cell_type == "convgru":
        return ConvGRUCell(channels, channels, kernel_size=3)
    if cell_type == "convrnn":
        return ConvRNNCell(channels, channels, kernel_size=3)
    if cell_type == "convlif":
        return ConvLIFCell(channels, channels, kernel_size=3, alpha=alpha, theta=theta)
    raise NotImplementedError(
        f"SOFE recurrent_cell={cell_type!r} is not implemented yet. "
        "Currently supported: 'convgru', 'convrnn', and 'convlif'."
    )


def _make_recurrent_cell_from_spec(
    spec: Dict[str, Any],
    cell_type: str,
    channels: int,
    *,
    alpha: float = 0.9,
    theta: float = 1.0,
) -> nn.Module:
    return _make_recurrent_cell(
        cell_type,
        channels,
        alpha=alpha,
        theta=theta,
        recurrent_backend=str(spec.get("recurrent_backend", "native")),
        spiking_neuron=str(spec.get("spiking_neuron", "lif")),
        surrogate_function=str(spec.get("surrogate_function", "sigmoid")),
        surrogate_alpha=float(spec.get("surrogate_alpha", 4.0)),
        detach_reset=bool(spec.get("detach_reset", True)),
        spiking_backend=str(spec.get("spiking_backend", "torch")),
        spiking_v_threshold=float(spec.get("spiking_v_threshold", theta)),
        spiking_v_reset=spec.get("spiking_v_reset", 0.0),
    )


class SOFECoreBase(nn.Module):
    """Base class for modular SOFE cores."""

    latent_channels: int
    pyramid_channels: List[int]

    def set_mode(self, mode: str) -> None:
        self._mode = str(mode)
        for module in self.modules():
            if module is self:
                continue
            child_set_mode = getattr(module, "set_mode", None)
            if callable(child_set_mode):
                child_set_mode(self._mode)

    def reset_state(self) -> None:
        return None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError


class SOFEFireCore(SOFECoreBase):
    """Single-scale ECM-like recurrent conv tower."""

    def __init__(
        self,
        *,
        spec: Dict[str, Any],
        input_channels: int,
        act: str = "silu",
        alpha: float = 0.9,
        theta: float = 1.0,
    ):
        super().__init__()
        stage_channels = list(spec["stage_channels"])
        stage_depths = list(spec["stage_depths"])
        separable = bool(spec["separable_conv"])
        placement = str(spec["recurrent_placement"])
        cell_type = str(spec["recurrent_cell"])

        self.stages = nn.ModuleList()
        self.recurrent_cells = nn.ModuleDict()

        in_ch = int(input_channels)
        for idx, (out_ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            blocks: List[nn.Module] = []
            for block_idx in range(depth):
                blocks.append(
                    _make_conv_block(
                        in_ch if block_idx == 0 else out_ch,
                        out_ch,
                        stride=1,
                        separable=separable,
                        act=act,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            if placement in {"every_stage", "encoder_only"} or (
                placement == "bottleneck_only" and idx == len(stage_channels) - 1
            ):
                self.recurrent_cells[str(idx)] = _make_recurrent_cell_from_spec(
                    spec,
                    cell_type,
                    out_ch,
                    alpha=alpha,
                    theta=theta,
                )
            in_ch = out_ch

        self.latent_channels = int(stage_channels[-1])
        self.pyramid_channels = [int(stage_channels[-1])]

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}
        features: List[torch.Tensor] = []

        out = x
        for idx, stage in enumerate(self.stages):
            out = stage(out)
            cell = self.recurrent_cells[str(idx)] if str(idx) in self.recurrent_cells else None
            if cell is not None:
                prev_hidden = _apply_decay_to_hidden(prev_state.get(str(idx)), decay)
                out, next_hidden = _run_recurrent_cell(cell, out, prev_hidden)
                next_state[str(idx)] = next_hidden
            features.append(out)

        return {"latent": out, "pyramid": [features[-1]]}, next_state


class _SOFEDownsampleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        stage_index: int,
        downsample_mode: str,
        separable: bool,
        act: str,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        if stage_index == 0:
            blocks.append(_make_conv_block(in_channels, out_channels, stride=1, separable=separable, act=act))
        elif downsample_mode == "stride_conv":
            blocks.append(_make_conv_block(in_channels, out_channels, stride=2, separable=separable, act=act))
        elif downsample_mode == "maxpool":
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(_make_conv_block(in_channels, out_channels, stride=1, separable=separable, act=act))
        else:
            raise NotImplementedError(
                f"SOFE downsample_mode={downsample_mode!r} is not implemented yet. "
                "Currently supported: 'stride_conv', 'maxpool'."
            )

        for _ in range(max(1, depth) - 1):
            blocks.append(_make_conv_block(out_channels, out_channels, stride=1, separable=separable, act=act))
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SOFEDecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        upsample_mode: str,
        skip_mode: str,
        separable: bool,
        act: str,
    ):
        super().__init__()
        self.upsample_mode = str(upsample_mode)
        self.skip_mode = str(skip_mode)
        self.out_channels = int(out_channels)

        if self.upsample_mode == "transposed_conv":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            up_out_channels = out_channels
        else:
            self.up = _make_conv_block(in_channels, out_channels, kernel_size=1, stride=1, separable=False, act=act)
            up_out_channels = out_channels

        if self.skip_mode == "concat":
            fuse_in_channels = up_out_channels + int(skip_channels)
            self.skip_proj = None
        elif self.skip_mode == "sum":
            self.skip_proj = None
            if int(skip_channels) != out_channels:
                self.skip_proj = _make_conv_block(skip_channels, out_channels, kernel_size=1, stride=1, separable=False, act=act)
            fuse_in_channels = out_channels
        elif self.skip_mode == "none":
            self.skip_proj = None
            fuse_in_channels = up_out_channels
        else:
            raise NotImplementedError(
                f"SOFE skip_mode={self.skip_mode!r} is not implemented yet. "
                "Currently supported: 'concat', 'sum', 'none'."
            )

        self.fuse = _make_conv_block(fuse_in_channels, out_channels, stride=1, separable=separable, act=act)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        if self.upsample_mode == "transposed_conv":
            x = self.up(x)
        else:
            size = skip.shape[2:] if skip is not None else (x.shape[2] * 2, x.shape[3] * 2)
            interp_mode = "nearest" if self.upsample_mode == "nearest_conv" else "bilinear"
            align = False if interp_mode == "bilinear" else None
            kwargs = {"size": size, "mode": interp_mode}
            if align is not None:
                kwargs["align_corners"] = False
            x = F.interpolate(x, **kwargs)
            x = self.up(x)

        if self.skip_mode == "none" or skip is None:
            return self.fuse(x)

        if self.skip_mode == "concat":
            x = torch.cat([x, skip], dim=1)
        else:
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            x = x + skip
        return self.fuse(x)


class SOFEUNetCore(SOFECoreBase):
    """Pyramidal encoder-decoder core closest to the existing MCM estimators."""

    def __init__(
        self,
        *,
        spec: Dict[str, Any],
        input_channels: int,
        act: str = "silu",
        alpha: float = 0.9,
        theta: float = 1.0,
    ):
        super().__init__()
        stage_channels = list(spec["stage_channels"])
        stage_depths = list(spec["stage_depths"])
        num_res_blocks = int(spec["num_res_blocks"])
        downsample_mode = str(spec["downsample_mode"])
        upsample_mode = str(spec["upsample_mode"])
        skip_mode = str(spec["skip_mode"])
        separable = bool(spec["separable_conv"])
        placement = str(spec["recurrent_placement"])
        cell_type = str(spec["recurrent_cell"])

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.recurrent_cells = nn.ModuleDict()

        in_ch = int(input_channels)
        for idx, (out_ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            self.encoders.append(
                _SOFEDownsampleStage(
                    in_ch,
                    out_ch,
                    depth=depth,
                    stage_index=idx,
                    downsample_mode=downsample_mode,
                    separable=separable,
                    act=act,
                )
            )
            if placement in {"every_stage", "encoder_only"}:
                self.recurrent_cells[f"enc_{idx}"] = _make_recurrent_cell_from_spec(
                    spec,
                    cell_type,
                    out_ch,
                    alpha=alpha,
                    theta=theta,
                )
            in_ch = out_ch

        bottleneck_channels = int(stage_channels[-1])
        for _ in range(num_res_blocks):
            self.bottleneck.append(
                _make_conv_block(
                    bottleneck_channels,
                    bottleneck_channels,
                    stride=1,
                    separable=separable,
                    act=act,
                )
            )
        if placement in {"every_stage", "bottleneck_only"}:
            self.recurrent_cells["bottleneck"] = _make_recurrent_cell_from_spec(
                spec,
                cell_type,
                bottleneck_channels,
                alpha=alpha,
                theta=theta,
            )

        rev_channels = list(reversed(stage_channels))
        decoder_channels: List[int] = []
        for idx in range(len(rev_channels) - 1):
            in_channels_stage = rev_channels[idx]
            skip_channels = rev_channels[idx + 1]
            out_channels = skip_channels
            self.decoders.append(
                _SOFEDecoderStage(
                    in_channels_stage,
                    skip_channels,
                    out_channels,
                    upsample_mode=upsample_mode,
                    skip_mode=skip_mode,
                    separable=separable,
                    act=act,
                )
            )
            if placement == "every_stage":
                self.recurrent_cells[f"dec_{idx}"] = _make_recurrent_cell_from_spec(
                    spec,
                    cell_type,
                    out_channels,
                    alpha=alpha,
                    theta=theta,
                )
            decoder_channels.append(out_channels)

        self.latent_channels = int(decoder_channels[-1] if decoder_channels else stage_channels[-1])
        # Keep metadata aligned with the runtime pyramid order emitted below:
        # [latent] + reversed(decoder_outputs[:-1]).
        self.pyramid_channels = [self.latent_channels] + list(reversed(decoder_channels[:-1]))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}
        skips: List[torch.Tensor] = []

        out = x
        for idx, encoder in enumerate(self.encoders):
            out = encoder(out)
            key = f"enc_{idx}"
            cell = self.recurrent_cells[key] if key in self.recurrent_cells else None
            if cell is not None:
                prev_hidden = _apply_decay_to_hidden(prev_state.get(key), decay)
                out, next_hidden = _run_recurrent_cell(cell, out, prev_hidden)
                next_state[key] = next_hidden
            skips.append(out)

        for block in self.bottleneck:
            out = block(out)

        bottleneck_cell = self.recurrent_cells["bottleneck"] if "bottleneck" in self.recurrent_cells else None
        if bottleneck_cell is not None:
            prev_hidden = _apply_decay_to_hidden(prev_state.get("bottleneck"), decay)
            out, next_hidden = _run_recurrent_cell(bottleneck_cell, out, prev_hidden)
            next_state["bottleneck"] = next_hidden

        decoder_outputs: List[torch.Tensor] = []
        skip_inputs = list(reversed(skips[:-1]))
        for idx, decoder in enumerate(self.decoders):
            skip = skip_inputs[idx] if idx < len(skip_inputs) else None
            out = decoder(out, skip)
            key = f"dec_{idx}"
            cell = self.recurrent_cells[key] if key in self.recurrent_cells else None
            if cell is not None:
                prev_hidden = _apply_decay_to_hidden(prev_state.get(key), decay)
                out, next_hidden = _run_recurrent_cell(cell, out, prev_hidden)
                next_state[key] = next_hidden
            decoder_outputs.append(out)

        latent = decoder_outputs[-1] if decoder_outputs else out
        pyramid = [latent] + list(reversed(decoder_outputs[:-1]))
        return {"latent": latent, "pyramid": pyramid}, next_state


class SOFESTFlowLiteCore(SOFECoreBase):
    """ST-FlowNet-inspired semi-pyramidal encoder with cross-scale fusion.

    Design notes:
    - Four encoder stages with spatial downsampling.
    - One recurrent augmentation cell at the first stage output.
    - Two residual bottleneck blocks on the coarsest feature map.
    - Cross-scale fusion by concatenating all encoder features resized to the
      coarsest resolution, then lightweight decoder upsampling.
    """

    def __init__(
        self,
        *,
        spec: Dict[str, Any],
        input_channels: int,
        act: str = "silu",
        alpha: float = 0.9,
        theta: float = 1.0,
    ):
        super().__init__()
        stage_channels = list(spec["stage_channels"])
        stage_depths = list(spec["stage_depths"])
        if len(stage_channels) < 4:
            raise ValueError("SOFE family='stflow_lite' requires at least 4 stage channels")
        if len(stage_depths) < 4:
            raise ValueError("SOFE family='stflow_lite' requires at least 4 stage depths")
        stage_channels = stage_channels[:4]
        stage_depths = stage_depths[:4]

        downsample_mode = str(spec["downsample_mode"])
        separable = bool(spec["separable_conv"])
        cell_type = str(spec["recurrent_cell"])
        use_aug = str(spec.get("recurrent_placement", "encoder_only")) != "none"

        self.encoders = nn.ModuleList()
        in_ch = int(input_channels)
        for idx in range(4):
            out_ch = int(stage_channels[idx])
            depth = int(stage_depths[idx])
            self.encoders.append(
                _SOFEDownsampleStage(
                    in_ch,
                    out_ch,
                    depth=depth,
                    stage_index=idx,
                    downsample_mode=downsample_mode,
                    separable=separable,
                    act=act,
                )
            )
            in_ch = out_ch

        self.aug_cell: Optional[nn.Module]
        if use_aug:
            self.aug_cell = _make_recurrent_cell_from_spec(
                spec,
                cell_type,
                int(stage_channels[0]),
                alpha=alpha,
                theta=theta,
            )
        else:
            self.aug_cell = None

        bottleneck_channels = int(stage_channels[-1])
        self.res_blocks = nn.ModuleList(
            [
                _make_conv_block(
                    bottleneck_channels,
                    bottleneck_channels,
                    stride=1,
                    separable=separable,
                    act=act,
                )
                for _ in range(max(int(spec["num_res_blocks"]), 2))
            ]
        )

        fusion_in_channels = int(sum(stage_channels))
        self.fusion_proj = _make_conv_block(
            fusion_in_channels,
            bottleneck_channels,
            stride=1,
            separable=False,
            act=act,
        )
        self.up_blocks = nn.ModuleList(
            [
                _make_conv_block(bottleneck_channels, int(stage_channels[2]), stride=1, separable=False, act=act),
                _make_conv_block(int(stage_channels[2]), int(stage_channels[1]), stride=1, separable=False, act=act),
                _make_conv_block(int(stage_channels[1]), int(stage_channels[0]), stride=1, separable=False, act=act),
            ]
        )

        self.latent_channels = int(stage_channels[0])
        self.pyramid_channels = [self.latent_channels]

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}
        enc_feats: List[torch.Tensor] = []

        out = x
        for idx, encoder in enumerate(self.encoders):
            out = encoder(out)
            if idx == 0 and self.aug_cell is not None:
                prev_hidden = _apply_decay_to_hidden(prev_state.get("aug"), decay)
                out, next_hidden = _run_recurrent_cell(self.aug_cell, out, prev_hidden)
                next_state["aug"] = next_hidden
            enc_feats.append(out)

        out = enc_feats[-1]
        for block in self.res_blocks:
            out = block(out)

        # Cross-scale fusion at coarsest scale.
        fuse_inputs = enc_feats[:-1] + [out]
        target_size = out.shape[2:]
        fuse_pieces: List[torch.Tensor] = []
        for feat in fuse_inputs:
            if tuple(feat.shape[2:]) != tuple(target_size):
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            fuse_pieces.append(feat)
        fused = self.fusion_proj(torch.cat(fuse_pieces, dim=1))

        # Lightweight decode from coarse to fine using encoder sizes as anchors.
        decode_targets = [enc_feats[2], enc_feats[1], enc_feats[0]]
        decoder_feats: List[torch.Tensor] = []
        dec = fused
        for idx, block in enumerate(self.up_blocks):
            dec = F.interpolate(
                dec,
                size=decode_targets[idx].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            dec = block(dec)
            decoder_feats.append(dec)

        latent = decoder_feats[-1]
        return {"latent": latent, "pyramid": [latent]}, next_state


def _apply_flow_activation(flow: torch.Tensor, activation: str) -> torch.Tensor:
    activation = str(activation).lower()
    if activation == "tanh":
        return torch.tanh(flow)
    if activation == "softsign":
        return F.softsign(flow)
    if activation in {"identity", "none", "linear"}:
        return flow
    raise NotImplementedError(
        f"SOFE flow_activation={activation!r} is not implemented yet. "
        "Currently supported: 'tanh', 'softsign', and 'identity'."
    )


def _initialize_flow_head(head: nn.Conv2d, *, init_mode: str, init_std: float) -> None:
    init_mode = str(init_mode).lower()
    init_std = float(init_std)
    if init_mode == "zero":
        nn.init.zeros_(head.weight)
    elif init_mode == "small_normal":
        nn.init.normal_(head.weight, mean=0.0, std=init_std)
    elif init_mode == "kaiming_normal":
        nn.init.kaiming_normal_(head.weight, mode="fan_in", nonlinearity="linear")
    elif init_mode == "xavier_uniform":
        nn.init.xavier_uniform_(head.weight)
    elif init_mode == "firenet_small_uniform":
        nn.init.uniform_(head.weight, -init_std, init_std)
    else:
        raise NotImplementedError(
            f"SOFE flow_head_init={init_mode!r} is not implemented yet. "
            "Currently supported: 'zero', 'small_normal', 'kaiming_normal', "
            "'xavier_uniform', 'firenet_small_uniform'."
        )
    if head.bias is not None:
        nn.init.zeros_(head.bias)


class SOFESingleScaleReadout(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        mid_channels: int,
        flow_activation: str,
        flow_head_init: str = "zero",
        flow_head_init_std: float = 1e-3,
        act: str = "silu",
    ):
        super().__init__()
        self.pre = _make_conv_block(in_channels, mid_channels, kernel_size=3, stride=1, separable=False, act=act)
        self.head = nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True)
        self.flow_activation = str(flow_activation)
        _initialize_flow_head(
            self.head,
            init_mode=flow_head_init,
            init_std=flow_head_init_std,
        )

    def forward(self, features: Dict[str, Any], *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = features["latent"]
        if tuple(x.shape[2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.pre(x)
        flow_logits = self.head(x)
        flow = _apply_flow_activation(flow_logits, self.flow_activation)
        return flow, {"flow_logits": flow_logits}


class SOFEMultiScaleReadout(nn.Module):
    def __init__(
        self,
        *,
        in_channels: Sequence[int],
        mid_channels: int,
        flow_activation: str,
        flow_head_init: str = "zero",
        flow_head_init_std: float = 1e-3,
        readout_scales: Optional[Sequence[int]] = None,
        act: str = "silu",
    ):
        super().__init__()
        all_indices = list(range(len(in_channels)))
        if readout_scales is None:
            self.scale_indices = all_indices
        else:
            self.scale_indices = [int(v) for v in readout_scales if int(v) in all_indices]
            if not self.scale_indices:
                self.scale_indices = all_indices

        self.pre_blocks = nn.ModuleList(
            [_make_conv_block(ch, mid_channels, kernel_size=3, stride=1, separable=False, act=act) for ch in in_channels]
        )
        self.heads = nn.ModuleList([nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True) for _ in in_channels])
        self.flow_activation = str(flow_activation)
        for head in self.heads:
            _initialize_flow_head(
                head,
                init_mode=flow_head_init,
                init_std=flow_head_init_std,
            )

    def forward(self, features: Dict[str, Any], *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pyramid = list(features.get("pyramid") or [])
        if not pyramid:
            pyramid = [features["latent"]]

        selected: List[Tuple[int, torch.Tensor]] = []
        for idx in self.scale_indices:
            if idx < len(pyramid):
                selected.append((idx, pyramid[idx]))

        if not selected:
            feat = features["latent"]
            if tuple(feat.shape[2:]) != tuple(output_size):
                feat = F.interpolate(feat, size=output_size, mode="bilinear", align_corners=False)
            flow_logits = self.heads[0](self.pre_blocks[0](feat))
            flow = _apply_flow_activation(flow_logits, self.flow_activation)
            return flow, {"flow_logits": flow_logits}

        # Decode coarse updates first, then upsample-and-refine toward finer maps.
        # This matches the intended "successive coarse updates" behavior more
        # closely than averaging all scales after resizing them to one grid.
        selected.sort(key=lambda item: (int(item[1].shape[2]) * int(item[1].shape[3]), item[0]))

        flow_logits: Optional[torch.Tensor] = None
        for idx, feat in selected:
            feat = self.pre_blocks[idx](feat)
            delta_logits = self.heads[idx](feat)
            if flow_logits is None:
                flow_logits = delta_logits
                continue
            if tuple(flow_logits.shape[2:]) != tuple(delta_logits.shape[2:]):
                flow_logits = F.interpolate(flow_logits, size=delta_logits.shape[2:], mode="nearest")
            flow_logits = flow_logits + delta_logits

        assert flow_logits is not None
        if tuple(flow_logits.shape[2:]) != tuple(output_size):
            flow_logits = F.interpolate(flow_logits, size=output_size, mode="nearest")
        flow = _apply_flow_activation(flow_logits, self.flow_activation)
        return flow, {"flow_logits": flow_logits}


class SOFEPersistentDeltaReadout(nn.Module):
    """Single-scale delta-flow readout head for persistent flow updates."""

    def __init__(
        self,
        *,
        in_channels: int,
        mid_channels: int,
        flow_activation: str,
        flow_head_init: str = "zero",
        flow_head_init_std: float = 1e-3,
        act: str = "silu",
    ):
        super().__init__()
        self.pre = _make_conv_block(in_channels, mid_channels, kernel_size=3, stride=1, separable=False, act=act)
        self.head = nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True)
        self.flow_activation = str(flow_activation)
        _initialize_flow_head(
            self.head,
            init_mode=flow_head_init,
            init_std=flow_head_init_std,
        )

    def forward(self, features: Dict[str, Any], *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = features["latent"]
        if tuple(x.shape[2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.pre(x)
        delta_flow_logits = self.head(x)
        delta_flow = _apply_flow_activation(delta_flow_logits, self.flow_activation)
        return delta_flow, {"flow_logits": delta_flow_logits, "delta_flow_logits": delta_flow_logits}


from smcfnet.model.ofe.firenet_lif import SOFEFireNetLIFCore, SOFEFireNetLIFReadout  # noqa: E402
from smcfnet.model.ofe.hagenaars_firenet_lif import (  # noqa: E402
    SOFEHagenaarsFireNetLIFCore,
    SOFEHagenaarsFireNetLIFReadout,
)
from smcfnet.model.ofe.ssnn_net import (  # noqa: E402
    SOFESSNNNetCore,
    SOFESSNNNetPyramidReadout,
    SOFESSNNNetReadout,
)


def build_sofe_modules(
    spec: Dict[str, Any],
    *,
    input_channels: int,
    act: str = "silu",
    alpha: float = 0.9,
    theta: float = 1.0,
) -> nn.ModuleDict:
    family = str(spec["family"]).lower()
    readout_type = str(spec["readout_type"]).lower()

    if family == "fire":
        core = SOFEFireCore(spec=spec, input_channels=input_channels, act=act, alpha=alpha, theta=theta)
    elif family == "firenet_lif":
        core = SOFEFireNetLIFCore(spec=spec, input_channels=input_channels, act=act, alpha=alpha, theta=theta)
    elif family == "hagenaars_firenet_lif":
        core = SOFEHagenaarsFireNetLIFCore(
            spec=spec,
            input_channels=input_channels,
            act=act,
            alpha=alpha,
            theta=theta,
        )
    elif family == "ssnn_net":
        core = SOFESSNNNetCore(
            spec=spec,
            input_channels=input_channels,
            act=act,
            alpha=alpha,
            theta=theta,
        )
    elif family == "unet":
        core = SOFEUNetCore(spec=spec, input_channels=input_channels, act=act, alpha=alpha, theta=theta)
    elif family == "stflow_lite":
        core = SOFESTFlowLiteCore(spec=spec, input_channels=input_channels, act=act, alpha=alpha, theta=theta)
    else:
        raise NotImplementedError(
            f"SOFE family={family!r} is not implemented yet. "
            "Currently supported: 'fire', 'firenet_lif', 'hagenaars_firenet_lif', "
            "'ssnn_net', 'unet', 'stflow_lite'."
        )

    if family == "firenet_lif":
        readout: nn.Module = SOFEFireNetLIFReadout(
            in_channels=core.latent_channels,
            flow_head_init_std=float(spec["flow_head_init_std"]),
        )
    elif family == "hagenaars_firenet_lif":
        readout = SOFEHagenaarsFireNetLIFReadout(
            in_channels=core.latent_channels,
            flow_activation=str(spec["flow_activation"]),
            flow_head_init_std=float(spec["flow_head_init_std"]),
            pred_kernel_size=int(spec.get("hagenaars_pred_kernel_size", 3)),
        )
    elif family == "ssnn_net":
        ssnn_readout_type = str(spec.get("ssnn_readout", "single")).lower()
        _readout_common = dict(
            flow_activation=str(spec["flow_activation"]),
            flow_head_init_std=float(spec["flow_head_init_std"]),
            pred_kernel_size=int(spec.get("ssnn_pred_kernel_size", 3)),
            flow_scale_x=float(spec.get("ssnn_flow_scale_x", 1.0)),
            flow_scale_y=float(spec.get("ssnn_flow_scale_y", 1.0)),
        )
        if ssnn_readout_type == "pyramid_residual":
            if len(core.pyramid_channels) < 4:
                raise ValueError(
                    "ssnn_readout='pyramid_residual' requires ssnn_version='v2' "
                    f"(pyramid_channels has {len(core.pyramid_channels)} entries, need 4)."
                )
            readout = SOFESSNNNetPyramidReadout(
                pyramid_channels=list(core.pyramid_channels),
                **_readout_common,
            )
        else:
            readout = SOFESSNNNetReadout(
                in_channels=core.latent_channels,
                **_readout_common,
            )
    elif readout_type == "single_scale":
        readout = SOFESingleScaleReadout(
            in_channels=core.latent_channels,
            mid_channels=int(spec["readout_mid_channels"]),
            flow_activation=str(spec["flow_activation"]),
            flow_head_init=str(spec["flow_head_init"]),
            flow_head_init_std=float(spec["flow_head_init_std"]),
            act=act,
        )
    elif readout_type == "multi_scale":
        readout = SOFEMultiScaleReadout(
            in_channels=list(core.pyramid_channels),
            mid_channels=int(spec["readout_mid_channels"]),
            flow_activation=str(spec["flow_activation"]),
            flow_head_init=str(spec["flow_head_init"]),
            flow_head_init_std=float(spec["flow_head_init_std"]),
            readout_scales=spec.get("readout_scales"),
            act=act,
        )
    elif readout_type == "persistent_delta":
        readout = SOFEPersistentDeltaReadout(
            in_channels=core.latent_channels,
            mid_channels=int(spec["readout_mid_channels"]),
            flow_activation=str(spec["flow_activation"]),
            flow_head_init=str(spec["flow_head_init"]),
            flow_head_init_std=float(spec["flow_head_init_std"]),
            act=act,
        )
    else:
        raise NotImplementedError(
            f"SOFE readout_type={readout_type!r} is not implemented yet. "
            "Currently supported: 'single_scale', 'multi_scale', 'persistent_delta'."
        )

    return nn.ModuleDict({"core": core, "readout": readout})
