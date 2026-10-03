"""Architecture-specific targets for the generic EA framework."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from smcfnet.config.config import Config
from smcfnet.evolution.cuadrado_unet.search_space import apply_peod_cuadrado_unet_guardrails
from smcfnet.evolution.firenet.search_space import apply_firenet_lif_guardrails
from smcfnet.evolution.ssnn_net.search_space import apply_ssnn_net_guardrails
from smcfnet.evolution.hagenaars_firenet.search_space import apply_hagenaars_firenet_lif_guardrails
from smcfnet.evolution.search_spaces import expand_parameter_grid
from smcfnet.model.ofe.sofe_builder import normalize_sofe_architecture_spec


def _omit_step_time_us(target: Any) -> bool:
    return bool(getattr(target, "omit_step_time_us", False))


@dataclass
class FireNetLIFTarget:
    """Search target for SOFE FireNet-LIF architecture sweeps."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "firenet_lif"

    def default_genome(self) -> Dict[str, Any]:
        return {
            "mcm": {
                "events_per_step_target": 15000,
                "bptt_steps": 5,
                "flow_scaling": 4.0,
                "smooth_weight": 1.0,
                "contrast_weight": 1.0,
                "loss_weight": 0.1,
            },
            "sofe_architecture": normalize_sofe_architecture_spec({"family": "firenet_lif"}),
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        genome_dict = dict(genome or {})
        base = self.default_genome()
        if any(key in genome_dict for key in ("mcm", "sofe_architecture", "training")):
            merged = _deep_merge(base, genome_dict)
        else:
            # Backward-compatible flat SOFE genome support.
            merged = copy.deepcopy(base)
            merged["sofe_architecture"] = _deep_merge(merged["sofe_architecture"], genome_dict)

        mcm = dict(merged.get("mcm") or {})
        normalized_mcm = {
            "events_per_step_target": int(mcm.get("events_per_step_target", 15000)),
            "bptt_steps": int(mcm.get("bptt_steps", 5)),
            "flow_scaling": float(mcm.get("flow_scaling", 4.0)),
            "smooth_weight": float(mcm.get("smooth_weight", 1.0)),
            "contrast_weight": float(mcm.get("contrast_weight", 1.0)),
            "loss_weight": float(mcm.get("loss_weight", 0.1)),
        }
        if not _omit_step_time_us(self):
            normalized_mcm["step_time_us"] = (
                None
                if mcm.get("step_time_us", None) is None
                else int(mcm.get("step_time_us"))
            )
        normalized = {
            "mcm": normalized_mcm,
            "sofe_architecture": normalize_sofe_architecture_spec(merged.get("sofe_architecture") or {}),
        }
        if "training" in merged:
            normalized["training"] = copy.deepcopy(dict(merged["training"]))
        guardrail_result = apply_firenet_lif_guardrails(normalized)
        self.last_repair_metadata = {
            "repairs": list(guardrail_result.repairs),
            "warnings": list(guardrail_result.warnings),
            "valid": bool(guardrail_result.valid),
        }
        guarded = guardrail_result.genome
        guarded["sofe_architecture"] = normalize_sofe_architecture_spec(guarded.get("sofe_architecture") or {})
        return guarded

    def expand_search_space(self) -> list[Dict[str, Any]]:
        return expand_parameter_grid(self.default_genome(), self.search_space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a Cuadrado genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        return self.normalize_genome(
            {
                "num_bins": int(cfg.data.event_bins),
                "mode": str(mcm_cfg.mode or "snn"),
                "image_height": int(cfg.data.image_height),
                "image_width": int(cfg.data.image_width),
                "kernel_size": int(getattr(mcm_cfg, "cuadrado_kernel_size", 7)),
                "multiply_factor": float(getattr(mcm_cfg, "cuadrado_multiply_factor", 10.0)),
            }
        )

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a FireNet-LIF EA genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        return self.normalize_genome(
            {
                "mcm": {
                    "step_time_us": (
                        None
                        if getattr(mcm_cfg, "step_time_us", None) is None
                        else int(getattr(mcm_cfg, "step_time_us"))
                    ),
                    "events_per_step_target": int(mcm_cfg.events_per_step_target or 15000),
                    "bptt_steps": int(mcm_cfg.bptt_steps or 5),
                    "flow_scaling": float(mcm_cfg.flow_scaling),
                    "smooth_weight": float(mcm_cfg.smooth_weight),
                    "contrast_weight": float(mcm_cfg.contrast_weight),
                    "loss_weight": float(mcm_cfg.loss_weight),
                },
                "sofe_architecture": copy.deepcopy(dict(mcm_cfg.sofe_architecture or {})),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        mcm = normalized["mcm"]
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.prefetcher_type = "streaming"
        cfg.model.mcm.step_time_us = mcm.get("step_time_us", None)
        cfg.model.mcm.events_per_step_target = (
            None if mcm["events_per_step_target"] is None else int(mcm["events_per_step_target"])
        )
        cfg.model.mcm.bptt_steps = int(mcm["bptt_steps"])
        cfg.model.mcm.flow_scaling = float(mcm["flow_scaling"])
        cfg.model.mcm.smooth_weight = float(mcm["smooth_weight"])
        cfg.model.mcm.contrast_weight = float(mcm["contrast_weight"])
        cfg.model.mcm.loss_weight = float(mcm["loss_weight"])
        cfg.model.mcm.sofe_architecture = copy.deepcopy(normalized["sofe_architecture"])
        cfg.model.mcm.sofe_search_space = copy.deepcopy(self.search_space)
        cfg.training.streaming_step_policy = "chunk_backward"
        for key, value in normalized.get("training", {}).items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)


@dataclass
class HagenaarsFireNetLIFTarget:
    """Search target for the full-resolution Hagenaars/Paredes-Valles FireNet-LIF SOFE."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "hagenaars_firenet_lif"

    def default_genome(self) -> Dict[str, Any]:
        return {
            "mcm": {
                "step_time_us": None,
                "events_per_step_target": 15000,
                "bptt_steps": 5,
                "decay_tau_us": 10000.0,
                "flow_scaling": 4.0,
                "smooth_weight": 1.0,
                "contrast_weight": 1.0,
                "loss_weight": 0.1,
            },
            "sofe_architecture": normalize_sofe_architecture_spec(
                {
                    "family": "hagenaars_firenet_lif",
                    "step_encoding": "hist2",
                    "normalize_counts": False,
                    "count_encoding": "raw",
                    "count_scale": 1.0,
                    "base_channels": 32,
                    "kernel_size": 3,
                    "spiking_v_threshold": 1.0,
                    "surrogate_function": "atan",
                    "surrogate_alpha": 1.0,
                    "detach_reset": True,
                    "firenet_spike_decay_mode": "decay",
                    "readout_type": "single_scale",
                    "flow_activation": "softsign",
                    "flow_head_init_std": 0.01,
                    "hagenaars_use_bias": True,
                    "hagenaars_pred_kernel_size": 3,
                }
            ),
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        genome_dict = dict(genome or {})
        base = self.default_genome()
        if any(key in genome_dict for key in ("mcm", "sofe_architecture", "training")):
            merged = _deep_merge(base, genome_dict)
        else:
            merged = copy.deepcopy(base)
            merged["sofe_architecture"] = _deep_merge(merged["sofe_architecture"], genome_dict)

        mcm = dict(merged.get("mcm") or {})
        decay_tau_us = mcm.get("decay_tau_us", 10000.0)
        normalized_mcm = {
            "events_per_step_target": (
                None
                if mcm.get("events_per_step_target", 15000) is None
                else int(mcm.get("events_per_step_target", 15000))
            ),
            "bptt_steps": int(mcm.get("bptt_steps", 5)),
            "decay_tau_us": None if decay_tau_us is None else float(decay_tau_us),
            "flow_scaling": float(mcm.get("flow_scaling", 4.0)),
            "smooth_weight": float(mcm.get("smooth_weight", 1.0)),
            "contrast_weight": float(mcm.get("contrast_weight", 1.0)),
            "loss_weight": float(mcm.get("loss_weight", 0.1)),
        }
        if not _omit_step_time_us(self):
            normalized_mcm["step_time_us"] = (
                None
                if mcm.get("step_time_us", None) is None
                else int(mcm.get("step_time_us"))
            )
        normalized = {
            "mcm": normalized_mcm,
            "sofe_architecture": normalize_sofe_architecture_spec(merged.get("sofe_architecture") or {}),
        }
        normalized["sofe_architecture"]["family"] = "hagenaars_firenet_lif"
        normalized["sofe_architecture"] = normalize_sofe_architecture_spec(normalized["sofe_architecture"])
        if "training" in merged:
            normalized["training"] = copy.deepcopy(dict(merged["training"]))
        guardrail_result = apply_hagenaars_firenet_lif_guardrails(normalized)
        self.last_repair_metadata = {
            "repairs": list(guardrail_result.repairs),
            "warnings": list(guardrail_result.warnings),
            "valid": bool(guardrail_result.valid),
        }
        guarded = guardrail_result.genome
        guarded["sofe_architecture"] = normalize_sofe_architecture_spec(guarded.get("sofe_architecture") or {})
        return guarded

    def expand_search_space(self) -> list[Dict[str, Any]]:
        return expand_parameter_grid(self.default_genome(), self.search_space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a Hagenaars FireNet-LIF EA genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        return self.normalize_genome(
            {
                "mcm": {
                    "step_time_us": (
                        None
                        if getattr(mcm_cfg, "step_time_us", None) is None
                        else int(getattr(mcm_cfg, "step_time_us"))
                    ),
                    "events_per_step_target": (
                        None
                        if getattr(mcm_cfg, "events_per_step_target", None) is None
                        else int(getattr(mcm_cfg, "events_per_step_target"))
                    ),
                    "bptt_steps": int(mcm_cfg.bptt_steps or 5),
                    "decay_tau_us": (
                        None
                        if getattr(mcm_cfg, "decay_tau_us", None) is None
                        else float(getattr(mcm_cfg, "decay_tau_us"))
                    ),
                    "flow_scaling": float(mcm_cfg.flow_scaling),
                    "smooth_weight": float(mcm_cfg.smooth_weight),
                    "contrast_weight": float(mcm_cfg.contrast_weight),
                    "loss_weight": float(mcm_cfg.loss_weight),
                },
                "sofe_architecture": copy.deepcopy(dict(mcm_cfg.sofe_architecture or {})),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        mcm = normalized["mcm"]
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.prefetcher_type = "streaming"
        cfg.model.mcm.step_time_us = mcm.get("step_time_us", None)
        cfg.model.mcm.events_per_step_target = (
            None if mcm["events_per_step_target"] is None else int(mcm["events_per_step_target"])
        )
        cfg.model.mcm.bptt_steps = int(mcm["bptt_steps"])
        cfg.model.mcm.decay_tau_us = mcm["decay_tau_us"]
        cfg.model.mcm.flow_scaling = float(mcm["flow_scaling"])
        cfg.model.mcm.smooth_weight = float(mcm["smooth_weight"])
        cfg.model.mcm.contrast_weight = float(mcm["contrast_weight"])
        cfg.model.mcm.loss_weight = float(mcm["loss_weight"])
        cfg.model.mcm.sofe_architecture = copy.deepcopy(normalized["sofe_architecture"])
        cfg.model.mcm.sofe_search_space = copy.deepcopy(self.search_space)
        cfg.training.streaming_step_policy = "chunk_backward"
        for key, value in normalized.get("training", {}).items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)


@dataclass
class CuadradoUNetTarget:
    """Search target for the Cuadrado/OF_EV_SNN style U-Net estimator."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "cuadrado_unet"

    def default_genome(self) -> Dict[str, Any]:
        return {
            "num_bins": 21,
            "mode": "snn",
            "image_height": 480,
            "image_width": 640,
            "kernel_size": 7,
            "multiply_factor": 10.0,
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(self.default_genome())
        merged.update(dict(genome or {}))
        return {
            "num_bins": int(merged["num_bins"]),
            "mode": str(merged["mode"]),
            "image_height": int(merged["image_height"]),
            "image_width": int(merged["image_width"]),
            "kernel_size": int(merged["kernel_size"]),
            "multiply_factor": float(merged["multiply_factor"]),
        }

    def expand_search_space(self) -> list[Dict[str, Any]]:
        return expand_parameter_grid(self.default_genome(), self.search_space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a Cuadrado genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        return self.normalize_genome(
            {
                "num_bins": int(cfg.data.event_bins),
                "mode": str(mcm_cfg.mode or "snn"),
                "image_height": int(cfg.data.image_height),
                "image_width": int(cfg.data.image_width),
                "kernel_size": int(getattr(mcm_cfg, "cuadrado_kernel_size", 7)),
                "multiply_factor": float(getattr(mcm_cfg, "cuadrado_multiply_factor", 10.0)),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "cuadrado_unet"
        cfg.model.mcm.estimator_representation = "cuadrado_histogram"
        cfg.model.mcm.input_representation = "voxel"
        cfg.model.mcm.prefetcher_type = "voxel"
        cfg.model.mcm.mode = str(normalized["mode"])
        cfg.model.mcm.cuadrado_kernel_size = int(normalized["kernel_size"])
        cfg.model.mcm.cuadrado_multiply_factor = float(normalized["multiply_factor"])
        cfg.data.event_bins = int(normalized["num_bins"])
        cfg.data.image_height = int(normalized["image_height"])
        cfg.data.image_width = int(normalized["image_width"])


@dataclass
class SSNNNetTarget:
    """Search target for streaming SSNN under SOFE/SMCM."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "ssnn_net"

    def default_genome(self) -> Dict[str, Any]:
        return {
            "mcm": {
                "step_time_us": None,
                "events_per_step_target": 15000,
                "bptt_steps": 5,
                "decay_tau_us": 10000.0,
                "flow_scaling": 4.0,
                "smooth_weight": 1.0,
                "contrast_weight": 1.0,
                "loss_weight": 0.1,
            },
            "sofe_architecture": normalize_sofe_architecture_spec(
                {
                    "family": "ssnn_net",
                    "step_encoding": "hist2",
                    "normalize_counts": False,
                    "count_encoding": "raw",
                    "count_scale": 1.0,
                    "base_channels": 24,
                    "kernel_size": 5,
                    "ssnn_multiply_factor": 10.0,
                    "spiking_v_threshold": 1.0,
                    "surrogate_alpha": 4.0,
                    "detach_reset": True,
                    "ssnn_hard_reset": False,
                    "ssnn_spike_decay_mode": "decay",
                    "readout_type": "single_scale",
                    "flow_activation": "softsign",
                    "flow_head_init_std": 0.01,
                    "ssnn_pred_kernel_size": 3,
                }
            ),
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        genome_dict = dict(genome or {})
        base = self.default_genome()
        if any(key in genome_dict for key in ("mcm", "sofe_architecture", "training")):
            merged = _deep_merge(base, genome_dict)
        else:
            merged = copy.deepcopy(base)
            merged["sofe_architecture"] = _deep_merge(merged["sofe_architecture"], genome_dict)

        mcm = dict(merged.get("mcm") or {})
        events_per_step_target = mcm.get("events_per_step_target", 15000)
        if "step_time_us" in mcm and mcm.get("step_time_us") is not None:
            events_per_step_target = None
        normalized_mcm = {
            "bptt_steps": int(mcm.get("bptt_steps", 5)),
            "decay_tau_us": (
                None
                if mcm.get("decay_tau_us", 10000.0) is None
                else float(mcm.get("decay_tau_us", 10000.0))
            ),
            "flow_scaling": float(mcm.get("flow_scaling", 4.0)),
            "smooth_weight": float(mcm.get("smooth_weight", 1.0)),
            "contrast_weight": float(mcm.get("contrast_weight", 1.0)),
            "loss_weight": float(mcm.get("loss_weight", 0.1)),
        }
        if "step_time_us" in mcm and mcm.get("step_time_us") is not None:
            normalized_mcm["step_time_us"] = int(mcm["step_time_us"])
        if events_per_step_target is not None:
            normalized_mcm["events_per_step_target"] = int(events_per_step_target)
        normalized = {
            "mcm": normalized_mcm,
            "sofe_architecture": normalize_sofe_architecture_spec(merged.get("sofe_architecture") or {}),
        }
        normalized["sofe_architecture"]["family"] = "ssnn_net"
        normalized["sofe_architecture"] = normalize_sofe_architecture_spec(normalized["sofe_architecture"])
        if "training" in merged:
            normalized["training"] = copy.deepcopy(dict(merged["training"]))
        guardrail_result = apply_ssnn_net_guardrails(normalized)
        self.last_repair_metadata = {
            "repairs": list(guardrail_result.repairs),
            "warnings": list(guardrail_result.warnings),
            "valid": bool(guardrail_result.valid),
        }
        guarded = guardrail_result.genome
        guarded["sofe_architecture"] = normalize_sofe_architecture_spec(guarded.get("sofe_architecture") or {})
        return guarded

    def expand_search_space(self) -> list[Dict[str, Any]]:
        return expand_parameter_grid(self.default_genome(), self.search_space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a streaming SSNN EA genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        events_per_step_target = getattr(mcm_cfg, "events_per_step_target", None)
        mcm_genome = {
            "bptt_steps": int(mcm_cfg.bptt_steps or 5),
            "decay_tau_us": (
                None
                if getattr(mcm_cfg, "decay_tau_us", None) is None
                else float(getattr(mcm_cfg, "decay_tau_us"))
            ),
            "flow_scaling": float(mcm_cfg.flow_scaling),
            "smooth_weight": float(mcm_cfg.smooth_weight),
            "contrast_weight": float(mcm_cfg.contrast_weight),
            "loss_weight": float(mcm_cfg.loss_weight),
        }
        if getattr(mcm_cfg, "step_time_us", None) is not None:
            mcm_genome["step_time_us"] = int(getattr(mcm_cfg, "step_time_us"))
            if events_per_step_target is None:
                mcm_genome["events_per_step_target"] = None
        if events_per_step_target is not None:
            mcm_genome["events_per_step_target"] = int(events_per_step_target)
        return self.normalize_genome(
            {
                "mcm": mcm_genome,
                "sofe_architecture": copy.deepcopy(dict(mcm_cfg.sofe_architecture or {})),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        mcm = normalized["mcm"]
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.prefetcher_type = "streaming"
        if "step_time_us" in mcm:
            cfg.model.mcm.step_time_us = mcm["step_time_us"]
        if "events_per_step_target" in mcm:
            cfg.model.mcm.events_per_step_target = int(mcm["events_per_step_target"])
        cfg.model.mcm.bptt_steps = int(mcm["bptt_steps"])
        cfg.model.mcm.decay_tau_us = mcm["decay_tau_us"]
        cfg.model.mcm.flow_scaling = float(mcm["flow_scaling"])
        cfg.model.mcm.smooth_weight = float(mcm["smooth_weight"])
        cfg.model.mcm.contrast_weight = float(mcm["contrast_weight"])
        cfg.model.mcm.loss_weight = float(mcm["loss_weight"])
        cfg.model.mcm.sofe_architecture = copy.deepcopy(normalized["sofe_architecture"])
        cfg.model.mcm.sofe_search_space = copy.deepcopy(self.search_space)
        cfg.training.streaming_step_policy = "chunk_backward"
        for key, value in normalized.get("training", {}).items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)


@dataclass
class SSNNNetV2Target:
    """Search target for the streaming SSNN-Net v2 architecture under SOFE/SMCM.

    Uses the v2 four-stage encoder with dilated bottleneck, configurable
    inter-stage signal, and optional high-resolution refinement.  The default
    genome reflects the hand-validated starting point from the Phase 1/2 rework.
    """

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "ssnn_net_v2"
    omit_step_time_us: bool = True

    def default_genome(self) -> Dict[str, Any]:
        return {
            "mcm": {
                "step_time_us": None,
                "events_per_step_target": 15000,
                "bptt_steps": 5,
                "decay_tau_us": 10000.0,
                "flow_scaling": 4.0,
                "smooth_weight": 1.0,
                "contrast_weight": 1.0,
                "loss_weight": 0.1,
            },
            "sofe_architecture": normalize_sofe_architecture_spec(
                {
                    "family": "ssnn_net",
                    "ssnn_version": "v2",
                    "step_encoding": "hist2",
                    "normalize_counts": False,
                    "count_encoding": "raw",
                    "count_scale": 1.0,
                    "base_channels": 24,
                    "kernel_size": 5,
                    "ssnn_stage_channels": [24, 48, 96, 192],
                    "ssnn_num_stages": 4,
                    "ssnn_bottleneck_dilations": [1, 2, 4],
                    "ssnn_encoder_signal": "membrane",
                    "ssnn_edge_tap": "hist",
                    "ssnn_refine_channels": 32,
                    "ssnn_refine_layers": 2,
                    "ssnn_readout": "single",
                    "ssnn_multiply_factor": 10.0,
                    "spiking_v_threshold": 1.0,
                    "surrogate_alpha": 4.0,
                    "detach_reset": True,
                    "ssnn_hard_reset": False,
                    "ssnn_spike_decay_mode": "hold",
                    "flow_activation": "softsign",
                    "flow_head_init_std": 0.01,
                    "ssnn_pred_kernel_size": 3,
                }
            ),
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        genome_dict = dict(genome or {})
        base = self.default_genome()
        if any(key in genome_dict for key in ("mcm", "sofe_architecture", "training")):
            merged = _deep_merge(base, genome_dict)
        else:
            merged = copy.deepcopy(base)
            merged["sofe_architecture"] = _deep_merge(merged["sofe_architecture"], genome_dict)

        mcm = dict(merged.get("mcm") or {})
        events_per_step_target = mcm.get("events_per_step_target", 15000)
        if "step_time_us" in mcm and mcm.get("step_time_us") is not None:
            events_per_step_target = None
        normalized_mcm = {
            "bptt_steps": int(mcm.get("bptt_steps", 5)),
            "decay_tau_us": (
                None
                if mcm.get("decay_tau_us", 10000.0) is None
                else float(mcm.get("decay_tau_us", 10000.0))
            ),
            "flow_scaling": float(mcm.get("flow_scaling", 4.0)),
            "smooth_weight": float(mcm.get("smooth_weight", 1.0)),
            "contrast_weight": float(mcm.get("contrast_weight", 1.0)),
            "loss_weight": float(mcm.get("loss_weight", 0.1)),
        }
        if not _omit_step_time_us(self):
            normalized_mcm["step_time_us"] = (
                None
                if mcm.get("step_time_us", None) is None
                else int(mcm.get("step_time_us"))
            )
        if events_per_step_target is not None:
            normalized_mcm["events_per_step_target"] = int(events_per_step_target)

        sofe_arch = dict(merged.get("sofe_architecture") or {})
        sofe_arch["family"] = "ssnn_net"
        sofe_arch["ssnn_version"] = "v2"
        normalized = {
            "mcm": normalized_mcm,
            "sofe_architecture": normalize_sofe_architecture_spec(sofe_arch),
        }
        if "training" in merged:
            normalized["training"] = copy.deepcopy(dict(merged["training"]))
        guardrail_result = apply_ssnn_net_guardrails(normalized)
        self.last_repair_metadata = {
            "repairs": list(guardrail_result.repairs),
            "warnings": list(guardrail_result.warnings),
            "valid": bool(guardrail_result.valid),
        }
        guarded = guardrail_result.genome
        guarded["sofe_architecture"]["ssnn_version"] = "v2"
        guarded["sofe_architecture"] = normalize_sofe_architecture_spec(guarded.get("sofe_architecture") or {})
        return guarded

    def expand_search_space(self) -> list[Dict[str, Any]]:
        from smcfnet.evolution.ssnn_net.search_space import SSNN_NET_V2_SEARCH_SPACE
        space = dict(self.search_space) if self.search_space else dict(SSNN_NET_V2_SEARCH_SPACE)
        return expand_parameter_grid(self.default_genome(), space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        """Extract a streaming SSNN v2 EA genome from an existing training config."""
        mcm_cfg = cfg.model.mcm
        events_per_step_target = getattr(mcm_cfg, "events_per_step_target", None)
        mcm_genome: Dict[str, Any] = {
            "bptt_steps": int(mcm_cfg.bptt_steps or 5),
            "decay_tau_us": (
                None
                if getattr(mcm_cfg, "decay_tau_us", None) is None
                else float(getattr(mcm_cfg, "decay_tau_us"))
            ),
            "flow_scaling": float(mcm_cfg.flow_scaling),
            "smooth_weight": float(mcm_cfg.smooth_weight),
            "contrast_weight": float(mcm_cfg.contrast_weight),
            "loss_weight": float(mcm_cfg.loss_weight),
        }
        if getattr(mcm_cfg, "step_time_us", None) is not None:
            mcm_genome["step_time_us"] = int(getattr(mcm_cfg, "step_time_us"))
            if events_per_step_target is None:
                mcm_genome["events_per_step_target"] = None
        if events_per_step_target is not None:
            mcm_genome["events_per_step_target"] = int(events_per_step_target)
        return self.normalize_genome(
            {
                "mcm": mcm_genome,
                "sofe_architecture": copy.deepcopy(dict(mcm_cfg.sofe_architecture or {})),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        mcm = normalized["mcm"]
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.prefetcher_type = "streaming"
        if "step_time_us" in mcm:
            cfg.model.mcm.step_time_us = mcm["step_time_us"]
        if "events_per_step_target" in mcm:
            cfg.model.mcm.events_per_step_target = int(mcm["events_per_step_target"])
        cfg.model.mcm.bptt_steps = int(mcm["bptt_steps"])
        cfg.model.mcm.decay_tau_us = mcm["decay_tau_us"]
        cfg.model.mcm.flow_scaling = float(mcm["flow_scaling"])
        cfg.model.mcm.smooth_weight = float(mcm["smooth_weight"])
        cfg.model.mcm.contrast_weight = float(mcm["contrast_weight"])
        cfg.model.mcm.loss_weight = float(mcm["loss_weight"])
        cfg.model.mcm.sofe_architecture = copy.deepcopy(normalized["sofe_architecture"])
        cfg.model.mcm.sofe_search_space = copy.deepcopy(self.search_space)
        cfg.training.streaming_step_policy = "chunk_backward"
        for key, value in normalized.get("training", {}).items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)


@dataclass
class PEODCuadradoUNetTarget:
    """Search target for the PEODCuadradoUNet architecture on the PEOD dataset.

    num_bins (21), mode (snn), image_height (720), and image_width (1280) are
    fixed by the PEOD input contract and must never be mutated as genes.
    Searchable genes: kernel_size, multiply_factor, smooth_weight,
    contrast_weight, loss_weight.
    """

    search_space: Dict[str, Any] = field(default_factory=dict)
    name: str = "cuadrado_unet_peod"
    architecture_name: str = "cuadrado_unet_peod"

    def default_genome(self) -> Dict[str, Any]:
        return {
            "num_bins": 21,
            "mode": "snn",
            "image_height": 720,
            "image_width": 1280,
            "kernel_size": 7,
            "multiply_factor": 10.0,
            "smooth_weight": 1.0,
            "loss_weight": 0.1,
        }

    def normalize_genome(self, genome: Mapping[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(self.default_genome())
        merged.update(dict(genome or {}))
        normalized = {
            "num_bins": 21,
            "mode": "snn",
            "image_height": 720,
            "image_width": 1280,
            "kernel_size": int(merged["kernel_size"]),
            "multiply_factor": float(merged["multiply_factor"]),
            "smooth_weight": float(merged["smooth_weight"]),
            "loss_weight": float(merged["loss_weight"]),
        }
        guardrail_result = apply_peod_cuadrado_unet_guardrails(normalized)
        self.last_repair_metadata = {
            "repairs": list(guardrail_result.repairs),
            "warnings": list(guardrail_result.warnings),
            "valid": bool(guardrail_result.valid),
        }
        return guardrail_result.genome

    def expand_search_space(self) -> list[Dict[str, Any]]:
        from smcfnet.evolution.cuadrado_unet.search_space import PEOD_CUADRADO_UNET_SEARCH_SPACE
        space = dict(self.search_space) if self.search_space else dict(PEOD_CUADRADO_UNET_SEARCH_SPACE)
        return expand_parameter_grid(self.default_genome(), space)

    def genome_from_config(self, cfg: Config) -> Dict[str, Any]:
        mcm_cfg = cfg.model.mcm
        return self.normalize_genome(
            {
                "kernel_size": int(getattr(mcm_cfg, "cuadrado_kernel_size", 7)),
                "multiply_factor": float(getattr(mcm_cfg, "cuadrado_multiply_factor", 10.0)),
                "smooth_weight": float(mcm_cfg.smooth_weight),
                "loss_weight": float(mcm_cfg.loss_weight),
            }
        )

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        normalized = self.normalize_genome(genome)
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = str(self.architecture_name)
        cfg.model.mcm.estimator_representation = "cuadrado_histogram"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.input_representation = "voxel"
        cfg.model.mcm.prefetcher_type = "voxel"
        cfg.model.mcm.mode = "snn"
        cfg.model.mcm.cuadrado_kernel_size = int(normalized["kernel_size"])
        cfg.model.mcm.cuadrado_multiply_factor = float(normalized["multiply_factor"])
        cfg.model.mcm.smooth_weight = float(normalized["smooth_weight"])
        cfg.model.mcm.loss_weight = float(normalized["loss_weight"])
        # contrast_weight is fixed at 1.0 (not a gene; correlated with smooth_weight).
        # Do NOT set event_bins: must remain 21 for PEOD input compatibility.
        # Do NOT set image_height/image_width: comes from data config.


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
