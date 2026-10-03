"""Streaming optical-flow estimator (SOFE) skeleton.

This module defines the planned event-native spiking flow estimator used by
SMCM.  The implementation is intentionally incomplete: the goal is to make the
streaming contract explicit before any architecture-specific code is written.

Key assumptions from the current plan:
- Input events arrive as packed tensors with absolute ``t_us`` timestamps.
- Temporal partitioning happens inside SMCM / SOFE, not in the DataLoader.
- Hidden state persists across step updates and is detached at configured
  truncated-BPTT boundaries.
- Readout is modular and may occur once per RGB frame or multiple times within
  the same detector-aligned sample.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.ofe import FlowEstimator, register_estimator
from smcfnet.model.ofe.sofe_builder import (
    build_sofe_modules,
    expand_sofe_search_space,
    normalize_sofe_architecture_spec,
)
from smcfnet.model.spiking_blocks import ConvGRUCell


@dataclass
class SOFEState:
    """Persistent estimator state carried across streaming updates.

    Planned responsibilities:
    - Hold all recurrent hidden state / membrane state required by the SNN.
    - Store any warm-start flow or decoded latent state needed by the readout.
    - Track the timestamp of the last processed step so elapsed-time decay can
      be computed from real ``t_us`` rather than assuming fixed-size updates.
    - Remain cheap to detach/reset so the trainer can enforce truncated BPTT
      and sequence boundaries without reconstructing the whole estimator.
    """

    hidden: Any = None
    readout_state: Any = None
    flow_warm: Optional[torch.Tensor] = None
    last_step_ref_us: Optional[torch.Tensor] = None


@dataclass
class StreamingStepBatch:
    """One temporally-partitioned step extracted from a packed event batch.

    Planned contents:
    - A compact step representation suitable for one SNN update.
    - The original packed events or per-step offsets when sparse-event warping
      or event-warping losses need exact per-event timing.
    - Per-sample timing metadata such as step start/end/reference timestamps.
    - Optional masks indicating which batch elements are active for this step.
    """

    step_input: Optional[torch.Tensor] = None
    events: Optional[torch.Tensor] = None
    events_offsets: Optional[torch.Tensor] = None
    window_start_us: Optional[torch.Tensor] = None
    step_start_us: Optional[torch.Tensor] = None
    step_end_us: Optional[torch.Tensor] = None
    step_ref_us: Optional[torch.Tensor] = None
    active_mask: Optional[torch.Tensor] = None


class StepInputEncoderBase(nn.Module):
    """Stateless step encoder boundary for streaming SOFE inputs."""

    output_channels: int

    def forward(self, step: StreamingStepBatch) -> torch.Tensor:
        raise NotImplementedError


class HistogramStepInputEncoder(StepInputEncoderBase):
    """Default v1 step encoder using split-polarity count histograms."""

    def __init__(
        self,
        *,
        encoding: str,
        image_height: int,
        image_width: int,
        normalize_counts: bool = True,
        count_encoding: str | None = None,
        count_scale: float = 1.0,
        mini_voxel_bins: int = 4,
    ):
        super().__init__()
        self.encoding = str(encoding).lower()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.normalize_counts = bool(normalize_counts)
        self.count_encoding = str(
            count_encoding if count_encoding is not None
            else ("normalized" if self.normalize_counts else "raw")
        ).lower()
        self.count_scale = float(count_scale)
        self.mini_voxel_bins = int(mini_voxel_bins)

        if self.encoding == "hist2":
            self.output_channels = 2
        elif self.encoding in {"hist4", "hist4_legacy"}:
            self.output_channels = 4
        elif self.encoding == "mini_voxel":
            self.output_channels = int(self.mini_voxel_bins)
        else:
            raise NotImplementedError(
                f"SOFE step_encoding={self.encoding!r} is not implemented; supported encodings "
                "are 'hist2', 'hist4', 'hist4_legacy', and 'mini_voxel'."
            )
        if self.count_encoding not in {"normalized", "raw", "log", "sqrt"}:
            raise NotImplementedError(
                f"SOFE count_encoding={self.count_encoding!r} is not implemented; supported encodings "
                "are 'normalized', 'raw', 'log', and 'sqrt'."
            )

    def _finalize_counts(
        self,
        out_b: torch.Tensor,
        *,
        total_events: int,
    ) -> torch.Tensor:
        """Map accumulated per-pixel counts to the tensor fed into the core."""
        if self.count_encoding == "normalized":
            denom = float(max(int(total_events), 1))
            out_b = out_b / denom
        elif self.count_encoding == "log":
            out_b = torch.log1p(out_b)
        elif self.count_encoding == "sqrt":
            out_b = torch.sqrt(out_b)
        elif self.count_encoding != "raw":
            raise NotImplementedError(
                f"Unsupported count_encoding={self.count_encoding!r}"
            )

        if self.count_scale != 1.0:
            out_b = out_b * self.count_scale
        return out_b

    def forward(self, step: StreamingStepBatch) -> torch.Tensor:
        if step.events is None or step.events_offsets is None:
            raise ValueError("StreamingStepBatch must include events and events_offsets")

        if self.encoding == "mini_voxel":
            raise NotImplementedError(
                "SOFE step_encoding='mini_voxel' is declared in the build system but "
                "is intentionally out of scope for the v1 streaming implementation."
            )

        events = step.events
        offsets = step.events_offsets.to(torch.int64)
        batch_size = int(offsets.numel() - 1)
        h = int(self.image_height)
        w = int(self.image_width)
        hw = int(h * w)
        channels = int(self.output_channels)
        out = events.new_zeros((batch_size, channels, h, w), dtype=torch.float32)

        def _accumulate(
            channel_index: int,
            out_b: torch.Tensor,
            flat_idx: torch.Tensor,
            values: torch.Tensor,
        ) -> None:
            if int(flat_idx.numel()) == 0:
                return
            out_b[channel_index].scatter_add_(0, flat_idx, values)

        def _accumulate_hist4_legacy(
            out_b: torch.Tensor,
            flat_idx: torch.Tensor,
            pos: torch.Tensor,
            neg: torch.Tensor,
        ) -> None:
            split = max(int(flat_idx.numel()) // 2, 1)
            first_idx = flat_idx[:split]
            second_idx = flat_idx[split:]
            _accumulate(0, out_b, first_idx, pos[:split])
            _accumulate(1, out_b, first_idx, neg[:split])
            _accumulate(2, out_b, second_idx, pos[split:])
            _accumulate(3, out_b, second_idx, neg[split:])

        for b in range(batch_size):
            start = int(offsets[b].item())
            end = int(offsets[b + 1].item())
            ev_b = events[start:end]
            if ev_b.numel() == 0:
                continue

            x = ev_b[:, 0].to(torch.int64)
            y = ev_b[:, 1].to(torch.int64)
            t = ev_b[:, 2].to(torch.int64)
            p = ev_b[:, 3].to(torch.float32)

            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if not torch.any(valid):
                continue

            x = x[valid]
            y = y[valid]
            t = t[valid]
            p = p[valid]
            flat_idx = y * w + x
            pos = (p > 0).to(torch.float32)
            neg = (p <= 0).to(torch.float32)
            out_b = out[b].view(channels, hw)

            if self.encoding == "hist2":
                _accumulate(0, out_b, flat_idx, pos)
                _accumulate(1, out_b, flat_idx, neg)
            elif self.encoding == "hist4_legacy":
                _accumulate_hist4_legacy(out_b, flat_idx, pos, neg)
            else:
                step_start_us = None
                if step.step_start_us is not None:
                    step_start_us = int(step.step_start_us[b].item())
                step_end_us = None
                if step.step_end_us is not None:
                    step_end_us = int(step.step_end_us[b].item())

                if step_start_us is None or step_end_us is None:
                    step_start_us = int(t.min().item())
                    step_end_us = int(t.max().item()) + 1

                if step_end_us <= step_start_us:
                    _accumulate_hist4_legacy(out_b, flat_idx, pos, neg)
                else:
                    mid_us = step_start_us + ((step_end_us - step_start_us) // 2)
                    early_mask = t < mid_us
                    late_mask = ~early_mask
                    _accumulate(0, out_b, flat_idx[early_mask], pos[early_mask])
                    _accumulate(1, out_b, flat_idx[early_mask], neg[early_mask])
                    _accumulate(2, out_b, flat_idx[late_mask], pos[late_mask])
                    _accumulate(3, out_b, flat_idx[late_mask], neg[late_mask])

            out[b] = self._finalize_counts(
                out[b],
                total_events=int(flat_idx.numel()),
            )

        return out


class SOFECoreAdapter(nn.Module):
    """Adapter boundary separating scheduler logic from the recurrent core."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def set_mode(self, mode: str) -> None:
        if hasattr(self.core, "set_mode"):
            self.core.set_mode(mode)

    def reset_state(self) -> None:
        if hasattr(self.core, "reset_state"):
            self.core.reset_state()

    def forward(
        self,
        step_input: torch.Tensor,
        core_state: Any,
        decay: torch.Tensor,
        *,
        step: Optional[StreamingStepBatch] = None,
    ) -> Tuple[Any, Any]:
        core_features, next_core_state = self.core(step_input, core_state, decay)
        return next_core_state, core_features


class _ReadoutAlignmentGRU(nn.Module):
    """Readout-side recurrent alignment module for dense flow hypotheses."""

    def __init__(self, *, state_channels: int):
        super().__init__()
        self.state_channels = int(max(int(state_channels), 1))
        self.cell = ConvGRUCell(in_channels=2, hidden_channels=self.state_channels, kernel_size=3)
        self.delta_head = nn.Conv2d(self.state_channels, 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.delta_head.weight)
        if self.delta_head.bias is not None:
            nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        flow_in: torch.Tensor,
        prev_hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.cell(flow_in, prev_hidden)
        delta = self.delta_head(hidden)
        flow_out = flow_in + delta
        return flow_out, hidden, delta


class PersistentSOFEReadout(nn.Module):
    """Readout boundary supporting persistent flow-hypothesis updates."""

    def __init__(
        self,
        *,
        readout_head: nn.Module,
        image_height: int,
        image_width: int,
        readout_type: str = "single_scale",
        readout_update_rule: str = "residual",
        readout_alpha_init: float = 1.0,
        readout_learn_alpha: bool = False,
        readout_state_channels: Optional[int] = None,
        stflow_alignment_enabled: bool = False,
        stflow_feedback_enabled: bool = False,
    ):
        super().__init__()
        self.readout_head = readout_head
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.readout_type = str(readout_type).lower()
        self.readout_update_rule = str(readout_update_rule).lower()
        self.readout_learn_alpha = bool(readout_learn_alpha)
        self.stflow_alignment_enabled = bool(stflow_alignment_enabled)
        self.stflow_feedback_enabled = bool(stflow_feedback_enabled)
        state_ch = int(readout_state_channels or 0)
        self.readout_state_channels = state_ch if state_ch > 0 else 16

        alpha_init = float(readout_alpha_init)
        alpha_init = max(min(alpha_init, 1.0), 1e-6)
        if self.readout_learn_alpha:
            alpha_clamped = max(min(alpha_init, 1.0 - 1e-6), 1e-6)
            self.readout_alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_clamped, dtype=torch.float32)))
        else:
            self.register_buffer("readout_alpha_fixed", torch.tensor(alpha_init, dtype=torch.float32), persistent=False)

        if self.stflow_feedback_enabled:
            raise NotImplementedError(
                "stflow_feedback_enabled=True is intentionally deferred until the "
                "no-feedback alignment baseline is stable."
            )
        if self.stflow_alignment_enabled:
            self.alignment = _ReadoutAlignmentGRU(state_channels=self.readout_state_channels)
        else:
            self.alignment = None

    def set_mode(self, mode: str) -> None:
        if hasattr(self.readout_head, "set_mode"):
            self.readout_head.set_mode(mode)

    def reset_state(self) -> None:
        if hasattr(self.readout_head, "reset_state"):
            self.readout_head.reset_state()

    def _resolve_alpha(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.readout_learn_alpha:
            return torch.sigmoid(self.readout_alpha_logit).to(device=device, dtype=dtype)
        return self.readout_alpha_fixed.to(device=device, dtype=dtype)

    @staticmethod
    def _as_readout_state_dict(readout_state: Any) -> Dict[str, Any]:
        if isinstance(readout_state, dict):
            return dict(readout_state)
        return {}

    def _decode(
        self,
        core_features: Any,
        *,
        t_ref_us: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if isinstance(core_features, dict):
            head_out = self.readout_head(
                core_features,
                output_size=(self.image_height, self.image_width),
            )
            if isinstance(head_out, tuple):
                flow, aux = head_out
                return flow, dict(aux or {})
            return head_out, {}

        if t_ref_us is None:
            return None, {}

        bsz = int(t_ref_us.numel())
        flow = torch.zeros(
            (bsz, 2, self.image_height, self.image_width),
            device=t_ref_us.device,
            dtype=torch.float32,
        )
        return flow, {}

    def update(
        self,
        core_features: Any,
        *,
        readout_state: Any = None,
        flow_warm: Optional[torch.Tensor] = None,
        t_ref_us: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, Optional[torch.Tensor], Dict[str, Any]]:
        readout_state_dict = self._as_readout_state_dict(readout_state)
        flow_candidate, aux = self._decode(core_features, t_ref_us=t_ref_us)
        if flow_candidate is None:
            if flow_warm is not None:
                flow_candidate = flow_warm
            elif t_ref_us is not None:
                bsz = int(t_ref_us.numel())
                flow_candidate = torch.zeros(
                    (bsz, 2, self.image_height, self.image_width),
                    device=t_ref_us.device,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    "PersistentSOFEReadout.update requires either decodable core features, "
                    "an existing flow_warm tensor, or t_ref_us for zero-flow fallback."
                )

        next_readout_state: Dict[str, Any] = {}
        if self.readout_type == "persistent_delta":
            if flow_warm is None:
                flow_prev = torch.zeros_like(flow_candidate)
            else:
                flow_prev = flow_warm
                if tuple(flow_prev.shape) != tuple(flow_candidate.shape):
                    flow_prev = F.interpolate(flow_prev, size=flow_candidate.shape[2:], mode="bilinear", align_corners=False)

            alpha = self._resolve_alpha(device=flow_candidate.device, dtype=flow_candidate.dtype)
            if self.readout_update_rule == "ema":
                flow_next = (1.0 - alpha) * flow_prev + alpha * flow_candidate
            else:
                flow_next = flow_prev + alpha * flow_candidate

            aux = dict(aux or {})
            if "flow_logits" in aux and "delta_flow_logits" not in aux:
                aux["delta_flow_logits"] = aux["flow_logits"]
            aux["delta_flow"] = flow_candidate.detach()
            aux["readout_alpha"] = alpha.detach()
            next_readout_state["readout_alpha"] = alpha.detach()
        else:
            flow_next = flow_candidate

        if self.stflow_alignment_enabled:
            if self.alignment is None:
                raise RuntimeError("stflow alignment enabled but alignment module is not initialized.")
            prev_align_hidden = readout_state_dict.get("align_hidden")
            flow_next, align_hidden, align_delta = self.alignment(flow_next, prev_align_hidden)
            next_readout_state["align_hidden"] = align_hidden
            aux["alignment_delta_flow"] = align_delta.detach()
            aux["alignment_flow"] = flow_next.detach()

        if self.stflow_alignment_enabled and "align_hidden" in readout_state_dict and "align_hidden" not in next_readout_state:
            next_readout_state["align_hidden"] = readout_state_dict["align_hidden"]

        if "readout_alpha" in readout_state_dict and "readout_alpha" not in next_readout_state:
            next_readout_state["readout_alpha"] = readout_state_dict["readout_alpha"]
        return next_readout_state, flow_next, dict(aux or {})

    def emit(
        self,
        *,
        readout_state: Any,
        flow_warm: Optional[torch.Tensor],
        t_ref_us: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if flow_warm is None:
            bsz = int(t_ref_us.numel())
            flow = torch.zeros(
                (bsz, 2, self.image_height, self.image_width),
                device=t_ref_us.device,
                dtype=torch.float32,
            )
        else:
            flow = flow_warm

        return flow, {}


@dataclass
class _SampleStepSpec:
    """Compact per-sample hybrid step metadata."""

    event_start: int
    event_end: int
    step_start_us: int
    step_end_us: int
    step_ref_us: int
    active: bool


@register_estimator("sofe")
class StreamingSpikingFlowEstimator(FlowEstimator):
    """Skeleton for the planned streaming spiking optical-flow estimator.

    This class exists to lock down names and responsibilities before the
    estimator is implemented.  Methods intentionally raise
    ``NotImplementedError``; the docstrings/comments describe what each method
    must eventually do.
    """

    def __init__(
        self,
        *,
        step_time_us: Optional[int] = None,
        partition_mode: str = "fixed_time",
        events_per_step_target: Optional[int] = None,
        step_dt_max_us: Optional[int] = None,
        step_dt_min_us: Optional[int] = None,
        readout_every_n_steps: Optional[int] = None,
        readout_interval_us: Optional[int] = None,
        decay_tau_us: Optional[float] = None,
        reset_on_sequence_change: bool = True,
        bptt_steps: Optional[int] = None,
        encoder_channels: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        activation: str = "silu",
        mode: str = "snn",
        alpha: float = 0.9,
        theta: float = 1.0,
        architecture_spec: Optional[Dict[str, Any]] = None,
        architecture_search_space: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.step_time_us = step_time_us
        self.partition_mode = partition_mode
        self.events_per_step_target = events_per_step_target
        self.step_dt_max_us = step_dt_max_us
        self.step_dt_min_us = step_dt_min_us
        self.readout_every_n_steps = readout_every_n_steps
        self.readout_interval_us = readout_interval_us
        self.decay_tau_us = decay_tau_us
        self.reset_on_sequence_change = reset_on_sequence_change
        self.bptt_steps = bptt_steps
        self._mode = mode
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.theta = float(theta)

        self.num_bins = int(kwargs.pop("num_bins", 5) or 5)
        self.image_height = int(kwargs.pop("image_height", kwargs.pop("height", 480)) or 480)
        self.image_width = int(kwargs.pop("image_width", kwargs.pop("width", 640)) or 640)
        self.max_steps_per_sample = kwargs.pop("max_steps_per_sample", None)
        self._legacy_encoder_channels = list(encoder_channels or [32, 64, 128, 256])
        self._legacy_num_res_blocks = int(num_res_blocks)
        self.architecture_search_space = copy.deepcopy(dict(architecture_search_space or {}))
        self.architecture_spec = normalize_sofe_architecture_spec(
            architecture_spec,
            encoder_channels=self._legacy_encoder_channels,
            num_res_blocks=self._legacy_num_res_blocks,
        )
        self.extra_config = dict(kwargs)

        self._state: Optional[SOFEState] = None
        self._last_flow: Optional[torch.Tensor] = None
        self._last_flow_history: Optional[List[torch.Tensor]] = None
        self._last_flow_logits: Optional[torch.Tensor] = None
        self._last_readout_debug: Optional[Dict[str, Any]] = None
        self._last_step_readout_aux: Optional[Dict[str, Any]] = None

        step_encoding = str(self.architecture_spec.get("step_encoding", "hist2")).lower()
        step_input_channels = self._step_input_channels()
        built_modules = build_sofe_modules(
            self.architecture_spec,
            input_channels=step_input_channels,
            act=self.activation,
            alpha=self.alpha,
            theta=self.theta,
        )
        self.step_encoder = step_encoding
        self.step_input_encoder = HistogramStepInputEncoder(
            encoding=step_encoding,
            image_height=self.image_height,
            image_width=self.image_width,
            normalize_counts=bool(self.architecture_spec.get("normalize_counts", True)),
            count_encoding=str(self.architecture_spec.get("count_encoding", "normalized")),
            count_scale=float(self.architecture_spec.get("count_scale", 1.0)),
            mini_voxel_bins=int(self.architecture_spec.get("mini_voxel_bins", self.num_bins)),
        )
        self.sofe_core = SOFECoreAdapter(built_modules["core"])
        self.sofe_readout = PersistentSOFEReadout(
            readout_head=built_modules["readout"],
            image_height=self.image_height,
            image_width=self.image_width,
            readout_type=str(self.architecture_spec.get("readout_type", "single_scale")),
            readout_update_rule=str(self.architecture_spec.get("readout_update_rule", "residual")),
            readout_alpha_init=float(self.architecture_spec.get("readout_alpha_init", 1.0)),
            readout_learn_alpha=bool(self.architecture_spec.get("readout_learn_alpha", False)),
            readout_state_channels=self.architecture_spec.get("readout_state_channels"),
            stflow_alignment_enabled=bool(self.architecture_spec.get("stflow_alignment_enabled", False)),
            stflow_feedback_enabled=bool(self.architecture_spec.get("stflow_feedback_enabled", False)),
        )
        self.architecture_modules = {
            "step_input": self.step_input_encoder,
            "core": self.sofe_core,
            "readout": self.sofe_readout,
        }

        self.placeholder_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.set_mode(mode)

    def _head_threshold(self, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        core = getattr(self.sofe_core, "core", None)
        head = getattr(core, "head", None)
        threshold = getattr(head, "thresh", None)
        if not isinstance(threshold, torch.Tensor) or threshold.numel() == 0:
            return None
        return threshold.detach().to(device=device, dtype=dtype).clamp_min(0.01)

    def collect_head_state_metrics(
        self,
        hidden: Any,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if not isinstance(hidden, dict):
            return {}
        head_state = hidden.get("head")
        if not isinstance(head_state, dict):
            return {}

        spike = head_state.get("spike")
        membrane = head_state.get("v")
        if not isinstance(spike, torch.Tensor) or spike.numel() == 0:
            return {}

        spike_f = spike.detach().float()
        membrane_f = membrane.detach().float() if isinstance(membrane, torch.Tensor) and membrane.numel() > 0 else None
        if active_mask is not None:
            active = torch.as_tensor(active_mask, dtype=torch.bool, device=spike_f.device).reshape(-1)
            if int(active.numel()) == int(spike_f.shape[0]) and bool(active.any().item()):
                spike_f = spike_f[active]
                if membrane_f is not None and int(membrane_f.shape[0]) == int(active.numel()):
                    membrane_f = membrane_f[active]
            elif int(active.numel()) == int(spike_f.shape[0]):
                return {}

        metrics: Dict[str, float] = {
            "head_spike_rate": float(spike_f.mean().item()),
        }

        if membrane_f is not None:
            threshold = self._head_threshold(device=membrane_f.device, dtype=membrane_f.dtype)
            if threshold is not None:
                margin = membrane_f - threshold
                margin_flat = margin.reshape(-1)
                metrics["head_v_minus_thresh_max"] = float(margin_flat.max().item())
                metrics["head_v_minus_thresh_p99"] = float(
                    torch.quantile(margin_flat, 0.99).item()
                )
        return metrics

    def set_mode(self, mode: str) -> None:
        """Switch estimator operating mode.

        Requirements for the future implementation:
        - Validate and store the requested mode (at minimum ``ann`` / ``snn``).
        - Propagate the mode into all spiking submodules so training and
          inference can share the same estimator wrapper.
        - Avoid resetting hidden state unless explicitly requested elsewhere.
        """
        self._mode = str(mode)
        for module in (self.sofe_core, self.sofe_readout):
          if hasattr(module, "set_mode"):
            module.set_mode(self._mode)

    def get_mode(self) -> str:
        """Return the currently configured estimator mode."""
        return self._mode

    def compile_for_inference(self, **kwargs) -> None:
        """Compile the per-step core and readout networks with torch.compile.

        Mirrors CuadradoUNet.compile_for_inference: applies torch.compile to
        the pure-tensor inner networks so every call inside the streaming step
        loop hits a compiled graph.

        ``sofe_core.core`` (SOFESSNNNetCore) receives (step_input, state_dict,
        decay) — all tensors — and has a fixed graph for a given architecture
        version, so it is a clean compile target. The readout head is similarly
        purely tensor-driven.

        Call this after loading the checkpoint and calling mcm.eval(), before
        the evaluation loop. Training must not use this path.
        """
        compile_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        core_net = getattr(self.sofe_core, "core", None)
        if isinstance(core_net, torch.nn.Module):
            self.sofe_core.core = torch.compile(core_net, **compile_kwargs)
        readout_head = getattr(
            getattr(self.sofe_readout, "readout_head", None), "readout_head", None
        ) or getattr(self.sofe_readout, "readout_head", None)
        if isinstance(readout_head, torch.nn.Module):
            self.sofe_readout.readout_head = torch.compile(readout_head, **compile_kwargs)

    def get_architecture_spec(self) -> Dict[str, Any]:
        """Return the normalized concrete SOFE architecture spec."""
        return copy.deepcopy(self.architecture_spec)

    def enumerate_architecture_candidates(self) -> List[Dict[str, Any]]:
        """Expand the configured SOFE search space into concrete specs."""
        return expand_sofe_search_space(
          base_spec=self.architecture_spec,
          search_space=self.architecture_search_space,
          encoder_channels=self._legacy_encoder_channels,
          num_res_blocks=self._legacy_num_res_blocks,
        )

    def _step_input_channels(self) -> int:
        encoding = str(self.architecture_spec.get("step_encoding", "hist2")).lower()
        if encoding == "hist2":
          return 2
        if encoding in {"hist4", "hist4_legacy"}:
          return 4
        if encoding == "mini_voxel":
          return int(self.architecture_spec.get("mini_voxel_bins", self.num_bins))
        raise NotImplementedError(
          f"SOFE step_encoding={encoding!r} is not implemented; supported encodings "
          "are 'hist2', 'hist4', 'hist4_legacy', and 'mini_voxel'."
        )

    def reset_state(self) -> None:
        """Clear all streaming state.

        Future implementation requirements:
        - Remove recurrent hidden state / membrane state.
        - Clear warm-start flow and readout caches.
        - Reset step counters / last-timestamp tracking so the next call starts
          a fresh sequence without stale temporal context.
        """
        self._state = None
        if hasattr(self.sofe_core, "reset_state"):
          self.sofe_core.reset_state()
        if hasattr(self.sofe_readout, "reset_state"):
          self.sofe_readout.reset_state()
        self._last_flow = None
        self._last_flow_history = None
        self._last_flow_logits = None
        self._last_readout_debug = None
        self._last_step_readout_aux = None

    def detach_state(self) -> None:
        """Detach carried streaming state from the current autograd graph.

        Future implementation requirements:
        - Preserve state values but break gradient history at a truncated-BPTT
          boundary.
        - Handle nested state structures safely (tensors, tuples, dicts).
        - Keep this operation cheap enough to call during the training loop.
        """
        def _detach(obj: Any) -> Any:
          if torch.is_tensor(obj):
            return obj.detach()
          if isinstance(obj, tuple):
            return tuple(_detach(v) for v in obj)
          if isinstance(obj, list):
            return [_detach(v) for v in obj]
          if isinstance(obj, dict):
            return {k: _detach(v) for k, v in obj.items()}
          return obj

        if self._state is not None:
          self._state = SOFEState(
            hidden=_detach(self._state.hidden),
            readout_state=_detach(self._state.readout_state),
            flow_warm=_detach(self._state.flow_warm),
            last_step_ref_us=_detach(self._state.last_step_ref_us),
          )

        if self._last_flow is not None:
          self._last_flow = _detach(self._last_flow)
        if self._last_flow_history is not None:
          self._last_flow_history = _detach(self._last_flow_history)
        if self._last_flow_logits is not None:
          self._last_flow_logits = _detach(self._last_flow_logits)
        if self._last_step_readout_aux is not None:
          self._last_step_readout_aux = _detach(self._last_step_readout_aux)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """Compatibility forward for the existing ``FlowEstimator`` contract.

        Planned role:
        - Preserve compatibility with code paths that still expect
          ``forward(voxel) -> flow``.
        - Either reject unsupported voxel-only usage explicitly or route through
          a future voxel-step baseline implementation.
        - The event-native streaming path should use ``forward_streaming``.
        """
        raise NotImplementedError(
            "SOFE forward() is only a compatibility placeholder. "
            "Streaming event-native execution should use forward_streaming()."
        )

    def forward_streaming(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
        state: Optional[SOFEState] = None,
        detach_state: bool = False,
        sequence_reset: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, SOFEState] | Tuple[torch.Tensor, SOFEState, Dict[str, Any]]:
        """Run streaming flow estimation over one detector-aligned sample batch.

        Future implementation requirements:
        - Partition packed ``t_us`` events into temporal steps according to the
          configured partition mode.
        - Compute elapsed-time-aware decay between steps using real
          microsecond timestamps.
        - Update recurrent state step-by-step.
        - Trigger readout at configured intervals and return the dense flow
          aligned with the requested reference time.
        - Preserve sparse-event timing information needed later by SMCM and the
          loss/warper path.
        """
        if detach_state:
          self.detach_state()

        batch_size = int(events_offsets.numel() - 1)
        work_state = state if state is not None else self._state
        if work_state is not None:
          last_ref = work_state.last_step_ref_us
          if isinstance(last_ref, torch.Tensor) and int(last_ref.numel()) != batch_size:
            work_state = None
        if work_state is None:
          work_state = SOFEState(last_step_ref_us=t_start_us.to(torch.int64).clone())

        if sequence_reset is not None:
          reset_mask = torch.as_tensor(sequence_reset, dtype=torch.bool, device=t_start_us.device).reshape(-1)
          if int(reset_mask.numel()) != batch_size:
            raise ValueError(
              "sequence_reset must match batch size: "
              f"got {int(reset_mask.numel())}, expected {batch_size}"
            )
          if bool(reset_mask.any()):
            work_state = self._apply_sequence_reset(work_state, reset_mask, t_start_us=t_start_us)

        steps = self.partition_steps(
          events=events,
          events_offsets=events_offsets,
          t_start_us=t_start_us,
          t_end_us=t_end_us,
          t_ref_us=t_ref_us,
        )

        flow_history: List[torch.Tensor] = []
        last_flow: Optional[torch.Tensor] = None
        # Diagnostic histories are only collected when debug output is requested
        # to avoid per-step Python overhead (tensor detach + float conversions +
        # torch.quantile calls) during frozen eval.
        decay_history: List[torch.Tensor] = []
        delta_t_history: List[torch.Tensor] = []
        step_event_count_history: List[torch.Tensor] = []
        step_duration_history: List[torch.Tensor] = []
        head_spike_active_rates: List[float] = []
        head_v_minus_thresh_active_max_values: List[float] = []
        head_v_minus_thresh_active_p99_values: List[float] = []
        head_spike_readout_rates: List[float] = []

        for step_index, step in enumerate(steps):
          step_input = self.build_step_input(step)
          step.step_input = step_input

          decay = self.compute_decay(
            step_ref_us=step.step_ref_us if step.step_ref_us is not None else t_ref_us,
            last_step_ref_us=work_state.last_step_ref_us,
            t_start_us=t_start_us,
          )

          if return_debug:
            step_ref_for_stats = step.step_ref_us if step.step_ref_us is not None else t_ref_us
            delta_t_history.append(
              self.compute_step_delta_t_us(
                step_ref_us=step_ref_for_stats,
                last_step_ref_us=work_state.last_step_ref_us,
                t_start_us=t_start_us,
              ).detach()
            )
            decay_history.append(decay.detach())
            if step.events_offsets is not None:
              offsets_i64 = step.events_offsets.to(torch.int64)
              step_event_count_history.append((offsets_i64[1:] - offsets_i64[:-1]).to(torch.float32).detach())
            if step.step_start_us is not None and step.step_end_us is not None:
              step_duration_history.append(
                torch.clamp(
                  step.step_end_us.to(torch.int64) - step.step_start_us.to(torch.int64),
                  min=0,
                ).to(torch.float32).detach()
              )

          work_state = self.update_step(
            step_input=step_input,
            step=step,
            state=work_state,
            decay=decay,
          )

          if return_debug:
            step_head_metrics = self.collect_head_state_metrics(
              getattr(work_state, "hidden", None),
              getattr(step, "active_mask", None),
            )
            if step_head_metrics:
              if "head_spike_rate" in step_head_metrics:
                head_spike_active_rates.append(float(step_head_metrics["head_spike_rate"]))
              if "head_v_minus_thresh_max" in step_head_metrics:
                head_v_minus_thresh_active_max_values.append(float(step_head_metrics["head_v_minus_thresh_max"]))
              if "head_v_minus_thresh_p99" in step_head_metrics:
                head_v_minus_thresh_active_p99_values.append(float(step_head_metrics["head_v_minus_thresh_p99"]))

          should_emit = self.should_readout(
            step_index=step_index,
            step=step,
            t_ref_us=t_ref_us,
            steps_processed=step_index + 1,
          )

          if should_emit:
            if return_debug:
              readout_head_metrics = self.collect_head_state_metrics(
                getattr(work_state, "hidden", None),
                getattr(step, "active_mask", None),
              )
              if "head_spike_rate" not in readout_head_metrics:
                readout_head_metrics = self.collect_head_state_metrics(
                  getattr(work_state, "hidden", None),
                  None,
                )
              if "head_spike_rate" in readout_head_metrics:
                head_spike_readout_rates.append(float(readout_head_metrics["head_spike_rate"]))
            last_flow = self.readout_flow(
              state=work_state,
              step=step,
              t_ref_us=t_ref_us,
            )
            flow_history.append(last_flow)

        if last_flow is None:
          last_step = steps[-1]
          if return_debug:
            readout_head_metrics = self.collect_head_state_metrics(
              getattr(work_state, "hidden", None),
              getattr(last_step, "active_mask", None),
            )
            if "head_spike_rate" not in readout_head_metrics:
              readout_head_metrics = self.collect_head_state_metrics(
                getattr(work_state, "hidden", None),
                None,
              )
            if "head_spike_rate" in readout_head_metrics:
              head_spike_readout_rates.append(float(readout_head_metrics["head_spike_rate"]))
          last_flow = self.readout_flow(
            state=work_state,
            step=last_step,
            t_ref_us=t_ref_us,
          )
          flow_history.append(last_flow)

        self._state = work_state
        self._last_flow = last_flow.detach()
        self._last_flow_history = [f.detach() for f in flow_history]

        if return_debug:
          debug: Dict[str, Any] = {
            "partition_mode": str(self.partition_mode),
            "sofe_family": str(self.architecture_spec.get("family", "fire")),
            "step_encoding": str(self.architecture_spec.get("step_encoding", "hist2")),
            "count_encoding": str(self.architecture_spec.get("count_encoding", "normalized")),
            "count_scale": float(self.architecture_spec.get("count_scale", 1.0)),
            "recurrent_backend": str(self.architecture_spec.get("recurrent_backend", "native")),
            "recurrent_cell": str(self.architecture_spec.get("recurrent_cell", "convgru")),
            "recurrent_placement": str(self.architecture_spec.get("recurrent_placement", "every_stage")),
            "spiking_neuron": str(self.architecture_spec.get("spiking_neuron", "lif")),
            "surrogate_function": str(self.architecture_spec.get("surrogate_function", "sigmoid")),
            "spiking_backend": str(self.architecture_spec.get("spiking_backend", "torch")),
            "readout_type": str(self.architecture_spec.get("readout_type", "single_scale")),
            "readout_update_rule": str(self.architecture_spec.get("readout_update_rule", "residual")),
            "readout_alpha_init": float(self.architecture_spec.get("readout_alpha_init", 1.0)),
            "readout_learn_alpha": bool(self.architecture_spec.get("readout_learn_alpha", False)),
            "readout_state_channels": int(self.architecture_spec.get("readout_state_channels") or 0),
            "stflow_alignment_enabled": bool(self.architecture_spec.get("stflow_alignment_enabled", False)),
            "stflow_feedback_enabled": bool(self.architecture_spec.get("stflow_feedback_enabled", False)),
            "flow_activation": str(self.architecture_spec.get("flow_activation", "tanh")),
            "flow_head_init": str(self.architecture_spec.get("flow_head_init", "zero")),
            "num_steps": int(len(steps)),
            "num_readouts": int(len(flow_history)),
            "step_time_us": int(self.step_time_us or 0),
            "readout_every_n_steps": int(self.readout_every_n_steps or 0),
            "readout_interval_us": int(self.readout_interval_us or 0),
            "empty_step_policy": "explicit_no_event_step",
          }
          final_head_metrics = self.collect_head_state_metrics(getattr(work_state, "hidden", None), None)
          if "head_spike_rate" in final_head_metrics:
            debug["head_spike_rate_final"] = float(final_head_metrics["head_spike_rate"])
          if head_spike_active_rates:
            debug["head_spike_rate_active_mean"] = sum(head_spike_active_rates) / float(len(head_spike_active_rates))
            debug["head_spike_rate_active_max"] = max(head_spike_active_rates)
          if head_spike_readout_rates:
            debug["head_spike_rate_readout_mean"] = sum(head_spike_readout_rates) / float(len(head_spike_readout_rates))
            debug["head_spike_rate_readout_last"] = head_spike_readout_rates[-1]
          if head_v_minus_thresh_active_max_values:
            debug["head_v_minus_thresh_active_max"] = max(head_v_minus_thresh_active_max_values)
          if head_v_minus_thresh_active_p99_values:
            debug["head_v_minus_thresh_active_p99"] = max(head_v_minus_thresh_active_p99_values)
          debug.update(
            self._summarize_streaming_stats(
              decay_history=decay_history,
              delta_t_history=delta_t_history,
              step_event_count_history=step_event_count_history,
              step_duration_history=step_duration_history,
            )
          )
        else:
          debug = {}
        self._last_readout_debug = debug

        if return_debug:
          return last_flow, work_state, debug
        return last_flow, work_state

    def partition_steps(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
    ) -> List[StreamingStepBatch]:
        """Partition one packed event batch into step updates.

        Future implementation requirements:
        - Support fixed-time and hybrid partitioning strategies.
        - Produce deterministic partitions for training, evaluation, and DDP.
        - Retain exact per-step timing metadata and sparse-event boundaries.
        - Respect any configured max/min duration and max-steps safeguards.
        """
        mode = str(self.partition_mode or "fixed_time").lower()
        if events.ndim != 2 or events.shape[1] != 4:
          raise ValueError(f"Expected packed events with shape [N,4], got {tuple(events.shape)}")
        if events_offsets.ndim != 1:
          raise ValueError("events_offsets must be a 1D prefix-sum tensor")

        batch_size = int(events_offsets.numel() - 1)
        if batch_size <= 0:
          return []

        t_start_us = t_start_us.to(torch.int64).reshape(-1)
        t_end_us = t_end_us.to(torch.int64).reshape(-1)
        t_ref_us = t_ref_us.to(torch.int64).reshape(-1)
        if int(t_start_us.numel()) != batch_size:
          raise ValueError("t_start_us must match events_offsets batch size")
        if int(t_end_us.numel()) != batch_size:
          raise ValueError("t_end_us must match events_offsets batch size")
        if int(t_ref_us.numel()) != batch_size:
          raise ValueError("t_ref_us must match events_offsets batch size")

        if mode == "fixed_time":
          return self._partition_steps_fixed_time(
            events=events,
            events_offsets=events_offsets,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            t_ref_us=t_ref_us,
          )
        if mode == "hybrid":
          return self._partition_steps_hybrid(
            events=events,
            events_offsets=events_offsets,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            t_ref_us=t_ref_us,
          )

        raise NotImplementedError(
          f"partition_mode={mode!r} is not implemented; supported modes are 'fixed_time' and 'hybrid'."
        )

    def _partition_steps_fixed_time(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
    ) -> List[StreamingStepBatch]:
        dt_us = int(self.step_time_us or 0)
        if dt_us < 0:
          raise ValueError("step_time_us must be >= 0")

        batch_size = int(events_offsets.numel() - 1)

        n_steps_per_sample: List[int] = []
        for b in range(batch_size):
          start_b = int(t_start_us[b].item())
          end_b = int(t_end_us[b].item())
          if end_b < start_b:
            end_b = start_b
          if dt_us <= 0:
            n_steps = 1
          else:
            dur = max(end_b - start_b, 1)
            n_steps = (dur + dt_us - 1) // dt_us
          if self.max_steps_per_sample is not None:
            n_steps = min(n_steps, int(self.max_steps_per_sample))
          n_steps_per_sample.append(max(1, int(n_steps)))

        global_steps = max(n_steps_per_sample)
        steps: List[StreamingStepBatch] = []

        for k in range(global_steps):
          per_sample_events: List[torch.Tensor] = []
          step_offsets = [0]
          window_start_list: List[int] = []
          step_start_list: List[int] = []
          step_end_list: List[int] = []
          step_ref_list: List[int] = []
          active_list: List[bool] = []

          for b in range(batch_size):
            start_b = int(t_start_us[b].item())
            end_b = int(t_end_us[b].item())
            if end_b < start_b:
              end_b = start_b

            if dt_us <= 0:
              start_k = start_b
              end_k = end_b
            else:
              start_k = start_b + k * dt_us
              end_k = min(start_b + (k + 1) * dt_us, end_b)

            start_i = int(events_offsets[b].item())
            end_i = int(events_offsets[b + 1].item())
            ev_i = events[start_i:end_i]

            if k >= int(n_steps_per_sample[b]) or ev_i.numel() == 0 or end_k <= start_k:
              chunk = events.new_zeros((0, 4))
            else:
              t_i = ev_i[:, 2].to(torch.int64)
              if k == int(n_steps_per_sample[b]) - 1:
                mask = (t_i >= start_k) & (t_i <= end_k)
              else:
                mask = (t_i >= start_k) & (t_i < end_k)
              chunk = ev_i[mask]

            per_sample_events.append(chunk)
            step_offsets.append(step_offsets[-1] + int(chunk.shape[0]))
            window_start_list.append(int(start_b))
            step_start_list.append(int(start_k))
            step_end_list.append(int(end_k))
            if int(chunk.shape[0]) > 0:
              step_ref_list.append(int(chunk[-1, 2].item()))
              active_list.append(True)
            else:
              # No events in this interval: use interval end as causal ref.
              step_ref_list.append(int(end_k))
              active_list.append(False)

          if step_offsets[-1] > 0:
            step_events = torch.cat(per_sample_events, dim=0)
          else:
            step_events = events.new_zeros((0, 4))

          steps.append(
            StreamingStepBatch(
              events=step_events,
              events_offsets=torch.as_tensor(step_offsets, dtype=torch.int64, device=events_offsets.device),
              window_start_us=torch.as_tensor(window_start_list, dtype=torch.int64, device=t_start_us.device),
              step_start_us=torch.as_tensor(step_start_list, dtype=torch.int64, device=t_start_us.device),
              step_end_us=torch.as_tensor(step_end_list, dtype=torch.int64, device=t_start_us.device),
              step_ref_us=torch.as_tensor(step_ref_list, dtype=torch.int64, device=t_start_us.device),
              active_mask=torch.as_tensor(active_list, dtype=torch.bool, device=t_start_us.device),
            )
          )

        return steps

    def _partition_steps_hybrid(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
    ) -> List[StreamingStepBatch]:
        target = self.events_per_step_target
        if target is None or int(target) <= 0:
          raise ValueError(
            "partition_mode='hybrid' requires events_per_step_target to be a positive integer."
          )

        target = int(target)
        max_dt = self.step_dt_max_us
        if max_dt is not None:
          max_dt = int(max_dt)
          if max_dt <= 0:
            raise ValueError("step_dt_max_us must be > 0 when partition_mode='hybrid'")

        min_dt = self.step_dt_min_us
        if min_dt is not None:
          min_dt = int(min_dt)
          if min_dt < 0:
            raise ValueError("step_dt_min_us must be >= 0 when partition_mode='hybrid'")
          if max_dt is not None and min_dt > max_dt:
            raise ValueError("step_dt_min_us cannot exceed step_dt_max_us")
        else:
          min_dt = 0

        batch_size = int(events_offsets.numel() - 1)
        sample_steps: List[List[_SampleStepSpec]] = []
        sample_base_offsets: List[int] = []
        packed_event_times_us = events[:, 2].to(torch.int64)

        for b in range(batch_size):
          start_i = int(events_offsets[b].item())
          end_i = int(events_offsets[b + 1].item())
          start_b = int(t_start_us[b].item())
          end_b = int(t_end_us[b].item())
          sample_base_offsets.append(start_i)
          sample_steps.append(
            self._partition_sample_hybrid(
              event_times_us=packed_event_times_us[start_i:end_i],
              t_start_us=start_b,
              t_end_us=end_b,
              target_events=target,
              max_dt_us=max_dt,
              min_dt_us=min_dt,
            )
          )

        return self._pack_sample_steps_from_specs(
          sample_steps=sample_steps,
          sample_base_offsets=sample_base_offsets,
          events=events,
          t_start_us=t_start_us,
          device=events_offsets.device,
        )

    def _partition_sample_hybrid(
        self,
        *,
        event_times_us: torch.Tensor,
        t_start_us: int,
        t_end_us: int,
        target_events: int,
        max_dt_us: Optional[int],
        min_dt_us: int,
    ) -> List[_SampleStepSpec]:
        if t_end_us < t_start_us:
          t_end_us = t_start_us

        times = event_times_us.reshape(-1).contiguous()
        num_events = int(times.numel())
        if num_events == 0:
          return [
            _SampleStepSpec(
              event_start=0,
              event_end=0,
              step_start_us=int(t_start_us),
              step_end_us=int(t_end_us),
              step_ref_us=int(t_end_us),
              active=False,
            )
          ]

        steps: List[_SampleStepSpec] = []

        chunk_start_idx = 0
        chunk_start_us = int(t_start_us)

        while chunk_start_idx < num_events:
          end_idx = min(chunk_start_idx + target_events - 1, num_events - 1)

          if min_dt_us > 0:
            min_threshold = int(chunk_start_us + min_dt_us)
            min_end_idx = int(
              torch.searchsorted(
                times,
                times.new_tensor(min_threshold, dtype=torch.int64),
                right=False,
              ).item()
            )
            min_end_idx = min(max(min_end_idx, chunk_start_idx), num_events - 1)
            if min_end_idx > end_idx:
              end_idx = min_end_idx

          if max_dt_us is not None:
            max_threshold = int(chunk_start_us + max_dt_us)
            max_end_idx = int(
              torch.searchsorted(
                times,
                times.new_tensor(max_threshold, dtype=torch.int64),
                right=True,
              ).item()
            ) - 1
            max_end_idx = min(max(max_end_idx, chunk_start_idx), num_events - 1)
            if max_end_idx < end_idx:
              end_idx = max_end_idx

          step_ref_us = int(times[end_idx].item())
          steps.append(
            _SampleStepSpec(
              event_start=int(chunk_start_idx),
              event_end=int(end_idx + 1),
              step_start_us=int(chunk_start_us),
              step_end_us=int(step_ref_us),
              step_ref_us=int(step_ref_us),
              active=True,
            )
          )
          chunk_start_idx = end_idx + 1
          if chunk_start_idx < num_events:
            chunk_start_us = int(times[chunk_start_idx].item())

        last_ref = int(steps[-1].step_ref_us)
        if last_ref < t_end_us:
          steps.append(
            _SampleStepSpec(
              event_start=num_events,
              event_end=num_events,
              step_start_us=int(last_ref),
              step_end_us=int(t_end_us),
              step_ref_us=int(t_end_us),
              active=False,
            )
          )

        if self.max_steps_per_sample is not None and len(steps) > int(self.max_steps_per_sample):
          keep = max(1, int(self.max_steps_per_sample))
          prefix = steps[: keep - 1]
          suffix = steps[keep - 1 :]
          active_suffix = [step for step in suffix if bool(step.active) and step.event_end > step.event_start]
          if active_suffix:
            merged_start = int(active_suffix[0].event_start)
            merged_end = int(active_suffix[-1].event_end)
            active = True
          else:
            merged_start = int(suffix[0].event_start)
            merged_end = int(merged_start)
            active = False
          prefix.append(
            _SampleStepSpec(
              event_start=merged_start,
              event_end=merged_end,
              step_start_us=int(suffix[0].step_start_us),
              step_end_us=int(suffix[-1].step_end_us),
              step_ref_us=int(suffix[-1].step_ref_us),
              active=bool(active),
            )
          )
          steps = prefix

        return steps

    def _pack_sample_steps_from_specs(
        self,
        sample_steps: List[List[_SampleStepSpec]],
        sample_base_offsets: List[int],
        *,
        events: torch.Tensor,
        t_start_us: torch.Tensor,
        device: torch.device,
    ) -> List[StreamingStepBatch]:
        if not sample_steps:
          return []

        global_steps = max(len(steps) for steps in sample_steps)
        packed_steps: List[StreamingStepBatch] = []
        batch_size = len(sample_steps)
        last_refs = [None] * batch_size
        empty_chunk = events.new_zeros((0, 4))

        for k in range(global_steps):
          per_sample_events: List[torch.Tensor] = []
          step_offsets = [0]
          window_start_list: List[int] = []
          step_start_list: List[int] = []
          step_end_list: List[int] = []
          step_ref_list: List[int] = []
          active_list: List[bool] = []

          for b, steps_b in enumerate(sample_steps):
            if k < len(steps_b):
              step_b = steps_b[k]
              base_b = int(sample_base_offsets[b])
              if step_b.event_end > step_b.event_start:
                chunk = events[base_b + step_b.event_start : base_b + step_b.event_end]
              else:
                chunk = empty_chunk
              start_k = int(step_b.step_start_us)
              end_k = int(step_b.step_end_us)
              ref_k = int(step_b.step_ref_us)
              active_k = bool(step_b.active)
              last_refs[b] = ref_k
            else:
              chunk = empty_chunk
              filler_ref = 0 if last_refs[b] is None else int(last_refs[b])
              start_k = filler_ref
              end_k = filler_ref
              ref_k = filler_ref
              active_k = False

            per_sample_events.append(chunk)
            step_offsets.append(step_offsets[-1] + int(chunk.shape[0]))
            window_start_list.append(int(t_start_us[b].item()))
            step_start_list.append(start_k)
            step_end_list.append(end_k)
            step_ref_list.append(ref_k)
            active_list.append(active_k)

          step_events = torch.cat(per_sample_events, dim=0) if step_offsets[-1] > 0 else empty_chunk
          packed_steps.append(
            StreamingStepBatch(
              events=step_events,
              events_offsets=torch.as_tensor(step_offsets, dtype=torch.int64, device=device),
              window_start_us=torch.as_tensor(window_start_list, dtype=torch.int64, device=device),
              step_start_us=torch.as_tensor(step_start_list, dtype=torch.int64, device=device),
              step_end_us=torch.as_tensor(step_end_list, dtype=torch.int64, device=device),
              step_ref_us=torch.as_tensor(step_ref_list, dtype=torch.int64, device=device),
              active_mask=torch.as_tensor(active_list, dtype=torch.bool, device=device),
            )
          )

        return packed_steps

    def build_step_input(self, step: StreamingStepBatch) -> torch.Tensor:
        """Convert one streaming step into the estimator's input representation.

        Future implementation requirements:
        - Support lightweight per-step encodings such as polarity histograms or
          former/latter splits before heavier mini-volumes are explored.
        - Preserve polarity and any intra-step temporal cues required by the
          chosen architecture.
        - Return a batch-aligned tensor suitable for one recurrent update.
        """
        return self.step_input_encoder(step)

    def compute_decay(
        self,
        *,
        step_ref_us: torch.Tensor,
        last_step_ref_us: Optional[torch.Tensor],
        t_start_us: torch.Tensor,
    ) -> torch.Tensor:
        """Compute continuous-time decay from elapsed microseconds.

        Future implementation requirements:
        - Derive ``delta_t_us`` from the current step reference time and the
          previous processed step.
        - Map elapsed time to a decay factor (for example using an exponential
          parameterized by ``tau_us``).
        - Support broadcasting over batch items and later over channels/layers
          if per-layer learned decay is introduced.
        """
        delta_t = self.compute_step_delta_t_us(
          step_ref_us=step_ref_us,
          last_step_ref_us=last_step_ref_us,
          t_start_us=t_start_us,
        )
        tau = float(self.decay_tau_us) if self.decay_tau_us is not None else 0.0
        if tau <= 0.0:
          return torch.ones_like(delta_t, dtype=torch.float32)
        return torch.exp(-delta_t / tau)

    def compute_step_delta_t_us(
        self,
        *,
        step_ref_us: torch.Tensor,
        last_step_ref_us: Optional[torch.Tensor],
        t_start_us: torch.Tensor,
    ) -> torch.Tensor:
        """Return elapsed microseconds used by continuous-time state decay."""
        step_ref_us = step_ref_us.to(torch.int64)
        t_start_us = t_start_us.to(torch.int64)

        if last_step_ref_us is None:
          delta_t = step_ref_us - t_start_us
        else:
          prev = last_step_ref_us.to(torch.int64)
          if int(prev.numel()) != int(step_ref_us.numel()):
            prev = t_start_us
          delta_t = step_ref_us - prev

        return torch.clamp(delta_t, min=0).to(torch.float32)

    @staticmethod
    def _summarize_tensor_history(prefix: str, values: List[torch.Tensor]) -> Dict[str, float]:
        if not values:
          return {}
        flat = torch.cat([v.reshape(-1).to(torch.float32).detach().cpu() for v in values], dim=0)
        if int(flat.numel()) == 0:
          return {}
        return {
          f"{prefix}_mean": float(flat.mean().item()),
          f"{prefix}_min": float(flat.min().item()),
          f"{prefix}_max": float(flat.max().item()),
        }

    def _summarize_streaming_stats(
        self,
        *,
        decay_history: List[torch.Tensor],
        delta_t_history: List[torch.Tensor],
        step_event_count_history: List[torch.Tensor],
        step_duration_history: List[torch.Tensor],
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        stats.update(self._summarize_tensor_history("decay", decay_history))
        stats.update(self._summarize_tensor_history("delta_t_us", delta_t_history))
        stats.update(self._summarize_tensor_history("step_event_count", step_event_count_history))
        stats.update(self._summarize_tensor_history("step_duration_us", step_duration_history))
        return stats

    def update_step(
        self,
        *,
        step_input: torch.Tensor,
        step: StreamingStepBatch,
        state: Optional[SOFEState],
        decay: torch.Tensor,
    ) -> SOFEState:
        """Apply one streaming update to the estimator state.

        Future implementation requirements:
        - Decay the previous hidden state using elapsed time.
        - Encode the current step input and feed it through the recurrent core.
        - Update any readout-side latent state or warm-start flow.
        - Return the new persistent state without forcing an immediate readout.
        """
        if state is None:
          state = SOFEState()

        prev_hidden = state.hidden
        next_hidden, core_features = self.sofe_core(
          step_input,
          prev_hidden,
          decay,
          step=step,
        )

        next_readout_state, next_flow_warm, readout_aux = self.sofe_readout.update(
          core_features,
          readout_state=state.readout_state,
          flow_warm=state.flow_warm,
          t_ref_us=step.step_ref_us,
        )
        self._last_step_readout_aux = readout_aux

        return SOFEState(
            hidden=next_hidden,
            readout_state=next_readout_state,
            flow_warm=next_flow_warm,
            last_step_ref_us=step.step_ref_us.to(torch.int64) if step.step_ref_us is not None else state.last_step_ref_us,
        )

    def should_readout(
        self,
        *,
        step_index: int,
        step: StreamingStepBatch,
        t_ref_us: torch.Tensor,
        steps_processed: int,
    ) -> bool:
        """Decide whether the current step should trigger a dense-flow readout.

        Future implementation requirements:
        - Support frame-end-only readout as the simplest baseline.
        - Support step-count-based and time-interval-based readout schedules.
        - Remain deterministic and easy to checkpoint/resume.
        """
        every_n = int(self.readout_every_n_steps or 0)
        if every_n > 0 and (steps_processed % every_n == 0):
            return True

        interval_us = int(self.readout_interval_us or 0)
        if (
            interval_us > 0
            and step.window_start_us is not None
            and step.step_start_us is not None
            and step.step_end_us is not None
        ):
            window_start = step.window_start_us.to(torch.int64)
            step_start = torch.maximum(step.step_start_us.to(torch.int64), window_start)
            step_end = torch.maximum(step.step_end_us.to(torch.int64), window_start)
            start_bucket = torch.div(step_start - window_start, interval_us, rounding_mode="floor")
            end_bucket = torch.div(step_end - window_start, interval_us, rounding_mode="floor")
            if bool(torch.any(end_bucket > start_bucket).item()):
                return True

        if step.step_end_us is None:
            return True

        step_end = step.step_end_us.to(torch.int64)
        return bool(torch.all(step_end >= t_ref_us.to(torch.int64)).item())

    def readout_flow(
        self,
        *,
        state: SOFEState,
        step: StreamingStepBatch,
        t_ref_us: torch.Tensor,
    ) -> torch.Tensor:
        """Decode dense optical flow from the current streaming state.

        Future implementation requirements:
        - Produce a dense flow field shaped ``(B, 2, H, W)`` in pixel units.
        - Align the readout with the requested reference time.
        - Optionally expose lightweight debug traces without storing giant
          histories by default.
        """
        bsz = int(t_ref_us.numel())
        h = int(self.image_height)
        w = int(self.image_width)

        flow_raw, aux = self.sofe_readout.emit(
          readout_state=state.readout_state if state is not None else None,
          flow_warm=state.flow_warm if state is not None else None,
          t_ref_us=t_ref_us,
        )
        combined_aux: Dict[str, Any] = {}
        if isinstance(aux, dict):
          combined_aux.update(aux)
        if isinstance(self._last_step_readout_aux, dict):
          combined_aux.update(self._last_step_readout_aux)
        flow_logits = combined_aux.get("flow_logits")
        self._last_flow_logits = (
          flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        )
        return flow_raw * self.placeholder_gain

    def _apply_sequence_reset(
        self,
        state: SOFEState,
        reset_mask: torch.Tensor,
        *,
        t_start_us: torch.Tensor,
    ) -> SOFEState:
        reset_mask = reset_mask.to(torch.bool)
        bsz = int(reset_mask.numel())

        def _masked_zero(val: Any) -> Any:
            if isinstance(val, torch.Tensor):
                if val.ndim == 0 or int(val.shape[0]) != bsz:
                    return val
                out = val.clone()
                out[reset_mask] = 0
                return out
            if isinstance(val, dict):
                return {k: _masked_zero(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_masked_zero(v) for v in val]
            if isinstance(val, tuple):
                return tuple(_masked_zero(v) for v in val)
            return val

        next_last = state.last_step_ref_us
        if isinstance(next_last, torch.Tensor) and int(next_last.numel()) == bsz:
            next_last = next_last.to(torch.int64).clone()
            next_last[reset_mask] = t_start_us.to(torch.int64)[reset_mask]
        else:
            next_last = t_start_us.to(torch.int64).clone()

        return SOFEState(
            hidden=_masked_zero(state.hidden),
            readout_state=_masked_zero(state.readout_state),
            flow_warm=_masked_zero(state.flow_warm),
            last_step_ref_us=next_last,
        )


# Short alias matching the research-plan terminology.
SOFE = StreamingSpikingFlowEstimator
