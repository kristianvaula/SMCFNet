"""Streaming motion-compensation module (SMCM).

This module is the streaming/event-native companion to the existing generic
MotionCompensationModule. It keeps the same outer contract:

    forward(...) -> (corrected, loss)

and focuses on:
- consuming packed t_us events
- delegating step partitioning/state updates to SOFE
- handling sequence reset / state detach semantics
- warping and loss computation in the streaming path
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from smcfnet.model.event_outputs import EventOutputRenderer
from smcfnet.model.mcm.base import BaseMotionCompensationModule
from smcfnet.model.ofe.sofe import SOFEState, StreamingSpikingFlowEstimator
from smcfnet.utils.profiling import CudaStageTimer


logger = logging.getLogger(__name__)


class StreamingMotionCompensationModule(BaseMotionCompensationModule):
    """Streaming motion-compensation wrapper.

    This class intentionally stays strict: it does not implement legacy voxel
    fallback behavior. Streaming execution must come through an estimator that
    implements forward_streaming().
    """

    def __init__(
        self,
        estimator: StreamingSpikingFlowEstimator,
        warper: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[..., object]] = None,
        flow_scaling: float = 1.0,
        reset_on_sequence_change: bool = True,
        output_renderer: Optional[nn.Module] = None,
        auxiliary_flow_loss: bool = False,
        auxiliary_flow_loss_weight: float = 0.25,
        auxiliary_flow_scales: Optional[List[str]] = None,
        auxiliary_warp_loss: bool = False,
        auxiliary_warp_loss_weight: float = 0.10,
    ):
        if not hasattr(estimator, "forward_streaming"):
            raise TypeError(
                "StreamingMotionCompensationModule requires an estimator "
                "implementing forward_streaming()."
            )

        super().__init__(
            estimator=estimator,
            warper=output_renderer or warper,
            loss_fn=loss_fn,
            flow_scaling=flow_scaling,
        )
        self.reset_on_sequence_change = bool(reset_on_sequence_change)
        self.auxiliary_flow_loss = bool(auxiliary_flow_loss)
        self.auxiliary_flow_loss_weight = float(auxiliary_flow_loss_weight)
        self.auxiliary_flow_scales: List[str] = list(
            auxiliary_flow_scales if auxiliary_flow_scales is not None
            else ["h16", "h8", "h4", "h2"]
        )
        self.auxiliary_warp_loss = bool(auxiliary_warp_loss)
        self.auxiliary_warp_loss_weight = float(auxiliary_warp_loss_weight)

        self._state: Optional[SOFEState] = None
        self._last_stream_debug: Optional[Dict[str, Any]] = None
        self._last_flow_logits: Optional[torch.Tensor] = None
        self._last_seq_idx: Optional[torch.Tensor] = None
        self._last_frame_index: Optional[torch.Tensor] = None

    @staticmethod
    def _detach_obj(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach()
        if isinstance(obj, tuple):
            return tuple(StreamingMotionCompensationModule._detach_obj(v) for v in obj)
        if isinstance(obj, list):
            return [StreamingMotionCompensationModule._detach_obj(v) for v in obj]
        if isinstance(obj, dict):
            return {k: StreamingMotionCompensationModule._detach_obj(v) for k, v in obj.items()}
        return obj

    @classmethod
    def _detach_sofe_state_value(cls, state: Optional[SOFEState]) -> Optional[SOFEState]:
        if state is None:
            return None
        return SOFEState(
            hidden=cls._detach_obj(state.hidden),
            readout_state=cls._detach_obj(state.readout_state),
            flow_warm=cls._detach_obj(state.flow_warm),
            last_step_ref_us=cls._detach_obj(state.last_step_ref_us),
        )

    def _prepare_estimator_state(
        self,
        *,
        t_start_us: torch.Tensor,
        events_offsets: torch.Tensor,
        sequence_reset: Optional[torch.Tensor],
    ) -> Optional[SOFEState]:
        batch_size = int(events_offsets.numel() - 1)
        work_state = self._state
        if work_state is not None:
            last_ref = work_state.last_step_ref_us
            if isinstance(last_ref, torch.Tensor) and int(last_ref.numel()) != batch_size:
                work_state = None
        if work_state is None:
            work_state = SOFEState(last_step_ref_us=t_start_us.to(torch.int64).clone())

        if sequence_reset is not None:
            reset_mask = torch.as_tensor(
                sequence_reset, dtype=torch.bool, device=t_start_us.device
            ).reshape(-1)
            if int(reset_mask.numel()) != batch_size:
                raise ValueError(
                    "sequence_reset must match batch size: "
                    f"got {int(reset_mask.numel())}, expected {batch_size}"
                )
            if bool(reset_mask.any()):
                work_state = self.estimator._apply_sequence_reset(
                    work_state, reset_mask, t_start_us=t_start_us
                )
        return work_state

    @staticmethod
    def _merge_step_chunk(steps: List[Any]) -> Dict[str, torch.Tensor]:
        if not steps:
            raise ValueError("TBPTT chunk must contain at least one step")

        ref_step = steps[0]
        if ref_step.events is None or ref_step.events_offsets is None:
            raise ValueError("StreamingStepBatch must include events and events_offsets")

        batch_size = int(ref_step.events_offsets.numel() - 1)
        event_template = ref_step.events
        chunk_events_per_sample: List[torch.Tensor] = []
        chunk_offsets = [0]

        for batch_index in range(batch_size):
            sample_chunks: List[torch.Tensor] = []
            total_events = 0
            for step in steps:
                assert step.events is not None
                assert step.events_offsets is not None
                start = int(step.events_offsets[batch_index].item())
                end = int(step.events_offsets[batch_index + 1].item())
                if end > start:
                    ev = step.events[start:end]
                    sample_chunks.append(ev)
                    total_events += int(ev.shape[0])
            if sample_chunks:
                merged = torch.cat(sample_chunks, dim=0)
            else:
                merged = event_template.new_zeros((0, 4))
            chunk_events_per_sample.append(merged)
            chunk_offsets.append(chunk_offsets[-1] + total_events)

        if chunk_offsets[-1] > 0:
            chunk_events = torch.cat(chunk_events_per_sample, dim=0)
        else:
            chunk_events = event_template.new_zeros((0, 4))

        active_masks = [step.active_mask for step in steps if step.active_mask is not None]
        if active_masks:
            chunk_active = torch.stack(active_masks, dim=0).any(dim=0)
        else:
            chunk_active = torch.ones(
                (batch_size,), dtype=torch.bool, device=ref_step.step_ref_us.device
            )

        return {
            "events": chunk_events,
            "events_offsets": torch.as_tensor(
                chunk_offsets, dtype=torch.int64, device=ref_step.events_offsets.device
            ),
            "t_start_us": steps[0].step_start_us.to(torch.int64),
            "t_end_us": steps[-1].step_end_us.to(torch.int64),
            "t_ref_us": steps[-1].step_ref_us.to(torch.int64),
            "active_mask": chunk_active,
        }

    @staticmethod
    def _reduce_loss_dicts(loss_dicts: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, torch.Tensor]]:
        values_by_key: Dict[str, List[torch.Tensor]] = {}
        for loss_dict in loss_dicts:
            if not isinstance(loss_dict, dict):
                continue
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    tensor_value = value.detach()
                else:
                    tensor_value = torch.as_tensor(value)
                values_by_key.setdefault(str(key), []).append(tensor_value)

        if not values_by_key:
            return None

        reduced: Dict[str, torch.Tensor] = {}
        for key, values in values_by_key.items():
            if len(values) == 1:
                reduced[key] = values[0]
            else:
                reduced[key] = torch.stack(values, dim=0).mean(dim=0)
        return reduced

    @staticmethod
    def _detach_loss_dict(loss_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(loss_dict, dict):
            return None

        detached: Dict[str, torch.Tensor] = {}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                detached[str(key)] = value.detach()
            elif isinstance(value, (int, float)):
                detached[str(key)] = torch.tensor(float(value), dtype=torch.float32)
        return detached

    @staticmethod
    def _step_event_count(step: Any) -> int:
        offsets = getattr(step, "events_offsets", None)
        if not isinstance(offsets, torch.Tensor) or int(offsets.numel()) <= 1:
            return 0
        return int((offsets[1:] - offsets[:-1]).sum().item())

    @staticmethod
    def _sequence_stats(values: Any) -> Dict[str, float]:
        if not isinstance(values, (list, tuple)) or not values:
            return {}
        vals = [float(v) for v in values]
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / float(len(vals)),
        }

    @staticmethod
    def _prepare_frozen_detector_output(
        corrected: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize frozen SMCM renderer output for detector consumption."""

        if not torch.is_floating_point(corrected):
            corrected = corrected.to(dtype=torch.float32)
        elif corrected.dtype != reference.dtype:
            corrected = corrected.to(dtype=reference.dtype)

        finite_mask = torch.isfinite(corrected)
        if not bool(finite_mask.all()):
            bad_count = int((~finite_mask).sum().detach().cpu().item())
            logger.warning(
                "Frozen SMCM renderer produced %d non-finite value(s); replacing with 0 before detector forward.",
                bad_count,
            )
            corrected = torch.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        return corrected

    def prepare_streaming_tbptt_plan(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
        flow_gt: Optional[torch.Tensor] = None,
        sequence_reset: Optional[torch.Tensor] = None,
        detach_state: bool = False,
        voxel: Optional[torch.Tensor] = None,
        **runtime_kwargs,
    ) -> Dict[str, Any]:
        """Prepare an explicit TBPTT chunk plan for trainer-owned backward.

        The regular ``_forward_streaming_tbptt`` path computes all chunk losses
        and lets the trainer call backward once. This plan API keeps the same
        partition/state semantics, but lets the trainer backward each chunk
        immediately so activation graphs can be released at TBPTT boundaries.
        """

        sequence_reset = self.prepare_sequence_reset(
            seq_idx=runtime_kwargs.get("seq_idx"),
            frame_index=runtime_kwargs.get("frame_index"),
            sequence_reset=sequence_reset,
        )

        if detach_state:
            self.detach_state()

        tbptt_steps = max(1, int(getattr(self.estimator, "bptt_steps", 0) or 1))
        work_state = self._prepare_estimator_state(
            t_start_us=t_start_us,
            events_offsets=events_offsets,
            sequence_reset=sequence_reset,
        )
        steps = self.estimator.partition_steps(
            events=events,
            events_offsets=events_offsets,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            t_ref_us=t_ref_us,
        )
        chunks = [steps[i : i + tbptt_steps] for i in range(0, len(steps), tbptt_steps)]
        chunk_event_counts = [
            sum(self._step_event_count(step) for step in chunk)
            for chunk in chunks
        ]
        total_events = int(sum(chunk_event_counts))

        return {
            "events": events,
            "events_offsets": events_offsets,
            "t_start_us": t_start_us,
            "t_end_us": t_end_us,
            "t_ref_us": t_ref_us,
            "flow_gt": flow_gt,
            "voxel": voxel,
            "runtime_kwargs": dict(runtime_kwargs),
            "tbptt_steps": int(tbptt_steps),
            "steps": steps,
            "chunks": chunks,
            "num_chunks": int(len(chunks)),
            "chunk_event_counts": chunk_event_counts,
            "chunk_loss_weights": [],
            "total_events": int(total_events),
            "work_state": work_state,
            "chunk_loss_dicts": [],
            "weighted_loss_total": None,
            "flow_history_unscaled": [],
            "head_spike_active_rates": [],
            "head_v_minus_thresh_active_max_values": [],
            "head_v_minus_thresh_active_p99_values": [],
            "head_spike_readout_rates": [],
            "last_flow": None,
            "last_flow_unscaled": None,
            "last_corrected": None,
        }

    def forward_streaming_tbptt_chunk(
        self,
        plan: Dict[str, Any],
        chunk_index: int,
    ) -> Dict[str, Any]:
        chunks = plan.get("chunks", [])
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise IndexError(f"TBPTT chunk index {chunk_index} out of range for {len(chunks)} chunks")

        chunk_steps = chunks[chunk_index]
        work_state = plan["work_state"]
        t_start_us = plan["t_start_us"]
        t_ref_us = plan["t_ref_us"]
        runtime_kwargs = plan["runtime_kwargs"]

        for step in chunk_steps:
            step_input = self.estimator.build_step_input(step)
            step.step_input = step_input
            decay = self.estimator.compute_decay(
                step_ref_us=step.step_ref_us if step.step_ref_us is not None else t_ref_us,
                last_step_ref_us=work_state.last_step_ref_us if work_state is not None else None,
                t_start_us=t_start_us,
            )
            work_state = self.estimator.update_step(
                step_input=step_input,
                step=step,
                state=work_state,
                decay=decay,
            )
            collect_head_metrics = getattr(self.estimator, "collect_head_state_metrics", None)
            if callable(collect_head_metrics):
                step_head_metrics = collect_head_metrics(
                    getattr(work_state, "hidden", None),
                    getattr(step, "active_mask", None),
                )
                if isinstance(step_head_metrics, dict) and step_head_metrics:
                    if "head_spike_rate" in step_head_metrics:
                        plan["head_spike_active_rates"].append(float(step_head_metrics["head_spike_rate"]))
                    if "head_v_minus_thresh_max" in step_head_metrics:
                        plan["head_v_minus_thresh_active_max_values"].append(float(step_head_metrics["head_v_minus_thresh_max"]))
                    if "head_v_minus_thresh_p99" in step_head_metrics:
                        plan["head_v_minus_thresh_active_p99_values"].append(float(step_head_metrics["head_v_minus_thresh_p99"]))

        chunk_batch = self._merge_step_chunk(chunk_steps)
        chunk_t_ref = chunk_batch["t_ref_us"]
        flow_unscaled = self.estimator.readout_flow(
            state=work_state,
            step=chunk_steps[-1],
            t_ref_us=chunk_t_ref,
        )
        collect_head_metrics = getattr(self.estimator, "collect_head_state_metrics", None)
        if callable(collect_head_metrics):
            readout_head_metrics = collect_head_metrics(
                getattr(work_state, "hidden", None),
                getattr(chunk_steps[-1], "active_mask", None),
            )
            if isinstance(readout_head_metrics, dict) and "head_spike_rate" not in readout_head_metrics:
                readout_head_metrics = collect_head_metrics(
                    getattr(work_state, "hidden", None),
                    None,
                )
            if isinstance(readout_head_metrics, dict) and "head_spike_rate" in readout_head_metrics:
                plan["head_spike_readout_rates"].append(float(readout_head_metrics["head_spike_rate"]))
        flow = flow_unscaled * self.flow_scaling if self.flow_scaling != 1.0 else flow_unscaled
        corrected = self.warp_events(
            flow=flow,
            voxel=plan["voxel"],
            events=chunk_batch["events"],
            events_offsets=chunk_batch["events_offsets"],
            t_start_us=chunk_batch["t_start_us"],
            t_end_us=chunk_batch["t_end_us"],
            t_ref_us=chunk_batch["t_ref_us"],
            **runtime_kwargs,
        )
        loss, loss_dict = self.compute_loss(
            corrected=corrected,
            flow=flow,
            flow_gt=plan["flow_gt"],
            flow_unscaled=flow_unscaled,
            voxel=plan["voxel"],
            events=chunk_batch["events"],
            events_offsets=chunk_batch["events_offsets"],
            t_start_us=chunk_batch["t_start_us"],
            t_end_us=chunk_batch["t_end_us"],
            t_ref_us=chunk_batch["t_ref_us"],
            **runtime_kwargs,
        )

        event_count = int(plan["chunk_event_counts"][chunk_index])
        total_events = int(plan["total_events"])
        if total_events > 0:
            loss_weight = float(event_count) / float(total_events)
        else:
            loss_weight = 1.0 / float(max(int(plan["num_chunks"]), 1))

        weighted_loss_detached = loss.detach() * float(loss_weight)
        prev_total = plan.get("weighted_loss_total")
        plan["weighted_loss_total"] = (
            weighted_loss_detached
            if prev_total is None
            else prev_total + weighted_loss_detached
        )
        plan["chunk_loss_dicts"].append(self._detach_loss_dict(loss_dict))
        plan["chunk_loss_weights"].append(float(loss_weight))
        plan["flow_history_unscaled"].append(flow_unscaled.detach())
        plan["work_state"] = work_state
        plan["last_flow"] = flow.detach()
        plan["last_flow_unscaled"] = flow_unscaled.detach()
        plan["last_corrected"] = corrected.detach()

        self._last_flow = flow.detach()
        flow_logits = getattr(self.estimator, "_last_flow_logits", None)
        self._last_flow_logits = flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        self._last_corrected = corrected.detach()

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "loss_weight": float(loss_weight),
            "event_count": int(event_count),
            "chunk_index": int(chunk_index),
            "num_chunks": int(plan["num_chunks"]),
        }

    def detach_streaming_tbptt_plan_state(self, plan: Dict[str, Any]) -> None:
        plan["work_state"] = self._detach_sofe_state_value(plan.get("work_state"))

    def finalize_streaming_tbptt_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        last_flow = plan.get("last_flow")
        last_flow_unscaled = plan.get("last_flow_unscaled")
        last_corrected = plan.get("last_corrected")
        if not isinstance(last_flow, torch.Tensor) or not isinstance(last_flow_unscaled, torch.Tensor):
            raise RuntimeError("TBPTT chunk-backward path produced no flow readout")
        if not isinstance(last_corrected, torch.Tensor):
            raise RuntimeError("TBPTT chunk-backward path produced no corrected output")

        loss_dict = self._reduce_loss_dicts(plan.get("chunk_loss_dicts", []))
        work_state = plan.get("work_state")

        self._state = work_state
        self._last_flow = last_flow.detach()
        flow_logits = getattr(self.estimator, "_last_flow_logits", None)
        self._last_flow_logits = flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        self._last_corrected = last_corrected.detach()
        self._store_loss_dict(loss_dict)

        self.estimator._state = work_state
        self.estimator._last_flow = last_flow_unscaled.detach()
        self.estimator._last_flow_history = [
            flow.detach() for flow in plan.get("flow_history_unscaled", [])
        ]

        event_count_stats = self._sequence_stats(plan.get("chunk_event_counts"))
        loss_weight_stats = self._sequence_stats(plan.get("chunk_loss_weights"))
        final_head_metrics: Dict[str, float] = {}
        collect_head_metrics = getattr(self.estimator, "collect_head_state_metrics", None)
        if callable(collect_head_metrics):
            maybe_metrics = collect_head_metrics(getattr(work_state, "hidden", None), None)
            if isinstance(maybe_metrics, dict):
                final_head_metrics = {str(k): float(v) for k, v in maybe_metrics.items()}

        stream_debug: Dict[str, Any] = {
            "partition_mode": str(self.estimator.partition_mode),
            "sofe_family": str(self.estimator.architecture_spec.get("family", "fire")),
            "step_encoding": str(self.estimator.architecture_spec.get("step_encoding", "hist2")),
            "recurrent_backend": str(self.estimator.architecture_spec.get("recurrent_backend", "native")),
            "recurrent_cell": str(self.estimator.architecture_spec.get("recurrent_cell", "convgru")),
            "recurrent_placement": str(self.estimator.architecture_spec.get("recurrent_placement", "every_stage")),
            "spiking_neuron": str(self.estimator.architecture_spec.get("spiking_neuron", "lif")),
            "surrogate_function": str(self.estimator.architecture_spec.get("surrogate_function", "sigmoid")),
            "spiking_backend": str(self.estimator.architecture_spec.get("spiking_backend", "torch")),
            "readout_type": str(self.estimator.architecture_spec.get("readout_type", "single_scale")),
            "num_steps": int(len(plan.get("steps", []))),
            "num_readouts": int(plan.get("num_chunks", 0)),
            "step_time_us": int(self.estimator.step_time_us or 0),
            "tbptt_steps": int(plan.get("tbptt_steps", 1)),
            "num_tbptt_chunks": int(plan.get("num_chunks", 0)),
            "streaming_step_policy": "chunk_backward",
            "streaming_step_policy_id": 1,
        }
        for stat_name, stat_value in event_count_stats.items():
            stream_debug[f"chunk_event_count_{stat_name}"] = float(stat_value)
        for stat_name, stat_value in loss_weight_stats.items():
            stream_debug[f"chunk_loss_weight_{stat_name}"] = float(stat_value)
        if "head_spike_rate" in final_head_metrics:
            stream_debug["head_spike_rate_final"] = float(final_head_metrics["head_spike_rate"])
        head_spike_active_rates = plan.get("head_spike_active_rates", [])
        if isinstance(head_spike_active_rates, list) and head_spike_active_rates:
            stream_debug["head_spike_rate_active_mean"] = sum(head_spike_active_rates) / float(len(head_spike_active_rates))
            stream_debug["head_spike_rate_active_max"] = max(head_spike_active_rates)
        head_spike_readout_rates = plan.get("head_spike_readout_rates", [])
        if isinstance(head_spike_readout_rates, list) and head_spike_readout_rates:
            stream_debug["head_spike_rate_readout_mean"] = sum(head_spike_readout_rates) / float(len(head_spike_readout_rates))
            stream_debug["head_spike_rate_readout_last"] = head_spike_readout_rates[-1]
        head_margin_max_values = plan.get("head_v_minus_thresh_active_max_values", [])
        if isinstance(head_margin_max_values, list) and head_margin_max_values:
            stream_debug["head_v_minus_thresh_active_max"] = max(head_margin_max_values)
        head_margin_p99_values = plan.get("head_v_minus_thresh_active_p99_values", [])
        if isinstance(head_margin_p99_values, list) and head_margin_p99_values:
            stream_debug["head_v_minus_thresh_active_p99"] = max(head_margin_p99_values)
        for key in (
            "cuda_memory_allocated_before_chunks_mb",
            "cuda_memory_allocated_after_chunks_mb",
            "cuda_peak_memory_allocated_before_chunks_mb",
            "cuda_peak_memory_allocated_after_chunks_mb",
        ):
            value = plan.get(key)
            if isinstance(value, (int, float)):
                stream_debug[key] = float(value)
        self.estimator._last_readout_debug = stream_debug
        self._last_stream_debug = stream_debug
        return stream_debug

    @property
    def primary_event_representation(self) -> str:
        return "streaming_events"

    @property
    def required_batch_keys(self) -> Tuple[str, ...]:
        return (
            "events",
            "events_offsets",
            "t_start_us",
            "t_end_us",
            "t_ref_us",
        )

    def forward(
        self,
        voxel: Optional[torch.Tensor] = None,
        flow_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        if "streaming_tbptt_plan" in kwargs:
            plan = kwargs["streaming_tbptt_plan"]
            chunk_index = int(kwargs.get("streaming_tbptt_chunk_index", 0))
            return self.forward_streaming_tbptt_chunk(plan, chunk_index)

        required = set(self.required_batch_keys)
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(
                "Streaming SMCM forward requires packed-event metadata keys "
                f"{sorted(required)}; missing {missing}"
            )

        consumed = set(self.required_batch_keys)
        consumed.update({"sequence_reset", "detach_state", "return_debug"})

        return self.forward_streaming(
            events=kwargs["events"],
            events_offsets=kwargs["events_offsets"],
            t_start_us=kwargs["t_start_us"],
            t_end_us=kwargs["t_end_us"],
            t_ref_us=kwargs["t_ref_us"],
            flow_gt=flow_gt,
            sequence_reset=kwargs.get("sequence_reset"),
            detach_state=bool(kwargs.get("detach_state", False)),
            return_debug=bool(kwargs.get("return_debug", False)),
            voxel=voxel,
            **{k: v for k, v in kwargs.items() if k not in consumed},
        )

    def forward_streaming(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
        flow_gt: Optional[torch.Tensor] = None,
        sequence_reset: Optional[torch.Tensor] = None,
        detach_state: bool = False,
        return_debug: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        runtime_kwargs = dict(kwargs)
        provided_voxel: Optional[torch.Tensor] = runtime_kwargs.pop("voxel", None)
        # Pop the profiling flag so it never leaks into estimator/warp/loss as an
        # unexpected kwarg, and use it to time the streaming stages (#110).
        profile_stages = bool(runtime_kwargs.pop("_profile_mcm_stages", False))

        sequence_reset = self.prepare_sequence_reset(
            seq_idx=runtime_kwargs.get("seq_idx"),
            frame_index=runtime_kwargs.get("frame_index"),
            sequence_reset=sequence_reset,
        )

        if detach_state:
            self.detach_state()

        if self.should_use_frozen_inference_path():
            return self.forward_streaming_inference(
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                sequence_reset=sequence_reset,
                return_debug=return_debug,
                voxel=provided_voxel,
                profile_stages=profile_stages,
                **runtime_kwargs,
            )

        tbptt_steps = int(getattr(self.estimator, "bptt_steps", 0) or 0)
        if self.training and self.loss_fn is not None and tbptt_steps > 0:
            # Per-stage timing is omitted for the autograd-heavy TBPTT path;
            # the attribute is kept present (and empty) for backbone profiling.
            self._last_profile_events = {}
            return self._forward_streaming_tbptt(
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                flow_gt=flow_gt,
                sequence_reset=sequence_reset,
                return_debug=return_debug,
                voxel=provided_voxel,
                **runtime_kwargs,
            )

        stage_timer = CudaStageTimer(enabled=profile_stages and bool(events.is_cuda))

        with stage_timer.section("smcm_estimator"):
            est_out = self.estimator.forward_streaming(
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                state=self._state,
                detach_state=False,
                sequence_reset=sequence_reset,
                return_debug=return_debug,
                **runtime_kwargs,
            )

        stream_debug: Dict[str, Any] = {}
        flow_unscaled: torch.Tensor
        next_state: Optional[SOFEState]
        if isinstance(est_out, tuple) and len(est_out) == 3:
            flow_unscaled, next_state, est_debug = est_out
            if isinstance(est_debug, dict):
                stream_debug.update(est_debug)
        elif isinstance(est_out, tuple) and len(est_out) == 2:
            flow_unscaled, next_state = est_out
        else:
            raise ValueError("Unexpected estimator.forward_streaming return signature")

        self._state = next_state
        flow = flow_unscaled * self.flow_scaling if self.flow_scaling != 1.0 else flow_unscaled

        with stage_timer.section("smcm_warp"):
            corrected = self.warp_events(
                flow=flow,
                voxel=provided_voxel,
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                **runtime_kwargs,
            )

        with stage_timer.section("smcm_loss"):
            loss, loss_dict = self.compute_loss(
                corrected=corrected,
                flow=flow,
                flow_gt=flow_gt,
                flow_unscaled=flow_unscaled,
                voxel=provided_voxel,
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                **runtime_kwargs,
            )

        self._last_profile_events = dict(stage_timer.events) if stage_timer.enabled else {}
        self._last_flow = flow.detach()
        flow_logits = getattr(self.estimator, "_last_flow_logits", None)
        self._last_flow_logits = flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        self._last_corrected = corrected.detach() if isinstance(corrected, torch.Tensor) else None
        self._store_loss_dict(loss_dict)
        self._last_stream_debug = stream_debug

        if return_debug:
            return corrected, loss, stream_debug
        return corrected, loss

    def forward_streaming_inference(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
        sequence_reset: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        voxel: Optional[torch.Tensor] = None,
        profile_stages: bool = False,
        **runtime_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run frozen streaming MCM without TBPTT or loss computation."""

        stage_timer = CudaStageTimer(enabled=profile_stages and bool(events.is_cuda))

        with torch.no_grad():
            with stage_timer.section("smcm_estimator"):
                est_out = self.estimator.forward_streaming(
                    events=events,
                    events_offsets=events_offsets,
                    t_start_us=t_start_us,
                    t_end_us=t_end_us,
                    t_ref_us=t_ref_us,
                    state=self._state,
                    detach_state=False,
                    sequence_reset=sequence_reset,
                    return_debug=return_debug,
                    **runtime_kwargs,
                )

            stream_debug: Dict[str, Any] = {}
            flow_unscaled: torch.Tensor
            next_state: Optional[SOFEState]
            if isinstance(est_out, tuple) and len(est_out) == 3:
                flow_unscaled, next_state, est_debug = est_out
                if isinstance(est_debug, dict):
                    stream_debug.update(est_debug)
            elif isinstance(est_out, tuple) and len(est_out) == 2:
                flow_unscaled, next_state = est_out
            else:
                raise ValueError("Unexpected estimator.forward_streaming return signature")

            self._state = next_state
            flow = flow_unscaled * self.flow_scaling if self.flow_scaling != 1.0 else flow_unscaled
            with stage_timer.section("smcm_warp"):
                corrected = self.warp_events(
                    flow=flow,
                    voxel=voxel,
                    events=events,
                    events_offsets=events_offsets,
                    t_start_us=t_start_us,
                    t_end_us=t_end_us,
                    t_ref_us=t_ref_us,
                    **runtime_kwargs,
                )
                corrected = self._prepare_frozen_detector_output(
                    corrected,
                    reference=flow,
                )

        self._last_profile_events = dict(stage_timer.events) if stage_timer.enabled else {}
        self._last_flow = flow.detach()
        flow_logits = getattr(self.estimator, "_last_flow_logits", None)
        self._last_flow_logits = flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        self._last_corrected = corrected.detach() if isinstance(corrected, torch.Tensor) else None
        self._clear_loss_caches()
        self._last_stream_debug = stream_debug

        zero = flow_unscaled.new_tensor(0.0)
        if return_debug:
            return corrected, zero, stream_debug
        return corrected, zero

    def _forward_streaming_tbptt(
        self,
        *,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        t_start_us: torch.Tensor,
        t_end_us: torch.Tensor,
        t_ref_us: torch.Tensor,
        flow_gt: Optional[torch.Tensor] = None,
        sequence_reset: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        voxel: Optional[torch.Tensor] = None,
        **runtime_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        tbptt_steps = max(1, int(getattr(self.estimator, "bptt_steps", 0) or 1))
        work_state = self._prepare_estimator_state(
            t_start_us=t_start_us,
            events_offsets=events_offsets,
            sequence_reset=sequence_reset,
        )
        steps = self.estimator.partition_steps(
            events=events,
            events_offsets=events_offsets,
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            t_ref_us=t_ref_us,
        )
        if not steps:
            zero = t_ref_us.new_zeros((), dtype=torch.float32)
            empty_corrected = self.warp_events(
                flow=t_ref_us.new_zeros((int(t_ref_us.numel()), 2, self.estimator.image_height, self.estimator.image_width), dtype=torch.float32),
                voxel=voxel,
                events=events,
                events_offsets=events_offsets,
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                t_ref_us=t_ref_us,
                **runtime_kwargs,
            )
            if return_debug:
                return empty_corrected, zero, {"num_steps": 0, "num_readouts": 0, "tbptt_steps": tbptt_steps}
            return empty_corrected, zero

        chunk_losses: List[torch.Tensor] = []
        chunk_loss_dicts: List[Optional[Dict[str, Any]]] = []
        flow_history_unscaled: List[torch.Tensor] = []
        last_flow: Optional[torch.Tensor] = None
        last_flow_unscaled: Optional[torch.Tensor] = None
        last_corrected: Optional[torch.Tensor] = None

        for chunk_start in range(0, len(steps), tbptt_steps):
            chunk_steps = steps[chunk_start : chunk_start + tbptt_steps]
            for step in chunk_steps:
                step_input = self.estimator.build_step_input(step)
                step.step_input = step_input
                decay = self.estimator.compute_decay(
                    step_ref_us=step.step_ref_us if step.step_ref_us is not None else t_ref_us,
                    last_step_ref_us=work_state.last_step_ref_us if work_state is not None else None,
                    t_start_us=t_start_us,
                )
                work_state = self.estimator.update_step(
                    step_input=step_input,
                    step=step,
                    state=work_state,
                    decay=decay,
                )

            chunk_batch = self._merge_step_chunk(chunk_steps)
            chunk_t_ref = chunk_batch["t_ref_us"]
            last_flow_unscaled = self.estimator.readout_flow(
                state=work_state,
                step=chunk_steps[-1],
                t_ref_us=chunk_t_ref,
            )
            flow_history_unscaled.append(last_flow_unscaled)
            last_flow = (
                last_flow_unscaled * self.flow_scaling
                if self.flow_scaling != 1.0
                else last_flow_unscaled
            )
            last_corrected = self.warp_events(
                flow=last_flow,
                voxel=voxel,
                events=chunk_batch["events"],
                events_offsets=chunk_batch["events_offsets"],
                t_start_us=chunk_batch["t_start_us"],
                t_end_us=chunk_batch["t_end_us"],
                t_ref_us=chunk_batch["t_ref_us"],
                **runtime_kwargs,
            )
            loss, loss_dict = self.compute_loss(
                corrected=last_corrected,
                flow=last_flow,
                flow_gt=flow_gt,
                flow_unscaled=last_flow_unscaled,
                voxel=voxel,
                events=chunk_batch["events"],
                events_offsets=chunk_batch["events_offsets"],
                t_start_us=chunk_batch["t_start_us"],
                t_end_us=chunk_batch["t_end_us"],
                t_ref_us=chunk_batch["t_ref_us"],
                **runtime_kwargs,
            )
            chunk_losses.append(loss)
            chunk_loss_dicts.append(loss_dict)

            if chunk_start + tbptt_steps < len(steps):
                work_state = self._detach_sofe_state_value(work_state)

        if last_flow is None or last_flow_unscaled is None or last_corrected is None:
            raise RuntimeError("TBPTT streaming path produced no readout")

        total_loss = torch.stack(chunk_losses, dim=0).mean()
        loss_dict = self._reduce_loss_dicts(chunk_loss_dicts)

        self._state = work_state
        self._last_flow = last_flow.detach()
        flow_logits = getattr(self.estimator, "_last_flow_logits", None)
        self._last_flow_logits = flow_logits.detach() if isinstance(flow_logits, torch.Tensor) else None
        self._last_corrected = last_corrected.detach() if isinstance(last_corrected, torch.Tensor) else None
        self._store_loss_dict(loss_dict)

        self.estimator._state = work_state
        self.estimator._last_flow = last_flow_unscaled.detach()
        self.estimator._last_flow_history = [flow.detach() for flow in flow_history_unscaled]

        stream_debug: Dict[str, Any] = {
            "partition_mode": str(self.estimator.partition_mode),
            "sofe_family": str(self.estimator.architecture_spec.get("family", "fire")),
            "step_encoding": str(self.estimator.architecture_spec.get("step_encoding", "hist2")),
            "recurrent_backend": str(self.estimator.architecture_spec.get("recurrent_backend", "native")),
            "recurrent_cell": str(self.estimator.architecture_spec.get("recurrent_cell", "convgru")),
            "recurrent_placement": str(self.estimator.architecture_spec.get("recurrent_placement", "every_stage")),
            "spiking_neuron": str(self.estimator.architecture_spec.get("spiking_neuron", "lif")),
            "surrogate_function": str(self.estimator.architecture_spec.get("surrogate_function", "sigmoid")),
            "spiking_backend": str(self.estimator.architecture_spec.get("spiking_backend", "torch")),
            "readout_type": str(self.estimator.architecture_spec.get("readout_type", "single_scale")),
            "num_steps": int(len(steps)),
            "num_readouts": int(len(flow_history_unscaled)),
            "step_time_us": int(self.estimator.step_time_us or 0),
            "tbptt_steps": int(tbptt_steps),
            "num_tbptt_chunks": int(len(flow_history_unscaled)),
            "streaming_step_policy": "outer_window",
            "streaming_step_policy_id": 0,
        }
        self.estimator._last_readout_debug = stream_debug
        self._last_stream_debug = stream_debug

        if return_debug:
            return last_corrected, total_loss, stream_debug
        return last_corrected, total_loss

    def prepare_sequence_reset(
        self,
        *,
        seq_idx: Optional[torch.Tensor] = None,
        frame_index: Optional[torch.Tensor] = None,
        sequence_reset: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        explicit_reset: Optional[torch.Tensor] = None
        if sequence_reset is not None:
            if isinstance(sequence_reset, torch.Tensor):
                explicit_reset = sequence_reset.to(dtype=torch.bool)
            else:
                explicit_reset = torch.as_tensor(sequence_reset, dtype=torch.bool)
            if explicit_reset.ndim != 1:
                explicit_reset = explicit_reset.reshape(-1)

        if not self.reset_on_sequence_change:
            return explicit_reset

        if seq_idx is None:
            return explicit_reset

        if isinstance(seq_idx, torch.Tensor):
            seq_idx_t = seq_idx.to(dtype=torch.int64)
        else:
            seq_idx_t = torch.as_tensor(seq_idx, dtype=torch.int64)
        if seq_idx_t.ndim != 1:
            seq_idx_t = seq_idx_t.reshape(-1)

        inferred = torch.ones_like(seq_idx_t, dtype=torch.bool)
        if self._last_seq_idx is not None and self._last_seq_idx.numel() == seq_idx_t.numel():
            prev_seq = self._last_seq_idx.to(device=seq_idx_t.device, dtype=torch.int64)
            inferred = seq_idx_t != prev_seq

            if frame_index is not None and self._last_frame_index is not None:
                cur_f = torch.as_tensor(frame_index, dtype=torch.int64, device=seq_idx_t.device).reshape(-1)
                if self._last_frame_index.numel() == cur_f.numel():
                    prev_f = self._last_frame_index.to(device=seq_idx_t.device, dtype=torch.int64)
                    inferred = inferred | (cur_f <= prev_f)

        self._last_seq_idx = seq_idx_t.detach()
        if frame_index is not None:
            self._last_frame_index = torch.as_tensor(
                frame_index, dtype=torch.int64, device=seq_idx_t.device
            ).reshape(-1).detach()

        if explicit_reset is None:
            return inferred
        return explicit_reset.to(device=seq_idx_t.device, dtype=torch.bool) | inferred

    def detach_state(self) -> None:
        if hasattr(self.estimator, "detach_state"):
            try:
                self.estimator.detach_state()
            except NotImplementedError:
                pass
        self._state = self._detach_sofe_state_value(self._state)

    def reset_state(self) -> None:
        self._state = None
        self._last_stream_debug = None
        self._last_flow_logits = None
        self._last_seq_idx = None
        self._last_frame_index = None
        super().reset_state()

    def warp_events(
        self,
        *,
        flow: torch.Tensor,
        voxel: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.warper is None:
            if voxel is None:
                raise ValueError(
                    "SMCM has no warper and no voxel was provided; cannot produce corrected representation."
                )
            return voxel

        if isinstance(self.output_renderer, EventOutputRenderer):
            return self.output_renderer(flow=flow, voxel=voxel, **kwargs)

        if self._warper_accepts_kwargs:
            warper_voxel = voxel
            if warper_voxel is None:
                # Sparse-event warpers often ignore voxel values and only use
                # tensor placement/shape from this positional argument.
                b, _, h, w = flow.shape
                warper_voxel = flow.new_zeros((b, 1, h, w))
            return self.warper(warper_voxel, flow, **kwargs)

        if voxel is None:
            raise ValueError(
                "Configured warper requires a dense voxel tensor; SMCM does not "
                "build voxels internally. Provide voxel explicitly or use a "
                "streaming-compatible warper."
            )
        return self.warper(voxel, flow)

    @staticmethod
    def _upsample_flow_to_size(
        flow: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Upsample a pyramid flow to target spatial size, rescaling displacement values."""
        src_h, src_w = int(flow.shape[2]), int(flow.shape[3])
        tgt_h, tgt_w = int(target_size[0]), int(target_size[1])
        if (src_h, src_w) == (tgt_h, tgt_w):
            return flow
        flow_up = torch.nn.functional.interpolate(
            flow, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
        )
        sx = float(tgt_w) / float(src_w)
        sy = float(tgt_h) / float(src_h)
        if sx != 1.0 or sy != 1.0:
            flow_up = flow_up.clone()
            flow_up[:, 0].mul_(sx)
            flow_up[:, 1].mul_(sy)
        return flow_up

    @staticmethod
    def _compute_auxiliary_pyramid_warp(
        aux: Dict[str, Any],
        corrected: torch.Tensor,
        loss_fn: Any,
        *,
        scales: List[str],
        ref_flow_hw: Tuple[int, int],
        **kwargs: Any,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Self-supervised event-warping loss for coarse pyramid flow levels.

        Each pyramid flow is upsampled to full resolution then evaluated with the
        same event-warping objective used for the main flow. ``flow_unscaled`` is
        stripped from kwargs so EventWarpingLoss uses the upsampled pyramid flow
        for its smoothness term rather than the main flow.
        """
        from smcfnet.model.losses import EventWarpingLoss, EventWarpingSupervisedLoss

        # Unwrap composed losses to reach the underlying EventWarpingLoss.
        if isinstance(loss_fn, EventWarpingLoss):
            warp_loss = loss_fn
        elif hasattr(loss_fn, "event_warping") and isinstance(loss_fn.event_warping, EventWarpingLoss):
            warp_loss = loss_fn.event_warping
        else:
            return None, {}

        # Drop flow_unscaled from kwargs so EventWarpingLoss picks up the
        # upsampled pyramid flow as its smoothness reference, not the main flow.
        kwargs_aux = {k: v for k, v in kwargs.items() if k != "flow_unscaled"}

        total_loss: Optional[torch.Tensor] = None
        count = 0
        loss_dict: Dict[str, Any] = {}

        for scale in scales:
            flow_pred = aux.get(f"flow_{scale}")
            if not (isinstance(flow_pred, torch.Tensor) and flow_pred.numel() > 0):
                continue

            flow_full = StreamingMotionCompensationModule._upsample_flow_to_size(
                flow_pred, ref_flow_hw
            )
            warp_val, _ = warp_loss(corrected, flow_full, None, **kwargs_aux)
            total_loss = warp_val if total_loss is None else (total_loss + warp_val)
            count += 1
            loss_dict[f"aux_warp_{scale}"] = warp_val.detach()

        if total_loss is None or count == 0:
            return None, {}

        avg_loss = total_loss / float(count)
        loss_dict["aux_warp_mean"] = avg_loss.detach()
        return avg_loss, loss_dict

    @staticmethod
    def _compute_auxiliary_pyramid_epe(
        aux: Dict[str, Any],
        flow_gt: torch.Tensor,
        *,
        scales: List[str],
        eps: float = 1e-6,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Compute supervised EPE against downsampled GT flow for each pyramid scale.

        GT flow is expected at full image resolution (pixel units at full res).
        For each scale level, GT is downsampled to the matching spatial size and
        the displacement values are scaled proportionally so that 1 unit = 1
        pixel at that level's resolution.
        """
        # Separate optional validity mask encoded in the 3rd channel.
        if flow_gt.shape[1] == 3:
            valid_full = flow_gt[:, 2:3] > 0.5
            flow_gt_xy = flow_gt[:, :2].float()
        else:
            valid_full = None
            flow_gt_xy = flow_gt.float()

        total_loss: Optional[torch.Tensor] = None
        count = 0
        loss_dict: Dict[str, Any] = {}

        for scale in scales:
            flow_pred = aux.get(f"flow_{scale}")
            if not (isinstance(flow_pred, torch.Tensor) and flow_pred.numel() > 0):
                continue

            tgt_h, tgt_w = int(flow_pred.shape[2]), int(flow_pred.shape[3])
            src_h, src_w = int(flow_gt_xy.shape[2]), int(flow_gt_xy.shape[3])

            # Downsample GT to the level's spatial size.
            if (tgt_h, tgt_w) != (src_h, src_w):
                gt_down = torch.nn.functional.interpolate(
                    flow_gt_xy, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
                )
            else:
                gt_down = flow_gt_xy

            # Scale displacement values to the level's pixel units.
            sx = float(tgt_w) / float(src_w)
            sy = float(tgt_h) / float(src_h)
            if sx != 1.0 or sy != 1.0:
                gt_down = gt_down.clone()
                gt_down[:, 0].mul_(sx)
                gt_down[:, 1].mul_(sy)

            # Valid mask at this level.
            if valid_full is not None:
                valid_down = torch.nn.functional.interpolate(
                    valid_full.float(), size=(tgt_h, tgt_w), mode="nearest"
                ) > 0.5
            else:
                valid_down = None

            diff = flow_pred.float() - gt_down
            epe_map = torch.sqrt(diff.pow(2).sum(dim=1, keepdim=True) + eps)

            if valid_down is not None and bool(valid_down.any()):
                epe_s = epe_map[valid_down].mean()
            else:
                epe_s = epe_map.mean()

            total_loss = epe_s if total_loss is None else (total_loss + epe_s)
            count += 1
            loss_dict[f"aux_epe_{scale}"] = epe_s.detach()

        if total_loss is None or count == 0:
            return None, {}

        avg_loss = total_loss / float(count)
        loss_dict["aux_epe_mean"] = avg_loss.detach()
        return avg_loss, loss_dict

    def compute_loss(
        self,
        *,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if self.loss_fn is None or not self.training or self.should_use_frozen_inference_path():
            z = flow.new_tensor(0.0)
            return z, None

        try:
            out = self.loss_fn(corrected, flow, flow_gt, **kwargs)
        except TypeError:
            try:
                out = self.loss_fn(corrected, flow, flow_gt)
            except TypeError:
                out = self.loss_fn(corrected, flow)

        if isinstance(out, tuple) and len(out) == 2:
            loss, loss_dict = out
        else:
            loss, loss_dict = out, None

        estimator_obj = getattr(self, "estimator", None)
        aux = getattr(estimator_obj, "_last_step_readout_aux", None)

        # Supervised auxiliary pyramid EPE loss (Phase 2).
        # Reads live pyramid flows from the estimator's last readout aux dict
        # and computes EPE against downsampled GT flow at each pyramid scale.
        if self.auxiliary_flow_loss and flow_gt is not None and isinstance(aux, dict):
            aux_loss, aux_loss_dict = self._compute_auxiliary_pyramid_epe(
                aux,
                flow_gt,
                scales=self.auxiliary_flow_scales,
            )
            if aux_loss is not None:
                loss = loss + self.auxiliary_flow_loss_weight * aux_loss
                if aux_loss_dict:
                    if loss_dict is None:
                        loss_dict = {}
                    loss_dict = dict(loss_dict)
                    loss_dict.update(aux_loss_dict)

        # Self-supervised auxiliary pyramid warping loss (Phase 3).
        # Upsamples each pyramid flow to full resolution and evaluates the same
        # event-warping objective used for the main flow, giving gradient signal
        # to the coarse encoder stages without needing GT flow.
        if self.auxiliary_warp_loss and isinstance(aux, dict) and self.loss_fn is not None:
            ref_flow_hw = (int(flow.shape[2]), int(flow.shape[3]))
            aux_warp_loss, aux_warp_dict = self._compute_auxiliary_pyramid_warp(
                aux,
                corrected,
                self.loss_fn,
                scales=self.auxiliary_flow_scales,
                ref_flow_hw=ref_flow_hw,
                **kwargs,
            )
            if aux_warp_loss is not None:
                loss = loss + self.auxiliary_warp_loss_weight * aux_warp_loss
                if aux_warp_dict:
                    if loss_dict is None:
                        loss_dict = {}
                    loss_dict = dict(loss_dict)
                    loss_dict.update(aux_warp_dict)

        return loss, loss_dict


# Alias kept to match earlier terminology in the repo/docs.
SpikingMotionCompensationModule = StreamingMotionCompensationModule
