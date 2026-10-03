"""Generic motion compensation module (MCM).

This is the shared compositor used by ECM/SMCM-style modules.
It is intentionally estimator-agnostic: the `FlowEstimator` decides whether
it is ANN or SNN and handles its own internal state.

Contract (matches existing backbone expectations):
  forward(voxel, flow_gt=None) -> (corrected, loss)

- `corrected` has the same shape as `voxel`.
- `loss` is a scalar tensor (0 if no loss_fn).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.ofe import FlowEstimator
from smcfnet.model.event_outputs import EventOutputRenderer
from smcfnet.model.mcm.base import BaseMotionCompensationModule
from smcfnet.model.warping import warp_voxel_bins
from smcfnet.utils.profiling import CudaStageTimer


class EventWarpingModule(nn.Module):
    """Warp each voxel bin to the reference time using predicted optical flow."""

    def forward(self, voxel: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return warp_voxel_bins(voxel, flow)

class MotionCompensationModule(BaseMotionCompensationModule):
    """Generic motion compensation compositor.

    Args:
        estimator: Optical flow estimator implementing the `FlowEstimator` contract.
        warper: Module that warps voxel bins using flow (default: warp_voxel_bins).
        loss_fn: Optional callable computing a scalar loss.
          Expected signatures:
            - loss_fn(corrected, flow) -> loss
            - loss_fn(corrected, flow, flow_gt) -> loss
            - MotionCompensationLoss-style: returns (loss, loss_dict)
        flow_scaling: Scalar multiplier applied to the raw flow estimator output
          before it is used for warping and passed to the loss function.
          ``1.0`` (default) = raw pixel units.
          ``640`` = original MCFNet-style where the estimator predicts
          small normalised values that are scaled to pixel displacements.
    """

    def __init__(
        self,
        estimator: FlowEstimator,
        warper: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[..., object]] = None,
        flow_scaling: float = 1.0,
        output_renderer: Optional[nn.Module] = None,
    ):
        super().__init__(
            estimator=estimator,
            warper=output_renderer or warper or EventWarpingModule(),
            loss_fn=loss_fn,
            flow_scaling=flow_scaling,
        )

    @property
    def primary_event_representation(self) -> str:
        return "voxel"

    @property
    def required_batch_keys(self) -> Tuple[str, ...]:
        keys = ["events_voxel"]
        if isinstance(self.output_renderer, EventOutputRenderer):
            keys.extend(["events", "events_offsets"])
        return tuple(keys)

    @staticmethod
    def _resolve_estimator_input(voxel: torch.Tensor, kwargs: dict) -> torch.Tensor:
        estimator_input = kwargs.pop("estimator_input", None)
        if estimator_input is None:
            estimator_input = kwargs.pop("mcm_estimator_input", None)
        if estimator_input is None:
            estimator_input = voxel
        return estimator_input

    @staticmethod
    def _match_flow_to_voxel_resolution(voxel: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        target_h, target_w = voxel.shape[-2:]
        source_h, source_w = flow.shape[-2:]
        if (source_h, source_w) == (target_h, target_w):
            return flow

        resized = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
        scale_x = float(target_w) / float(max(source_w, 1))
        scale_y = float(target_h) / float(max(source_h, 1))
        resized = resized.clone()
        resized[:, 0].mul_(scale_x)
        resized[:, 1].mul_(scale_y)
        return resized

    def _render_corrected(
        self,
        *,
        voxel: torch.Tensor,
        flow: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        if isinstance(self.output_renderer, EventOutputRenderer):
            return self.output_renderer(flow=flow, voxel=voxel, **kwargs)
        if kwargs and self._warper_accepts_kwargs:
            return self.warper(voxel, flow, **kwargs)
        return self.warper(voxel, flow)

    def forward_inference(
        self,
        voxel: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run frozen MCM as a pure event-correction module.

        This path is for detector training with a frozen MCM. It produces the
        downstream corrected representation, but intentionally skips MCM loss
        computation and avoids retaining an autograd graph through the frozen
        estimator/renderer.
        """

        runtime_kwargs = dict(kwargs)
        profile_stages = bool(runtime_kwargs.pop("_profile_mcm_stages", False))
        stage_timer = CudaStageTimer(enabled=profile_stages and bool(voxel.is_cuda))
        estimator_input = self._resolve_estimator_input(voxel, runtime_kwargs)

        with torch.no_grad():
            with stage_timer.section("mcm_estimator"):
                flow = self.estimator(estimator_input)
            flow_unscaled = flow
            with stage_timer.section("mcm_flow_resize"):
                flow = self._match_flow_to_voxel_resolution(voxel, flow)
            with stage_timer.section("mcm_flow_scale"):
                if self.flow_scaling != 1.0:
                    flow = flow * self.flow_scaling
            with stage_timer.section("mcm_renderer"):
                corrected = self._render_corrected(
                    voxel=voxel,
                    flow=flow,
                    kwargs=runtime_kwargs,
                )

        self._last_flow = flow.detach()
        self._last_corrected = corrected.detach()
        self._last_profile_events = dict(stage_timer.events) if stage_timer.enabled else {}
        self._clear_loss_caches()
        return corrected, flow_unscaled.new_tensor(0.0)

    def forward(
        self,
        voxel: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        runtime_kwargs = dict(kwargs)
        profile_stages = bool(runtime_kwargs.pop("_profile_mcm_stages", False))
        stage_timer = CudaStageTimer(enabled=profile_stages and bool(voxel.is_cuda))
        if self.should_use_frozen_inference_path():
            if profile_stages:
                runtime_kwargs["_profile_mcm_stages"] = True
            return self.forward_inference(voxel, **runtime_kwargs)

        estimator_input = self._resolve_estimator_input(voxel, runtime_kwargs)
        with stage_timer.section("mcm_estimator"):
            flow = self.estimator(estimator_input)
        flow_unscaled = flow

        flow_history = getattr(self.estimator, "_last_flow_history", None)

        with stage_timer.section("mcm_flow_resize"):
            flow = self._match_flow_to_voxel_resolution(voxel, flow)

        # Apply flow scaling: 1.0 = raw pixel units (default),
        with stage_timer.section("mcm_flow_scale"):
            if self.flow_scaling != 1.0:
                flow = flow * self.flow_scaling

        with stage_timer.section("mcm_renderer"):
            corrected = self._render_corrected(
                voxel=voxel,
                flow=flow,
                kwargs=runtime_kwargs,
            )

        loss = torch.tensor(0.0, device=voxel.device)
        loss_dict = None
        if self.loss_fn is not None and self.training:
            with stage_timer.section("mcm_loss"):
                loss_kwargs = dict(runtime_kwargs)
                loss_kwargs["flow_unscaled"] = flow_unscaled
                if flow_history is not None:
                    loss_kwargs["flow_history"] = flow_history
                try:
                    out = self.loss_fn(corrected, flow, flow_gt, voxel=voxel, **loss_kwargs)
                except TypeError:
                    try:
                        out = self.loss_fn(corrected, flow, flow_gt)
                    except TypeError:
                        out = self.loss_fn(corrected, flow)

                if isinstance(out, tuple) and len(out) == 2:
                    loss, loss_dict = out
                else:
                    loss = out

        # Caches for logging/visualization (best-effort)
        self._last_flow = flow.detach()
        self._last_corrected = corrected.detach()
        self._last_profile_events = dict(stage_timer.events) if stage_timer.enabled else {}
        self._store_loss_dict(loss_dict)

        return corrected, loss
