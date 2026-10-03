"""Shared abstractions for motion-compensation modules.

The goal of this module is to keep the outer MCM/SMCM contract consistent
across:
- classic voxel-first MCM implementations
- future streaming/event-native SMCM implementations

This lets trainers, backbones, and checkpoint code reason about a small common
interface even when the internal event representation differs.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from smcfnet.model.ofe import FlowEstimator


def _accepts_var_kwargs(module: nn.Module) -> bool:
    """Return whether a module forward accepts arbitrary keyword arguments."""

    warper_forward = getattr(module, "forward", None)
    if warper_forward is None:
        return False

    sig = inspect.signature(warper_forward)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )


class BaseMotionCompensationModule(nn.Module, ABC):
    """Common outer contract for MCM-like modules.

    Shared responsibilities:
    - expose a uniform ``forward(...)->(corrected, loss)`` style interface
    - delegate basic estimator control (mode, reset, optional inference mode)
    - expose common logging caches used by trainers and backbones
    - declare the primary event-representation contract expected by the module
    """

    def __init__(
        self,
        estimator: FlowEstimator,
        warper: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[..., object]] = None,
        flow_scaling: float = 1.0,
    ):
        super().__init__()
        self.estimator = estimator
        self.warper = warper
        self._warper_accepts_kwargs = _accepts_var_kwargs(self.warper) if self.warper is not None else False
        self.loss_fn = loss_fn
        self.flow_scaling = float(flow_scaling)

        self._reset_logging_caches()

    @property
    def output_renderer(self) -> Optional[nn.Module]:
        """Transition alias: the detector-output renderer lives in warper slot."""

        return self.warper

    @output_renderer.setter
    def output_renderer(self, module: Optional[nn.Module]) -> None:
        self.warper = module
        self._warper_accepts_kwargs = _accepts_var_kwargs(module) if module is not None else False

    @property
    def output_channels(self) -> Optional[int]:
        renderer = self.output_renderer
        value = getattr(renderer, "output_channels", None)
        return int(value) if value is not None else None

    @property
    def output_type(self) -> str:
        renderer = self.output_renderer
        return str(getattr(renderer, "output_type", "legacy_warper"))

    def _reset_logging_caches(self) -> None:
        """Clear best-effort diagnostic caches used elsewhere in the codebase."""

        self._last_flow: Optional[torch.Tensor] = None
        self._last_corrected: Optional[torch.Tensor] = None
        # Per-stage CUDA timing events consumed by backbone profiling. Default to
        # an empty dict so getattr(self.mcm, "_last_profile_events", None) never
        # returns None for streaming variants that don't time every path (#110).
        self._last_profile_events: Dict[str, object] = {}
        self._clear_loss_caches()

    def _clear_loss_caches(self) -> None:
        """Clear cached loss diagnostics without touching flow/output caches."""

        self._last_contrast_loss: Optional[torch.Tensor] = None
        self._last_smooth_loss: Optional[torch.Tensor] = None
        self._last_smooth_loss_raw: Optional[torch.Tensor] = None
        self._last_magnitude_loss: Optional[torch.Tensor] = None
        self._last_epe: Optional[torch.Tensor] = None
        self._last_schedule_ew_weight: Optional[torch.Tensor] = None
        self._last_schedule_sup_weight: Optional[torch.Tensor] = None
        self._last_center_loss: Optional[torch.Tensor] = None
        self._last_centered_ratio: Optional[torch.Tensor] = None
        self._last_loss_dict: Optional[Dict[str, torch.Tensor]] = None

    def _store_loss_dict(self, loss_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        """Populate shared diagnostic caches from a per-term loss dict."""

        self._clear_loss_caches()
        self._last_loss_dict = loss_dict
        if not isinstance(loss_dict, dict):
            return
        if "contrast" in loss_dict:
            self._last_contrast_loss = loss_dict["contrast"]
        if "smooth" in loss_dict:
            self._last_smooth_loss = loss_dict["smooth"]
        if "smooth_raw" in loss_dict:
            self._last_smooth_loss_raw = loss_dict["smooth_raw"]
        if "magnitude" in loss_dict:
            self._last_magnitude_loss = loss_dict["magnitude"]
        if "epe" in loss_dict:
            self._last_epe = loss_dict["epe"]
        if "sup_epe" in loss_dict:
            self._last_epe = loss_dict["sup_epe"]
        if "schedule_ew_weight" in loss_dict:
            self._last_schedule_ew_weight = loss_dict["schedule_ew_weight"]
        if "schedule_sup_weight" in loss_dict:
            self._last_schedule_sup_weight = loss_dict["schedule_sup_weight"]
        if "center" in loss_dict:
            self._last_center_loss = loss_dict["center"]
        if "centered_ratio" in loss_dict:
            self._last_centered_ratio = loss_dict["centered_ratio"]

    def all_parameters_frozen(self) -> bool:
        """Return true when this MCM has parameters and none require gradients."""

        saw_parameter = False
        for param in self.parameters():
            saw_parameter = True
            if param.requires_grad:
                return False
        return saw_parameter

    def should_use_frozen_inference_path(self) -> bool:
        """Use the lightweight forward path when MCM is frozen in a detector run."""

        return self.all_parameters_frozen()

    def all_parameters_frozen(self) -> bool:
        """Return true when this MCM has parameters and none require gradients."""

        saw_parameter = False
        for param in self.parameters():
            saw_parameter = True
            if param.requires_grad:
                return False
        return saw_parameter

    def should_use_frozen_inference_path(self) -> bool:
        """Use the lightweight forward path when MCM is frozen in a detector run."""

        return self.all_parameters_frozen()

    @property
    @abstractmethod
    def primary_event_representation(self) -> str:
        """Return the primary event input contract for this module.

        Expected values for the current roadmap:
        - ``"voxel"`` for classic MCM
        - ``"streaming_events"`` for event-native SMCM
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def required_batch_keys(self) -> Tuple[str, ...]:
        """Return the minimum batch keys the module expects.

        This is intended as a future trainer/build hook so one trainer can
        request the right loader/prefetch path without hardcoding module type.
        """
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        self.estimator.set_mode(mode)

    def get_mode(self) -> str:
        return self.estimator.get_mode()

    def set_inference_mode(self, mode: str) -> None:
        """Switch estimator inference mode when supported by the estimator."""

        if hasattr(self.estimator, "set_inference_mode"):
            self.estimator.set_inference_mode(mode)

    def get_inference_mode(self) -> Optional[str]:
        if hasattr(self.estimator, "get_inference_mode"):
            return self.estimator.get_inference_mode()
        return None

    def reset_state(self) -> None:
        """Reset estimator state and clear shared logging caches."""

        self._reset_logging_caches()
        self.estimator.reset_state()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run motion compensation and return ``(corrected, loss)``."""
        raise NotImplementedError
