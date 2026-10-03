"""Hagenaars/Paredes-Valles FireNet-LIF SOFE implementation.

This module implements the original full-resolution FireNet-style recurrent
SNN topology as a separate SOFE family.  It intentionally does not replace the
existing Xu et al. ``firenet_lif`` implementation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_spikingjelly_modules():
    try:
        from spikingjelly.activation_based import neuron as sj_neuron
        from spikingjelly.activation_based import surrogate as sj_surrogate
    except ImportError as exc:
        raise ImportError(
            "SOFE family='hagenaars_firenet_lif' requires the "
            "'spikingjelly' package in the active environment."
        ) from exc
    return sj_neuron, sj_surrogate


def _make_surrogate(name: str, alpha: float):
    _sj_neuron, sj_surrogate = _import_spikingjelly_modules()
    name = str(name).lower()
    if name == "atan":
        return sj_surrogate.ATan(alpha=float(alpha), spiking=True)
    if name == "sigmoid":
        return sj_surrogate.Sigmoid(alpha=float(alpha), spiking=True)
    if name == "softsign":
        return sj_surrogate.SoftSign(alpha=float(alpha), spiking=True)
    raise NotImplementedError(
        f"Hagenaars FireNet-LIF surrogate_function={name!r} is not implemented; "
        "supported values are 'atan', 'sigmoid', and 'softsign'."
    )


def _apply_decay_to_state(
    state: Optional[Any],
    decay: Optional[torch.Tensor],
    *,
    spike_decay_mode: str = "hold",
) -> Optional[Any]:
    if state is None or decay is None:
        return state
    if torch.is_tensor(state):
        decay_view = decay.view(-1, *([1] * max(state.ndim - 1, 0))).to(
            device=state.device,
            dtype=state.dtype,
        )
        return state * decay_view
    if isinstance(state, dict):
        decayed: Dict[str, Any] = {}
        for key, value in state.items():
            if key == "spike":
                if spike_decay_mode == "hold":
                    decayed[key] = value
                elif spike_decay_mode == "decay":
                    decayed[key] = _apply_decay_to_state(
                        value,
                        decay,
                        spike_decay_mode=spike_decay_mode,
                    )
                elif spike_decay_mode == "zero":
                    decayed[key] = torch.zeros_like(value) if torch.is_tensor(value) else value
                else:
                    raise ValueError(
                        f"Invalid spike_decay_mode={spike_decay_mode!r}; "
                        "expected 'hold', 'decay', or 'zero'."
                    )
            else:
                decayed[key] = _apply_decay_to_state(
                    value,
                    decay,
                    spike_decay_mode=spike_decay_mode,
                )
        return decayed
    return state


class _HagenaarsConvLIF(nn.Module):
    """One full-resolution Conv2d + explicit-state SpikingJelly IF/LIF step."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        threshold: float,
        surrogate_function: str,
        surrogate_alpha: float,
        detach_reset: bool,
        recurrent: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        sj_neuron, _sj_surrogate = _import_spikingjelly_modules()

        self.out_channels = int(out_channels)
        self.recurrent = bool(recurrent)
        self._mode = "snn"

        padding = int(kernel_size) // 2
        self.ff = nn.Conv2d(
            int(in_channels),
            self.out_channels,
            int(kernel_size),
            padding=padding,
            bias=bool(bias),
        )
        self.rec: Optional[nn.Conv2d]
        if self.recurrent:
            self.rec = nn.Conv2d(
                self.out_channels,
                self.out_channels,
                int(kernel_size),
                padding=padding,
                bias=False,
            )
        else:
            self.rec = None

        self.register_buffer("thresh", torch.tensor(float(threshold)), persistent=False)
        surrogate_fn = _make_surrogate(surrogate_function, surrogate_alpha)
        # Continuous-time decay is applied explicitly by the SOFE scheduler
        # before this node is called, so an IFNode gives the desired
        # mem <- alpha * mem + drive update while still using SpikingJelly's
        # spike/reset/surrogate implementation.
        self.node = sj_neuron.IFNode(
            v_threshold=float(threshold),
            v_reset=0.0,
            surrogate_function=surrogate_fn,
            detach_reset=bool(detach_reset),
            step_mode="s",
            backend="torch",
            store_v_seq=False,
        )

        w_scale = math.sqrt(1.0 / max(int(in_channels), 1))
        nn.init.uniform_(self.ff.weight, -w_scale, w_scale)
        if self.ff.bias is not None:
            nn.init.zeros_(self.ff.bias)
        if self.rec is not None:
            rec_scale = math.sqrt(1.0 / max(self.out_channels, 1))
            nn.init.uniform_(self.rec.weight, -rec_scale, rec_scale)

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"ann", "snn"}:
            raise ValueError(f"Invalid Hagenaars FireNet-LIF mode: {mode!r}")
        self._mode = mode

    def reset_state(self) -> None:
        reset = getattr(self.node, "reset", None)
        if callable(reset):
            reset()

    def _zero_state(self, drive: torch.Tensor) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros_like(drive)
        return {"mem": zeros, "spike": torch.zeros_like(zeros)}

    def _normalize_state(self, drive: torch.Tensor, prev_state: Optional[Any]) -> Dict[str, torch.Tensor]:
        if isinstance(prev_state, dict):
            mem = prev_state.get("mem")
            spike = prev_state.get("spike")
        elif torch.is_tensor(prev_state):
            mem = prev_state
            spike = None
        else:
            mem = None
            spike = None

        if (
            mem is None
            or not torch.is_tensor(mem)
            or mem.shape[0] != drive.shape[0]
            or mem.shape[1] != drive.shape[1]
            or mem.shape[2:] != drive.shape[2:]
        ):
            return self._zero_state(drive)

        mem = mem.to(device=drive.device, dtype=drive.dtype)
        if spike is None or not torch.is_tensor(spike) or tuple(spike.shape) != tuple(mem.shape):
            spike = torch.zeros_like(mem)
        else:
            spike = spike.to(device=drive.device, dtype=drive.dtype)
        return {"mem": mem, "spike": spike}

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Any],
        *,
        residual: torch.Tensor | int | float = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        drive = self.ff(x)
        state = self._normalize_state(drive, prev_state)
        if self.rec is not None:
            drive = drive + self.rec(state["spike"])

        if self._mode == "snn":
            self.node.v = state["mem"]
            out = self.node(drive)
            mem = self.node.v
            self.reset_state()
        else:
            mem = state["mem"] + drive
            out = torch.relu(mem)

        if not isinstance(residual, (int, float)) or float(residual) != 0.0:
            out = out + residual
        return out, {"mem": mem, "spike": out}


class SOFEHagenaarsFireNetLIFCore(nn.Module):
    """Original FireNet-LIF full-resolution recurrent tower.

    Topology:
        E1 -> G1 -> E2 -> E3 -> G2 -> E4 -> E5
    with 3x3 convolutions, 32 hidden channels by default, no downsampling, and
    recurrent state carried explicitly by the streaming SOFE wrapper.
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
        del act, alpha
        channels = int(spec.get("base_channels", 32))
        kernel_size = int(spec.get("kernel_size", 3))
        threshold = float(spec.get("spiking_v_threshold", theta))
        self.spike_decay_mode = str(spec.get("firenet_spike_decay_mode", "hold")).lower()
        if self.spike_decay_mode not in {"hold", "decay", "zero"}:
            raise ValueError(
                f"Invalid firenet_spike_decay_mode={self.spike_decay_mode!r}; "
                "expected 'hold', 'decay', or 'zero'."
            )

        common = dict(
            kernel_size=kernel_size,
            threshold=threshold,
            surrogate_function=str(spec.get("surrogate_function", "atan")),
            surrogate_alpha=float(spec.get("surrogate_alpha", 1.0)),
            detach_reset=bool(spec.get("detach_reset", True)),
            bias=bool(spec.get("hagenaars_use_bias", True)),
        )
        self.E1 = _HagenaarsConvLIF(int(input_channels), channels, recurrent=False, **common)
        self.G1 = _HagenaarsConvLIF(channels, channels, recurrent=True, **common)
        self.E2 = _HagenaarsConvLIF(channels, channels, recurrent=False, **common)
        self.E3 = _HagenaarsConvLIF(channels, channels, recurrent=False, **common)
        self.G2 = _HagenaarsConvLIF(channels, channels, recurrent=True, **common)
        self.E4 = _HagenaarsConvLIF(channels, channels, recurrent=False, **common)
        self.E5 = _HagenaarsConvLIF(channels, channels, recurrent=False, **common)

        self.latent_channels = channels
        self.pyramid_channels = [channels]

    def set_mode(self, mode: str) -> None:
        self._mode = str(mode)
        for module in self.modules():
            if module is self:
                continue
            child_set_mode = getattr(module, "set_mode", None)
            if callable(child_set_mode):
                child_set_mode(self._mode)

    def reset_state(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            child_reset = getattr(module, "reset_state", None)
            if callable(child_reset):
                child_reset()

    def _prev(self, state: Dict[str, Any], name: str, decay: Optional[torch.Tensor]) -> Optional[Any]:
        return _apply_decay_to_state(
            state.get(name),
            decay,
            spike_decay_mode=self.spike_decay_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}

        x1, next_state["E1"] = self.E1(x, self._prev(prev_state, "E1", decay))
        x2, next_state["G1"] = self.G1(x1, self._prev(prev_state, "G1", decay))
        x3, next_state["E2"] = self.E2(x2, self._prev(prev_state, "E2", decay))
        x4, next_state["E3"] = self.E3(x3, self._prev(prev_state, "E3", decay))
        x5, next_state["G2"] = self.G2(x4, self._prev(prev_state, "G2", decay))
        x6, next_state["E4"] = self.E4(x5, self._prev(prev_state, "E4", decay))
        x7, next_state["E5"] = self.E5(x6, self._prev(prev_state, "E5", decay))

        return {"latent": x7, "pyramid": [x7]}, next_state


class SOFEHagenaarsFireNetLIFReadout(nn.Module):
    """Real-valued dense optical-flow readout for Hagenaars FireNet-LIF."""

    def __init__(
        self,
        *,
        in_channels: int,
        flow_activation: str = "tanh",
        flow_head_init_std: float = 0.01,
        pred_kernel_size: int = 3,
    ):
        super().__init__()
        kernel_size = int(pred_kernel_size)
        self.pred = nn.Conv2d(
            int(in_channels),
            2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.flow_activation = str(flow_activation).lower()
        nn.init.xavier_uniform_(self.pred.weight, gain=float(flow_head_init_std))
        if self.pred.bias is not None:
            nn.init.zeros_(self.pred.bias)

    def _activate(self, flow_logits: torch.Tensor) -> torch.Tensor:
        if self.flow_activation == "tanh":
            return torch.tanh(flow_logits)
        if self.flow_activation == "softsign":
            return F.softsign(flow_logits)
        if self.flow_activation in {"identity", "none", "linear"}:
            return flow_logits
        raise NotImplementedError(
            f"Hagenaars FireNet-LIF flow_activation={self.flow_activation!r} is not implemented."
        )

    def forward(self, features: Dict[str, Any], *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = features["latent"]
        flow_logits = self.pred(x)
        flow_low = self._activate(flow_logits)
        scale_x = 1.0
        scale_y = 1.0
        if tuple(flow_low.shape[2:]) != tuple(output_size):
            low_h, low_w = flow_low.shape[2:]
            out_h, out_w = output_size
            flow = F.interpolate(flow_low, size=output_size, mode="bilinear", align_corners=False)
            scale_x = float(out_w) / float(low_w)
            scale_y = float(out_h) / float(low_h)
            flow[:, 0] = flow[:, 0] * scale_x
            flow[:, 1] = flow[:, 1] * scale_y
        else:
            flow = flow_low
        return flow, {
            "flow_logits": flow_logits,
            "flow_low": flow_low.detach(),
            "flow_full": flow.detach(),
            "flow_upsample_scale_x": float(scale_x),
            "flow_upsample_scale_y": float(scale_y),
        }
