"""Xu et al 2025 based FireNet-LIF SOFE architecture.

This module keeps the reference-style FireNet-LIF recurrent tower separate
from the generic SOFE architecture builder. Runtime state remains explicit so
the streaming scheduler can reset, detach, and mask sequence boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FireNetArctanSpike(torch.autograd.Function):
    """Reference ANN_vs_SNN_evflow arctan spike surrogate."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, width)
        return (x > 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x, width = ctx.saved_tensors
        return grad_output * (1.0 / (1.0 + width * x * x)), None


def _firenet_arctan_spike(
    membrane: torch.Tensor,
    threshold: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    return _FireNetArctanSpike.apply(membrane - threshold, width)


def _as_firenet_pair(value: Any, *, default: Tuple[float, float], name: str) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"FireNet-LIF {name} must be a two-element sequence, got {value!r}")


class _FireNetConvLIF(nn.Module):
    """Reference-compatible ConvLIF block from ANN_vs_SNN_evflow."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        *,
        act_width: float,
        leak: Sequence[float],
        thresh: Sequence[float],
        learn_leak: bool,
        learn_thresh: bool,
        hard_reset: bool,
        detach_reset: bool,
        recurrent: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.recurrent = bool(recurrent)
        self.hard_reset = bool(hard_reset)
        self.detach_reset = bool(detach_reset)
        self._mode = "snn"

        padding = int(kernel_size) // 2
        self.ff = nn.Conv2d(
            self.input_size,
            self.hidden_size,
            int(kernel_size),
            padding=padding,
            bias=False,
        )
        if self.recurrent:
            self.rec = nn.Conv2d(
                self.hidden_size,
                self.hidden_size,
                int(kernel_size),
                padding=padding,
                bias=False,
            )
        else:
            self.rec = None

        leak_mu, leak_sigma = _as_firenet_pair(leak, default=(-4.0, 0.1), name="leak")
        thresh_mu, thresh_sigma = _as_firenet_pair(thresh, default=(0.8, 0.0), name="thresh")
        leak_init = torch.randn(self.hidden_size, 1, 1) * leak_sigma + leak_mu
        thresh_init = torch.randn(self.hidden_size, 1, 1) * thresh_sigma + thresh_mu
        if learn_leak:
            self.leak = nn.Parameter(leak_init)
        else:
            self.register_buffer("leak", leak_init)
        if learn_thresh:
            self.thresh = nn.Parameter(thresh_init)
        else:
            self.register_buffer("thresh", thresh_init)
        self.register_buffer("act_width", torch.tensor(float(act_width)), persistent=False)

        w_scale_ff = math.sqrt(1.0 / max(self.input_size, 1))
        nn.init.uniform_(self.ff.weight, -w_scale_ff, w_scale_ff)
        if self.rec is not None:
            w_scale_rec = math.sqrt(1.0 / max(self.hidden_size, 1))
            nn.init.uniform_(self.rec.weight, -w_scale_rec, w_scale_rec)

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"ann", "snn"}:
            raise ValueError(f"Invalid FireNet-LIF mode: {mode!r}")
        self._mode = mode

    def reset_state(self) -> None:
        return None

    def _zero_state(self, ff: torch.Tensor) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros_like(ff)
        return {"v": zeros, "spike": torch.zeros_like(zeros)}

    def _normalize_state(self, ff: torch.Tensor, prev_state: Optional[Any]) -> Dict[str, torch.Tensor]:
        if isinstance(prev_state, dict):
            v = prev_state.get("v")
            spike = prev_state.get("spike")
        elif torch.is_tensor(prev_state) and prev_state.ndim >= 5 and int(prev_state.shape[0]) >= 2:
            v = prev_state[0]
            spike = prev_state[1]
        else:
            v = None
            spike = None

        if v is None or not torch.is_tensor(v) or tuple(v.shape) != tuple(ff.shape):
            return self._zero_state(ff)

        v = v.to(device=ff.device, dtype=ff.dtype)
        if spike is None or not torch.is_tensor(spike) or tuple(spike.shape) != tuple(ff.shape):
            spike = torch.zeros_like(ff)
        else:
            spike = spike.to(device=ff.device, dtype=ff.dtype)
        return {"v": v, "spike": spike}

    def forward(
        self,
        input_: torch.Tensor,
        prev_state: Optional[Any],
        *,
        residual: torch.Tensor | int | float = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ff = self.ff(input_)
        state = self._normalize_state(ff, prev_state)
        v = state["v"]
        z = state["spike"]

        drive = ff
        if self.rec is not None:
            drive = drive + self.rec(z)

        threshold = self.thresh.clamp_min(0.01).to(device=ff.device, dtype=ff.dtype)
        leak = torch.sigmoid(self.leak).to(device=ff.device, dtype=ff.dtype)
        z_for_reset = z.detach() if self.detach_reset else z
        if self.hard_reset:
            v_out = v * leak * (1.0 - z_for_reset) + (1.0 - leak) * drive
        else:
            v_out = v * leak + (1.0 - leak) * drive - z_for_reset * threshold

        if self._mode == "snn":
            out = _firenet_arctan_spike(
                v_out,
                threshold,
                self.act_width.to(device=ff.device, dtype=ff.dtype),
            )
        else:
            out = torch.relu(v_out)

        if not isinstance(residual, (int, float)) or float(residual) != 0.0:
            out = out + residual
        return out, {"v": v_out, "spike": out}


class _FireNetPredictor(nn.Module):
    """Reference FireNet predictor: 1x1 conv followed by Softsign."""

    def __init__(self, in_channels: int, *, w_scale: float = 0.01):
        super().__init__()
        self.conv2d = nn.Conv2d(int(in_channels), 2, kernel_size=1, bias=True)
        nn.init.uniform_(self.conv2d.weight, -float(w_scale), float(w_scale))
        if self.conv2d.bias is not None:
            nn.init.zeros_(self.conv2d.bias)
        self.softsign = nn.Softsign()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.conv2d(x)
        return self.softsign(logits), logits


def _apply_decay_to_hidden(
    hidden: Optional[Any],
    decay: Optional[torch.Tensor],
    *,
    spike_decay_mode: str = "hold",
) -> Optional[Any]:
    if hidden is None or decay is None:
        return hidden
    if torch.is_tensor(hidden):
        decay_view = decay.view(-1, *([1] * max(hidden.ndim - 1, 0))).to(device=hidden.device, dtype=hidden.dtype)
        return hidden * decay_view
    if isinstance(hidden, tuple):
        return tuple(_apply_decay_to_hidden(v, decay, spike_decay_mode=spike_decay_mode) for v in hidden)
    if isinstance(hidden, list):
        return [_apply_decay_to_hidden(v, decay, spike_decay_mode=spike_decay_mode) for v in hidden]
    if isinstance(hidden, dict):
        decayed: Dict[str, Any] = {}
        for key, value in hidden.items():
            if key == "spike":
                if spike_decay_mode == "hold":
                    decayed[key] = value
                elif spike_decay_mode == "decay":
                    decayed[key] = _apply_decay_to_hidden(value, decay, spike_decay_mode=spike_decay_mode)
                elif spike_decay_mode == "zero":
                    decayed[key] = torch.zeros_like(value) if torch.is_tensor(value) else value
                else:
                    raise ValueError(
                        f"Invalid FireNet-LIF spike_decay_mode={spike_decay_mode!r}; "
                        "expected 'hold', 'decay', or 'zero'."
                    )
            else:
                decayed[key] = _apply_decay_to_hidden(value, decay, spike_decay_mode=spike_decay_mode)
        return decayed
    return hidden


class SOFEFireNetLIFCore(nn.Module):
    """Reference-style LIFFireNet recurrent tower for streaming SOFE.

    The layer names mirror ANN_vs_SNN_evflow's LIFFireNet so model summaries
    and future key remapping stay straightforward: head, G1, R1a, R1b, G2,
    R2a, R2b.
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
        del act, alpha, theta
        base_channels = int(spec["base_channels"])
        kernel_size = int(spec.get("kernel_size", 3))
        self.input_downsample = int(spec.get("firenet_input_downsample", 1) or 1)
        self.downsample_mode = str(spec.get("firenet_downsample_mode", "sum")).lower()
        self.spike_decay_mode = str(spec.get("firenet_spike_decay_mode", "hold")).lower()
        if self.spike_decay_mode not in {"hold", "decay", "zero"}:
            raise ValueError(
                f"Invalid FireNet-LIF spike_decay_mode={self.spike_decay_mode!r}; "
                "expected 'hold', 'decay', or 'zero'."
            )
        common = dict(
            kernel_size=kernel_size,
            act_width=float(spec.get("firenet_act_width", 10.0)),
            leak=spec.get("firenet_leak", [-4.0, 0.1]),
            thresh=spec.get("firenet_thresh", [0.8, 0.0]),
            learn_leak=bool(spec.get("firenet_learn_leak", True)),
            learn_thresh=bool(spec.get("firenet_learn_thresh", True)),
            hard_reset=bool(spec.get("firenet_hard_reset", True)),
            detach_reset=bool(spec.get("detach_reset", True)),
        )

        self.head = _FireNetConvLIF(
            int(input_channels),
            base_channels,
            recurrent=False,
            **common,
        )
        self.G1 = _FireNetConvLIF(base_channels, base_channels, recurrent=True, **common)
        self.R1a = _FireNetConvLIF(base_channels, base_channels, recurrent=False, **common)
        self.R1b = _FireNetConvLIF(base_channels, base_channels, recurrent=False, **common)
        self.G2 = _FireNetConvLIF(base_channels, base_channels, recurrent=True, **common)
        self.R2a = _FireNetConvLIF(base_channels, base_channels, recurrent=False, **common)
        self.R2b = _FireNetConvLIF(base_channels, base_channels, recurrent=False, **common)

        self.latent_channels = base_channels
        self.pyramid_channels = [base_channels]

    def _downsample_input(self, x: torch.Tensor) -> torch.Tensor:
        factor = int(self.input_downsample)
        if factor <= 1:
            return x

        if self.downsample_mode == "sum":
            return F.avg_pool2d(x, kernel_size=factor, stride=factor) * float(factor * factor)
        if self.downsample_mode == "avg":
            return F.avg_pool2d(x, kernel_size=factor, stride=factor)
        if self.downsample_mode == "max":
            return F.max_pool2d(x, kernel_size=factor, stride=factor)
        raise ValueError(
            f"Invalid FireNet-LIF downsample_mode={self.downsample_mode!r}; "
            "expected 'sum', 'avg', or 'max'."
        )

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
        x = self._downsample_input(x)
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}

        x1, next_state["head"] = self.head(
            x,
            _apply_decay_to_hidden(
                prev_state.get("head"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x2, next_state["G1"] = self.G1(
            x1,
            _apply_decay_to_hidden(
                prev_state.get("G1"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x3, next_state["R1a"] = self.R1a(
            x2,
            _apply_decay_to_hidden(
                prev_state.get("R1a"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x4, next_state["R1b"] = self.R1b(
            x3,
            _apply_decay_to_hidden(
                prev_state.get("R1b"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x5, next_state["G2"] = self.G2(
            x4,
            _apply_decay_to_hidden(
                prev_state.get("G2"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x6, next_state["R2a"] = self.R2a(
            x5,
            _apply_decay_to_hidden(
                prev_state.get("R2a"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )
        x7, next_state["R2b"] = self.R2b(
            x6,
            _apply_decay_to_hidden(
                prev_state.get("R2b"),
                decay,
                spike_decay_mode=self.spike_decay_mode,
            ),
        )

        return {"latent": x7, "pyramid": [x7]}, next_state


class SOFEFireNetLIFReadout(nn.Module):
    """LIFFireNet predictor head: 1x1 conv with Softsign output."""

    def __init__(
        self,
        *,
        in_channels: int,
        flow_head_init_std: float = 0.01,
    ):
        super().__init__()
        self.pred = _FireNetPredictor(in_channels, w_scale=float(flow_head_init_std))

    def forward(self, features: Dict[str, Any], *, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = features["latent"]
        flow_low, flow_logits = self.pred(x)
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
