"""Streaming SSNN-Net components for SOFE/SMCM.

This module contains the event-native SSNN-Net core and readout used through
``model.mcm.architecture='sofe'`` with
``model.mcm.sofe_architecture.family='ssnn_net'``. The old direct voxel-grid
``architecture='ssnn_net'`` estimator has been removed so this family is only
used through the streaming SOFE partitioning path.

Two architecture versions are selected via ``sofe_architecture.ssnn_version``:

* **v1** (default): the original three-stage encoder described below.
* **v2**: four-stage encoder, dilated bottleneck, membrane inter-stage
  signaling, and a non-spiking high-resolution refinement stage with an
  optional event-density edge tap.

For v1:
* **(A) 2D-recurrent encoder.** The original 3D convolutions treat the temporal
  bin axis as a spatial-style dimension with a kernel of 5. Here each
  partitioned sparse-event step is one recurrent SNN update through a 2D
  spiking encoder, and explicit membrane/spike state carries temporal context.
* **(B) Reduced U-Net depth.** Three downsampling stages (to H/8, 256 ch) instead
  of five (to H/16, 512 ch). Motion compensation only needs flow that is smooth
  and roughly correct; it does not need flow-benchmark accuracy at H/16.
* **(C) Lightweight decoder.** A single skip from the shallowest encoder feature
  (high-frequency edges, which matter most for warping) instead of the full
  cat-skip ladder. Decoder channel counts stay small.
* **(D) Streaming readout.** The readout is owned by SOFE's persistent readout
  wrapper, so flow is emitted from the carried streaming state instead of from
  a one-shot voxel forward.

For v2 (Phase 1 implementation):
* **(A) Four-stage encoder** downsampling to H/16, all stages use
  ``_StreamingSSNNEncoderStage`` which exposes both the emitted signal and the
  membrane for decoder handoff.
* **(B) Dilated bottleneck** at H/16 using dilation rates [1, 2, 4] to expand the
  receptive field without additional downsampling.
* **(C) Configurable inter-stage signal** via ``ssnn_encoder_signal``:
  ``"membrane"`` (default, graded — best for training stability),
  ``"spike"`` (binary — required for strict neuromorphic ablation),
  ``"mixed"`` (straight-through: spike + mem.detach()).
* **(D) Non-spiking decoder** from H/16 to H/4 followed by a
  ``_SSNNHighResRefinement`` stage that combines the decoded H/4 feature,
  the enc1 membrane (naturally at H/2), and an optional event-density edge tap.
* **(E) Pyramid output** ``[bott_h16, dec1_h8, dec2_h4, latent_h2]`` for future
  coarse-to-fine residual readout (Phase 2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplyBy(nn.Module):
    """Scale activations before IF neurons."""

    def __init__(self, scale_value: float = 10.0):
        super().__init__()
        self.scale_value = float(scale_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.scale_value)


class _SSNNSurrogateSpike(torch.autograd.Function):
    """Hard IF spike with a sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, alpha)
        return (x >= 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x, alpha = ctx.saved_tensors
        sig = torch.sigmoid(alpha * x)
        return grad_output * alpha * sig * (1.0 - sig), None


def _ssnn_spike(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return _SSNNSurrogateSpike.apply(x, alpha)


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
                        f"Invalid ssnn_spike_decay_mode={spike_decay_mode!r}; "
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


# ---------------------------------------------------------------------------
# 2D building blocks — shared between v1 and v2
# ---------------------------------------------------------------------------


class SeparableConv2d(nn.Sequential):
    """Depthwise-separable 2D conv: depthwise k×k followed by 1×1 pointwise.

    Supports optional ``dilation`` for use inside the v2 dilated bottleneck.
    With ``dilation > 1`` the depthwise padding is set to
    ``dilation * (kernel_size // 2)`` so the spatial output size is
    preserved (stride must be 1 when dilation > 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        pad = int(dilation) * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size, stride=stride, padding=pad,
                dilation=dilation, groups=in_channels, bias=False,
                padding_mode="replicate",
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class _StreamingSSNNIFBlock(nn.Module):
    """Explicit-state Conv-IF block for SOFE/SMCM streaming execution.

    The ``dilation`` parameter is forwarded to ``SeparableConv2d`` and is used
    by the v2 dilated bottleneck.  All encoder and refinement usages keep the
    default ``dilation=1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        multiply_factor: float = 1.0,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
        hard_reset: bool = False,
        detach_reset: bool = True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            SeparableConv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=int(stride),
                dilation=int(dilation),
            ),
            MultiplyBy(float(multiply_factor)),
        )
        self.threshold = float(threshold)
        self.hard_reset = bool(hard_reset)
        self.detach_reset = bool(detach_reset)
        self._mode = "snn"
        self.register_buffer(
            "surrogate_alpha",
            torch.tensor(float(surrogate_alpha), dtype=torch.float32),
            persistent=False,
        )

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"ann", "snn"}:
            raise ValueError(f"Invalid SSNN mode: {mode!r}")
        self._mode = mode

    def reset_state(self) -> None:
        return None

    @staticmethod
    def _zero_state(drive: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        drive = self.conv(x)
        state = self._normalize_state(drive, prev_state)

        if self._mode == "ann":
            mem = state["mem"] + drive
            out = F.relu(mem)
            return out, {"mem": mem, "spike": out}

        mem = state["mem"] + drive
        threshold = torch.as_tensor(self.threshold, device=mem.device, dtype=mem.dtype)
        spike = _ssnn_spike(
            mem - threshold,
            self.surrogate_alpha.to(device=mem.device, dtype=mem.dtype),
        )
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.hard_reset:
            mem_next = mem * (1.0 - spike_for_reset)
        else:
            mem_next = mem - spike_for_reset * threshold
        return spike, {"mem": mem_next, "spike": spike}


class _StreamingSSNNResBlock(nn.Module):
    """Residual explicit-state SSNN block used at the v1 bottleneck."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        multiply_factor: float,
        threshold: float,
        surrogate_alpha: float,
        hard_reset: bool,
        detach_reset: bool,
    ):
        super().__init__()
        common = dict(
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
        )
        self.block1 = _StreamingSSNNIFBlock(channels, channels, **common)
        self.block2 = _StreamingSSNNIFBlock(channels, channels, **common)

    def set_mode(self, mode: str) -> None:
        self.block1.set_mode(mode)
        self.block2.set_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prev_state = dict(state or {})
        y, s1 = self.block1(x, prev_state.get("block1"))
        y, s2 = self.block2(y, prev_state.get("block2"))
        return y + x, {"block1": s1, "block2": s2}


# ---------------------------------------------------------------------------
# v2-specific building blocks
# ---------------------------------------------------------------------------


class _StreamingSSNNEncoderStage(nn.Module):
    """Single strided encoder stage for the v2 SSNN core.

    Wraps ``_StreamingSSNNIFBlock`` and controls what signal propagates to the
    next encoder stage via ``encoder_signal``:

    * ``"spike"``    — binary spike output.  Required for strict neuromorphic
      deployment fidelity; harder to train.
    * ``"membrane"`` — (default) graded membrane potential.  Makes every
      inter-stage connection graded, greatly easing gradient flow and
      convergence.  This is the recommended default for motion-compensation
      quality experiments.
    * ``"mixed"``    — ``spike + mem.detach()``.  Straight-through
      approximation: the forward path sees a graded signal while backward
      receives gradient primarily through the spike branch.  Do not expose
      this to the EA until hand ablations confirm it is useful.

    The membrane is always returned separately for decoder skip/handoff use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        encoder_signal: str,
        kernel_size: int,
        stride: int = 2,
        multiply_factor: float,
        threshold: float,
        surrogate_alpha: float,
        hard_reset: bool,
        detach_reset: bool,
    ):
        super().__init__()
        self.block = _StreamingSSNNIFBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            multiply_factor=multiply_factor,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
        )
        self.encoder_signal = str(encoder_signal).lower()
        if self.encoder_signal not in {"spike", "membrane", "mixed"}:
            raise ValueError(
                f"encoder_signal={self.encoder_signal!r}; "
                "expected 'spike', 'membrane', or 'mixed'."
            )

    def set_mode(self, mode: str) -> None:
        self.block.set_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (next_stage_input, membrane, next_state)."""
        spike, next_state = self.block(x, prev_state)
        mem = next_state["mem"]
        if self.encoder_signal == "spike":
            next_input = spike
        elif self.encoder_signal == "membrane":
            next_input = mem
        else:
            # mixed: straight-through — graded forward, spike-dominated backward
            next_input = spike + mem.detach()
        return next_input, mem, next_state


class _StreamingSSNNDilatedBottleneck(nn.Module):
    """Multi-dilation residual bottleneck for the v2 SSNN core.

    Runs a sequence of ``_StreamingSSNNIFBlock`` layers with increasing
    dilation rates so that the effective receptive field grows without
    adding another spatial downsampling level.  A residual connection adds
    the input to the final block output, matching ``_StreamingSSNNResBlock``
    semantics but with configurable dilation.

    Each dilation index carries its own membrane/spike state keyed by
    str(index) in the returned state dict, so that
    ``_apply_decay_to_state`` can apply independent decay per block.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: List[int],
        multiply_factor: float,
        threshold: float,
        surrogate_alpha: float,
        hard_reset: bool,
        detach_reset: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _StreamingSSNNIFBlock(
                channels, channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=int(d),
                multiply_factor=multiply_factor,
                threshold=threshold,
                surrogate_alpha=surrogate_alpha,
                hard_reset=hard_reset,
                detach_reset=detach_reset,
            )
            for d in dilations
        ])

    def set_mode(self, mode: str) -> None:
        for block in self.blocks:
            block.set_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prev = dict(state or {})
        next_s: Dict[str, Any] = {}
        out = x
        for i, block in enumerate(self.blocks):
            out, ns = block(out, prev.get(str(i)))
            next_s[str(i)] = ns
        return out + x, next_s


class _SSNNHighResRefinement(nn.Module):
    """Non-spiking high-resolution refinement for the v2 SSNN core.

    Takes the upsampled H/4 decoder feature, the enc1 membrane at H/2, and an
    optional event-density edge tap at H/2, fuses them through a small
    non-spiking conv stack, and returns a latent feature at H/2 that is fed
    to the SOFE readout head.

    Keeping this stage non-spiking is intentional: the temporal recurrence
    requirement is satisfied by the spiking encoder path.  The refinement's
    only job is to snap the smooth coarse features onto high-frequency edge
    structure, and that does not benefit from IF quantisation.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        refine_channels: int,
        num_layers: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        c_in = int(in_channels)
        for _ in range(max(int(num_layers), 1)):
            layers.append(SeparableConv2d(c_in, int(refine_channels), kernel_size))
            layers.append(nn.SiLU(inplace=True))
            c_in = int(refine_channels)
        self.net = nn.Sequential(*layers)
        self.out_channels = int(refine_channels)

    def forward(
        self,
        dec_h4: torch.Tensor,
        enc1_mem: torch.Tensor,
        edge_tap: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target_size = enc1_mem.shape[2:]
        if tuple(dec_h4.shape[2:]) != tuple(target_size):
            dec_h4 = F.interpolate(dec_h4, size=target_size, mode="bilinear", align_corners=False)
        parts = [dec_h4, enc1_mem]
        if edge_tap is not None:
            if tuple(edge_tap.shape[2:]) != tuple(target_size):
                edge_tap = F.interpolate(edge_tap, size=target_size, mode="nearest")
            parts.append(edge_tap)
        return self.net(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
# Main public classes
# ---------------------------------------------------------------------------


class SOFESSNNNetCore(nn.Module):
    """Streaming SSNN core for ``StreamingMotionCompensationModule``.

    Dispatches between v1 (legacy three-stage encoder) and v2 (four-stage
    encoder with dilated bottleneck and high-res refinement) based on
    ``spec["ssnn_version"]``.  The public contract — ``forward`` returning
    ``(features_dict, next_state_dict)`` and ``latent_channels`` /
    ``pyramid_channels`` attributes — is identical for both versions so that
    the surrounding SOFE readout and scheduler remain unchanged.

    v1 state keys: ``stem``, ``enc1``, ``enc2``, ``enc3``, ``bottleneck``
    v2 state keys: ``enc1``, ``enc2``, ``enc3``, ``enc4``, ``bottleneck``
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

        self.version = str(spec.get("ssnn_version", "v1")).lower()
        if self.version not in {"v1", "v2"}:
            raise ValueError(
                f"ssnn_version={self.version!r} is not supported; expected 'v1' or 'v2'."
            )

        # Shared hyperparameters
        base_channels = int(spec.get("base_channels", 32))
        kernel_size = int(spec.get("kernel_size", 3))
        multiply_factor = float(spec.get("ssnn_multiply_factor", spec.get("multiply_factor", 10.0)))
        threshold = float(spec.get("spiking_v_threshold", theta))
        surrogate_alpha = float(spec.get("surrogate_alpha", 4.0))
        hard_reset = bool(spec.get("ssnn_hard_reset", False))
        detach_reset = bool(spec.get("detach_reset", True))
        self.spike_decay_mode = str(spec.get("ssnn_spike_decay_mode", "hold")).lower()
        if self.spike_decay_mode not in {"hold", "decay", "zero"}:
            raise ValueError(
                f"Invalid ssnn_spike_decay_mode={self.spike_decay_mode!r}; "
                "expected 'hold', 'decay', or 'zero'."
            )

        common = dict(
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
        )

        if self.version == "v1":
            self._build_v1(spec, input_channels, base_channels, kernel_size, common)
        else:
            self._build_v2(spec, input_channels, base_channels, kernel_size, common)

        self._mode = "snn"

    # ------------------------------------------------------------------
    # v1 build
    # ------------------------------------------------------------------

    def _build_v1(
        self,
        spec: Dict[str, Any],
        input_channels: int,
        base_channels: int,
        kernel_size: int,
        common: Dict[str, Any],
    ) -> None:
        c = base_channels
        self.stem = _StreamingSSNNIFBlock(int(input_channels), c, kernel_size=5, **{
            k: v for k, v in common.items() if k != "kernel_size"
        })
        self.enc1 = _StreamingSSNNIFBlock(c, c * 2, stride=2, **common)
        self.enc2 = _StreamingSSNNIFBlock(c * 2, c * 4, stride=2, **common)
        self.enc3 = _StreamingSSNNIFBlock(c * 4, c * 8, stride=2, **common)
        self.bottleneck = _StreamingSSNNResBlock(c * 8, **common)
        self.dec1 = nn.Sequential(
            SeparableConv2d(c * 8, c * 4, kernel_size),
            nn.SiLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            SeparableConv2d(c * 4, c * 2, kernel_size),
            nn.SiLU(inplace=True),
        )
        # H/2 decoder feature + H/2 enc1 membrane + pooled full-res stem membrane.
        self.fuse = nn.Sequential(
            SeparableConv2d(c * 2 + c * 2 + c, c * 2, kernel_size),
            nn.SiLU(inplace=True),
        )
        self.latent_channels = c * 2
        self.pyramid_channels = [self.latent_channels]

    # ------------------------------------------------------------------
    # v2 build
    # ------------------------------------------------------------------

    def _build_v2(
        self,
        spec: Dict[str, Any],
        input_channels: int,
        base_channels: int,
        kernel_size: int,
        common: Dict[str, Any],
    ) -> None:
        # Stage channel sizes: explicit list or auto-derived from base_channels.
        raw_stage_ch = spec.get("ssnn_stage_channels")
        if (
            raw_stage_ch is not None
            and isinstance(raw_stage_ch, (list, tuple))
            and len(raw_stage_ch) >= 4
        ):
            stage_ch = [int(c) for c in raw_stage_ch[:4]]
        else:
            c = base_channels
            stage_ch = [c, c * 2, c * 4, c * 8]

        raw_dilations = spec.get("ssnn_bottleneck_dilations", [1, 2, 4])
        if not isinstance(raw_dilations, (list, tuple)):
            raw_dilations = [1, 2, 4]
        bottleneck_dilations = [int(d) for d in raw_dilations]

        self.encoder_signal = str(spec.get("ssnn_encoder_signal", "membrane")).lower()
        if self.encoder_signal not in {"spike", "membrane", "mixed"}:
            raise ValueError(
                f"ssnn_encoder_signal={self.encoder_signal!r}; "
                "expected 'spike', 'membrane', or 'mixed'."
            )

        self.edge_tap = str(spec.get("ssnn_edge_tap", "hist")).lower()
        if self.edge_tap not in {"hist", "none"}:
            raise ValueError(f"ssnn_edge_tap={self.edge_tap!r}; expected 'hist' or 'none'.")

        refine_channels = int(spec.get("ssnn_refine_channels", 32))
        refine_layers = int(spec.get("ssnn_refine_layers", 2))

        enc_kw = dict(encoder_signal=self.encoder_signal, **common)
        in_c = int(input_channels)

        # Four spiking encoder stages, each downsampling by 2.
        self.enc1 = _StreamingSSNNEncoderStage(in_c, stage_ch[0], stride=2, **enc_kw)
        self.enc2 = _StreamingSSNNEncoderStage(stage_ch[0], stage_ch[1], stride=2, **enc_kw)
        self.enc3 = _StreamingSSNNEncoderStage(stage_ch[1], stage_ch[2], stride=2, **enc_kw)
        self.enc4 = _StreamingSSNNEncoderStage(stage_ch[2], stage_ch[3], stride=2, **enc_kw)

        # Dilated spiking bottleneck at H/16.
        self.bottleneck = _StreamingSSNNDilatedBottleneck(
            stage_ch[3], dilations=bottleneck_dilations, **common
        )

        # Non-spiking lightweight decoder: H/16 → H/8 → H/4.
        self.dec1 = nn.Sequential(
            SeparableConv2d(stage_ch[3], stage_ch[2], kernel_size),
            nn.SiLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            SeparableConv2d(stage_ch[2], stage_ch[1], kernel_size),
            nn.SiLU(inplace=True),
        )

        # High-resolution refinement: dec2_up (H/2) + enc1_mem (H/2) + edge_tap → latent.
        edge_tap_ch = 1 if self.edge_tap == "hist" else 0
        refine_in_ch = stage_ch[1] + stage_ch[0] + edge_tap_ch
        self.refinement = _SSNNHighResRefinement(
            in_channels=refine_in_ch,
            refine_channels=refine_channels,
            num_layers=refine_layers,
            kernel_size=kernel_size,
        )

        self.latent_channels = refine_channels
        # Ordered coarse-to-fine: H/16, H/8, H/4, H/2 — for Phase 2 pyramid readout.
        self.pyramid_channels = [stage_ch[3], stage_ch[2], stage_ch[1], refine_channels]
        self._v2_stage_channels = stage_ch

    # ------------------------------------------------------------------
    # Shared interface
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = str(mode).lower()
        for module in self.modules():
            if module is self:
                continue
            child_set_mode = getattr(module, "set_mode", None)
            if callable(child_set_mode):
                child_set_mode(self._mode)

    def reset_state(self) -> None:
        return None

    def _prev(self, state: Dict[str, Any], name: str, decay: Optional[torch.Tensor]) -> Optional[Any]:
        return _apply_decay_to_state(
            state.get(name),
            decay,
            spike_decay_mode=self.spike_decay_mode,
        )

    @staticmethod
    def _mem(state: Dict[str, torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        mem = state.get("mem") if isinstance(state, dict) else None
        return mem if torch.is_tensor(mem) else fallback

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.version == "v1":
            return self._forward_v1(x, state, decay)
        return self._forward_v2(x, state, decay)

    # ------------------------------------------------------------------
    # v1 forward (original implementation, unchanged)
    # ------------------------------------------------------------------

    def _forward_v1(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}

        stem_spk, next_state["stem"] = self.stem(x, self._prev(prev_state, "stem", decay))
        enc1_spk, next_state["enc1"] = self.enc1(stem_spk, self._prev(prev_state, "enc1", decay))
        enc2_spk, next_state["enc2"] = self.enc2(enc1_spk, self._prev(prev_state, "enc2", decay))
        enc3_spk, next_state["enc3"] = self.enc3(enc2_spk, self._prev(prev_state, "enc3", decay))
        bott, next_state["bottleneck"] = self.bottleneck(
            enc3_spk,
            self._prev(prev_state, "bottleneck", decay),
        )

        # Decoder uses integrated membrane features, not just the final sparse spike.
        stem_mem = self._mem(next_state["stem"], stem_spk)
        enc1_mem = self._mem(next_state["enc1"], enc1_spk)
        bott_mem = bott

        d = F.interpolate(bott_mem, scale_factor=2.0, mode="nearest")
        d = self.dec1(d)
        d = F.interpolate(d, scale_factor=2.0, mode="nearest")
        d = self.dec2(d)
        stem_skip = F.avg_pool2d(stem_mem, kernel_size=2, stride=2)
        if enc1_mem.shape[-2:] != d.shape[-2:]:
            enc1_mem = F.interpolate(enc1_mem, size=d.shape[-2:], mode="nearest")
        if stem_skip.shape[-2:] != d.shape[-2:]:
            stem_skip = F.interpolate(stem_skip, size=d.shape[-2:], mode="nearest")
        latent = self.fuse(torch.cat((d, enc1_mem, stem_skip), dim=1))
        return {"latent": latent, "pyramid": [latent]}, next_state

    # ------------------------------------------------------------------
    # v2 forward
    # ------------------------------------------------------------------

    def _forward_v2(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, Any]],
        decay: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prev_state = dict(state or {})
        next_state: Dict[str, Any] = {}

        # Event-density edge tap at full resolution, downsampled to H/2.
        # Provides high-frequency event-location structure to the refinement
        # stage without passing through the lossy spiking encoder.
        if self.edge_tap == "hist":
            edge_h2 = F.avg_pool2d(x.abs().sum(dim=1, keepdim=True), kernel_size=2, stride=2)
        else:
            edge_h2 = None

        # Four spiking encoder stages: H → H/2 → H/4 → H/8 → H/16.
        e1, enc1_mem, next_state["enc1"] = self.enc1(x, self._prev(prev_state, "enc1", decay))
        e2, _enc2_mem, next_state["enc2"] = self.enc2(e1, self._prev(prev_state, "enc2", decay))
        e3, _enc3_mem, next_state["enc3"] = self.enc3(e2, self._prev(prev_state, "enc3", decay))
        e4, _enc4_mem, next_state["enc4"] = self.enc4(e3, self._prev(prev_state, "enc4", decay))

        # Dilated spiking bottleneck at H/16.
        bott, next_state["bottleneck"] = self.bottleneck(
            e4, self._prev(prev_state, "bottleneck", decay)
        )

        # Non-spiking decoder: H/16 → H/8 → H/4.
        d1 = F.interpolate(bott, scale_factor=2.0, mode="nearest")
        d1 = self.dec1(d1)
        d2 = F.interpolate(d1, scale_factor=2.0, mode="nearest")
        d2 = self.dec2(d2)

        # High-resolution refinement: d2 upsampled to H/2 + enc1_mem (H/2) + edge_tap → latent.
        latent = self.refinement(d2, enc1_mem, edge_h2)

        # Pyramid is coarse-to-fine: bott(H/16), d1(H/8), d2(H/4), latent(H/2).
        # Phase 2 will attach per-level residual readout heads to these.
        pyramid = [bott, d1, d2, latent]
        return {
            "latent": latent,
            "pyramid": pyramid,
            "edge": edge_h2,
            "debug": {"encoder_signal": self.encoder_signal},
        }, next_state


class SOFESSNNNetPyramidReadout(nn.Module):
    """Coarse-to-fine residual pyramid readout for the v2 SSNN core (Phase 2).

    Consumes ``features["pyramid"]`` produced by ``SOFESSNNNetCore`` v2:
    ``[bott_h16, dec1_h8, dec2_h4, latent_h2]``.

    Prediction is coarse-to-fine residual:

    * H/16 head predicts an absolute coarse flow (bounded).
    * H/8, H/4, H/2 heads each predict a residual correction on top of the
      upsampled previous level.  Upsampling also multiplies the flow values by
      the spatial scale factor so that all intermediate flows are expressed in
      their own level's pixel units.

    The final H/2 flow is upsampled to ``output_size`` and multiplied by the
    corresponding spatial scale, matching the convention in
    ``SOFESSNNNetReadout``.

    Auxiliary pyramid flows (``flow_h16``, ``flow_h8``, ``flow_h4``,
    ``flow_h2``) are returned live (not detached) in the aux dict so that
    supervised EPE losses can be computed against downsampled GT flow in
    ``StreamingMotionCompensationModule.compute_loss``.
    """

    def __init__(
        self,
        *,
        pyramid_channels: List[int],
        flow_activation: str = "softsign",
        flow_head_init_std: float = 0.01,
        pred_kernel_size: int = 3,
        flow_scale_x: float = 1.0,
        flow_scale_y: float = 1.0,
    ):
        super().__init__()
        if len(pyramid_channels) < 4:
            raise ValueError(
                f"SOFESSNNNetPyramidReadout requires 4 pyramid_channels "
                f"(h16, h8, h4, h2); got {len(pyramid_channels)}."
            )
        self.flow_activation = str(flow_activation).lower()
        self.flow_scale_x = float(flow_scale_x)
        self.flow_scale_y = float(flow_scale_y)
        kernel_size = int(pred_kernel_size)

        self.heads = nn.ModuleList()
        for ch in pyramid_channels[:4]:
            head = nn.Conv2d(
                int(ch), 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            nn.init.normal_(head.weight, mean=0.0, std=float(flow_head_init_std))
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            self.heads.append(head)

    def _activate(self, flow_logits: torch.Tensor) -> torch.Tensor:
        if self.flow_activation == "tanh":
            return torch.tanh(flow_logits)
        if self.flow_activation == "softsign":
            return F.softsign(flow_logits)
        if self.flow_activation in {"identity", "none", "linear"}:
            return flow_logits
        raise NotImplementedError(
            f"SSNN flow_activation={self.flow_activation!r} is not implemented."
        )

    def _scale_flow(self, flow: torch.Tensor) -> torch.Tensor:
        if self.flow_scale_x == 1.0 and self.flow_scale_y == 1.0:
            return flow
        out = flow.clone()
        out[:, 0].mul_(self.flow_scale_x)
        out[:, 1].mul_(self.flow_scale_y)
        return out

    @staticmethod
    def _upsample_and_rescale(
        flow: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Upsample flow to ``target_size`` and scale values to the new pixel units."""
        src_h, src_w = flow.shape[2], flow.shape[3]
        tgt_h, tgt_w = int(target_size[0]), int(target_size[1])
        up = F.interpolate(flow, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        if tgt_w != src_w or tgt_h != src_h:
            up = up.clone()
            up[:, 0].mul_(float(tgt_w) / float(src_w))
            up[:, 1].mul_(float(tgt_h) / float(src_h))
        return up

    def forward(
        self,
        features: Dict[str, Any],
        *,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pyramid = list(features.get("pyramid") or [])
        latent = features["latent"]
        # Pad pyramid to length 4 with the finest available feature.
        while len(pyramid) < 4:
            pyramid.append(latent)

        # --- Level 0: H/16 — absolute coarse flow ---
        logits_h16 = self.heads[0](pyramid[0])
        flow_h16 = self._scale_flow(self._activate(logits_h16))

        # --- Level 1: H/8 — residual on upsampled H/16 ---
        flow_up = self._upsample_and_rescale(flow_h16, pyramid[1].shape[2:])
        delta_h8 = self._scale_flow(self._activate(self.heads[1](pyramid[1])))
        flow_h8 = flow_up + delta_h8

        # --- Level 2: H/4 — residual on upsampled H/8 ---
        flow_up = self._upsample_and_rescale(flow_h8, pyramid[2].shape[2:])
        delta_h4 = self._scale_flow(self._activate(self.heads[2](pyramid[2])))
        flow_h4 = flow_up + delta_h4

        # --- Level 3: H/2 — residual on upsampled H/4 ---
        flow_up = self._upsample_and_rescale(flow_h4, pyramid[3].shape[2:])
        logits_h2 = self.heads[3](pyramid[3])
        delta_h2 = self._scale_flow(self._activate(logits_h2))
        flow_h2 = flow_up + delta_h2

        # --- Upsample H/2 → output_size ---
        scale_x = self.flow_scale_x
        scale_y = self.flow_scale_y
        if tuple(flow_h2.shape[2:]) != tuple(output_size):
            low_h, low_w = flow_h2.shape[2], flow_h2.shape[3]
            out_h, out_w = int(output_size[0]), int(output_size[1])
            flow_full = F.interpolate(
                flow_h2, size=(out_h, out_w), mode="bilinear", align_corners=False
            )
            upsample_sx = float(out_w) / float(low_w)
            upsample_sy = float(out_h) / float(low_h)
            flow_full = flow_full.clone()
            flow_full[:, 0].mul_(upsample_sx)
            flow_full[:, 1].mul_(upsample_sy)
            scale_x = self.flow_scale_x * upsample_sx
            scale_y = self.flow_scale_y * upsample_sy
        else:
            flow_full = flow_h2

        return flow_full, {
            # Live tensors for auxiliary EPE loss computation.
            "flow_h16": flow_h16,
            "flow_h8": flow_h8,
            "flow_h4": flow_h4,
            "flow_h2": flow_h2,
            # Existing diagnostic contract expected by SOFE/MCM trainer.
            "flow_logits": logits_h2,
            "flow_low": flow_h2.detach(),
            "flow_full": flow_full.detach(),
            "flow_upsample_scale_x": float(scale_x),
            "flow_upsample_scale_y": float(scale_y),
        }


class SOFESSNNNetReadout(nn.Module):
    """Bounded SSNN flow readout used by the streaming SOFE wrapper.

    ``flow_scale_x`` and ``flow_scale_y`` set the maximum representable
    displacement (in pixels at the latent resolution) after the bounded
    activation.  The spatial upsampling scale is then applied on top to
    convert latent-resolution pixel displacements to full-resolution pixels.
    Both default to 1.0 to preserve the original behaviour when unset.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        flow_activation: str = "softsign",
        flow_head_init_std: float = 0.01,
        pred_kernel_size: int = 3,
        flow_scale_x: float = 1.0,
        flow_scale_y: float = 1.0,
    ):
        super().__init__()
        kernel_size = int(pred_kernel_size)
        self.flow_activation = str(flow_activation).lower()
        self.flow_scale_x = float(flow_scale_x)
        self.flow_scale_y = float(flow_scale_y)
        self.pred = nn.Conv2d(
            int(in_channels),
            2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        nn.init.normal_(self.pred.weight, mean=0.0, std=float(flow_head_init_std))
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
            f"SSNN flow_activation={self.flow_activation!r} is not implemented."
        )

    def forward(
        self,
        features: Dict[str, Any],
        *,
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = features["latent"]
        flow_logits = self.pred(x)
        # Apply bounded activation then configured flow scale (max displacement in pixels
        # at the latent resolution).
        flow_low = self._activate(flow_logits)
        if self.flow_scale_x != 1.0 or self.flow_scale_y != 1.0:
            flow_low = flow_low.clone()
            flow_low[:, 0].mul_(self.flow_scale_x)
            flow_low[:, 1].mul_(self.flow_scale_y)

        scale_x = self.flow_scale_x
        scale_y = self.flow_scale_y
        if tuple(flow_low.shape[2:]) != tuple(output_size):
            low_h, low_w = flow_low.shape[2:]
            out_h, out_w = output_size
            flow = F.interpolate(flow_low, size=output_size, mode="bilinear", align_corners=False)
            upsample_sx = float(out_w) / float(low_w)
            upsample_sy = float(out_h) / float(low_h)
            flow = flow.clone()
            flow[:, 0].mul_(upsample_sx)
            flow[:, 1].mul_(upsample_sy)
            scale_x = self.flow_scale_x * upsample_sx
            scale_y = self.flow_scale_y * upsample_sy
        else:
            flow = flow_low

        return flow, {
            "flow_logits": flow_logits,
            "flow_low": flow_low.detach(),
            "flow_full": flow.detach(),
            "flow_upsample_scale_x": float(scale_x),
            "flow_upsample_scale_y": float(scale_y),
        }
