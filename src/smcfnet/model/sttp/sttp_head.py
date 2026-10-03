"""Short-Term Trajectory Prediction heads.

Three variants are colocated here, all sharing the forward signature
``forward(stem_features: List[Tensor], flow: Optional[Tensor]) -> List[Tensor]``
so they are interchangeable at the call site:

- ``STTPHead`` (Designs B and C)
    fut branch on stem features, optionally with explicit OFE flow injection.
    ``flow_channels=0`` (Design B) skips the flow branch entirely; the fut
    stem conv learns any implicit flow cues from the stem features alone.
    ``flow_channels=2`` (Design C) projects (dx, dy) via a 1x1 conv, concats
    with stem features, and feeds the fut stem conv.

- ``FlowOnlySTTPHead`` (Design D Stage 2)
    Trainable, flow only. Drops the stem detection features Design C uses.
    Tests whether a learned head can extract trajectory information from
    flow alone. ~3-5k params per scale (<20k total) vs ~150k for Design C.

- ``FlowProjectionHead`` (Stage 1 baseline)
    Zero-parameter analytic projection: ``future_center = anchor_center +
    k * flow_at_anchor``. The "what does flow alone explain" floor for the
    matched-pair STTP table.

Dispatch lives in ``smcfnet.model.build`` (build-time, between Designs C/B
and flow_only) and ``smcfnet.core.train._maybe_swap_to_trivial_sttp_head``
(eval/viz-only swap to the trivial baseline).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.network_blocks import BaseConv


def _resize_flow(flow: torch.Tensor, size: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    return F.interpolate(flow.to(dtype), size=size, mode="bilinear", align_corners=False)


def _init_pred_layer(pred: nn.Conv2d) -> None:
    nn.init.normal_(pred.weight, std=0.01)
    nn.init.zeros_(pred.bias)


class STTPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_scales: int = 3,
        flow_channels: int = 0,
        flow_proj_dim: int = 16,
        act: str = "silu",
        future_steps: int = 1,
    ):
        """
        Args:
            in_channels: Stem feature channels (same for all scales).
            num_scales: Number of FPN scales (P3/P4/P5).
            flow_channels: 0 for Design B (features only), 2 for Design C (with OFE flow).
            flow_proj_dim: Flow projection width after the 1x1 flow encoder. Design C only.
            act: Activation for BaseConv layers.
            future_steps: trajectory length k. The head emits one box per step
                t+1..t+k, so the per-scale pred layer outputs 4*k channels.
        """
        super().__init__()
        self.num_scales = num_scales
        self.flow_channels = flow_channels
        self.flow_proj_dim = flow_proj_dim
        self.future_steps = int(future_steps)

        self.flow_encoders = nn.ModuleList()
        self.fut_stems = nn.ModuleList()
        self.fut_towers = nn.ModuleList()
        self.fut_preds = nn.ModuleList()

        for _ in range(num_scales):
            if flow_channels > 0:
                self.flow_encoders.append(
                    BaseConv(flow_channels, flow_proj_dim, ksize=1, stride=1, act=act)
                )
                stem_in = in_channels + flow_proj_dim
            else:
                self.flow_encoders.append(None)
                stem_in = in_channels

            self.fut_stems.append(
                BaseConv(stem_in, in_channels, ksize=1, stride=1, act=act)
            )

            self.fut_towers.append(
                nn.Sequential(
                    BaseConv(in_channels, in_channels, ksize=3, stride=1, act=act),
                    BaseConv(in_channels, in_channels, ksize=3, stride=1, act=act),
                )
            )

            # fut_preds outputs raw YOLOX-style (tx, ty, tw, th) per step, packed as
            # 4*k channels ordered [step1_txtytwth, step2_..., ...]. Decoded downstream
            # via grid + stride: (cx, cy) = (tx + grid) * stride, (w, h) = exp(tw, th) * stride.
            # See sttp_loss.STTPLoss._decode_absolute (train) and
            # validation.decode_sttp_predictions (eval).
            self.fut_preds.append(
                nn.Conv2d(in_channels, 4 * self.future_steps, kernel_size=1, stride=1, padding=0)
            )

        self._init_weights()

    def _init_weights(self):
        for pred in self.fut_preds:
            _init_pred_layer(pred)

    def forward(
        self,
        stem_features: List[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            stem_features: [[B, C, H_k, W_k] for k in range(num_scales)]
            flow: OFE flow [B, flow_channels, H_flow, W_flow], resized per-scale to (H_k, W_k).
                Only used when flow_channels > 0.
        Returns:
            Per-scale fut predictions, each [B, 4, H_k, W_k] (tx, ty, tw, th).
        """
        if len(stem_features) != self.num_scales:
            raise ValueError(
                f"STTPHead expected {self.num_scales} stem feature maps, "
                f"got {len(stem_features)}"
            )

        fut_outputs = []

        for k in range(self.num_scales):
            x = stem_features[k]
            H_k, W_k = x.shape[2], x.shape[3]

            if self.flow_channels > 0:
                if flow is not None:
                    f = _resize_flow(flow, (H_k, W_k), x.dtype)
                    f = self.flow_encoders[k](f)
                else:
                    f = torch.zeros(
                        x.shape[0],
                        self.flow_proj_dim,
                        H_k,
                        W_k,
                        device=x.device,
                        dtype=x.dtype,
                    )

                x = torch.cat([x, f], dim=1)

            x = self.fut_stems[k](x)
            x = self.fut_towers[k](x)
            out = self.fut_preds[k](x)
            fut_outputs.append(out)

        return fut_outputs


class FlowOnlySTTPHead(nn.Module):
    """Trainable flow-only STTP head (Design D Stage 2).

    Drop-in replacement for ``STTPHead`` that consumes ONLY flow. Loss + target
    plumbing is unchanged from Design C, so the existing ``STTPLoss`` and
    ``compute_ade`` pipeline drop in unmodified.

    Per-scale (P3, P4, P5):

        flow [B, 2, H_full, W_full]
            --> bilinear-resize to [B, 2, H_k, W_k]
            --> flow_encoder    (BaseConv 2 -> F, k=1)
            --> fut_tower       (2x BaseConv F -> F, k=3)
            --> fut_pred        (Conv2d F -> 4, k=1)
        output: [B, 4, H_k, W_k]

    Args:
        num_scales: number of FPN scales (3 for P3/P4/P5).
        flow_channels: flow input channels, expected 2 for (dx, dy).
        flow_proj_dim: encoder + tower channel width F.
        act: activation type passed to BaseConv.
    """

    def __init__(
        self,
        num_scales: int = 3,
        flow_channels: int = 2,
        flow_proj_dim: int = 16,
        act: str = "silu",
        future_steps: int = 1,
    ) -> None:
        super().__init__()
        if flow_channels <= 0:
            raise ValueError(
                f"FlowOnlySTTPHead requires flow_channels > 0; got {flow_channels}. "
                "Set sttp.flow_channels: 2 in the config (Design D operates on flow)."
            )
        self.num_scales = int(num_scales)
        self.flow_channels = int(flow_channels)
        self.flow_proj_dim = int(flow_proj_dim)
        self.future_steps = int(future_steps)

        self.flow_encoders = nn.ModuleList()
        self.fut_towers = nn.ModuleList()
        self.fut_preds = nn.ModuleList()

        for _ in range(num_scales):
            self.flow_encoders.append(
                BaseConv(flow_channels, flow_proj_dim, ksize=1, stride=1, act=act)
            )
            self.fut_towers.append(
                nn.Sequential(
                    BaseConv(flow_proj_dim, flow_proj_dim, ksize=3, stride=1, act=act),
                    BaseConv(flow_proj_dim, flow_proj_dim, ksize=3, stride=1, act=act),
                )
            )
            self.fut_preds.append(
                nn.Conv2d(flow_proj_dim, 4 * self.future_steps, kernel_size=1, stride=1, padding=0)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for pred in self.fut_preds:
            _init_pred_layer(pred)

    def forward(
        self,
        stem_features: List[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            stem_features: list of ``num_scales`` tensors ``[B, C, H_k, W_k]``.
                Only used for output shape and dtype/device inference; the head
                does not consume the features.
            flow: full-res ``[B, 2, H, W]`` OFE flow field. If ``None``, a
                zero-flow tensor is substituted (mirrors ``STTPHead`` zero-pad
                behaviour in Design C).

        Returns:
            fut_outputs: list of ``num_scales`` tensors ``[B, 4, H_k, W_k]`` in
            the raw decoder space ``STTPLoss._decode_absolute`` expects.
        """
        if len(stem_features) != self.num_scales:
            raise ValueError(
                f"expected {self.num_scales} stem features, got {len(stem_features)}"
            )

        outputs: List[torch.Tensor] = []
        for k in range(self.num_scales):
            ref = stem_features[k]
            B, _, Hk, Wk = ref.shape

            if flow is not None:
                f = _resize_flow(flow, (Hk, Wk), ref.dtype)
            else:
                f = torch.zeros(
                    (B, self.flow_channels, Hk, Wk),
                    device=ref.device,
                    dtype=ref.dtype,
                )

            f = self.flow_encoders[k](f)
            f = self.fut_towers[k](f)
            out = self.fut_preds[k](f)
            outputs.append(out)

        return outputs


class FlowProjectionHead(nn.Module):
    """Zero-parameter trivial flow-projection head (Stage 1 baseline).

    Produces dense per-anchor future-box predictions analytically:
    ``future_center = anchor_center + k * flow_at_anchor``. No trainable
    parameters; the "what does flow alone explain" floor for the matched-pair
    STTP table.

    The output tensors are shaped ``[B, 4, H_k, W_k]`` per FPN scale, in the
    same **raw** space the ``STTPLoss._decode_absolute`` decoder expects:

        decoded_cx = (raw_cx + grid_x) * stride
        decoded_cy = (raw_cy + grid_y) * stride
        decoded_w  = exp(raw_w) * stride
        decoded_h  = exp(raw_h) * stride

    To make ``decoded_cxcy`` equal ``anchor_center + k * flow``, where the
    anchor center is ``(grid + 0.5) * stride``, we set
    ``raw_cxcy = 0.5 + k * flow / stride``. ``raw_wh`` are zeros (decoded
    w,h = stride). The width/height channels are ignored by
    ``TrajectoryMetricsAccumulator`` (it only reads decoded_cxcy), so their
    values are operationally irrelevant.

    Args:
        future_steps: ``k`` in ``t+k``. Multiplied with the per-anchor flow
            sample to produce displacement.
        num_scales: number of FPN scales (default 3 for P3/P4/P5).
        strides: per-scale strides matching the FPN levels.
    """

    def __init__(
        self,
        future_steps: int,
        num_scales: int = 3,
        strides: Tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        if len(strides) != num_scales:
            raise ValueError(
                f"len(strides)={len(strides)} must equal num_scales={num_scales}"
            )
        self.future_steps = int(future_steps)
        self.num_scales = int(num_scales)
        # Buffer so the module moves with .to(device) consistently.
        self.register_buffer(
            "strides",
            torch.tensor(strides, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        stem_features: List[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return raw [B, 4*k, H_k, W_k] per scale, step s decoding to anchor + s*flow.

        Channels are packed [step1_txtytwth, step2_..., ...]; step s displaces the
        anchor by ``s * flow`` (the linear constant-velocity trajectory t+1..t+k).

        Args:
            stem_features: list of ``num_scales`` tensors ``[B, C, H_k, W_k]``.
                Only used for shape and dtype/device inference.
            flow: optional ``[B, 2, H, W]`` full-resolution OFE flow field
                in pixels-per-window. If None, the head returns the no-op
                projection (predicted future == current anchor center) at every step.
        """
        if len(stem_features) != self.num_scales:
            raise ValueError(
                f"expected {self.num_scales} stem features, got {len(stem_features)}"
            )

        K = self.future_steps
        outputs: List[torch.Tensor] = []
        for k in range(self.num_scales):
            ref = stem_features[k]
            B, _, Hk, Wk = ref.shape
            stride = float(self.strides[k].item())

            raw = torch.zeros((B, 4 * K, Hk, Wk), device=ref.device, dtype=ref.dtype)
            f = _resize_flow(flow, (Hk, Wk), ref.dtype) if flow is not None else None
            for s in range(1, K + 1):
                base = 4 * (s - 1)
                # raw_cx/cy default 0.5 (anchor center); add s*flow displacement.
                raw[:, base + 0].fill_(0.5)
                raw[:, base + 1].fill_(0.5)
                if f is not None:
                    # Bilinear preserves displacement vectors: one stride-unit at a
                    # coarse scale still corresponds to ``stride`` pixels at full res.
                    raw[:, base + 0] = raw[:, base + 0] + (s * f[:, 0]) / stride
                    raw[:, base + 1] = raw[:, base + 1] + (s * f[:, 1]) / stride

            outputs.append(raw)

        return outputs


# ---------------------------------------------------------------------------
# Spiking Recurrent STTP Head
# ---------------------------------------------------------------------------

from smcfnet.model.spiking_blocks import PLIFNeuron, LIFNeuron


def _make_neuron(neuron_model: str, hidden_dim: int):
    if neuron_model == "plif":
        return PLIFNeuron(alpha=0.9, theta=1.0, num_channels=hidden_dim)
    if neuron_model == "lif":
        return LIFNeuron(alpha=0.9, theta=1.0)
    raise ValueError(f"neuron_model must be 'plif' or 'lif', got {neuron_model!r}")


class SpikingRecurrentSTTPHead(nn.Module):
    """Temporal STTP head: unroll a LIF over a K-frame stem-feature window.

    Per FPN scale: a 1x1 BaseConv encodes each frame's stem features into the
    LIF input current; the neuron membrane carries across the K window steps
    (that persistence is the temporal memory); the graded final-step membrane
    potential is read out by a 1x1 conv into the raw (tx,ty,tw,th) decoder
    space STTPLoss._decode_absolute expects.

    forward(stem_window, flow=None):
        stem_window: list of num_scales tensors [B, K_ctx, C, H_k, W_k].
        flow: ignored (kept for call-site signature parity with STTPHead).
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int = 3,
        hidden_dim: int = 64,
        neuron_model: str = "plif",
        act: str = "silu",
        future_steps: int = 1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.neuron_model = neuron_model
        self.future_steps = int(future_steps)

        self.encoders = nn.ModuleList()
        self.neurons = nn.ModuleList()
        self.fut_preds = nn.ModuleList()
        for _ in range(num_scales):
            self.encoders.append(BaseConv(in_channels, hidden_dim, ksize=1, stride=1, act=act))
            self.neurons.append(_make_neuron(neuron_model, hidden_dim))
            self.fut_preds.append(nn.Conv2d(hidden_dim, 4 * self.future_steps, kernel_size=1, stride=1, padding=0))
        self._init_weights()

    def _init_weights(self):
        for pred in self.fut_preds:
            _init_pred_layer(pred)

    def forward(
        self,
        stem_window: List[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if len(stem_window) != self.num_scales:
            raise ValueError(
                f"SpikingRecurrentSTTPHead expected {self.num_scales} window "
                f"tensors, got {len(stem_window)}"
            )

        outputs: List[torch.Tensor] = []
        for k in range(self.num_scales):
            window = stem_window[k]  # [B, K, C, H, W]
            if window.dim() != 5:
                raise ValueError(
                    f"scale {k}: expected 5D window [B,K,C,H,W], got shape {tuple(window.shape)}"
                )
            K = window.shape[1]
            neuron = self.neurons[k]
            neuron.reset_membrane()
            for t in range(K):
                I_t = self.encoders[k](window[:, t])  # [B, hidden, H, W]
                neuron(I_t)  # updates neuron.v in place
            v_final = neuron.v  # graded membrane potential [B, hidden, H, W]
            outputs.append(self.fut_preds[k](v_final))
        return outputs


# ---------------------------------------------------------------------------
# ANN-Recurrent STTP Head (ConvGRU)
# ---------------------------------------------------------------------------


class _ConvGRUCell(nn.Module):
    """Minimal convolutional GRU cell (3x3, same padding)."""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_zr = nn.Conv2d(in_channels + hidden_dim, 2 * hidden_dim, 3, padding=1)
        self.conv_h = nn.Conv2d(in_channels + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class ConvGRUSTTPHead(nn.Module):
    """ANN-recurrent control: same window contract as SpikingRecurrentSTTPHead,
    but a ConvGRU replaces the LIF. Isolates 'recurrence helps' from 'spiking'."""

    def __init__(
        self,
        in_channels: int,
        num_scales: int = 3,
        hidden_dim: int = 64,
        act: str = "silu",
        future_steps: int = 1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.future_steps = int(future_steps)
        self.encoders = nn.ModuleList()
        self.cells = nn.ModuleList()
        self.fut_preds = nn.ModuleList()
        for _ in range(num_scales):
            self.encoders.append(BaseConv(in_channels, hidden_dim, ksize=1, stride=1, act=act))
            self.cells.append(_ConvGRUCell(hidden_dim, hidden_dim))
            self.fut_preds.append(nn.Conv2d(hidden_dim, 4 * self.future_steps, kernel_size=1, stride=1, padding=0))
        for pred in self.fut_preds:
            _init_pred_layer(pred)

    def forward(
        self,
        stem_window: List[torch.Tensor],
        flow: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if len(stem_window) != self.num_scales:
            raise ValueError(
                f"ConvGRUSTTPHead expected {self.num_scales} window tensors, got {len(stem_window)}"
            )
        outputs: List[torch.Tensor] = []
        for k in range(self.num_scales):
            window = stem_window[k]
            B, K, C, H, W = window.shape
            h = torch.zeros(B, self.hidden_dim, H, W, device=window.device, dtype=window.dtype)
            for t in range(K):
                x_t = self.encoders[k](window[:, t])
                h = self.cells[k](x_t, h)
            outputs.append(self.fut_preds[k](h))
        return outputs
