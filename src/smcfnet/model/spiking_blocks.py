"""Spiking neural network building blocks for MCM.

Pure PyTorch implementation — no external SNN frameworks required.
Implements the BISNN (Bi-directional SNN) training strategy:
  - ANN mode: uses ClipFloor quantization with straight-through estimator
  - SNN mode: uses LIF neurons with surrogate gradient (arctan)

References:
  - ST-FlowNet + BISNN (Sun et al., 2025)
  - Surrogate gradient: Fang et al. (2021) arctan surrogate
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Surrogate gradient for spiking (SNN mode)
# ---------------------------------------------------------------------------

class SurrogateHeaviside(torch.autograd.Function):
    """Heaviside step with arctan surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Arctan surrogate: 1 / (pi * (1 + (pi * x)^2))
        grad = 1.0 / (math.pi * (1.0 + (math.pi * x) ** 2))
        return grad_output * grad


surrogate_heaviside = SurrogateHeaviside.apply


# ---------------------------------------------------------------------------
# ClipFloor activation for ANN mode (BISNN pre-training)
# ---------------------------------------------------------------------------

class ClipFloorActivation(torch.autograd.Function):
    """Quantized activation for ANN-mode BISNN pre-training.

    Forward:  lambda_l * clip(floor(x * L / lambda_l) / L, 0, 1)
    Backward: straight-through estimator (pass gradient through unchanged)

    Args (via forward):
        x: input tensor
        L: number of quantization levels
        lambda_l: scaling factor (typically equal to LIF threshold theta)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, L: int, lambda_l: float) -> torch.Tensor:
        quantized = lambda_l * torch.clamp(
            torch.floor(x * L / lambda_l) / L, 0.0, 1.0
        )
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # Straight-through estimator
        return grad_output, None, None


clip_floor = ClipFloorActivation.apply


# ---------------------------------------------------------------------------
# Spiking Neuron base class + implementations (Axis B)
# ---------------------------------------------------------------------------

class SpikingNeuron(nn.Module):
    """Base class for all spiking neuron models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset_membrane(self) -> None:
        raise NotImplementedError


class LIFNeuron(SpikingNeuron):
    """Leaky Integrate-and-Fire neuron with soft reset.

    Update rule:
        v_tilde = v - s * theta        (soft reset)
        v = alpha * v_tilde + I         (leaky integration)
        s = Heaviside(v - theta)        (spike with surrogate grad)

    Args:
        alpha: membrane decay constant (0 < alpha < 1).
        theta: firing threshold.
    """

    def __init__(self, alpha: float = 0.9, theta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.theta = theta
        self.v: torch.Tensor | None = None

    def reset_membrane(self) -> None:
        self.v = None

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(I)
        # Soft reset
        s_prev = surrogate_heaviside(self.v - self.theta)
        v_tilde = self.v - s_prev * self.theta
        # Leaky integration
        self.v = self.alpha * v_tilde + I
        # Spike
        s = surrogate_heaviside(self.v - self.theta)
        return s


class PLIFNeuron(SpikingNeuron):
    """Parametric LIF neuron with learnable decay constant per channel.

    Same dynamics as LIF, but alpha is a learnable parameter (one per
    output channel) initialized from the provided value and clamped to (0, 1)
    via sigmoid reparametrization.

    Reference: Fang et al. (2021), "Incorporating Learnable Membrane Time
    Constants to Enhance Learning of Spiking Neural Networks"

    Args:
        alpha: initial membrane decay constant.
        theta: firing threshold.
        num_channels: number of output channels (one alpha per channel).
    """

    def __init__(self, alpha: float = 0.9, theta: float = 1.0, num_channels: int = 1):
        super().__init__()
        self.theta = theta
        # Store alpha in logit space so sigmoid maps back to (0, 1)
        init_logit = math.log(alpha / (1.0 - alpha + 1e-8))
        self._alpha_logit = nn.Parameter(
            torch.full((num_channels, 1, 1), init_logit)
        )
        self.v: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit)

    def reset_membrane(self) -> None:
        self.v = None

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(I)
        alpha = self.alpha  # (C, 1, 1)
        s_prev = surrogate_heaviside(self.v - self.theta)
        v_tilde = self.v - s_prev * self.theta
        self.v = alpha * v_tilde + I
        s = surrogate_heaviside(self.v - self.theta)
        return s


class AdaptiveLIFNeuron(SpikingNeuron):
    """LIF neuron with homeostatic threshold adaptation.

    The firing threshold adjusts dynamically based on recent activity,
    implementing a form of homeostatic plasticity that stabilizes firing
    rates under varying input densities (e.g. day/night event streams).

    Formulation from Zhang et al. 2023, SpikeFPN (arXiv:2307.12900):
        A_th(t) = u_th + beta * a(t)
        a(t) = tau_a * a(t-1) + y(t-1)

    where a(t) is an activity trace that accumulates past spikes, tau_a
    is a learnable time constant, and beta scales the threshold adaptation.
    The effective threshold ranges from u_th (silent neuron) to
    u_th + beta / (1 - tau_a) (saturated firing).

    Args:
        alpha: membrane decay constant (0 < alpha < 1).
        theta: base firing threshold u_th.
        beta: threshold adaptation scaling coefficient.
        tau_a_init: initial value for tau_a (constrained to [tau_a_min, tau_a_max]).
        tau_a_min: lower bound for tau_a.
        tau_a_max: upper bound for tau_a.
        num_channels: number of output channels (one tau_a per channel).
    """

    def __init__(
        self,
        alpha: float = 0.9,
        theta: float = 0.3,
        beta: float = 0.07,
        tau_a_init: float = 0.3,
        tau_a_min: float = 0.2,
        tau_a_max: float = 0.4,
        num_channels: int = 1,
    ):
        super().__init__()
        self.alpha = alpha
        self.u_th = theta
        self.beta = beta
        self.tau_a_min = tau_a_min
        self.tau_a_max = tau_a_max

        # Learnable tau_a via sigmoid reparameterization to [tau_a_min, tau_a_max].
        # sigmoid(logit) maps to (0, 1), then scale to [min, max].
        # Solve for init logit: sigmoid(x) = (tau_a_init - min) / (max - min)
        frac = (tau_a_init - tau_a_min) / (tau_a_max - tau_a_min + 1e-8)
        frac = max(min(frac, 0.999), 0.001)
        init_logit = math.log(frac / (1.0 - frac))
        self._tau_a_logit = nn.Parameter(
            torch.full((num_channels, 1, 1), init_logit)
        )

        self.v: torch.Tensor | None = None
        self.a: torch.Tensor | None = None  # activity trace
        self.s: torch.Tensor | None = None  # previous spike

    @property
    def tau_a(self) -> torch.Tensor:
        """Learnable time constant, constrained to [tau_a_min, tau_a_max]."""
        return self.tau_a_min + (self.tau_a_max - self.tau_a_min) * torch.sigmoid(
            self._tau_a_logit
        )

    def reset_membrane(self) -> None:
        self.v = None
        self.a = None
        self.s = None

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(I)
        if self.a is None:
            self.a = torch.zeros_like(I)
        if self.s is None:
            self.s = torch.zeros_like(I)

        # Adaptive threshold for this timestep
        A_th = self.u_th + self.beta * self.a

        # Soft reset using stored previous spike (not recomputed,
        # since A_th may have changed between timesteps)
        v_tilde = self.v - self.s * A_th

        # Leaky integration
        self.v = self.alpha * v_tilde + I

        # Spike
        s = surrogate_heaviside(self.v - A_th)

        # Update activity trace: a(t) = tau_a * a(t-1) + s(t)
        tau_a = self.tau_a  # (C, 1, 1)
        self.a = tau_a * self.a + s

        # Store spike for next timestep's soft reset
        self.s = s

        return s


# ---------------------------------------------------------------------------
# ANN activation base class + implementations
# ---------------------------------------------------------------------------

class AnnActivation(nn.Module):
    """Base class for ANN-mode activations (BISNN pre-training)."""

    def forward(self, x: torch.Tensor, L: int, theta: float) -> torch.Tensor:
        raise NotImplementedError


class ClipFloorAnnActivation(AnnActivation):
    """Standard ClipFloor quantization for BISNN ANN mode."""

    def forward(self, x: torch.Tensor, L: int, theta: float) -> torch.Tensor:
        return clip_floor(x, L, theta)


# ---------------------------------------------------------------------------
# Neuron and activation factories
# ---------------------------------------------------------------------------

_NEURON_REGISTRY: dict[str, type[SpikingNeuron]] = {
    "lif": LIFNeuron,
    "plif": PLIFNeuron,
    "adaptive_lif": AdaptiveLIFNeuron,
}

_ANN_ACT_REGISTRY: dict[str, type[AnnActivation]] = {
    "clip_floor": ClipFloorAnnActivation,
}


def create_neuron(neuron_type: str, **kwargs) -> SpikingNeuron:
    """Create a spiking neuron by registry key."""
    if neuron_type not in _NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron type '{neuron_type}'. "
            f"Available: {list(_NEURON_REGISTRY.keys())}"
        )
    return _NEURON_REGISTRY[neuron_type](**kwargs)


def create_ann_activation(ann_type: str = "clip_floor", **kwargs) -> AnnActivation:
    """Create an ANN-mode activation by registry key."""
    if ann_type not in _ANN_ACT_REGISTRY:
        raise ValueError(
            f"Unknown ANN activation type '{ann_type}'. "
            f"Available: {list(_ANN_ACT_REGISTRY.keys())}"
        )
    return _ANN_ACT_REGISTRY[ann_type](**kwargs)


# ---------------------------------------------------------------------------
# ConvGRU Cell
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for temporal aggregation.

    Standard Conv2d-based gates (not spiking). Used for temporal
    augmentation and alignment in MCM flow estimators.

    Args:
        in_channels: input feature channels.
        hidden_channels: hidden state channels.
        kernel_size: convolution kernel size.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        self.gate_conv = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=True,
        )
        self.candidate_conv = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=pad, bias=True,
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        if h_prev is None or h_prev.shape[0] != x.shape[0] or h_prev.shape[2:] != x.shape[2:]:
            h_prev = torch.zeros(
                x.shape[0], self.hidden_channels, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
        combined = torch.cat([x, h_prev], dim=1)
        gates = torch.sigmoid(self.gate_conv(combined))
        r, z = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate_conv(torch.cat([x, r * h_prev], dim=1)))
        h = (1 - z) * h_prev + z * candidate
        return h


# ---------------------------------------------------------------------------
# Spiking Conv Block
# ---------------------------------------------------------------------------

class SpikingConvBlock(nn.Module):
    """Conv2d + BatchNorm + (spiking neuron or ClipFloor depending on mode).

    In ANN mode, uses ClipFloor quantization (BISNN pre-training).
    In SNN mode, uses spiking neuron (LIF/PLIF/AdaptiveLIF) with surrogate gradient.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        alpha: LIF decay constant (used when neuron_type/neuron_kwargs not set).
        theta: LIF firing threshold (also used as ClipFloor lambda_l).
        neuron_type: registry key for spiking neuron (default ``"lif"``).
        neuron_kwargs: extra kwargs for neuron constructor. Overrides alpha/theta
            if provided. For PLIF and AdaptiveLIF, ``num_channels`` is
            auto-set to out_channels.
        ann_activation: registry key for ANN-mode activation (default ``"clip_floor"``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        alpha: float = 0.9,
        theta: float = 1.0,
        neuron_type: str = "lif",
        neuron_kwargs: dict | None = None,
        ann_activation: str = "clip_floor",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.theta = theta
        self._mode = "ann"
        # ClipFloor quantization levels (BISNN: L = T, typically 4-8)
        self._clip_L = 4

        # Build neuron via factory
        nk = dict(neuron_kwargs or {})
        nk.setdefault("alpha", alpha)
        nk.setdefault("theta", theta)
        # PLIF and AdaptiveLIF need num_channels for per-channel learnable params
        if neuron_type in ("plif", "adaptive_lif"):
            nk.setdefault("num_channels", out_channels)
        self.snn_act = create_neuron(neuron_type, **nk)

        # Build ANN activation via factory
        self.ann_act = create_ann_activation(ann_activation)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        assert value in ("ann", "snn"), f"Invalid mode: {value}"
        self._mode = value

    def reset_membrane(self) -> None:
        self.snn_act.reset_membrane()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self._mode == "snn":
            return self.snn_act(x)
        else:
            return self.ann_act(x, self._clip_L, self.theta)


# ---------------------------------------------------------------------------
# Spiking Residual Block
# ---------------------------------------------------------------------------

class SpikingResBlock(nn.Module):
    """Two SpikingConvBlocks with residual skip connection.

    Args:
        channels: number of channels (in == out).
        alpha: LIF decay constant.
        theta: LIF firing threshold.
        neuron_type: registry key for spiking neuron.
        neuron_kwargs: extra kwargs for neuron constructor.
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 0.9,
        theta: float = 1.0,
        neuron_type: str = "lif",
        neuron_kwargs: dict | None = None,
    ):
        super().__init__()
        self.conv1 = SpikingConvBlock(
            channels, channels, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=neuron_kwargs,
        )
        self.conv2 = SpikingConvBlock(
            channels, channels, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=neuron_kwargs,
        )

    @property
    def mode(self) -> str:
        return self.conv1.mode

    @mode.setter
    def mode(self, value: str) -> None:
        self.conv1.mode = value
        self.conv2.mode = value

    def reset_membrane(self) -> None:
        self.conv1.reset_membrane()
        self.conv2.reset_membrane()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) # TODO: check performance - in SNN mode, conv2 outputs binary spikes {0,1} and x is also spikes. the sum can be 0,1,2, so no longer valid spike train. downstream LIF neuron will handle this, but output is not purely spiking. to keep it pure spiking, add LIF after addition. But may hurt BISNN as ANN does not have same activation.
