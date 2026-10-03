"""ST-FlowNet flow estimator (primary architecture).

Semi-pyramidal encoder-decoder
with cross-scale feature fusion, ConvGRU temporal modules, and BISNN
training strategy (ANN pre-train then SNN fine-tune).

References:
  - ST-FlowNet + BISNN (Sun et al., 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.spiking_blocks import (
    SpikingConvBlock,
    SpikingResBlock,
    ConvGRUCell,
)
from smcfnet.model.ofe import FlowEstimator, register_estimator


@register_estimator("st_flownet")
class STFlowNet(FlowEstimator):
    """ST-FlowNet: spiking optical flow estimator with ConvGRU temporal modules.

    Architecture::

        V_t (B, num_bins, H, W)
          |
        ConvGRU1 — temporal augmentation (state <- projected GRU2 output)
          |
        Enc1: SpikingConv stride=2  (B, ch[0], H/2, W/2)  --> down_f1 -+
        Enc2: SpikingConv stride=2  (B, ch[1], H/4, W/4)  --> down_f2 -|
        Enc3: SpikingConv stride=2  (B, ch[2], H/8, W/8)  --> down_f3 -|
        Enc4: SpikingConv stride=2  (B, ch[3], H/16, W/16)            |
          |                                                            |
        num_res_blocks x SpikingResBlock(ch[3]) -----------------------|
          |                                                            |
        Cross-scale fusion: concat all at H/16 -> upsample to H/8     |
          (B, sum(ch), H/8, W/8)                                  <---+
          |
        Decoder: SpikingConv (B, ch[2], H/8, W/8)
          |
        Bilinear upsample to (B, ch[2], H, W)
          |
        Generator: Conv2d -> (B, 2, H, W)
          |
        ConvGRU2 — temporal alignment / refinement
          |
        SO_t (B, 2, H, W)

    Args:
        num_bins: Number of temporal bins in the voxel grid.
        encoder_channels: Channel counts for each encoder stage.
        num_res_blocks: Number of residual blocks in the bottleneck.
        alpha: LIF membrane decay constant.
        theta: LIF firing threshold.
        mode: Initial operating mode, ``"ann"`` or ``"snn"``.
        neuron_type: Registry key for spiking neuron model (default ``"lif"``).
        neuron_kwargs: Extra kwargs forwarded to neuron constructor.
    """

    def __init__(
        self,
        num_bins: int = 5,
        encoder_channels: list | None = None,
        num_res_blocks: int = 2,
        alpha: float = 0.9,
        theta: float = 1.0,
        mode: str = "ann",
        neuron_type: str = "lif",
        neuron_kwargs: dict | None = None,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        assert len(encoder_channels) == 4, "Need exactly 4 encoder channel values"

        self.num_bins = num_bins
        ch = encoder_channels

        nk = dict(neuron_kwargs or {})

        # ConvGRU1: temporal augmentation
        self.gru1 = ConvGRUCell(num_bins, num_bins, kernel_size=3)

        # Encoder 1-4: SpikingConvBlocks with stride 2
        self.enc1 = SpikingConvBlock(
            num_bins, ch[0], stride=2, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=nk,
        )
        self.enc2 = SpikingConvBlock(
            ch[0], ch[1], stride=2, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=nk,
        )
        self.enc3 = SpikingConvBlock(
            ch[1], ch[2], stride=2, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=nk,
        )
        self.enc4 = SpikingConvBlock(
            ch[2], ch[3], stride=2, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=nk,
        )

        # Downsample paths for cross-scale fusion
        self.down_f1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(ch[0], ch[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(ch[0], ch[0], kernel_size=3, stride=2, padding=1),
        )
        self.down_f2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(ch[1], ch[1], kernel_size=3, stride=2, padding=1),
        )
        self.down_f3 = nn.Conv2d(ch[2], ch[2], kernel_size=3, stride=2, padding=1)

        fused_ch = sum(ch)
        self.fuse_up = nn.ConvTranspose2d(fused_ch, fused_ch, kernel_size=4, stride=2, padding=1)
        self.fuse_bn = nn.BatchNorm2d(fused_ch)

        # Decoder
        self.decoder = SpikingConvBlock(
            fused_ch, ch[2], stride=1, alpha=alpha, theta=theta,
            neuron_type=neuron_type, neuron_kwargs=nk,
        )

        # Generator: flow output
        self.generator = nn.Conv2d(ch[2], 2, kernel_size=3, padding=1, bias=True)

        # Bottleneck residual blocks
        self.res_blocks = nn.ModuleList([
            SpikingResBlock(
                ch[3], alpha=alpha, theta=theta,
                neuron_type=neuron_type, neuron_kwargs=nk,
            )
            for _ in range(num_res_blocks)
        ])

        # Zero-init so module starts as identity (near-zero flow)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

        # ConvGRU2: temporal refinement
        self.gru2 = ConvGRUCell(2, 2, kernel_size=3)

        # Hidden states
        self._h1: torch.Tensor | None = None
        self._h2: torch.Tensor | None = None
        # Output is always .detach()'d (truncated BPTT), so freeze to avoid
        # DDP "unused parameter" errors.
        self.flow_to_hidden = nn.Conv2d(2, num_bins, kernel_size=1)
        self.flow_to_hidden.weight.requires_grad_(False)
        self.flow_to_hidden.bias.requires_grad_(False)

        # Set initial mode
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Switch between ``'ann'`` and ``'snn'`` mode for all spiking blocks."""
        assert mode in ("ann", "snn"), f"Invalid mode: {mode}"
        for module in self.modules():
            if isinstance(module, (SpikingConvBlock, SpikingResBlock)):
                module.mode = mode

    def get_mode(self) -> str:
        return self.enc1.mode

    def reset_state(self) -> None:
        """Clear ConvGRU hidden states and LIF membrane potentials."""
        self._h1 = None
        self._h2 = None
        for module in self.modules():
            if isinstance(module, (SpikingConvBlock, SpikingResBlock)):
                module.reset_membrane()

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            voxel: Raw event voxel grid, shape ``(B, num_bins, H, W)``.

        Returns:
            flow: Dense optical flow SO_t, shape ``(B, 2, H, W)``.
        """
        flow = self._predict_flow(voxel)
        self._last_flow = flow.detach()
        return flow

    def _predict_flow(self, voxel: torch.Tensor) -> torch.Tensor:
        """Predict dense optical flow from the voxel grid."""
        H, W = voxel.shape[2], voxel.shape[3]

        # Temporal augmentation via ConvGRU1
        x = self.gru1(voxel, self._h1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        x = e4
        for res in self.res_blocks:
            x = res(x)

        target = x.shape[2:]
        f1 = F.interpolate(self.down_f1(e1), size=target, mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.down_f2(e2), size=target, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.down_f3(e3), size=target, mode='bilinear', align_corners=False)

        fused = torch.cat([x, f1, f2, f3], dim=1)
        fused = F.silu(self.fuse_bn(self.fuse_up(fused)))

        # Decode + upsample + generate flow
        x = self.decoder(fused)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        flow = self.generator(x)

        flow = self.gru2(flow, self._h2)
        self._h2 = flow.detach()

        self._h1 = self.flow_to_hidden(flow).detach()

        return flow
