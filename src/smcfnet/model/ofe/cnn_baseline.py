"""CNN baseline flow estimator (non-spiking variant of ST-FlowNet).

Same architecture as STFlowNet but uses standard Conv+BN+ReLU blocks
instead of SpikingConvBlock. Direct comparison for RQ4: does the
spiking architecture improve accuracy or just energy efficiency?

set_mode() is a no-op since there are no spiking blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.spiking_blocks import ConvGRUCell
from smcfnet.model.ofe import FlowEstimator, register_estimator


class ConvBNReLU(nn.Module):
    """Standard Conv2d + BatchNorm + ReLU block (non-spiking)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNResBlock(nn.Module):
    """Two ConvBNReLU blocks with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = ConvBNReLU(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


@register_estimator("cnn_baseline")
class CNNFlowNet(FlowEstimator):
    """Non-spiking ST-FlowNet variant for RQ4 comparison.

    Same architecture (4-stage encoder, cross-scale fusion, ConvGRU,
    residual bottleneck) but with standard Conv+BN+ReLU instead of
    SpikingConvBlock/LIF.

    Args:
        num_bins: Number of temporal bins in the voxel grid.
        encoder_channels: Channel counts for each encoder stage.
        num_res_blocks: Number of residual blocks in the bottleneck.
        alpha: Ignored (no spiking neurons).
        theta: Ignored.
        mode: Ignored (always ANN).
        neuron_type: Ignored.
        neuron_kwargs: Ignored.
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
        assert len(encoder_channels) == 4

        ch = encoder_channels

        # ConvGRU1: temporal augmentation
        self.gru1 = ConvGRUCell(num_bins, num_bins, kernel_size=3)

        # Encoder (standard conv)
        self.enc1 = ConvBNReLU(num_bins, ch[0], stride=2)
        self.enc2 = ConvBNReLU(ch[0], ch[1], stride=2)
        self.enc3 = ConvBNReLU(ch[1], ch[2], stride=2)
        self.enc4 = ConvBNReLU(ch[2], ch[3], stride=2)

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

        # Bottleneck
        self.res_blocks = nn.ModuleList([
            CNNResBlock(ch[3]) for _ in range(num_res_blocks)
        ])

        # Decoder
        self.decoder = ConvBNReLU(fused_ch, ch[2], stride=1)

        # Flow generator
        self.generator = nn.Conv2d(ch[2], 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

        # ConvGRU2: temporal refinement
        self.gru2 = ConvGRUCell(2, 2, kernel_size=3)

        self._h1: torch.Tensor | None = None
        self._h2: torch.Tensor | None = None
        # Output is always .detach()'d (truncated BPTT), so freeze to avoid
        # DDP "unused parameter" errors.
        self.flow_to_hidden = nn.Conv2d(2, num_bins, kernel_size=1)
        self.flow_to_hidden.weight.requires_grad_(False)
        self.flow_to_hidden.bias.requires_grad_(False)

    def set_mode(self, mode: str) -> None:
        """No-op — CNN baseline has no spiking blocks."""
        pass

    def get_mode(self) -> str:
        return "ann"

    def reset_state(self) -> None:
        self._h1 = None
        self._h2 = None

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        H, W = voxel.shape[2], voxel.shape[3]

        x = self.gru1(voxel, self._h1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        x = e4
        for res in self.res_blocks:
            x = res(x)

        target = x.shape[2:]
        f1 = F.interpolate(self.down_f1(e1), size=target, mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.down_f2(e2), size=target, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.down_f3(e3), size=target, mode='bilinear', align_corners=False)

        fused = torch.cat([x, f1, f2, f3], dim=1)
        fused = F.silu(self.fuse_bn(self.fuse_up(fused)))

        x = self.decoder(fused)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        flow = self.generator(x)

        flow = self.gru2(flow, self._h2)
        self._h2 = flow.detach()
        self._h1 = self.flow_to_hidden(flow).detach()

        self._last_flow = flow.detach()
        return flow
