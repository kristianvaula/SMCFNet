"""ECM optical-flow estimator.

This is the flow network previously embedded inside `smcfnet.model.ecm.EventCorrectionModule`,
extracted into a standalone `FlowEstimator` so it can be used by a generic
motion compensation compositor.

It predicts dense optical flow from an event voxel grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from smcfnet.model.network_blocks import BaseConv
from smcfnet.model.ofe import FlowEstimator, register_estimator


@register_estimator("ecm")
class ECMFlowEstimator(FlowEstimator):
    """Lightweight CNN flow predictor used by ECM."""

    def __init__(
        self,
        num_bins: int = 5,
        mid_channels: int = 32,
        act: str = "silu",
        mode: str = "ann",
        **kwargs,
    ):
        super().__init__()
        self.num_bins = num_bins

        self.conv1 = BaseConv(num_bins, mid_channels, ksize=3, stride=1, act=act)
        self.conv2 = BaseConv(mid_channels, mid_channels * 2, ksize=3, stride=1, act=act)
        self.conv3 = BaseConv(mid_channels * 2, mid_channels, ksize=3, stride=1, act=act)
        self.flow_head = nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1, bias=True)

        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """No-op — ECM is ANN-only."""
        self._mode = "ann"

    def get_mode(self) -> str:
        return "ann"

    def reset_state(self) -> None:
        """No temporal state to clear."""
        pass

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        x = self.conv1(voxel)
        x = self.conv2(x)
        x = self.conv3(x)
        flow = self.flow_head(x)
        self._last_flow = flow.detach()
        return flow


@register_estimator("ecm_peod")
class PEODECMFlowEstimator(ECMFlowEstimator):
    """PEOD full-resolution ECM estimator.

    ECM is fully convolutional, so PEOD compatibility is primarily a clear
    registry alias with 720x1280 geometry metadata for configs and diagnostics.
    The inherited parameter names are unchanged, so ECM checkpoints remain
    compatible between ``ecm`` and ``ecm_peod`` when channel counts match.
    """

    def __init__(
        self,
        num_bins: int = 5,
        mid_channels: int = 32,
        act: str = "silu",
        mode: str = "ann",
        image_height: int = 720,
        image_width: int = 1280,
        **kwargs,
    ):
        super().__init__(
            num_bins=num_bins,
            mid_channels=mid_channels,
            act=act,
            mode=mode,
            **kwargs,
        )
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        if self.image_height != 720 or self.image_width != 1280:
            raise ValueError(
                "ecm_peod expects PEOD event geometry (height=720, width=1280), "
                f"got ({self.image_height}, {self.image_width})"
            )
