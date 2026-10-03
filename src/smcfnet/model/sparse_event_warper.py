"""Sparse event warping module following the MCFNet ECM strategy.

Warps raw events directly using optical flow, then re-voxelizes them
into a dense tensor for backbone processing. This preserves temporal
information during warping, unlike voxel-grid warping.

Key difference from voxel-grid warping:
- Events are warped directly (forward scatter) in sparse domain
- Temporal information is preserved per-event
- Events are then re-voxelized to create the backbone input
- This matches the original MCFNet ECM pipeline
"""

from __future__ import annotations

from typing import Tuple

import torch

from smcfnet.model.event_outputs import VoxelGridEventOutput


class SparseEventWarpingModule(VoxelGridEventOutput):
    """Warp raw events directly using optical flow and re-voxelize.

    This module implements the ECM (Event Correction Module) strategy from
    the MCFNet paper: events are warped directly (forward scatter) and then
    re-voxelized into a dense tensor for backbone processing.

    The key difference from voxel-grid warping (warp_voxel_bins):
    - Warps raw events (preserving per-event timestamps and polarities)
    - Uses forward splatting (push) instead of backward sampling (pull)
    - Re-voxelizes warped events to reconstruct temporal bins

    Args:
        num_bins: Number of temporal bins in the output voxel grid.
        res: (H, W) image resolution.
        flow_scaling: Scaling factor applied to optical flow. The original
            MCFNet uses ``max(H, W) = 640``. Set to ``1.0`` if flow is
            already in pixel units (pre-scaled by MCM).
        t_ref: Reference time for warping. ``1.0`` (default) = forward
            (events warped toward end of window), ``0.0`` = backward
            (warped toward start).
    """

    def __init__(
        self,
        num_bins: int = 5,
        res: Tuple[int, int] = (480, 640),
        flow_scaling: float = 1.0,
        t_ref: float = 1.0,
    ):
        super().__init__(
            num_bins=num_bins,
            res=res,
            flow_scaling=flow_scaling,
            t_ref=t_ref,
        )

    def forward(self, voxel: torch.Tensor, flow: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(flow=flow, voxel=voxel, **kwargs)
