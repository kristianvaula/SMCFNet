"""Shared voxel-grid warping utilities.

Used by MCM to warp temporal bins to a reference time
using predicted optical flow.
"""

import torch
import torch.nn.functional as F


def warp_voxel_bins(
    voxel: torch.Tensor, flow: torch.Tensor, t_ref: float = 1.0,
) -> torch.Tensor:
    """Warp each temporal bin to a reference time.

    Assumes constant-velocity motion over the event window.
    Bin *k* is displaced by ``flow * (t_ref - t_k)`` where
    ``t_k = k / (num_bins - 1)``.

    Args:
        voxel: Event voxel grid, shape ``(B, C, H, W)`` where C = num_bins.
        flow: Dense optical flow, shape ``(B, 2, H, W)`` in pixel units.
        t_ref: Reference time to warp towards.  ``1.0`` (default) warps to
            the last bin (forward), ``0.0`` warps to the first bin (backward).

    Returns:
        Warped voxel grid with the same shape as *voxel*.
    """
    B, C, H, W = voxel.shape

    # Build a normalised identity grid once (cached across bins)
    # grid_sample expects (B, H, W, 2) with ordering (x, y) in [-1, 1]
    gy = torch.linspace(-1.0, 1.0, H, device=voxel.device, dtype=voxel.dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=voxel.device, dtype=voxel.dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) [x, y]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # Convert pixel flow to normalised flow (grid_sample coords are -1..1)
    # flow[:, 0] = dx (pixels), flow[:, 1] = dy (pixels)
    norm_flow_x = flow[:, 0:1] * (2.0 / max(W - 1, 1))  # (B, 1, H, W)
    norm_flow_y = flow[:, 1:2] * (2.0 / max(H - 1, 1))  # (B, 1, H, W)

    corrected_bins = []
    for k in range(C):
        t_k = k / max(C - 1, 1)
        delta_t = t_ref - t_k  # >0 for earlier bins

        # Displacement for this bin (normalised coords)
        dx = (norm_flow_x * delta_t).squeeze(1)  # (B, H, W)
        dy = (norm_flow_y * delta_t).squeeze(1)

        # grid_sample expects (B, H, W, 2) with ordering (x, y)
        sample_grid = base_grid.clone()
        sample_grid[..., 0] = sample_grid[..., 0] + dx  # x-coord
        sample_grid[..., 1] = sample_grid[..., 1] + dy  # y-coord

        # padding_mode="zeros": events warped outside the image boundary
        # contribute nothing.  "border" would replicate edge pixels,
        # enabling a trivial CMax solution (Hagenaars et al., 2021).
        warped = F.grid_sample(
            voxel[:, k : k + 1],
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        corrected_bins.append(warped)

    return torch.cat(corrected_bins, dim=1)
