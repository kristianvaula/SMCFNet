"""
Event Dynamic Upsampling Module (EDUM).

CARAFE-style content-aware upsampling for event voxel grids, guided by
RGB spatial structure.  Upsamples from native event camera resolution
(e.g. 480x640) to full RGB resolution (e.g. 1080x1440) using a two-stage
approach:

  1. Integer-scale CARAFE reassembly (2x) with learned per-pixel kernels.
  2. Bilinear interpolation to the exact target resolution.

The kernel predictor fuses compressed event features with an RGB edge
encoder so that upsampling preserves edges visible in the RGB image and
suppresses noise amplification in sparse event regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.network_blocks import BaseConv


class EventDynamicUpsamplingModule(nn.Module):
    """CARAFE-style upsampling for event voxel grids with RGB guidance.

    Args:
        num_bins: Number of temporal bins in the voxel grid.
        target_height: Target output height (RGB resolution).
        target_width: Target output width (RGB resolution).
        mid_channels: Internal feature channel width.
        kernel_size: Size of the CARAFE reassembly kernel (k).
        scale_factor: Integer upsampling factor for CARAFE stage.
        rgb_in_channels: Number of RGB input channels.
        act: Activation function for BaseConv layers.
    """

    def __init__(
        self,
        num_bins: int = 5,
        target_height: int = 1080,
        target_width: int = 1440,
        mid_channels: int = 32,
        kernel_size: int = 3,
        scale_factor: int = 2,
        rgb_in_channels: int = 3,
        act: str = "silu",
    ):
        super().__init__()
        self.num_bins = num_bins
        self.target_height = target_height
        self.target_width = target_width
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # --- A. Event channel compressor ---
        self.event_compressor = BaseConv(
            num_bins, mid_channels, ksize=1, stride=1, act=act,
        )

        # --- B. RGB edge encoder (downsample to event resolution) ---
        self.rgb_encoder = nn.Sequential(
            BaseConv(rgb_in_channels, mid_channels, ksize=3, stride=2, act=act),
            BaseConv(mid_channels, mid_channels, ksize=3, stride=1, act=act),
            # AdaptiveAvgPool is applied dynamically in forward to match
            # the exact event spatial size.
        )

        # --- C. Kernel predictor ---
        # Input: concatenated event + RGB features (2 * mid_channels)
        # Output: k^2 * scale^2 kernel weights per spatial location
        kernel_channels = kernel_size * kernel_size * scale_factor * scale_factor
        self.kernel_predictor = nn.Sequential(
            BaseConv(mid_channels * 2, mid_channels * 2, ksize=3, stride=1, act=act),
            nn.Conv2d(
                mid_channels * 2,
                kernel_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, event_voxel: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_voxel: ``(B, num_bins, H_ev, W_ev)`` at native event resolution.
            rgb: ``(B, 3, H_rgb, W_rgb)`` at full RGB resolution.

        Returns:
            Upsampled voxel grid ``(B, num_bins, target_height, target_width)``.
        """
        B, C, H_ev, W_ev = event_voxel.shape

        # 1. Compress event channels
        event_feat = self.event_compressor(event_voxel)  # (B, mid, H_ev, W_ev)

        # 2. Extract RGB guidance features at event resolution
        rgb_feat = self.rgb_encoder(rgb)  # (B, mid, ~H_rgb/2, ~W_rgb/2)
        rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (H_ev, W_ev))  # (B, mid, H_ev, W_ev)

        # 3. Predict reassembly kernels
        combined = torch.cat([event_feat, rgb_feat], dim=1)  # (B, 2*mid, H_ev, W_ev)
        kernels = self.kernel_predictor(combined)  # (B, k^2*s^2, H_ev, W_ev)
        kernels = F.softmax(kernels, dim=1)  # normalise over kernel neighbourhood

        # 4. CARAFE reassembly (integer scale)
        upsampled = self._carafe_reassemble(event_voxel, kernels, H_ev, W_ev)
        # → (B, num_bins, H_ev * scale, W_ev * scale)

        # 5. Bilinear fine-adjust to exact target resolution
        if upsampled.shape[2] != self.target_height or upsampled.shape[3] != self.target_width:
            upsampled = F.interpolate(
                upsampled,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False,
            )

        return upsampled

    # ------------------------------------------------------------------
    # CARAFE reassembly
    # ------------------------------------------------------------------

    def _carafe_reassemble(
        self,
        x: torch.Tensor,
        kernels: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Content-aware reassembly (CARAFE core operation).

        For each output pixel the predicted kernel specifies weights over a
        ``kernel_size x kernel_size`` neighbourhood in the input.  The
        kernels are shared across all ``num_bins`` channels.

        Args:
            x: Input tensor ``(B, C, H, W)``.
            kernels: Predicted kernel weights ``(B, k^2 * s^2, H, W)``
                     (already softmax-normalised).
            H, W: Spatial dimensions of x.

        Returns:
            Upsampled tensor ``(B, C, H * scale, W * scale)``.
        """
        B, C, _, _ = x.shape
        k = self.kernel_size
        s = self.scale_factor
        pad = k // 2

        # Unfold input into overlapping patches: (B, C, H, W) → (B, C*k*k, H*W)
        x_padded = F.pad(x, [pad] * 4, mode="replicate")
        x_unfolded = F.unfold(x_padded, kernel_size=k, stride=1)  # (B, C*k*k, H*W)
        x_unfolded = x_unfolded.view(B, C, k * k, H * W)  # (B, C, k*k, H*W)

        # Reshape kernels: (B, k*k*s*s, H, W) → (B, k*k, s*s, H*W)
        kernels = kernels.view(B, k * k, s * s, H * W)

        # Weighted sum: for each of the s*s sub-pixel positions, compute
        # the weighted combination over the k*k neighbourhood.
        # x_unfolded: (B, C, k*k, H*W)
        # kernels:    (B, 1, k*k, s*s * H*W)   (after rearrange)
        # → output:   (B, C, s*s, H*W)

        # Rearrange kernels for broadcasting: (B, k*k, s*s, H*W) → (B, 1, k*k, s*s*H*W)
        # Actually simpler: einsum over the k*k dimension
        # out_{b,c,p,n} = sum_j  x_{b,c,j,n} * kernel_{b,j,p,n}
        # where j=k*k, p=s*s, n=H*W
        out = torch.einsum("bcjn,bjpn->bcpn", x_unfolded, kernels)  # (B, C, s*s, H*W)

        # Reshape to spatial: (B, C, s*s, H, W) → pixel shuffle → (B, C, H*s, W*s)
        out = out.view(B, C, s * s, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, s*s)
        out = out.view(B, C, H, W, s, s)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, H, s, W, s)
        out = out.view(B, C, H * s, W * s)

        return out
