"""Event Correction Module (ECM) as a MotionCompensationModule implementation.

ECM is a specialization of the generic motion compensation compositor:

  voxel -> ECMFlowEstimator -> warper -> corrected -> ECM loss

The flow network itself lives in `smcfnet.model.ofe.ecm_ofe.ECMFlowEstimator`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from smcfnet.model.mcm.mcm import MotionCompensationModule
from smcfnet.model.ofe.ecm_ofe import ECMFlowEstimator, PEODECMFlowEstimator


class _ECMContrastSmoothLoss(nn.Module):
    """Legacy ECM loss: -Var(IWE) + TV(flow).

    Kept to preserve historical ECM behavior (raw variance, not log-variance).
    """

    def __init__(
        self,
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.01,
        contrast_clamp: float = 50.0,
    ):
        super().__init__()
        self.contrast_weight = contrast_weight
        self.smooth_weight = smooth_weight
        self.contrast_clamp = contrast_clamp

    def forward(self, corrected: torch.Tensor, flow: torch.Tensor):
        iwe = corrected.sum(dim=1)  # (B, H, W)
        iwe_flat = iwe.view(iwe.shape[0], -1)
        contrast_loss = -iwe_flat.var(dim=1).mean()
        if self.contrast_clamp > 0:
            contrast_loss = contrast_loss.clamp(min=-self.contrast_clamp)

        flow_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
        flow_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
        smooth_loss = flow_dx + flow_dy

        weighted_contrast = self.contrast_weight * contrast_loss
        weighted_smooth = self.smooth_weight * smooth_loss
        total = weighted_contrast + weighted_smooth

        return total, {
            "contrast": weighted_contrast.detach(),
            "smooth": weighted_smooth.detach(),
        }


class EventCorrectionModule(MotionCompensationModule):
    """Pre-backbone event voxel correction via contrast maximization.

    This is a convenience subclass that wires ECMFlowEstimator + legacy
    contrast-smooth loss into the generic MotionCompensationModule.

    For new code, prefer building the MCM directly via ``_build_mcm(config, variant='ecm')``
    which uses the registry-based flow estimator and CMaxLoss.
    """

    def __init__(
        self,
        num_bins: int = 5,
        mid_channels: int = 32,
        act: str = "silu",
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.01,
        contrast_clamp: float = 50.0,
    ):
        estimator = ECMFlowEstimator(num_bins=num_bins, mid_channels=mid_channels, act=act)
        loss_fn = _ECMContrastSmoothLoss(
            contrast_weight=contrast_weight,
            smooth_weight=smooth_weight,
            contrast_clamp=contrast_clamp,
        )
        super().__init__(estimator=estimator, warper=None, loss_fn=loss_fn)

    def forward(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        corrected, loss = super().forward(voxel)
        return corrected, loss

    # Backward-compat helper used by tests
    def _predict_flow(self, voxel: torch.Tensor) -> torch.Tensor:
        """Predict optical flow without warping (for testing/visualization)."""
        return self.estimator(voxel)


class PEODEventCorrectionModule(MotionCompensationModule):
    """Legacy ECM wrapper configured for PEOD 720x1280 events.

    Prefer ``model.mcm.architecture: ecm_peod`` in new YAML configs. This class
    exists for direct construction paths that still import the concrete ECM
    module rather than using the registry-based MCM builder.
    """

    def __init__(
        self,
        num_bins: int = 5,
        mid_channels: int = 32,
        act: str = "silu",
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.01,
        contrast_clamp: float = 50.0,
        image_height: int = 720,
        image_width: int = 1280,
    ):
        estimator = PEODECMFlowEstimator(
            num_bins=num_bins,
            mid_channels=mid_channels,
            act=act,
            image_height=image_height,
            image_width=image_width,
        )
        loss_fn = _ECMContrastSmoothLoss(
            contrast_weight=contrast_weight,
            smooth_weight=smooth_weight,
            contrast_clamp=contrast_clamp,
        )
        super().__init__(estimator=estimator, warper=None, loss_fn=loss_fn)

    def forward(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        corrected, loss = super().forward(voxel)
        return corrected, loss

    def _predict_flow(self, voxel: torch.Tensor) -> torch.Tensor:
        return self.estimator(voxel)
