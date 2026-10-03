#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from __future__ import annotations

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


# ---------------------------------------------------------------------------
# Motion Compensation Losses (Axis C) — used by MCM
# ---------------------------------------------------------------------------

class MotionCompensationLoss(nn.Module):
    """Base class for motion compensation loss functions."""

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss.

        Args:
            corrected: Motion-corrected voxel, ``(B, C, H, W)``.
            flow: Predicted optical flow, ``(B, 2, H, W)``.
            flow_gt: Optional ground-truth flow for supervised losses.
            **kwargs: Additional context (e.g. ``voxel`` for bidirectional
                contrast).

        Returns:
            Tuple of (scalar loss, dict of named loss components for logging).
        """
        raise NotImplementedError


class CMaxLoss(MotionCompensationLoss):
    """Contrast maximization + Charbonnier smoothness (self-supervised).

    Loss = contrast_weight * L_contrast + smooth_weight * L_smooth
         + magnitude_weight * L_magnitude
         + magnitude_weight * L_magnitude

    - L_contrast = -log(Var(IWE) + eps), clamped to [-contrast_clamp, 0]
    - L_smooth = mean(sqrt(flow_dx^2 + flow_dy^2 + eps))
    - L_magnitude = mean(||flow||_2)   (flow magnitude regularization)
    - L_magnitude = mean(||flow||_2)   (flow magnitude regularization)

    Uses log-variance instead of raw variance to produce self-normalizing
    gradients: dL/dI_j = 2(I_j - mu) / (N * Var(I)), which shrinks as
    variance grows — preventing the gradient explosions seen with raw
    variance when events align sharply.

    The magnitude term acts as a prior that favors smaller flows, replacing
    the implicit magnitude constraint that supervised L1 loss provides in
    the original IDNet training.  Without it the self-supervised contrast
    loss can drive flow magnitudes to diverge because CMax only rewards
    event alignment sharpness, not small flow.

    When ``use_backward=True`` (default), events are warped in **both**
    temporal directions (forward to t=1 and backward to t=0) following
    the bidirectional formulation from Zhu et al., CVPR'19 used in the
    original MCFNet.  This prevents degenerate solutions where the network
    pushes all events in one direction to trivially maximize forward
    contrast while producing incorrect flow.

    Args:
        contrast_weight: Weight for contrast (sharpness) loss.
        smooth_weight: Weight for Charbonnier smoothness loss.
        contrast_clamp: Clamp contrast loss to prevent spikes.
        magnitude_weight: Weight for L2 flow magnitude penalty (0 = disabled).
        use_backward: Bidirectional contrast (forward + backward warping).
    """

    def __init__(
        self,
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.1,
        contrast_clamp: float = 10.0,
        magnitude_weight: float = 0.0,
        use_backward: bool = True,
    ):
        super().__init__()
        self.contrast_weight = contrast_weight
        self.smooth_weight = smooth_weight
        self.contrast_clamp = contrast_clamp
        self.magnitude_weight = magnitude_weight
        self.use_backward = use_backward

    @staticmethod
    def _iwe_contrast(corrected: torch.Tensor) -> torch.Tensor:
        """Compute -log(Var(IWE) / active_frac + eps) for a warped voxel."""
        iwe = corrected.sum(dim=1)  # (B, H, W)
        iwe_flat = iwe.view(iwe.shape[0], -1)
        active = (iwe_flat.abs() > 1e-8).float()
        active_frac = active.mean(dim=1).clamp(min=0.01)
        return -(iwe_flat.var(dim=1) / active_frac + 1e-6).log().mean()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        # Forward contrast: events warped to t_ref=1 (already computed)
        fwd_contrast = self._iwe_contrast(corrected)

        # Backward contrast: events warped to t_ref=0
        # Requires the original (unwarped) voxel, passed via kwargs by MCM.
        voxel = kwargs.get("voxel", None)
        if self.use_backward and voxel is not None:
            from smcfnet.model.warping import warp_voxel_bins
            corrected_bwd = warp_voxel_bins(voxel, flow, t_ref=0.0)
            bwd_contrast = self._iwe_contrast(corrected_bwd)
            contrast_loss = (fwd_contrast + bwd_contrast) * 0.5
        else:
            contrast_loss = fwd_contrast

        if self.contrast_clamp > 0:
            contrast_loss = contrast_loss.clamp(min=-self.contrast_clamp)

        # Charbonnier smoothness loss
        eps = 1e-6
        flow_dx = flow[:, :, :-1, 1:] - flow[:, :, :-1, :-1]
        flow_dy = flow[:, :, 1:, :-1] - flow[:, :, :-1, :-1]
        smooth_loss = torch.sqrt(flow_dx.pow(2) + flow_dy.pow(2) + eps).mean()

        weighted_contrast = self.contrast_weight * contrast_loss
        weighted_smooth = self.smooth_weight * smooth_loss
        total = weighted_contrast + weighted_smooth

        loss_dict = {
            "contrast": weighted_contrast.detach(),
            "smooth": weighted_smooth.detach(),
        }

        # Flow magnitude regularization: mean per-pixel L2 norm
        if self.magnitude_weight > 0:
            magnitude_loss = torch.norm(flow, p=2, dim=1).mean()
            weighted_magnitude = self.magnitude_weight * magnitude_loss
            total = total + weighted_magnitude
            loss_dict["magnitude"] = weighted_magnitude.detach()

        return total, loss_dict

        # Flow magnitude regularization: mean per-pixel L2 norm
        if self.magnitude_weight > 0:
            magnitude_loss = torch.norm(flow, p=2, dim=1).mean()
            weighted_magnitude = self.magnitude_weight * magnitude_loss
            total = total + weighted_magnitude
            loss_dict["magnitude"] = weighted_magnitude.detach()

        return total, loss_dict


class SupervisedFlowLoss(MotionCompensationLoss):
    """End-Point Error (EPE) loss for supervised flow training.

    Loss = mean(||flow - flow_gt||_2)  over valid pixels only.

    When ``max_epe`` is set, per-pixel EPE values beyond that threshold are
    clamped.  This prevents the enormous loss values that occur when the
    flow estimator predicts maximum-magnitude flow early in training
    (e.g. EPE ≈ max_flow ≈ 400), which cause gradient explosions and
    eventual NaN divergence.

    Requires ground-truth flow (``flow_gt``) to be provided.
    The flow_gt tensor can be ``(B, 2, H, W)`` or ``(B, 3, H, W)`` where
    the 3rd channel is a validity mask (1 = valid, 0 = invalid).
    """

    def __init__(self, max_epe: float = 128.0, eps: float = 1e-6):
        super().__init__()
        self.max_epe = max_epe
        self.eps = float(eps)

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if flow_gt is None:
            raise ValueError("SupervisedFlowLoss requires flow_gt")

        # Separate validity mask when encoded in the 3rd channel.
        if flow_gt.shape[1] == 3:
            valid = flow_gt[:, 2:3, :, :] > 0.5  # (B,1,H,W)
            flow_gt = flow_gt[:, :2, :, :]
        else:
            valid = None

        diff = flow - flow_gt
        # ``torch.norm`` has an undefined derivative at exactly zero EPE. DSEC
        # flow masks commonly leave zero-valued invalid regions, and even when
        # the scalar loss is finite those pixels can poison backward with NaNs.
        epe_map = torch.sqrt(diff.pow(2).sum(dim=1, keepdim=True) + self.eps)  # (B,1,H,W)

        # Soft clamp: linear below max_epe, logarithmic above.
        # A hard clamp has zero gradient when EPE > max_epe, which kills
        # learning entirely at initialization (all pixels exceed the cap).
        # The log transition keeps gradient alive: grad = max_epe / epe,
        # decaying smoothly but never reaching zero.
        if self.max_epe > 0:
            epe_map = torch.where(
                epe_map <= self.max_epe,
                epe_map,
                self.max_epe * (1.0 + torch.log(epe_map / self.max_epe)),
            )

        if valid is not None and valid.any():
            epe = epe_map[valid].mean()
        else:
            epe = epe_map.mean()

        return epe, {"epe": epe.detach()}


class SupervisedFlowL1Loss(MotionCompensationLoss):
    """L1 endpoint error for supervised flow training.

    Loss = mean(||flow - flow_gt||_1) over valid pixels only.

    This matches the L1 supervision used by several OFE papers.
    """

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if flow_gt is None:
            raise ValueError("SupervisedFlowL1Loss requires flow_gt")

        if flow_gt.shape[1] == 3:
            valid = flow_gt[:, 2:3, :, :] > 0.5
            flow_gt = flow_gt[:, :2, :, :]
        else:
            valid = None

        l1_map = torch.abs(flow - flow_gt).sum(dim=1, keepdim=True)

        if valid is not None and valid.any():
            l1 = l1_map[valid].mean()
        else:
            l1 = l1_map.mean()

        return l1, {"l1": l1.detach()}


class SupervisedFlowL1TIDLoss(MotionCompensationLoss):
    """Temporal iterative deblurring (TID) L1 supervision.

    Applies a weighted L1 loss across a sequence of flow predictions
    using a geometric series with factor ``gamma``. If flow history is
    unavailable, falls back to the final flow L1 loss.
    """

    def __init__(self, gamma: float = 0.8):
        super().__init__()
        self.gamma = float(gamma)
        self.l1 = SupervisedFlowL1Loss()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if flow_gt is None:
            raise ValueError("SupervisedFlowL1TIDLoss requires flow_gt")

        flow_history = kwargs.get("flow_history")
        if flow_history is None:
            loss, loss_dict = self.l1(corrected, flow, flow_gt)
            return loss, {"tid_l1": loss_dict["l1"]}

        if flow_gt.shape[1] == 3:
            valid = flow_gt[:, 2:3, :, :] > 0.5
            flow_gt = flow_gt[:, :2, :, :]
        else:
            valid = None

        num_steps = int(flow_history.shape[1])
        weights = [self.gamma ** (num_steps - 1 - idx) for idx in range(num_steps)]
        weights_t = flow_history.new_tensor(weights).view(1, num_steps, 1, 1, 1)
        weights_sum = float(sum(weights)) if num_steps > 0 else 1.0

        diff = torch.abs(flow_history - flow_gt.unsqueeze(1)).sum(dim=2, keepdim=True)

        if valid is not None and valid.any():
            valid_exp = valid.unsqueeze(1).float()
            diff = diff * valid_exp
            denom = valid_exp.sum(dim=(2, 3, 4)).clamp(min=1.0)
            per_step = diff.sum(dim=(2, 3, 4)) / denom
        else:
            per_step = diff.mean(dim=(2, 3, 4))

        weighted = (per_step * weights_t.view(1, num_steps)).sum(dim=1) / weights_sum
        weighted = weighted.mean()

        return weighted, {"tid_l1": weighted.detach()}


class AngularLoss(MotionCompensationLoss):
    """Angular error loss for optical flow (direction only, ignores magnitude).

    Loss = 1 - (flow · flow_gt) / (||flow|| * ||flow_gt||)

    This penalizes directional mismatch while being invariant to flow magnitude.
    Useful as a weak regularizer to prevent flow collapse while preserving
    motion-compensation-driven optimization (e.g., via CMax).

    Only operates on valid GT pixels. Returns 0 when flow_gt is unavailable.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        if flow_gt is None:
            return torch.tensor(0.0, device=flow.device), {"angular": torch.tensor(0.0, device=flow.device)}

        # Separate validity mask if encoded in 3rd channel
        if flow_gt.shape[1] == 3:
            valid = flow_gt[:, 2:3, :, :] > 0.5  # (B,1,H,W)
            flow_gt = flow_gt[:, :2, :, :]
        else:
            valid = None

        # Reshape to (B*H*W, 2) for cosine similarity
        B, C, H, W = flow.shape
        flow_flat = flow.view(B, 2, -1).transpose(1, 2)  # (B, H*W, 2)
        flow_gt_flat = flow_gt.view(B, 2, -1).transpose(1, 2)  # (B, H*W, 2)

        # Compute cosine similarity: (pred · gt) / (||pred|| * ||gt||)
        # torch.nn.functional.cosine_similarity returns (B, H*W)
        cos_sim = torch.nn.functional.cosine_similarity(
            flow_flat, flow_gt_flat, dim=2
        )  # (B, H*W)

        # Angular loss: 1 - cosine_similarity
        angular_error = 1.0 - cos_sim  # (B, H*W)

        # Apply validity mask if present
        if valid is not None:
            valid_flat = valid.view(B, -1)  # (B, H*W)
            if valid_flat.any():
                angular_error = angular_error[valid_flat]
        else:
            angular_error = angular_error.view(-1)

        angular_loss = angular_error.mean()

        return angular_loss, {"angular": angular_loss.detach()}


class SupervisedAngularLoss(MotionCompensationLoss):
    """Weighted combination of supervised EPE and angular loss.

    Loss = alpha * EPE + beta * Angular
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, max_epe: float = 128.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.supervised = SupervisedFlowLoss(max_epe=max_epe)
        self.angular = AngularLoss()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        sup_loss, sup_dict = self.supervised(corrected, flow, flow_gt)
        total = self.alpha * sup_loss
        log = {f"sup_{k}": v for k, v in sup_dict.items()}

        if flow_gt is not None:
            ang_loss, ang_dict = self.angular(corrected, flow, flow_gt)
            total = total + self.beta * ang_loss
            log.update({f"ang_{k}": v for k, v in ang_dict.items()})

        return total, log


class CombinedLoss(MotionCompensationLoss):
    """Weighted combination of self-supervised (CMax) and supervised (EPE) loss.

    Loss = alpha * CMax + beta * EPE

    Falls back to CMax-only when flow_gt is not available.

    Args:
        alpha: Weight for CMax loss.
        beta: Weight for supervised EPE loss.
        contrast_weight: Passed to CMaxLoss.
        smooth_weight: Passed to CMaxLoss.
        contrast_clamp: Passed to CMaxLoss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.1,
        contrast_clamp: float = 10.0,
        magnitude_weight: float = 0.0,
        use_backward: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cmax = CMaxLoss(
            contrast_weight, smooth_weight, contrast_clamp,
            magnitude_weight=magnitude_weight, use_backward=use_backward,
        )
        self.supervised = SupervisedFlowLoss()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        cmax_loss, cmax_dict = self.cmax(corrected, flow, flow_gt, **kwargs)
        total = self.alpha * cmax_loss
        log = {f"cmax_{k}": v for k, v in cmax_dict.items()}

        if flow_gt is not None:
            sup_loss, sup_dict = self.supervised(corrected, flow, flow_gt)
            total = total + self.beta * sup_loss
            log.update({f"sup_{k}": v for k, v in sup_dict.items()})

        return total, log


class CMaxAngularLoss(MotionCompensationLoss):
    """Weighted combination of self-supervised (CMax) and angular loss.

    Loss = alpha * CMax + beta * Angular

    CMax is the primary objective (motion compensation / event sharpness).
    Angular loss is a weak directional regularizer that prevents collapse
    without constraining flow magnitude.

    Falls back to CMax-only when flow_gt is not available.

    Args:
        alpha: Weight for CMax loss (primary objective).
        beta: Weight for angular loss (directional regularizer).
        contrast_weight: Passed to CMaxLoss.
        smooth_weight: Passed to CMaxLoss.
        contrast_clamp: Passed to CMaxLoss.
        magnitude_weight: Passed to CMaxLoss.
        use_backward: Passed to CMaxLoss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.01,
        contrast_weight: float = 1.0,
        smooth_weight: float = 0.1,
        contrast_clamp: float = 10.0,
        magnitude_weight: float = 0.0,
        use_backward: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cmax = CMaxLoss(
            contrast_weight, smooth_weight, contrast_clamp,
            magnitude_weight=magnitude_weight, use_backward=use_backward,
        )
        self.angular = AngularLoss()

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        cmax_loss, cmax_dict = self.cmax(corrected, flow, flow_gt, **kwargs)
        total = self.alpha * cmax_loss
        log = {f"cmax_{k}": v for k, v in cmax_dict.items()}

        if flow_gt is not None:
            ang_loss, ang_dict = self.angular(corrected, flow, flow_gt)
            total = total + self.beta * ang_loss
            log.update({f"ang_{k}": v for k, v in ang_dict.items()})

        return total, log


class EventWarpingLoss(MotionCompensationLoss):
    """Original MCFNet averaged-timestamp loss (Zhu et al., CVPR'19).

    Operates on **raw events** via forward splatting, preserving per-event
    timing and polarity information.  This is the loss used in the original
    MCFNet ECM pipeline.

    For each direction (forward ``tref=1``, backward ``tref=0``):

    1. Look up per-event flow from the dense flow map.
    2. Forward-warp events via bilinear splatting to produce per-polarity
       IWEs (Image of Warped Events) and timestamp-weighted IWEs.
    3. Compute per-pixel average timestamp: ``t_avg = sum(w*t) / sum(w)``.
    4. Loss = sum of squared average timestamps, normalized by the number
       of non-zero pixels.

    Default total loss (bidirectional + normalized smoothness)::

        L = L_fw + L_bw + smooth_weight * (mean(abs(dx)) + mean(abs(dy)))

    With ``smoothness_mode="original"``, the smoothness term follows the
    original MCFNet ECM implementation::

        L_smooth = smooth_weight * (sum(sqrt(dx^2 + 1e-6)) + sum(sqrt(dy^2 + 1e-6))) / C

    Requires ``events`` (packed or batched) and ``pol_mask`` to be passed
    via kwargs by the MCM compositor.

    Args:
        res: (H, W) image resolution.
        smooth_weight: Weight for ECM-style mean-gradient smoothness regularization.
        flow_scaling: Scalar multiplier for optical flow.  The original
            MCFNet uses ``max(H, W) = 640``.  Set to ``1.0`` if the flow
            is already in pixel units (pre-scaled by the MCM).
        loss_scaling: Normalize loss by the number of non-zero pixels
            (default True, matching the original MCFNet).
        smoothness_mode: ``"mean_l1"`` keeps the current normalized L1
            smoothness. ``"original"`` restores the original MCFNet ECM
            smoothness: summed Charbonnier gradients divided by channel count.
            ``"original_edge_aware"`` keeps that same summed-Charbonnier,
            divide-by-channel-count scale (so ``smooth_weight`` values tuned
            for ``"original"`` stay roughly valid) but multiplies each
            per-pixel Charbonnier term by the same density-based edge weight
            used in ``"edge_aware_l1"`` before summing.
        center_weight: Weight for the mean-flow centering penalty (0.0 = off,
            matching all existing configs). Penalizes predictions where the
            flow is dominated by its own spatial mean (DC/global-shift
            collapse) rather than spatially-varying structure. This does
            **not** force zero-mean flow — a real global ego-motion
            component is legitimate — it only penalizes the *ratio* of
            locally-varying flow to total flow magnitude staying below
            ``center_target_ratio``.
        center_min_flow_mag: Gate threshold (in flow_scaling-applied pixel
            units). The penalty is only applied to samples whose mean flow
            magnitude exceeds this value, so it cannot suppress genuine
            signal during early training when predicted flow is still near
            zero everywhere (gate is computed with ``.detach()``, so it does
            not itself contribute gradient).
        center_target_ratio: Target floor for centered_ratio (locally-varying
            flow magnitude / total flow magnitude). Matches
            ``evolution.metrics.FLOW_CENTERED_RATIO_MIN`` by convention so the
            training-time penalty and the EA fitness guardrail push toward
            the same operating point.
    """

    def __init__(
        self,
        res: tuple[int, int] = (480, 640),
        smooth_weight: float = 0.001,
        flow_scaling: float = 1.0,
        loss_scaling: bool = True,
        smoothness_mode: str = "mean_l1",
        edge_weight_gamma: float = 5.0,
        center_weight: float = 0.0,
        center_min_flow_mag: float = 0.05,
        center_target_ratio: float = 0.25,
    ):
        super().__init__()
        self.res = list(res)
        self.flow_scaling = float(flow_scaling)
        self.smooth_weight = float(smooth_weight)
        self.loss_scaling = bool(loss_scaling)
        self.edge_weight_gamma = float(edge_weight_gamma)
        self.center_weight = float(center_weight)
        if self.center_weight < 0.0:
            raise ValueError("center_weight must be >= 0")
        self.center_min_flow_mag = float(center_min_flow_mag)
        if self.center_min_flow_mag < 0.0:
            raise ValueError("center_min_flow_mag must be >= 0")
        self.center_target_ratio = float(center_target_ratio)
        if not (0.0 <= self.center_target_ratio <= 1.0):
            raise ValueError("center_target_ratio must be in [0, 1]")
        mode = str(smoothness_mode or "mean_l1").lower()
        aliases = {
            "mean": "mean_l1",
            "l1_mean": "mean_l1",
            "current": "mean_l1",
            "original_ecm": "original",
            "mcfnet": "original",
            "sum_charbonnier": "original",
            "legacy": "original",
            "edge_aware": "edge_aware_l1",
        }
        self.smoothness_mode = aliases.get(mode, mode)
        if self.smoothness_mode not in {
            "mean_l1", "original", "edge_aware_l1", "original_edge_aware",
        }:
            raise ValueError(
                "smoothness_mode must be one of {'mean_l1', 'original', "
                "'edge_aware_l1', 'original_edge_aware'}, "
                f"got {smoothness_mode!r}"
            )

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        from smcfnet.model.event_warping import (
            get_interpolation,
            interpolate_to_iwe,
            prepare_events_for_warping,
        )

        # Accept pre-batched events or packed events + offsets
        events = kwargs.get("events")
        pol_mask = kwargs.get("pol_mask")

        if events is None:
            raise ValueError(
                "EventWarpingLoss requires 'events' (packed (N,4) tensor) "
                "and 'events_offsets' in kwargs."
            )

        # If packed (N, 4) events with offsets — convert to batched format
        if events.dim() == 2:
            offsets = kwargs.get("events_offsets")
            if offsets is None:
                raise ValueError(
                    "Packed events require 'events_offsets' in kwargs."
                )
            events, pol_mask = prepare_events_for_warping(events, offsets)
        elif pol_mask is None:
            raise ValueError(
                "Batched events require 'pol_mask' in kwargs."
            )

        events = events.to(torch.float32)

        # Guard against stale/mismatched res configuration.
        if flow.ndim == 4 and (int(flow.shape[2]) != self.res[0] or int(flow.shape[3]) != self.res[1]):
            raise ValueError(
                f"EventWarpingLoss.res={self.res} does not match flow spatial dims "
                f"({int(flow.shape[2])}, {int(flow.shape[3])}). "
                "Ensure the loss is configured with the event camera resolution."
            )

        # Replicate pol_mask and timestamps for the 4 bilinear corners
        pol_mask_4 = torch.cat([pol_mask] * 4, dim=1)
        ts_list = torch.cat([events[:, :, 0:1]] * 4, dim=1)

        # ---- Per-event flow lookup ----
        flow_idx = events[:, :, 1:3].clone()       # (y, x)
        flow_idx[:, :, 0] *= self.res[1]            # y * W
        flow_idx = torch.sum(flow_idx, dim=2)        # y * W + x = linear index
        # Clamp to valid range for gather (padded events have (-1,-1) → negative idx)
        flow_idx = flow_idx.clamp(0, self.res[0] * self.res[1] - 1)

        flow_flat = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow_flat[:, 1, :], 1, flow_idx.long())
        event_flowx = torch.gather(flow_flat[:, 0, :], 1, flow_idx.long())
        event_flow = torch.cat([
            event_flowy.unsqueeze(2),
            event_flowx.unsqueeze(2),
        ], dim=2)  # (B, N, 2)

        # ---- Forward warping (tref=1) ----
        fw_idx, fw_weights = get_interpolation(
            events, event_flow, 1.0, self.res, self.flow_scaling,
        )

        fw_iwe_pos = interpolate_to_iwe(
            fw_idx.long(), fw_weights, self.res,
            polarity_mask=pol_mask_4[:, :, 0:1],
        )
        fw_iwe_neg = interpolate_to_iwe(
            fw_idx.long(), fw_weights, self.res,
            polarity_mask=pol_mask_4[:, :, 1:2],
        )

        # Timestamp-weighted IWEs
        fw_iwe_pos_ts = interpolate_to_iwe(
            fw_idx.long(), fw_weights * ts_list, self.res,
            polarity_mask=pol_mask_4[:, :, 0:1],
        )
        fw_iwe_neg_ts = interpolate_to_iwe(
            fw_idx.long(), fw_weights * ts_list, self.res,
            polarity_mask=pol_mask_4[:, :, 1:2],
        )

        # Averaged timestamps → squared
        fw_iwe_pos_ts = fw_iwe_pos_ts / (fw_iwe_pos + 1e-9)
        fw_iwe_neg_ts = fw_iwe_neg_ts / (fw_iwe_neg + 1e-9)
        fw_loss = torch.sum(fw_iwe_pos_ts ** 2, dim=1) + torch.sum(fw_iwe_neg_ts ** 2, dim=1)

        # ---- Backward warping (tref=0) ----
        bw_idx, bw_weights = get_interpolation(
            events, event_flow, 0.0, self.res, self.flow_scaling,
        )

        bw_iwe_pos = interpolate_to_iwe(
            bw_idx.long(), bw_weights, self.res,
            polarity_mask=pol_mask_4[:, :, 0:1],
        )
        bw_iwe_neg = interpolate_to_iwe(
            bw_idx.long(), bw_weights, self.res,
            polarity_mask=pol_mask_4[:, :, 1:2],
        )

        # Backward residual to tref=0: (1 - ts)
        bw_iwe_pos_ts = interpolate_to_iwe(
            bw_idx.long(), bw_weights * (1 - ts_list), self.res,
            polarity_mask=pol_mask_4[:, :, 0:1],
        )
        bw_iwe_neg_ts = interpolate_to_iwe(
            bw_idx.long(), bw_weights * (1 - ts_list), self.res,
            polarity_mask=pol_mask_4[:, :, 1:2],
        )

        bw_iwe_pos_ts = bw_iwe_pos_ts / (bw_iwe_pos + 1e-9)
        bw_iwe_neg_ts = bw_iwe_neg_ts / (bw_iwe_neg + 1e-9)
        bw_loss = torch.sum(bw_iwe_pos_ts ** 2, dim=1) + torch.sum(bw_iwe_neg_ts ** 2, dim=1)

        # ---- Loss scaling by nonzero pixel count ----
        # Spatially flatten before dividing to avoid shape mismatch when B > 1.
        # fw_loss / bw_loss are (B, H, W); normalize per sample then reduce.
        fw_loss_flat = fw_loss.view(fw_loss.shape[0], -1).sum(dim=1)   # (B,)
        bw_loss_flat = bw_loss.view(bw_loss.shape[0], -1).sum(dim=1)   # (B,)

        if self.loss_scaling:
            fw_nonzero_px = (fw_iwe_pos + fw_iwe_neg).view(fw_iwe_pos.shape[0], -1)
            fw_nonzero_count = (fw_nonzero_px > 0).float().sum(dim=1)  # (B,)
            fw_loss_flat = fw_loss_flat / (fw_nonzero_count + 1e-9)

            bw_nonzero_px = (bw_iwe_pos + bw_iwe_neg).view(bw_iwe_pos.shape[0], -1)
            bw_nonzero_count = (bw_nonzero_px > 0).float().sum(dim=1)  # (B,)
            bw_loss_flat = bw_loss_flat / (bw_nonzero_count + 1e-9)

        fw_loss = fw_loss_flat.sum()
        bw_loss = bw_loss_flat.sum()

        # ---- ECM-style smoothness ----
        # Use unscaled flow when provided to match the original MCFNet setup.
        smooth_weight_scale = kwargs.get("smooth_weight_scale", 1.0)
        if smooth_weight_scale is None:
            smooth_weight_scale = 1.0
        smooth_weight_scale = float(smooth_weight_scale)
        if smooth_weight_scale < 0.0:
            raise ValueError(f"smooth_weight_scale must be non-negative, got {smooth_weight_scale}")
        flow_for_smooth = kwargs.get("flow_unscaled", flow)
        if flow_for_smooth.ndim == 4:
            if flow_for_smooth.shape[1] != 2:
                raise ValueError(
                    "flow_unscaled must have shape (B, 2, H, W) when provided as a dense tensor, "
                    f"got {tuple(flow_for_smooth.shape)}"
                )
            flow_reshaped = flow_for_smooth
        elif flow_for_smooth.ndim == 2:
            flow_reshaped = flow_for_smooth.view(flow_for_smooth.shape[0], 2, self.res[0], self.res[1])
        else:
            raise ValueError(
                "flow_unscaled must be either a dense (B, 2, H, W) tensor or a flattened "
                f"(B, 2*H*W) tensor, got {tuple(flow_for_smooth.shape)}"
            )
        if self.smoothness_mode == "original":
            flow_dx = flow_reshaped[:, :, :-1, :] - flow_reshaped[:, :, 1:, :]
            flow_dy = flow_reshaped[:, :, :, :-1] - flow_reshaped[:, :, :, 1:]
            flow_dx = torch.sqrt(flow_dx.pow(2) + 1e-6)
            flow_dy = torch.sqrt(flow_dy.pow(2) + 1e-6)
            smooth_raw = (flow_dx.sum() + flow_dy.sum()) / flow_dx.shape[1]
        elif self.smoothness_mode == "original_edge_aware":
            # Same density-based edge weight as "edge_aware_l1", but applied
            # to the unnormalized Charbonnier terms and summed like
            # "original" so the smooth_weight scale stays comparable.
            with torch.no_grad():
                density = corrected.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
                d_max = density.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-3)
                edge_weight = torch.exp(-self.edge_weight_gamma * density / d_max)
            flow_dx = flow_reshaped[:, :, :-1, :] - flow_reshaped[:, :, 1:, :]
            flow_dy = flow_reshaped[:, :, :, :-1] - flow_reshaped[:, :, :, 1:]
            flow_dx = torch.sqrt(flow_dx.pow(2) + 1e-6)
            flow_dy = torch.sqrt(flow_dy.pow(2) + 1e-6)
            ew_dx = (edge_weight[:, :, :-1, :] + edge_weight[:, :, 1:, :]) * 0.5
            ew_dy = (edge_weight[:, :, :, :-1] + edge_weight[:, :, :, 1:]) * 0.5
            smooth_raw = (
                (flow_dx * ew_dx).sum() + (flow_dy * ew_dy).sum()
            ) / flow_dx.shape[1]
        elif self.smoothness_mode == "edge_aware_l1":
            # Weight smoothness penalty by event sparsity: pixels with many events
            # (moving edges) are allowed sharp flow discontinuities; sparse regions
            # are penalized normally.
            with torch.no_grad():
                density = corrected.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
                d_max = density.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-3)
                # edge_weight → 1 (full penalty) where sparse, → 0 where event-dense
                edge_weight = torch.exp(-self.edge_weight_gamma * density / d_max)
            flow_dx_map = (flow_reshaped[:, :, :, 1:] - flow_reshaped[:, :, :, :-1]).abs()
            flow_dy_map = (flow_reshaped[:, :, 1:, :] - flow_reshaped[:, :, :-1, :]).abs()
            ew_x = (edge_weight[:, :, :, 1:] + edge_weight[:, :, :, :-1]) * 0.5
            ew_y = (edge_weight[:, :, 1:, :] + edge_weight[:, :, :-1, :]) * 0.5
            smooth_raw = (
                (flow_dx_map * ew_x.expand_as(flow_dx_map)).mean()
                + (flow_dy_map * ew_y.expand_as(flow_dy_map)).mean()
            )
        else:
            flow_dx = (flow_reshaped[:, :, :, 1:] - flow_reshaped[:, :, :, :-1]).abs().mean()
            flow_dy = (flow_reshaped[:, :, 1:, :] - flow_reshaped[:, :, :-1, :]).abs().mean()
            smooth_raw = flow_dx + flow_dy
        smooth_loss = self.smooth_weight * smooth_weight_scale * smooth_raw

        # ---- Mean-flow centering penalty (DC/global-collapse guardrail) ----
        # Penalizes flow fields whose spatial structure is dominated by their
        # own per-sample mean (a uniform global shift) rather than
        # locally-varying motion. Does not force zero-mean flow: a real
        # global ego-motion component stays unpenalized as long as enough
        # locally-varying structure rides on top of it. Gated by flow
        # magnitude (detached) so it cannot suppress legitimate signal while
        # the network is still near-silent early in training.
        center_loss = flow_reshaped.new_zeros(())
        centered_ratio_mean = flow_reshaped.new_zeros(())
        if self.center_weight > 0.0:
            eps = 1e-8
            mean_flow = flow_reshaped.mean(dim=(2, 3), keepdim=True)  # (B, 2, 1, 1)
            centered = flow_reshaped - mean_flow
            flow_mag_map = torch.sqrt(
                flow_reshaped[:, 0] ** 2 + flow_reshaped[:, 1] ** 2 + eps
            )  # (B, H, W)
            centered_mag_map = torch.sqrt(
                centered[:, 0] ** 2 + centered[:, 1] ** 2 + eps
            )
            flow_mag = flow_mag_map.mean(dim=(1, 2))  # (B,)
            centered_mag = centered_mag_map.mean(dim=(1, 2))  # (B,)
            centered_ratio = centered_mag / (flow_mag + eps)
            gate = (flow_mag.detach() > self.center_min_flow_mag).to(flow_reshaped.dtype)
            center_penalty = torch.relu(self.center_target_ratio - centered_ratio)
            center_loss = self.center_weight * (gate * center_penalty).mean()
            centered_ratio_mean = centered_ratio.detach().mean()

        total = fw_loss + bw_loss + smooth_loss + center_loss

        loss_dict = {
            "contrast": (fw_loss + bw_loss).detach(),
            "smooth": smooth_loss.detach(),
            "smooth_raw": smooth_raw.detach(),
            "center": center_loss.detach(),
            "centered_ratio": centered_ratio_mean,
        }

        return total, loss_dict


class EventWarpingSupervisedLoss(MotionCompensationLoss):
    """Weighted combination of EventWarpingLoss and supervised EPE.

    Loss = alpha * EventWarping + beta * EPE

    Falls back to EventWarping-only when ``flow_gt`` is not available.
    This makes it usable in mixed pipelines where some batches may not
    contain ground-truth flow.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        res: tuple[int, int] = (480, 640),
        smooth_weight: float = 0.001,
        flow_scaling: float = 1.0,
        loss_scaling: bool = True,
        max_epe: float = 128.0,
        smoothness_mode: str = "mean_l1",
        edge_weight_gamma: float = 5.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.event_warping = EventWarpingLoss(
            res=res,
            smooth_weight=smooth_weight,
            flow_scaling=flow_scaling,
            loss_scaling=loss_scaling,
            smoothness_mode=smoothness_mode,
            edge_weight_gamma=edge_weight_gamma,
        )
        self.supervised = SupervisedFlowLoss(max_epe=max_epe)

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        ew_loss, ew_dict = self.event_warping(corrected, flow, flow_gt, **kwargs)
        total = self.alpha * ew_loss
        log = {f"ew_{k}": v for k, v in ew_dict.items()}

        if flow_gt is not None:
            sup_loss, sup_dict = self.supervised(corrected, flow, flow_gt)
            total = total + self.beta * sup_loss
            log.update({f"sup_{k}": v for k, v in sup_dict.items()})

        if "smooth_raw" in ew_dict:
            log["smooth_raw"] = ew_dict["smooth_raw"].detach()

        return total, log


class ScheduledEventWarpingSupervisedLoss(MotionCompensationLoss):
    """Scheduled transition from supervised flow loss to event-warping loss.

    This starts with mostly supervised training and linearly shifts weight to
    self-supervised event warping. With the default 30-epoch schedule and
    ``self_supervised_only_epochs=3``, epochs 27, 28, and 29 are fully
    self-supervised.

    Loss = ew_weight(epoch) * alpha * EventWarping
         + sup_weight(epoch) * beta * EPE

    If ``flow_gt`` is unavailable, the loss falls back to fully
    self-supervised event warping for that batch.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        res: tuple[int, int] = (480, 640),
        smooth_weight: float = 0.001,
        flow_scaling: float = 1.0,
        loss_scaling: bool = True,
        max_epe: float = 128.0,
        smoothness_mode: str = "mean_l1",
        edge_weight_gamma: float = 5.0,
        schedule_total_epochs: int = 30,
        self_supervised_only_epochs: int = 3,
        schedule_start_epoch: int = 0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.schedule_total_epochs = max(int(schedule_total_epochs), 1)
        self.self_supervised_only_epochs = max(int(self_supervised_only_epochs), 0)
        self.schedule_start_epoch = max(int(schedule_start_epoch), 0)
        self.event_warping = EventWarpingLoss(
            res=res,
            smooth_weight=smooth_weight,
            flow_scaling=flow_scaling,
            loss_scaling=loss_scaling,
            smoothness_mode=smoothness_mode,
            edge_weight_gamma=edge_weight_gamma,
        )
        self.supervised = SupervisedFlowLoss(max_epe=max_epe)

    @staticmethod
    def _kwarg_int(kwargs: dict, *names: str) -> int | None:
        for name in names:
            if name not in kwargs or kwargs[name] is None:
                continue
            value = kwargs[name]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                return int(value.detach().flatten()[0].item())
            return int(value)
        return None

    def schedule_weights(self, epoch: int | None = None, num_epochs: int | None = None) -> tuple[float, float]:
        total_epochs = max(int(num_epochs or self.schedule_total_epochs), 1)
        final_selfsup_epochs = min(max(int(self.self_supervised_only_epochs), 0), total_epochs)
        start_epoch = min(max(int(self.schedule_start_epoch), 0), total_epochs - 1)
        selfsup_start_epoch = max(total_epochs - final_selfsup_epochs, start_epoch)

        if epoch is None:
            epoch_i = start_epoch
        else:
            epoch_i = max(int(epoch), 0)

        if epoch_i >= selfsup_start_epoch:
            progress = 1.0
        elif epoch_i <= start_epoch:
            progress = 0.0
        else:
            denom = max(selfsup_start_epoch - start_epoch, 1)
            progress = float(epoch_i - start_epoch) / float(denom)

        progress = min(max(progress, 0.0), 1.0)
        return progress, 1.0 - progress

    def forward(
        self,
        corrected: torch.Tensor,
        flow: torch.Tensor,
        flow_gt: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        epoch = self._kwarg_int(kwargs, "training_epoch", "current_epoch", "epoch")
        num_epochs = self._kwarg_int(kwargs, "training_num_epochs", "num_epochs")
        ew_weight, sup_weight = self.schedule_weights(epoch=epoch, num_epochs=num_epochs)

        ew_loss, ew_dict = self.event_warping(corrected, flow, flow_gt, **kwargs)
        log = {f"ew_{k}": v for k, v in ew_dict.items()}

        if flow_gt is None:
            total = self.alpha * ew_loss
            effective_ew_weight = 1.0
            effective_sup_weight = 0.0
        else:
            sup_loss, sup_dict = self.supervised(corrected, flow, flow_gt)
            total = self.alpha * ew_weight * ew_loss + self.beta * sup_weight * sup_loss
            log.update({f"sup_{k}": v for k, v in sup_dict.items()})
            effective_ew_weight = ew_weight
            effective_sup_weight = sup_weight

        if "contrast" in ew_dict:
            log["contrast"] = (self.alpha * effective_ew_weight * ew_dict["contrast"]).detach()
        if "smooth" in ew_dict:
            log["smooth"] = (self.alpha * effective_ew_weight * ew_dict["smooth"]).detach()
        if "smooth_raw" in ew_dict:
            log["smooth_raw"] = ew_dict["smooth_raw"].detach()
        log["schedule_ew_weight"] = torch.tensor(effective_ew_weight, device=flow.device)
        log["schedule_sup_weight"] = torch.tensor(effective_sup_weight, device=flow.device)
        return total, log


# Legacy EventWarpingResidualLoss implementation is intentionally kept in
# docs/process/2026-03-24-residual-loss.md (appendix snapshot), not in runtime code.

# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

_MC_LOSS_REGISTRY: Dict[str, Type[MotionCompensationLoss]] = {
    "cmax": CMaxLoss,
    "supervised": SupervisedFlowLoss,
    "supervised_l1": SupervisedFlowL1Loss,
    "supervised_l1_tid": SupervisedFlowL1TIDLoss,
    "angular": AngularLoss,
    "supervised_angular": SupervisedAngularLoss,
    "combined": CombinedLoss,
    "cmax_angular": CMaxAngularLoss,
    "event_warping": EventWarpingLoss,
    "event_warping_supervised": EventWarpingSupervisedLoss,
    "event_warping_supervised_scheduled": ScheduledEventWarpingSupervisedLoss,
    "scheduled_event_warping_supervised": ScheduledEventWarpingSupervisedLoss,
}


def create_mc_loss(loss_type: str, **kwargs) -> MotionCompensationLoss:
    """Create a motion compensation loss by registry key."""
    if loss_type not in _MC_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown MC loss type '{loss_type}'. "
            f"Available: {list(_MC_LOSS_REGISTRY.keys())}"
        )
    return _MC_LOSS_REGISTRY[loss_type](**kwargs)
