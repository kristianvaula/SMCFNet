import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
from typing import Dict, Optional, Sequence, Union
from smcfnet.utils.profiling import CudaStageTimer

_logger = logging.getLogger(__name__)

class DualStreamYOLOX(nn.Module):
  """
  Complete YOLOX model with dual-stream backbone for RGB-Event fusion.

  Architecture:
      RGB + Event Inputs
          ↓
      Dual-Stream Backbone (CSPDarknet)
          ↓
      Feature Fusion (Concatenation)
          ↓
      PAFPN Neck
          ↓
      YOLOX Head  ──stem features──►  STTP Head (optional)
          ↓                               ↓
      Detections / Losses           Future box predictions
  """

  def __init__(self, backbone, neck, head, sttp_head=None, sttp_loss=None):
    super().__init__()
    self.backbone = backbone
    self.neck = neck
    self.head = head
    self.sttp_head = sttp_head
    self.sttp_loss = sttp_loss
    if sttp_head is not None:
      from .yolo_head import YOLOXHead
      if not isinstance(head, YOLOXHead):
        raise ValueError(
          f"STTP head requires YOLOXHead (got {type(head).__name__}); "
          "DETR head does not expose per-scale stem features."
        )
    # When True, backbone/neck/head forward runs under torch.no_grad()
    # to avoid storing activations for frozen parameters (saves ~50% VRAM).
    # Set by Trainer when config.model.sttp.freeze_detector is enabled.
    self.detector_frozen = False
    # Auxiliary loss weights — registered as buffers so they survive
    # state_dict() / load_state_dict() round-trips.  Overwritten by
    # build_model() from config values.
    self.register_buffer('_loss_mcm_weight', torch.tensor(0.0), persistent=False)
    self.register_buffer('_mcm_contrast_clamp', torch.tensor(10.0), persistent=False)
    self.register_buffer('_loss_sttp_weight', torch.tensor(0.0), persistent=False)

  def forward(
    self,
    rgb,
    event,
    targets=None,
    future_targets=None,
    flow=None,
    stem_window=None,
    loss_mode: bool = False,
    return_detections: bool = False,
    **kwargs,
  ):
    """
    Forward pass.

    Args:
      rgb: RGB images (B, 3, H, W)
      event: Event voxel grids (B, 5, H, W)
      targets: Ground truth for training (optional)
      future_targets: GT boxes at t+k for STTP training. Dict with:
        - "boxes": [B, max_gt, 4] future targets (absolute cxcywh or normalized delta)
        - "validity": [B, max_gt] bool, True if object still in frame at t+k
        - "current_boxes": [B, max_gt, 4] current-frame GT boxes in absolute cxcywh
        None when STTP is disabled.
      flow: [B, 2, H, W] OFE optical flow field (Design C only, optional).
      **kwargs: Extra context forwarded to the backbone / MCM (e.g.
        ``events``, ``events_offsets`` for EventWarpingLoss).

    Returns:
      During training: loss dict
      During inference: detection results (+ future box predictions if STTP enabled)
    """
    compute_loss = bool(self.training or loss_mode)

    # When the detector is frozen (STTP-only training), skip gradient
    # tracking for backbone/neck/head to avoid storing activations.
    det_ctx = torch.no_grad() if (self.training and self.detector_frozen) else nullcontext()

    with det_ctx:
      # Extract fused features from dual-stream backbone
      backbone_features = self.backbone(rgb, event, **kwargs)

      # Extract profiling events independently of MCM state.
      profile_events = backbone_features.pop("_profile_events", None)

      # MCM is active only when the backbone emits its auxiliary keys; gate
      # the pop + side-effect assignment so the forward path is MCFNet-faithful
      # when MCM is disabled.
      has_mcm = "_mcm_loss" in backbone_features
      if has_mcm:
        mcm_loss = backbone_features.pop("_mcm_loss", None)
        mcm_contrast = backbone_features.pop("_mcm_contrast", None)
        mcm_smooth = backbone_features.pop("_mcm_smooth", None)
        # Live tensor for STTP, detached copy for viz
        mcm_flow = backbone_features.pop("_mcm_flow", None)
        self._mcm_viz = {
          "flow": mcm_flow.detach() if mcm_flow is not None else None,
          "input": backbone_features.pop("_mcm_input", None),
          "output": backbone_features.pop("_mcm_output", None),
        }
      else:
        mcm_loss = mcm_contrast = mcm_smooth = mcm_flow = None

      # Get multi-scale features from PAFPN neck
      # backbone_features: {dark3, dark4, dark5}
      # Convert dict to list for PAFPN
      fpn_features = [
        backbone_features['dark3'],
        backbone_features['dark4'],
        backbone_features['dark5']
      ]

      # PAFPN forward - creates feature pyramid
      neck_outputs = self.neck(fpn_features)

      # Detection head forward
      if compute_loss:
        assert targets is not None, "Targets required during training/loss_mode"
        # YOLOXHead returns (outputs, stem_features); DETR head returns a
        # bare loss_dict. Guard the unpack so DETR runs don't crash.
        head_out = self.head(
          neck_outputs,
          targets,
          loss_mode=bool(loss_mode),
          return_detections=bool(return_detections),
        )
        detections = None
        if (
          return_detections
          and isinstance(head_out, tuple)
          and len(head_out) == 2
          and isinstance(head_out[0], dict)
          and isinstance(head_out[1], dict)
        ):
          outputs, stem_features, detections = head_out[0], None, head_out[1]
        elif isinstance(head_out, tuple) and len(head_out) == 3:
          outputs, stem_features, detections = head_out
        elif isinstance(head_out, tuple) and len(head_out) == 2:
          outputs, stem_features = head_out
        else:
          outputs, stem_features = head_out, None

        # DETR head returns a loss dict directly; YOLOX head returns a tuple
        if isinstance(outputs, dict):
          # DETR head — loss dict already contains 'total'
          loss_dict = outputs
          total = loss_dict['total']
          assignments = None
        else:
          # YOLOX head — unpack tuple into loss dict
          assignments = outputs[6]      # SimOTA assignments for STTP
          loss_dict = {
            'loss_obj': outputs[2],      # objectness loss
            'loss_cls': outputs[3],      # classification loss
            'loss_box': outputs[1],      # box regression loss
            'loss_l1': outputs[4],       # L1 loss (usually 0.0 unless head.use_l1=True)
            'fg_ratio': outputs[5],      # num_fg / max(num_gts, 1)
            'loss_iou': outputs[1] / 5.0,
          }
          total = outputs[0]

      # --- end of training detection head ---

      else:
        # YOLOXHead returns (detections, stem_features); DETR head returns a
        # bare detections dict. Guard the unpack so DETR runs don't crash.
        head_out = self.head(neck_outputs)
        if isinstance(head_out, tuple):
          detections, stem_features = head_out
        else:
          detections, stem_features = head_out, None

    # --- STTP forward + loss (outside no_grad so STTP head gets gradients) ---
    if compute_loss:
      sttp_w = float(self._loss_sttp_weight)
      if self.training and self.sttp_head is not None and assignments is not None and sttp_w > 0:
        sttp_flow = flow if flow is not None else mcm_flow
        sttp_input = stem_window if stem_window is not None else stem_features
        fut_preds = self.sttp_head(sttp_input, flow=sttp_flow)

        if self.sttp_loss is not None and future_targets is not None:
          # Build per-foreground future GT targets from assignments
          fut_targets, validity, current_boxes = self._gather_future_targets(
            future_targets, assignments,
          )

          # Build grids from shifts for decoding
          grids = torch.stack([
            assignments["x_shifts"],
            assignments["y_shifts"],
          ], dim=-1)  # [1, n_anchors_all, 2]

          sttp_losses = self.sttp_loss(
            fut_preds=fut_preds,
            fut_targets=fut_targets,
            fg_masks=assignments["fg_masks"],
            validity_mask=validity,
            expanded_strides=assignments["expanded_strides"],
            grids=grids,
            num_fg=assignments["num_fg"],
            current_boxes=current_boxes,
          )

          loss_dict['loss_fut'] = sttp_losses['loss_fut']
          total = total + sttp_w * sttp_losses['loss_fut']

          if 'loss_tc' in sttp_losses:
            loss_dict['loss_tc'] = sttp_losses['loss_tc']
            total = total + sttp_w * sttp_losses['loss_tc']

      # Inject MCM diagnostics only when MCM is active.
      if has_mcm:
        loss_dict['loss_mcm'] = mcm_loss if mcm_loss is not None else torch.tensor(0.0, device=rgb.device)
        loss_dict['mcm_contrast'] = mcm_contrast if mcm_contrast is not None else torch.tensor(0.0, device=rgb.device)
        loss_dict['mcm_smooth'] = mcm_smooth if mcm_smooth is not None else torch.tensor(0.0, device=rgb.device)

      # Total weighted loss (detection + optional MCM contribution)
      mcm_w = float(self._loss_mcm_weight)
      if mcm_w > 0 and mcm_loss is not None:
        if torch.isfinite(mcm_loss):
          total = total + mcm_w * mcm_loss
        else:
          if not getattr(self, '_mcm_nan_warned', False):
            _logger.warning(
              "MCM loss is non-finite (%.4f) — using clamped fallback. "
              "Check flow estimator for NaN.",
              mcm_loss.item() if mcm_loss.numel() == 1 else float('nan'),
            )
            self._mcm_nan_warned = True
          fallback = self._mcm_contrast_clamp.to(device=total.device, dtype=total.dtype)
          total = total + mcm_w * fallback
      loss_dict['total'] = total
      if profile_events is not None:
        loss_dict["_profile_events"] = profile_events

      if return_detections:
        return loss_dict, detections
      return loss_dict
    else:
      # STTP inference: append future predictions
      if self.sttp_head is not None:
        sttp_flow = flow if flow is not None else mcm_flow
        sttp_input = stem_window if stem_window is not None else stem_features
        fut_preds = self.sttp_head(sttp_input, flow=sttp_flow)
        return detections, fut_preds

      return detections

  def _gather_future_targets(self, future_targets, assignments):
    """Index future GT boxes using SimOTA matched_gt_inds.

    Args:
      future_targets: dict with "boxes" [B, max_gt, k, 4], "validity" [B, max_gt, k],
        and "current_boxes" [B, max_gt, 4].
      assignments: dict from YOLOXHead with "matched_gt_inds" (list per batch)
        and "fg_masks" [B * n_anchors_all].

    Returns:
      fut_targets: [num_fg_total, k, 4] future GT targets for matched anchors.
      validity: [num_fg_total, k] per-step bool mask.
      current_boxes: [num_fg_total, 4] current GT boxes for matched anchors.
    """
    fut_boxes = future_targets["boxes"]       # [B, max_gt, k, 4]
    fut_valid = future_targets["validity"]    # [B, max_gt, k]
    curr_boxes = future_targets["current_boxes"]  # [B, max_gt, 4]
    k = int(fut_boxes.shape[2])
    matched_gt_inds_list = assignments["matched_gt_inds"]

    targets_list = []
    validity_list = []
    current_list = []
    for batch_idx, gt_inds in enumerate(matched_gt_inds_list):
      if gt_inds.numel() == 0:
        continue
      targets_list.append(fut_boxes[batch_idx][gt_inds])   # [num_fg_img, k, 4]
      validity_list.append(fut_valid[batch_idx][gt_inds])  # [num_fg_img, k]
      current_list.append(curr_boxes[batch_idx][gt_inds])  # [num_fg_img, 4]

    if len(targets_list) == 0:
      device = fut_boxes.device
      return (
        torch.zeros(0, k, 4, device=device),
        torch.zeros(0, k, dtype=torch.bool, device=device),
        torch.zeros(0, 4, device=device),
      )

    return torch.cat(targets_list, 0), torch.cat(validity_list, 0), torch.cat(current_list, 0)
