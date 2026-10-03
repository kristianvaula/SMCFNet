import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class STTPLoss(nn.Module):
    """Loss for the STTP future-box prediction head.

    Computes SmoothL1 between predicted future targets and GT future targets,
    masked by:
      - fg_mask: only foreground anchors (from SimOTA assignment on current frame)
      - validity_mask: only objects still visible at t+k

    ``target_mode="absolute"`` uses decoded absolute future cxcywh targets.
    ``target_mode="delta"`` uses normalized deltas relative to the current box:
    `(dcx / W, dcy / H, dw / W, dh / H)`.

    Optionally adds a temporal consistency term:
      L_tc = SmoothL1(fut_pred(t), reg_pred(t+k).detach())
    which ties the future prediction to what detection actually sees at t+k.
    """

    def __init__(
        self,
        temporal_consistency_weight: float = 0.0,
        target_mode: str = "absolute",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        """
        Args:
            temporal_consistency_weight: weight for temporal consistency loss.
                0.0 disables it (start here). Positive value enables the RED-style
                self-supervised term.
        """
        super().__init__()
        if target_mode not in {"absolute", "delta"}:
            raise ValueError(f"target_mode must be 'absolute' or 'delta', got {target_mode!r}")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.tc_weight = temporal_consistency_weight
        self.target_mode = target_mode
        self.image_width = image_width
        self.image_height = image_height

    def _decode_absolute(
        self,
        flat_preds: torch.Tensor,
        expanded_strides: torch.Tensor,
        grids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode raw (tx,ty,tw,th) to absolute cxcywh.

        Accepts ``[B, A, 4]`` (single step) or ``[B, A, k, 4]`` (k-step trajectory);
        grids ``[1, A, 2]`` and strides ``[1, A]`` broadcast over the optional step axis.
        """
        decoded = flat_preds.clone()
        if flat_preds.dim() == 4:
            g = grids.unsqueeze(2)                              # [1, A, 1, 2]
            s = expanded_strides.unsqueeze(-1).unsqueeze(-1)    # [1, A, 1, 1]
        else:
            g = grids                                          # [1, A, 2]
            s = expanded_strides.unsqueeze(-1)                 # [1, A, 1]
        decoded[..., :2] = (flat_preds[..., :2] + g) * s
        decoded[..., 2:4] = torch.exp(flat_preds[..., 2:4].clamp(max=10.0)) * s
        return decoded

    def _absolute_to_delta(self, boxes: torch.Tensor, current_boxes: torch.Tensor) -> torch.Tensor:
        if self.image_width is None or self.image_height is None:
            raise ValueError("delta target mode requires image_width and image_height")
        out = torch.empty_like(boxes)
        out[:, 0] = (boxes[:, 0] - current_boxes[:, 0]) / float(self.image_width)
        out[:, 1] = (boxes[:, 1] - current_boxes[:, 1]) / float(self.image_height)
        out[:, 2] = (boxes[:, 2] - current_boxes[:, 2]) / float(self.image_width)
        out[:, 3] = (boxes[:, 3] - current_boxes[:, 3]) / float(self.image_height)
        return out

    def forward(
        self,
        fut_preds: List[torch.Tensor],
        fut_targets: torch.Tensor,
        fg_masks: torch.Tensor,
        validity_mask: torch.Tensor,
        expanded_strides: torch.Tensor,
        grids: torch.Tensor,
        num_fg: float,
        reg_preds_future: torch.Tensor = None,
        current_boxes: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fut_preds: list of 3 tensors [B, 4*k, H_k, W_k] from the STTP head
                (k = future_steps; channels packed [step1_4, step2_4, ...]).
            fut_targets: [num_fg_total, k, 4] GT future targets for matched anchors.
                In ``target_mode="absolute"``: cxcywh in absolute pixel coords (same
                format as reg_targets in YOLOXHead).
                In ``target_mode="delta"``: normalized deltas vs the current box,
                ``(dcx / W, dcy / H, dw / W, dh / H)``, as produced by the loader/mapper.
            fg_masks: [B * n_anchors_all] bool, foreground anchors from SimOTA.
            validity_mask: [num_fg_total, k] bool, per-step True if object is still
                in frame at that future step.
            expanded_strides: [1, n_anchors_all] stride per anchor position.
            grids: [1, n_anchors_all, 2] grid offsets (x, y) for decoding.
            num_fg: number of foreground anchors (for normalization).
            reg_preds_future: [num_fg_total, 4] detection reg output at t+k (detached).
                Required only if temporal_consistency_weight > 0; compared against the
                final-step (t+k) prediction.

        Returns:
            dict with 'loss_fut' and optionally 'loss_tc'.
        """
        flat_preds = []
        for pred in fut_preds:
            B, C, H, W = pred.shape
            flat_preds.append(pred.permute(0, 2, 3, 1).reshape(B, -1, C))
        flat_preds = torch.cat(flat_preds, dim=1)  # [B, n_anchors_all, 4*k]
        k = flat_preds.shape[-1] // 4
        flat_preds = flat_preds.reshape(flat_preds.shape[0], flat_preds.shape[1], k, 4)  # [B, A, k, 4]

        if self.target_mode == "absolute":
            pred_space = self._decode_absolute(flat_preds, expanded_strides, grids)
        else:
            pred_space = flat_preds

        fg_preds = pred_space.reshape(-1, k, 4)[fg_masks]  # [num_fg_total, k, 4]

        # Per-step validity mask; normalize by the number of valid (object, step) pairs.
        valid = validity_mask.bool()              # [num_fg_total, k]
        num_valid = max(valid.sum().item(), 1.0)

        # Supervised loss: SmoothL1 on valid foreground predictions vs GT future boxes
        loss_fut_all = self.smooth_l1(fg_preds, fut_targets)  # [num_fg_total, k, 4]
        loss_fut = (loss_fut_all * valid.unsqueeze(-1)).sum() / num_valid

        result = {"loss_fut": loss_fut}

        # Temporal consistency loss (optional): tie the final-step (t+k) prediction to
        # what detection sees at t+k. Detection gives a single box, so TC supervises the
        # last trajectory step only.
        if self.tc_weight > 0 and reg_preds_future is not None:
            tc_target = reg_preds_future.detach()
            if self.target_mode == "delta":
                if current_boxes is None:
                    raise ValueError("delta target mode with temporal consistency requires current_boxes")
                tc_target = self._absolute_to_delta(tc_target, current_boxes)
            fg_preds_last = fg_preds[:, -1, :]        # [num_fg_total, 4]
            valid_last = valid[:, -1]                 # [num_fg_total]
            num_valid_last = max(valid_last.sum().item(), 1.0)
            loss_tc_all = self.smooth_l1(fg_preds_last, tc_target)
            loss_tc = (loss_tc_all * valid_last.unsqueeze(-1)).sum() / num_valid_last
            result["loss_tc"] = self.tc_weight * loss_tc

        return result
