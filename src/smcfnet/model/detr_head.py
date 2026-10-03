"""
DETR Decoder Head for SMCM-MCFNet.

Replaces the YOLOX detection head with an RF-DETR transformer decoder.
Uses deformable cross-attention, Hungarian matching, and set prediction
losses (focal + L1 + GIoU).

Key differences from YOLOX head:
  - No NMS required (set prediction)
  - Outputs normalized [cx, cy, w, h] boxes instead of absolute pixel coords
  - Uses Hungarian matching instead of SimOTA
  - Returns loss dict directly during training
"""

import logging
import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from rfdetr.models.transformer import Transformer
from rfdetr.models.position_encoding import PositionEmbeddingSine
from rfdetr.models.lwdetr import SetCriterion, MLP
from rfdetr.models.matcher import HungarianMatcher

_logger = logging.getLogger(__name__)


class ChannelLayerNorm(nn.Module):
    """LayerNorm applied per-channel on (B, C, H, W) tensors.

    Equivalent to GroupNorm with num_groups=1 but more explicit.
    Preferred over BatchNorm for small batch sizes (RF-DETR paper, Appendix A).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class FeatureProjector(nn.Module):
    """Project multi-scale PAFPN features to uniform hidden_dim.

    PAFPN outputs 3 feature maps with channels [256*w, 512*w, 1024*w]
    (e.g. [128, 256, 512] for YOLOX-S). The transformer decoder expects
    all levels at hidden_dim (typically 256).
    """

    def __init__(self, in_channels: List[int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleList()
        for in_ch in in_channels:
            self.projections.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                ChannelLayerNorm(hidden_dim),
            ))

    def forward(self, features: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.projections, features)]


class DETRDecoderHead(nn.Module):
    """RF-DETR transformer decoder head.

    Wraps RF-DETR's transformer decoder, classification/bbox heads,
    and SetCriterion loss into a single module compatible with DualStreamYOLOX.

    Args:
        num_classes: Number of object classes.
        in_channels: Base channel list [256, 512, 1024] before width scaling.
        width: YOLOX width multiplier (e.g. 0.5 for size "s").
        hidden_dim: Transformer hidden dimension.
        num_queries: Number of detection queries.
        num_select: Top-k queries for inference.
        dec_layers: Number of decoder layers.
        sa_nheads: Self-attention heads.
        ca_nheads: Cross-attention heads.
        dec_n_points: Deformable attention sampling points per level.
        dim_feedforward: FFN hidden dim.
        dropout: Dropout rate.
        group_detr: Number of query groups during training.
        aux_loss: Use auxiliary losses at each decoder layer.
        two_stage: Use two-stage proposals (start False).
        bbox_reparam: Bbox reparameterization (delta to reference).
        lite_refpoint_refine: Lite reference point refinement.
        focal_alpha: Focal loss alpha.
        ia_bce_loss: IoU-aligned BCE loss.
        cls_loss_coef: Classification loss weight.
        bbox_loss_coef: Bbox L1 loss weight.
        giou_loss_coef: GIoU loss weight.
        set_cost_class: Hungarian matcher class cost.
        set_cost_bbox: Hungarian matcher bbox L1 cost.
        set_cost_giou: Hungarian matcher GIoU cost.
        img_h: Input image height (for target normalization).
        img_w: Input image width (for target normalization).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: List[int],
        width: float = 0.5,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_select: int = 300,
        dec_layers: int = 3,
        sa_nheads: int = 8,
        ca_nheads: int = 16,
        dec_n_points: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        group_detr: int = 13,
        aux_loss: bool = True,
        two_stage: bool = False,
        bbox_reparam: bool = True,
        lite_refpoint_refine: bool = True,
        focal_alpha: float = 0.25,
        ia_bce_loss: bool = True,
        cls_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        set_cost_class: float = 2.0,
        set_cost_bbox: float = 5.0,
        set_cost_giou: float = 2.0,
        img_h: int = 480,
        img_w: int = 640,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_select = num_select
        self.hidden_dim = hidden_dim
        self.group_detr = group_detr
        self.aux_loss = aux_loss
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.img_h = img_h
        self.img_w = img_w

        # Scale in_channels by width multiplier (PAFPN output channels)
        scaled_channels = [int(ch * width) for ch in in_channels]
        num_feature_levels = len(scaled_channels)

        # 1. Feature projector: PAFPN channels → hidden_dim
        self.input_proj = FeatureProjector(scaled_channels, hidden_dim)

        # 2. Position encoding
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
        )

        # 3. Transformer decoder
        self.transformer = Transformer(
            d_model=hidden_dim,
            sa_nhead=sa_nheads,
            ca_nhead=ca_nheads,
            num_queries=num_queries,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,  # Need all layers for aux_loss
            group_detr=group_detr,
            two_stage=two_stage,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        # 4. Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # 5. Query embeddings (scaled by group_detr for training)
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, 4)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)

        # Wire bbox_embed to transformer decoder for iterative refinement
        if not lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        # 6. Loss: Hungarian matcher + SetCriterion
        matcher = HungarianMatcher(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou,
            focal_alpha=focal_alpha,
        )

        weight_dict = {"loss_ce": cls_loss_coef, "loss_bbox": bbox_loss_coef, "loss_giou": giou_loss_coef}
        if aux_loss:
            for i in range(dec_layers - 1):
                weight_dict[f"loss_ce_{i}"] = cls_loss_coef
                weight_dict[f"loss_bbox_{i}"] = bbox_loss_coef
                weight_dict[f"loss_giou_{i}"] = giou_loss_coef
        if two_stage:
            weight_dict["loss_ce_enc"] = cls_loss_coef
            weight_dict["loss_bbox_enc"] = bbox_loss_coef
            weight_dict["loss_giou_enc"] = giou_loss_coef

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=focal_alpha,
            losses=["labels", "boxes", "cardinality"],
            group_detr=group_detr,
            ia_bce_loss=ia_bce_loss,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize detection head weights."""
        # Class head: bias init for focal loss (prior prob ~ 0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

        # Bbox head: zero init for last layer
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0.0)

        # Reference point embedding: uniform init
        nn.init.uniform_(self.refpoint_embed.weight, 0, 1)

    def _get_queries(self, *, loss_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get query embeddings, handling group_detr for train vs eval."""
        if self.training or bool(loss_mode):
            refpoint = self.refpoint_embed.weight
            query = self.query_feat.weight
        else:
            # At inference, use only first num_queries (drop group duplicates)
            refpoint = self.refpoint_embed.weight[:self.num_queries]
            query = self.query_feat.weight[:self.num_queries]
        return refpoint, query

    def _prepare_features(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Project features and generate position encodings + masks.

        Args:
            features: Tuple of PAFPN feature maps.

        Returns:
            srcs: Projected features (list of (B, hidden_dim, H, W)).
            masks: Padding masks (list of (B, H, W), all False = no padding).
            pos_embeds: Positional encodings (list of (B, hidden_dim, H, W)).
        """
        srcs = self.input_proj(features)
        masks = []
        pos_embeds = []

        for src in srcs:
            bs, _, h, w = src.shape
            # No padding in our pipeline — all pixels are valid
            mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)
            masks.append(mask)

            # PositionEmbeddingSine expects a NestedTensor-like object
            nt = types.SimpleNamespace(tensors=src, mask=mask)
            pos = self.pos_embed(nt, align_dim_orders=False)  # (B, C, H, W)
            pos_embeds.append(pos)

        return srcs, masks, pos_embeds

    def forward(
        self,
        features: Tuple[torch.Tensor, ...],
        targets: Optional[torch.Tensor] = None,
        loss_mode: bool = False,
        return_detections: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Tuple of PAFPN feature maps.
            targets: Ground truth (B, M, 5) [class_id, cx, cy, w, h] in absolute pixels.
                     Required during training, ignored during inference.

        Returns:
            During training: loss dict with keys {total, loss_ce, loss_bbox, loss_giou, ...}
            During inference: dict with keys {pred_logits, pred_boxes}
        """
        # 1. Project features and generate pos encodings
        srcs, masks, pos_embeds = self._prepare_features(features)

        # 2. Get query embeddings
        compute_loss = bool(self.training or loss_mode)
        refpoint_embed, query_feat = self._get_queries(loss_mode=compute_loss)

        # 3. Run transformer decoder
        hs, references, memory_ts, boxes_ts = self.transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat,
        )
        # hs: (num_layers, B, num_queries, hidden_dim)
        # references: (num_layers, B, num_queries, 4) — sigmoid-normalized

        # 4. Apply detection heads to each decoder layer output
        outputs_class = self.class_embed(hs)  # (num_layers, B, nq, num_classes)
        outputs_coord = self.bbox_embed(hs)   # (num_layers, B, nq, 4)

        # Apply bbox reparameterization if enabled
        if self.bbox_reparam:
            outputs_coord = outputs_coord + references
        outputs_coord = outputs_coord.sigmoid()

        # Use last layer as primary output
        out = {
            "pred_logits": outputs_class[-1],  # (B, nq, num_classes)
            "pred_boxes": outputs_coord[-1],    # (B, nq, 4) normalized [cx,cy,w,h]
        }

        # Auxiliary outputs from intermediate decoder layers
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        # Two-stage encoder outputs
        if self.two_stage and memory_ts is not None and boxes_ts is not None:
            enc_outputs_class = self.class_embed(memory_ts)
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": boxes_ts,
            }

        # 5. Compute loss during training
        if compute_loss:
            assert targets is not None, "Targets required during training"
            detr_targets = self._convert_targets(targets)
            loss_dict = self.criterion(out, detr_targets)

            # Weight losses and compute total
            weight_dict = self.criterion.weight_dict
            total = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict
                if k in weight_dict
            )
            loss_dict["total"] = total

            if return_detections:
                return loss_dict, out
            return loss_dict

        # 6. Inference: return raw outputs (postprocessed in validation.py)
        return out

    def _convert_targets(
        self, targets: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert YOLOX-format targets to DETR format.

        Args:
            targets: (B, M, 5) [class_id, cx, cy, w, h] in absolute pixels.
                     Padded rows have class_id == 0 and all-zero boxes.

        Returns:
            List of dicts per batch element:
                {"labels": (N,) int64, "boxes": (N, 4) normalized [cx,cy,w,h]}
        """
        batch_size = targets.shape[0]
        img_h, img_w = self.img_h, self.img_w
        detr_targets = []

        for i in range(batch_size):
            t = targets[i]  # (M, 5)

            # Filter out padding rows (all zeros): check w (col 3) and h (col 4)
            valid_mask = (t[:, 3] > 0) | (t[:, 4] > 0)  # w > 0 or h > 0
            t = t[valid_mask]

            if len(t) == 0:
                detr_targets.append({
                    "labels": torch.zeros(0, dtype=torch.int64, device=targets.device),
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device=targets.device),
                })
                continue

            labels = t[:, 0].long()

            # Normalize boxes: absolute pixels → [0, 1]
            boxes = t[:, 1:5].clone()
            boxes[:, 0] /= img_w  # cx
            boxes[:, 1] /= img_h  # cy
            boxes[:, 2] /= img_w  # w
            boxes[:, 3] /= img_h  # h

            detr_targets.append({
                "labels": labels,
                "boxes": boxes,
            })

        return detr_targets
