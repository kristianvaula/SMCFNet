"""
Validation metrics for object detection.
Implements COCO-standard mAP calculation for YOLOX-style predictions.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def postprocess_predictions(
    prediction: torch.Tensor,
    num_classes: int,
    conf_thre: float = 0.01,
    nms_thre: float = 0.65,
    return_anchor_indices: bool = False,
):
    """
    Postprocess YOLOX predictions with NMS.

    Args:
        prediction: (batch_size, num_anchors, 5 + num_classes)
                   Format: [x, y, w, h, obj_conf, cls_conf_0, ..., cls_conf_n]
        num_classes: Number of object classes
        conf_thre: Confidence threshold
        nms_thre: NMS IoU threshold
        return_anchor_indices: When True, also return per-image LongTensors mapping
            each surviving detection back to its source anchor index in the
            pre-NMS ``prediction`` tensor. Used by STTP trajectory metrics so the
            future-box decoded at each anchor can be looked up directly without a
            reverse-nearest-neighbor cdist match.

    Returns:
        If ``return_anchor_indices`` is False:
            List of detections per image, each (num_dets, 7).
            Format: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        Otherwise:
            Tuple of (detections_per_image, anchor_indices_per_image), where
            ``anchor_indices_per_image[i]`` is a LongTensor of shape ``[num_dets_i]``
            (or None if no detections survived for image ``i``).
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output: List[Optional[torch.Tensor]] = [None for _ in range(len(prediction))]
    output_anchor_idx: List[Optional[torch.Tensor]] = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        n_anchors = image_pred.shape[0]
        anchor_ids = torch.arange(n_anchors, device=image_pred.device)

        # Filter by object confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + num_classes], 1, keepdim=True
        )
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        # Get detections above threshold
        detections = image_pred[conf_mask]
        if not detections.size(0):
            continue
        anchor_ids = anchor_ids[conf_mask]

        # Get class confidences
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        # Detections: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((
            detections[:, :5],
            class_conf,
            class_pred.float()
        ), 1)

        # Get unique classes
        unique_labels = detections[:, -1].unique()

        # Perform NMS per class
        for c in unique_labels:
            class_mask = detections[:, -1] == c
            detections_class = detections[class_mask]
            anchor_class = anchor_ids[class_mask]

            # Sort by confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4] * detections_class[:, 5],
                descending=True
            )
            detections_class = detections_class[conf_sort_index]
            anchor_class = anchor_class[conf_sort_index]

            # NMS
            max_detections = []
            max_anchor_ids = []
            while detections_class.size(0):
                # Take detection with highest confidence
                max_detections.append(detections_class[0].unsqueeze(0))
                max_anchor_ids.append(anchor_class[0:1])

                if len(detections_class) == 1:
                    break

                # Calculate IoU with remaining boxes
                ious = bbox_iou(
                    max_detections[-1][:, :4],
                    detections_class[1:, :4]
                )

                # Remove overlapping detections
                keep = ious[0] < nms_thre
                detections_class = detections_class[1:][keep]
                anchor_class = anchor_class[1:][keep]

            if len(max_detections):
                max_detections_t = torch.cat(max_detections)
                max_anchor_ids_t = torch.cat(max_anchor_ids)
                if output[i] is None:
                    output[i] = max_detections_t
                    output_anchor_idx[i] = max_anchor_ids_t
                else:
                    output[i] = torch.cat((output[i], max_detections_t))
                    output_anchor_idx[i] = torch.cat((output_anchor_idx[i], max_anchor_ids_t))

    if return_anchor_indices:
        return output, output_anchor_idx
    return output


def postprocess_detr_predictions(
    outputs: Dict[str, torch.Tensor],
    img_h: int,
    img_w: int,
    num_select: int = 300,
    conf_thre: float = 0.01,
) -> List[Optional[torch.Tensor]]:
    """
    Convert DETR outputs to same format as postprocess_predictions.

    No NMS needed — DETR outputs are already deduplicated via set prediction.

    Args:
        outputs: Dict with "pred_logits" (B, Q, C) and "pred_boxes" (B, Q, 4).
                 Logits are raw (pre-sigmoid). Boxes are normalized [cx,cy,w,h].
        img_h: Image height for denormalization.
        img_w: Image width for denormalization.
        num_select: Top-k queries to keep.
        conf_thre: Confidence threshold.

    Returns:
        List of detections per image (same format as postprocess_predictions).
        Each detection: (N, 7) [x1, y1, x2, y2, obj_conf, class_conf, class_id]
    """
    pred_logits = outputs["pred_logits"]  # (B, Q, C)
    pred_boxes = outputs["pred_boxes"]    # (B, Q, 4) normalized [cx,cy,w,h]
    batch_size = pred_logits.shape[0]

    # Sigmoid for focal-loss style logits
    scores = pred_logits.sigmoid()  # (B, Q, C)

    result = []
    for i in range(batch_size):
        s = scores[i]    # (Q, C)
        b = pred_boxes[i] # (Q, 4)

        # Get max class confidence per query
        class_conf, class_id = s.max(dim=-1)  # (Q,), (Q,)

        # Top-k selection
        if num_select < s.shape[0]:
            topk_idx = class_conf.topk(num_select).indices
            class_conf = class_conf[topk_idx]
            class_id = class_id[topk_idx]
            b = b[topk_idx]

        # Confidence threshold
        keep = class_conf >= conf_thre
        if not keep.any():
            result.append(None)
            continue

        class_conf = class_conf[keep]
        class_id = class_id[keep]
        b = b[keep]

        # Convert normalized [cx,cy,w,h] → absolute [x1,y1,x2,y2]
        cx = b[:, 0] * img_w
        cy = b[:, 1] * img_h
        w = b[:, 2] * img_w
        h = b[:, 3] * img_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # DETR has no separate objectness — set obj_conf=1.0
        obj_conf = torch.ones_like(class_conf)

        # Stack into (N, 7): [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        dets = torch.stack([x1, y1, x2, y2, obj_conf, class_conf, class_id.float()], dim=-1)
        result.append(dets)

    return result


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) in xyxy format
        box2: (M, 4) in xyxy format
    
    Returns:
        iou: (N, M)
    """
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area

    return inter_area / (union_area + eps)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        recall: Recall values
        precision: Precision values
    
    Returns:
        ap: Average Precision
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def _compute_ap_at_iou(
    pred_img_ids: np.ndarray,
    best_iou_arr: np.ndarray,
    best_gt_idx_arr: np.ndarray,
    n_gt_boxes: int,
    iou_threshold: float,
    cls: int,
) -> float:
    """
    Compute AP for a single class at a single IoU threshold.

    Predictions must already be sorted by descending confidence.
    ``best_iou_arr`` and ``best_gt_idx_arr`` are pre-computed from the IoU
    matrix so that we avoid re-computing IoU for every threshold.
    """
    n_preds = int(pred_img_ids.shape[0])
    tp = np.zeros(n_preds, dtype=np.float32)
    fp = np.zeros(n_preds, dtype=np.float32)
    detected: set = set()

    for pred_idx in range(n_preds):
        best_iou = float(best_iou_arr[pred_idx])
        best_gt_idx = int(best_gt_idx_arr[pred_idx])
        pred_img_id = int(pred_img_ids[pred_idx])

        if best_gt_idx >= 0 and best_iou >= iou_threshold and (pred_img_id, best_gt_idx) not in detected:
            tp[pred_idx] = 1.0
            detected.add((pred_img_id, best_gt_idx))
        else:
            fp[pred_idx] = 1.0

    tp_cumsum = np.cumsum(tp, axis=0)
    fp_cumsum = np.cumsum(fp, axis=0)

    recalls = tp_cumsum / (n_gt_boxes + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    return compute_ap(recalls, precisions)


def _count_matches_at_iou(
    pred_img_ids: np.ndarray,
    best_iou_arr: np.ndarray,
    best_gt_idx_arr: np.ndarray,
    iou_threshold: float,
) -> int:
    """Count unique GT matches at a given IoU threshold for recall diagnostics."""
    detected: set = set()
    tp_count = 0

    for pred_idx in range(int(pred_img_ids.shape[0])):
        best_iou = float(best_iou_arr[pred_idx])
        best_gt_idx = int(best_gt_idx_arr[pred_idx])
        pred_img_id = int(pred_img_ids[pred_idx])

        if best_gt_idx >= 0 and best_iou >= float(iou_threshold) and (pred_img_id, best_gt_idx) not in detected:
            detected.add((pred_img_id, best_gt_idx))
            tp_count += 1

    return int(tp_count)


# COCO IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95  (10 values)
_COCO_IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def _targets_cxcywh_to_xyxy(target: np.ndarray) -> np.ndarray:
    """Convert target boxes from [class_id, cx, cy, w, h] to xyxy."""
    boxes_cxcywh = np.asarray(target[:, 1:5], dtype=np.float32)
    boxes_xyxy = np.empty_like(boxes_cxcywh, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    return boxes_xyxy


def _bbox_iou_numpy(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Numpy equivalent of ``bbox_iou`` for CPU-side metric computation."""
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    area1 = np.clip(box1[:, 2] - box1[:, 0], a_min=0.0, a_max=None) * np.clip(
        box1[:, 3] - box1[:, 1], a_min=0.0, a_max=None
    )
    area2 = np.clip(box2[:, 2] - box2[:, 0], a_min=0.0, a_max=None) * np.clip(
        box2[:, 3] - box2[:, 1], a_min=0.0, a_max=None
    )
    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + eps)


def _normalize_prediction_entry(pred_entry) -> Optional[np.ndarray]:
    """Normalize validation prediction entries to a single detection tensor."""
    if pred_entry is None:
        return None
    if isinstance(pred_entry, list):
        if len(pred_entry) == 0:
            return None
        pred_entry = pred_entry[0]
    if pred_entry is None:
        return None
    if isinstance(pred_entry, torch.Tensor):
        pred_np = pred_entry.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
    else:
        pred_np = np.asarray(pred_entry, dtype=np.float32)
    if pred_np.size == 0:
        return None
    pred_np = np.ascontiguousarray(pred_np, dtype=np.float32)
    if pred_np.ndim != 2:
        pred_np = pred_np.reshape(-1, pred_np.shape[-1])
    return pred_np


class StreamingMAPAccumulator:
    """Exact COCO-style mAP accumulator with batch-wise ingestion.

    This keeps only the reduced information required for AP computation:
    per-class prediction boxes/scores/image ids and per-class GT boxes grouped
    by image. It avoids retaining a second full copy of all per-image
    predictions/targets in the trainer.
    """

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self._next_img_id = 0
        self._pred_boxes: List[List[np.ndarray]] = [[] for _ in range(self.num_classes)]
        self._pred_scores: List[List[np.ndarray]] = [[] for _ in range(self.num_classes)]
        self._pred_img_ids: List[List[np.ndarray]] = [[] for _ in range(self.num_classes)]
        self._gt_boxes_by_img: List[Dict[int, np.ndarray]] = [{} for _ in range(self.num_classes)]
        self._n_gt_boxes: List[int] = [0 for _ in range(self.num_classes)]
        self._pred_count_per_img: List[int] = []
        self._pred_score_max_per_img: List[float] = []

    def update(
        self,
        predictions_per_image: List[Optional[torch.Tensor]],
        targets_per_image: List[torch.Tensor],
    ) -> None:
        if len(predictions_per_image) != len(targets_per_image):
            raise ValueError(
                f"Predictions/targets length mismatch: {len(predictions_per_image)} vs {len(targets_per_image)}"
            )

        for pred_entry, target in zip(predictions_per_image, targets_per_image):
            img_id = int(self._next_img_id)
            self._next_img_id += 1

            pred = _normalize_prediction_entry(pred_entry)
            pred_count = int(pred.shape[0]) if pred is not None else 0
            self._pred_count_per_img.append(pred_count)
            if pred is not None and pred_count > 0:
                pred_scores_img = np.ascontiguousarray(pred[:, 4] * pred[:, 5], dtype=np.float32)
                self._pred_score_max_per_img.append(float(np.max(pred_scores_img)))
            else:
                self._pred_score_max_per_img.append(0.0)
            if pred is not None:
                pred_classes = pred[:, -1].astype(np.int64, copy=False)
                for cls in np.unique(pred_classes):
                    cls = int(cls)
                    if cls < 0 or cls >= self.num_classes:
                        continue
                    class_mask = pred_classes == cls
                    pred_cls = np.ascontiguousarray(pred[class_mask], dtype=np.float32)
                    self._pred_boxes[cls].append(np.ascontiguousarray(pred_cls[:, :4], dtype=np.float32))
                    self._pred_scores[cls].append(
                        np.ascontiguousarray(pred_cls[:, 4] * pred_cls[:, 5], dtype=np.float32)
                    )
                    self._pred_img_ids[cls].append(
                        np.full((pred_cls.shape[0],), img_id, dtype=np.int64)
                    )

            if target is None:
                continue
            if isinstance(target, torch.Tensor):
                target_np = target.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
            else:
                target_np = np.asarray(target, dtype=np.float32)
            if target_np.size == 0:
                continue

            target_np = np.ascontiguousarray(target_np, dtype=np.float32).reshape(-1, 5)
            target_classes = target_np[:, 0].astype(np.int64, copy=False)
            boxes_xyxy = _targets_cxcywh_to_xyxy(target_np)
            for cls in np.unique(target_classes):
                cls = int(cls)
                if cls < 0 or cls >= self.num_classes:
                    continue
                class_mask = target_classes == cls
                gt_cls_boxes = np.ascontiguousarray(boxes_xyxy[class_mask], dtype=np.float32)
                self._gt_boxes_by_img[cls][img_id] = gt_cls_boxes
                self._n_gt_boxes[cls] += int(gt_cls_boxes.shape[0])

    def compute(
        self,
        *,
        verbose: bool = False,
        class_names: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, float]:
        logger.info(
            f"Starting COCO mAP calculation for {self.num_classes} classes "
            f"at {len(_COCO_IOU_THRESHOLDS)} IoU thresholds"
        )
        if verbose:
            logger.info(f"Processing {self._next_img_id} streamed validation images")

        per_class_aps: Dict[int, List[float]] = {}
        best_ious_all: List[np.ndarray] = []
        matched_gt_iou50 = 0
        total_gt_boxes = int(sum(self._n_gt_boxes))

        for cls in range(self.num_classes):
            if verbose:
                logger.info(f"Processing class {cls}/{self.num_classes}...")

            n_gt_boxes = int(self._n_gt_boxes[cls])
            pred_boxes_groups = self._pred_boxes[cls]
            pred_scores_groups = self._pred_scores[cls]
            pred_img_ids_groups = self._pred_img_ids[cls]
            gt_boxes_by_img = self._gt_boxes_by_img[cls]

            if verbose:
                logger.info(
                    f"Class {cls}: {len(pred_boxes_groups)} prediction groups, {n_gt_boxes} GT boxes"
                )

            if n_gt_boxes == 0:
                if verbose:
                    logger.info(f"Class {cls}: No ground truth boxes, skipping")
                continue

            if len(pred_boxes_groups) == 0:
                if verbose:
                    logger.info(f"Class {cls}: No predictions, AP=0.0")
                per_class_aps[cls] = [0.0] * len(_COCO_IOU_THRESHOLDS)
                continue

            pred_boxes = np.ascontiguousarray(np.concatenate(pred_boxes_groups, axis=0), dtype=np.float32)
            pred_scores = np.ascontiguousarray(np.concatenate(pred_scores_groups, axis=0), dtype=np.float32)
            pred_img_ids = np.ascontiguousarray(np.concatenate(pred_img_ids_groups, axis=0), dtype=np.int64)

            if verbose:
                logger.info(f"Class {cls}: Total {int(pred_boxes.shape[0])} predictions to match")

            sort_idx = np.argsort(-pred_scores, kind="stable")
            pred_boxes = pred_boxes[sort_idx]
            pred_img_ids = pred_img_ids[sort_idx]

            best_iou_arr = np.zeros(pred_boxes.shape[0], dtype=np.float32)
            best_gt_idx_arr = np.full(pred_boxes.shape[0], -1, dtype=np.int64)

            for img_id in np.unique(pred_img_ids):
                img_id = int(img_id)
                gt_stacked = gt_boxes_by_img.get(img_id)
                if gt_stacked is None or gt_stacked.size == 0:
                    continue
                pred_mask = pred_img_ids == img_id
                preds_for_img = pred_boxes[pred_mask]
                iou_matrix = _bbox_iou_numpy(preds_for_img, gt_stacked)
                max_local_idx = np.argmax(iou_matrix, axis=1)
                max_iou = iou_matrix[np.arange(iou_matrix.shape[0]), max_local_idx]
                best_iou_arr[pred_mask] = max_iou
                best_gt_idx_arr[pred_mask] = max_local_idx.astype(np.int64, copy=False)

            best_ious_all.append(best_iou_arr)
            matched_gt_iou50 += _count_matches_at_iou(
                pred_img_ids,
                best_iou_arr,
                best_gt_idx_arr,
                iou_threshold=0.5,
            )

            cls_aps = []
            for iou_thr in _COCO_IOU_THRESHOLDS:
                ap = _compute_ap_at_iou(
                    pred_img_ids,
                    best_iou_arr,
                    best_gt_idx_arr,
                    n_gt_boxes,
                    float(iou_thr),
                    cls,
                )
                cls_aps.append(ap)

            per_class_aps[cls] = cls_aps

            if verbose:
                ap50 = cls_aps[0]
                ap_mean = float(np.mean(cls_aps))
                logger.info(f"Class {cls}: AP50={ap50:.4f}, AP={ap_mean:.4f}")

        idx_50 = 0
        idx_75 = 5

        all_ap50 = [aps[idx_50] for aps in per_class_aps.values()]
        all_ap75 = [aps[idx_75] for aps in per_class_aps.values()]
        all_ap_mean = [float(np.mean(aps)) for aps in per_class_aps.values()]

        metrics: Dict[str, float] = {
            'mAP': float(np.mean(all_ap_mean)) if all_ap_mean else 0.0,
            'mAP50': float(np.mean(all_ap50)) if all_ap50 else 0.0,
            'mAP75': float(np.mean(all_ap75)) if all_ap75 else 0.0,
        }

        if len(best_ious_all) > 0:
            best_ious_concat = np.ascontiguousarray(np.concatenate(best_ious_all, axis=0), dtype=np.float32)
            metrics["dbg_best_iou_mean"] = float(np.mean(best_ious_concat))
            metrics["dbg_best_iou_median"] = float(np.median(best_ious_concat))
        else:
            metrics["dbg_best_iou_mean"] = 0.0
            metrics["dbg_best_iou_median"] = 0.0

        pred_count_arr = np.asarray(self._pred_count_per_img, dtype=np.float32)
        pred_score_max_arr = np.asarray(self._pred_score_max_per_img, dtype=np.float32)
        metrics["dbg_recall_iou50"] = float(matched_gt_iou50 / total_gt_boxes) if total_gt_boxes > 0 else 0.0
        metrics["dbg_pred_count_mean"] = float(np.mean(pred_count_arr)) if pred_count_arr.size > 0 else 0.0
        metrics["dbg_pred_has_any_frac"] = float(np.mean(pred_count_arr > 0)) if pred_count_arr.size > 0 else 0.0
        metrics["dbg_pred_score_max_p95"] = float(np.percentile(pred_score_max_arr, 95)) if pred_score_max_arr.size > 0 else 0.0

        for cls, aps in per_class_aps.items():
            if class_names is not None and cls < len(class_names):
                key = f"AP50/{class_names[cls]}"
            else:
                key = f"AP50/cls_{cls}"
            metrics[key] = aps[idx_50]

        logger.info(
            f"Final metrics: mAP={metrics['mAP']:.4f}, "
            f"mAP50={metrics['mAP50']:.4f}, mAP75={metrics['mAP75']:.4f}"
        )
        return metrics


def decode_sttp_predictions(
    fut_preds: List[torch.Tensor],
    hw: List[Tuple[int, int]],
    strides: List[int],
    *,
    target_mode: str = "absolute",
    current_boxes: Optional[torch.Tensor] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> torch.Tensor:
    """Decode STTP predictions to absolute cxcywh coordinates in pixel space.

    In ``absolute`` mode this uses the same grid/stride decoding as STTPLoss / YOLOXHead.
    In ``delta`` mode the raw head output is interpreted as normalized displacement /
    size delta relative to the current box and is decoded back into absolute pixel-space
    cxcywh for evaluation.

    Args:
        fut_preds: List of 3 tensors [B, 4*k, H_k, W_k] from the STTP head
            (k = future_steps, channels packed [step1_4, step2_4, ...]).
        hw: List of (H, W) tuples for each FPN level (from head.hw).
        strides: List of stride ints for each level (from head.strides).

    Returns:
        [B, n_anchors_all, k, 4] decoded future-trajectory boxes in absolute cxcywh.
    """
    if target_mode not in {"absolute", "delta"}:
        raise ValueError(f"target_mode must be 'absolute' or 'delta', got {target_mode!r}")
    dtype = fut_preds[0].dtype
    device = fut_preds[0].device

    grids = []
    stride_tensors = []
    for (hsize, wsize), stride in zip(hw, strides):
        yv, xv = torch.meshgrid(
            torch.arange(hsize, device=device),
            torch.arange(wsize, device=device),
            indexing="ij",
        )
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype)
        grids.append(grid)
        stride_tensors.append(torch.full((*grid.shape[:2], 1), stride, dtype=dtype, device=device))

    grids_cat = torch.cat(grids, dim=1)          # [1, n_anchors_all, 2]
    strides_cat = torch.cat(stride_tensors, dim=1)  # [1, n_anchors_all, 1]

    # Flatten multi-scale predictions, then split the packed step axis: [B, A, k, 4].
    flat = []
    for pred in fut_preds:
        B, C, H, W = pred.shape
        flat.append(pred.permute(0, 2, 3, 1).reshape(B, -1, C))
    flat = torch.cat(flat, dim=1)  # [B, n_anchors_all, 4*k]
    k = flat.shape[-1] // 4
    flat = flat.reshape(flat.shape[0], flat.shape[1], k, 4)  # [B, A, k, 4]
    grids_b = grids_cat.unsqueeze(2)      # [1, A, 1, 2]
    strides_b = strides_cat.unsqueeze(2)  # [1, A, 1, 1]

    if target_mode == "absolute":
        decoded = flat.clone()
        decoded[..., :2] = (flat[..., :2] + grids_b) * strides_b
        decoded[..., 2:4] = torch.exp(flat[..., 2:4].clamp(max=10.0)) * strides_b
        return decoded

    if current_boxes is None or image_width is None or image_height is None:
        raise ValueError("delta target mode requires current_boxes, image_width, and image_height")

    # Delta mode: same current box broadcast across the k steps.
    cb = current_boxes.to(device=device, dtype=dtype).unsqueeze(2)  # [B, A, 1, 4]
    decoded = cb.expand(-1, -1, k, -1).clone()
    decoded[..., 0] = cb[..., 0] + flat[..., 0] * float(image_width)
    decoded[..., 1] = cb[..., 1] + flat[..., 1] * float(image_height)
    decoded[..., 2] = cb[..., 2] + flat[..., 2] * float(image_width)
    decoded[..., 3] = cb[..., 3] + flat[..., 3] * float(image_height)
    decoded[..., 2:4] = decoded[..., 2:4].clamp(min=1e-6)
    return decoded


class TrajectoryMetricsAccumulator:
    """Accumulates decoded future-center errors for ADE/FDE over a validation epoch.

    For each image, matches post-NMS detections to GT objects via IoU, traces
    each detection back to its source anchor, and computes the L2 center error
    between the decoded future prediction and the GT future box in pixel space.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._l2_errors: List[float] = []  # one entry per matched (object, valid step)
        self._steps: List[int] = []        # 0-based trajectory step, parallel to _l2_errors
        self._classes: List[int] = []      # GT class id, parallel to _l2_errors
        self._tags: List[int] = []         # optional per-pred tag, parallel to _l2_errors
        self._k: int = 0                   # trajectory length (set on first update)

    def update(
        self,
        det_pre_nms_cxcywh: torch.Tensor,
        fut_decoded: torch.Tensor,
        predictions_post_nms: List[Optional[torch.Tensor]],
        targets_per_image: List[torch.Tensor],
        future_targets: Dict[str, torch.Tensor],
        anchor_indices_per_image: Optional[List[Optional[torch.Tensor]]] = None,
        pred_tag_per_image: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        """Accumulate one batch.

        Args:
            det_pre_nms_cxcywh: [B, n_anchors, 4] decoded detection boxes (cx,cy,w,h)
                before NMS modifies them.
            fut_decoded: [B, n_anchors, k, 4] decoded future-trajectory boxes (cx,cy,w,h).
            predictions_post_nms: per-image post-NMS detections [N,7] in xyxy format.
            targets_per_image: per-image GT [M,5] (class_id, cx,cy,w,h), already filtered.
            future_targets: dict with "boxes" [B, max_gt, k, 4] and "validity" [B, max_gt, k].
            anchor_indices_per_image: optional per-image LongTensor of source anchor
                indices for each surviving detection (from
                ``postprocess_predictions(..., return_anchor_indices=True)``). When
                provided, replaces the legacy reverse-NN cdist match.
        """
        fut_boxes_all = future_targets["boxes"]   # [B, max_gt, k, 4]
        fut_valid_all = future_targets["validity"]  # [B, max_gt, k]
        self._k = int(fut_decoded.shape[2])
        bsz = det_pre_nms_cxcywh.shape[0]

        for i in range(bsz):
            dets = predictions_post_nms[i]
            if dets is None or dets.shape[0] == 0:
                continue

            gt = targets_per_image[i]
            if gt.numel() == 0:
                continue
            gt = gt.to(dets.device)

            n_gt = gt.shape[0]
            gt_fut_boxes = fut_boxes_all[i, :n_gt]     # [n_gt, k, 4] cxcywh
            gt_fut_valid = fut_valid_all[i, :n_gt]      # [n_gt, k] bool

            # Skip if no GT objects have any valid future step
            if not gt_fut_valid.any():
                continue

            # 1) Map post-NMS detections → source anchors. Prefer the exact indices
            # plumbed through ``postprocess_predictions``; fall back to a reverse-NN
            # cdist match only if they are unavailable (e.g. DETR head path).
            anchor_idx_provided = (
                anchor_indices_per_image[i]
                if anchor_indices_per_image is not None and i < len(anchor_indices_per_image)
                else None
            )
            if anchor_idx_provided is not None:
                anchor_idx = anchor_idx_provided.to(dets.device)
            else:
                det_cx = (dets[:, 0] + dets[:, 2]) / 2
                det_cy = (dets[:, 1] + dets[:, 3]) / 2
                det_w = dets[:, 2] - dets[:, 0]
                det_h = dets[:, 3] - dets[:, 1]
                det_cxcywh = torch.stack([det_cx, det_cy, det_w, det_h], dim=1)  # [N_det, 4]

                pre_nms = det_pre_nms_cxcywh[i]  # [n_anchors, 4]
                dist = torch.cdist(det_cxcywh.float(), pre_nms.float(), p=1)
                anchor_idx = dist.argmin(dim=1)  # [N_det]

            # 2) Match detections → GT via IoU
            gt_xyxy = _targets_cxcywh_to_xyxy_torch(gt[:, 1:5])  # [n_gt, 4]
            iou = bbox_iou(dets[:, :4], gt_xyxy)                  # [N_det, n_gt]

            # Greedy assignment: best det for each GT
            matched_det = {}  # gt_idx → det_idx
            used_dets = set()
            # Sort GT matches by IoU descending for greedy assignment
            if iou.numel() > 0:
                max_iou_per_det, best_gt_per_det = iou.max(dim=1)
                sorted_det_idx = max_iou_per_det.argsort(descending=True)
                for d_idx in sorted_det_idx.tolist():
                    g_idx = int(best_gt_per_det[d_idx].item())
                    if max_iou_per_det[d_idx] < self.iou_threshold:
                        continue
                    if g_idx in matched_det or d_idx in used_dets:
                        continue
                    matched_det[g_idx] = d_idx
                    used_dets.add(d_idx)

            # 3) Compute per-step displacement errors for matched pairs (valid steps only)
            fut_dec_i = fut_decoded[i]  # [n_anchors, k, 4]
            tag_i = (
                pred_tag_per_image[i]
                if pred_tag_per_image is not None and i < len(pred_tag_per_image)
                else None
            )
            for g_idx, d_idx in matched_det.items():
                step_valid = gt_fut_valid[g_idx]  # [k] bool
                if not step_valid.any():
                    continue

                a_idx = int(anchor_idx[d_idx].item())
                pred_centers = fut_dec_i[a_idx, :, :2]       # [k, 2]
                gt_centers = gt_fut_boxes[g_idx, :, :2]      # [k, 2]
                l2_steps = (pred_centers - gt_centers).norm(dim=1)  # [k]
                cls = int(gt[g_idx, 0].item())
                tag = int(tag_i[d_idx]) if tag_i is not None else -1
                for s in range(self._k):
                    if not bool(step_valid[s]):
                        continue
                    self._l2_errors.append(float(l2_steps[s].item()))
                    self._steps.append(s)
                    self._classes.append(cls)
                    # Optional per-prediction tag (e.g. tracker age) kept parallel to the
                    # error list so callers can stratify (warm vs cold-start) post hoc.
                    self._tags.append(tag)

    def compute(self) -> Dict[str, float]:
        """Return decoded-pixel displacement metrics over matched objects.

        The head emits a k-step trajectory (t+1..t+k), so ADE averages over all
        valid (object, step) pairs while FDE is the error at the final step t+k
        (over objects valid at that step); the two are now genuinely distinct.
        ``ADE_step{j}`` (1-based) gives the per-step displacement curve, and
        ``ADE_cls{id}`` the per-class ADE (DSEC-Det class ids).
        """
        if not self._l2_errors:
            return {}
        import numpy as _np
        errors = _np.array(self._l2_errors, dtype=_np.float64)
        steps = _np.array(self._steps, dtype=_np.int64)
        classes = _np.array(self._classes, dtype=_np.int64)
        ade = float(errors.mean())
        last = self._k - 1
        final_mask = steps == last
        # FDE = error at final step t+k; fall back to ADE if no object survived to k.
        fde = float(errors[final_mask].mean()) if final_mask.any() else ade
        out: Dict[str, float] = {"ADE": ade, "FDE": fde}
        for j in range(self._k):
            sel = steps == j
            if sel.any():
                out[f"ADE_step{j + 1}"] = float(errors[sel].mean())
        for cid in _np.unique(classes):
            out[f"ADE_cls{int(cid)}"] = float(errors[classes == cid].mean())
        return out


def _targets_cxcywh_to_xyxy_torch(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [N,4] cxcywh to xyxy (torch version for GPU tensors)."""
    out = torch.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def calculate_map(
    predictions: List[List[torch.Tensor]],
    targets: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
    verbose: bool = False,
    class_names: Optional[Tuple[str, ...]] = None,
) -> Dict[str, float]:
    """
    Calculate COCO-standard mAP metrics for object detection.

    Evaluates at 10 IoU thresholds [0.50, 0.55, ..., 0.95] and returns:
      - ``mAP``   — mean AP averaged over all thresholds (COCO primary metric)
      - ``mAP50`` — AP at IoU=0.5
      - ``mAP75`` — AP at IoU=0.75
      - ``AP50/<class_name>`` — per-class AP at IoU=0.5

    Args:
        predictions: List of predictions per image, each is list of detections per image.
                    Each detection: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        targets: List of ground truth per image, shape (num_boxes, 5).
                Format: [class_id, cx, cy, w, h]
        num_classes: Number of classes
        iou_threshold: Kept for backward compatibility (unused — all COCO thresholds are evaluated)
        verbose: If True, log detailed per-class statistics
        class_names: Optional tuple of class name strings. When provided, per-class
                    keys use ``AP50/<name>``; otherwise ``AP50/cls_<i>``.

    Returns:
        Dictionary with mAP metrics
    """
    accumulator = StreamingMAPAccumulator(num_classes=num_classes)
    accumulator.update(predictions, targets)
    return accumulator.compute(verbose=verbose, class_names=class_names)
