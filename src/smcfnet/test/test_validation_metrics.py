import unittest

import numpy as np
import torch

from smcfnet.utils.validation import (
    StreamingMAPAccumulator,
    _COCO_IOU_THRESHOLDS,
    bbox_iou,
    calculate_map,
    compute_ap,
)


def _xyxy_to_target_row(class_id: int, x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = float(x2 - x1)
    h = float(y2 - y1)
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return [float(class_id), cx, cy, w, h]


def _reference_calculate_map(predictions, targets, num_classes: int, class_names=None):
    per_class_aps = {}
    pred_count_per_img = []
    pred_score_max_per_img = []
    best_ious_global = []
    matched_gt_iou50 = 0
    total_gt_boxes_all = 0

    for cls in range(num_classes):
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_img_ids = []
        all_true_boxes = []
        n_gt_boxes = 0

        for img_idx, (pred_list, target) in enumerate(zip(predictions, targets)):
            pred = pred_list[0] if isinstance(pred_list, list) else pred_list
            if cls == 0:
                if pred is None or len(pred) == 0:
                    pred_count_per_img.append(0)
                    pred_score_max_per_img.append(0.0)
                else:
                    pred_count_per_img.append(int(len(pred)))
                    pred_score_max_per_img.append(float((pred[:, 4] * pred[:, 5]).max().item()))
            if pred is not None and len(pred) > 0:
                class_mask = pred[:, -1] == cls
                if class_mask.any():
                    pred_cls = pred[class_mask]
                    all_pred_boxes.append(pred_cls[:, :4].cpu())
                    all_pred_scores.append((pred_cls[:, 4] * pred_cls[:, 5]).cpu())
                    all_pred_img_ids.extend([img_idx] * len(pred_cls))

            if target is not None and len(target) > 0:
                class_mask = target[:, 0] == cls
                if class_mask.any():
                    gt_cls = target[class_mask]
                    boxes_cxcywh = gt_cls[:, 1:5].cpu()
                    boxes_xyxy = torch.zeros_like(boxes_cxcywh)
                    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
                    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
                    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
                    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

                    all_true_boxes.extend((img_idx, box) for box in boxes_xyxy)
                    n_gt_boxes += len(boxes_xyxy)

        if n_gt_boxes == 0:
            continue
        total_gt_boxes_all += n_gt_boxes

        if len(all_pred_boxes) == 0:
            per_class_aps[cls] = [0.0] * len(_COCO_IOU_THRESHOLDS)
            continue

        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)
        pred_img_ids = torch.tensor(all_pred_img_ids, dtype=torch.long)

        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_img_ids = pred_img_ids[sort_idx]

        gt_boxes_by_img = {}
        for gt_idx, (img_idx, gt_box) in enumerate(all_true_boxes):
            if img_idx not in gt_boxes_by_img:
                gt_boxes_by_img[img_idx] = ([], [])
            gt_boxes_by_img[img_idx][0].append(gt_box)
            gt_boxes_by_img[img_idx][1].append(gt_idx)

        gt_boxes_by_img_stacked = {}
        for img_idx, (boxes_list, idx_list) in gt_boxes_by_img.items():
            gt_boxes_by_img_stacked[img_idx] = (torch.stack(boxes_list), idx_list)

        best_iou_arr = torch.zeros(len(pred_boxes), dtype=torch.float32)
        best_gt_idx_arr = torch.full((len(pred_boxes),), -1, dtype=torch.long)

        for img_id_t in pred_img_ids.unique():
            img_id = int(img_id_t.item())
            if img_id not in gt_boxes_by_img_stacked:
                continue
            gt_stacked, gt_indices = gt_boxes_by_img_stacked[img_id]
            pred_mask = pred_img_ids == img_id_t
            preds_for_img = pred_boxes[pred_mask]
            iou_matrix = bbox_iou(preds_for_img, gt_stacked)
            max_iou, max_local_idx = iou_matrix.max(dim=1)
            gt_idx_tensor = torch.tensor(gt_indices, dtype=torch.long)
            max_gt_idx = gt_idx_tensor[max_local_idx]
            best_iou_arr[pred_mask] = max_iou
            best_gt_idx_arr[pred_mask] = max_gt_idx

        best_ious_global.append(best_iou_arr.numpy())

        detected_iou50 = set()
        tp_iou50 = 0
        for pred_idx in range(len(pred_boxes)):
            best_iou = float(best_iou_arr[pred_idx].item())
            best_gt_idx = int(best_gt_idx_arr[pred_idx].item())
            pred_img_id = int(pred_img_ids[pred_idx].item())
            if best_gt_idx >= 0 and best_iou >= 0.5 and (pred_img_id, best_gt_idx) not in detected_iou50:
                detected_iou50.add((pred_img_id, best_gt_idx))
                tp_iou50 += 1
        matched_gt_iou50 += tp_iou50

        cls_aps = []
        for iou_thr in _COCO_IOU_THRESHOLDS:
            tp = torch.zeros(len(pred_boxes), dtype=torch.float32)
            fp = torch.zeros(len(pred_boxes), dtype=torch.float32)
            detected = set()

            for pred_idx in range(len(pred_boxes)):
                best_iou = float(best_iou_arr[pred_idx].item())
                best_gt_idx = int(best_gt_idx_arr[pred_idx].item())
                pred_img_id = int(pred_img_ids[pred_idx].item())

                if best_gt_idx >= 0 and best_iou >= float(iou_thr) and (pred_img_id, best_gt_idx) not in detected:
                    tp[pred_idx] = 1.0
                    detected.add((pred_img_id, best_gt_idx))
                else:
                    fp[pred_idx] = 1.0

            recalls = torch.cumsum(tp, dim=0).numpy() / (n_gt_boxes + 1e-6)
            precisions = torch.cumsum(tp, dim=0).numpy() / (
                torch.cumsum(tp, dim=0).numpy() + torch.cumsum(fp, dim=0).numpy() + 1e-6
            )
            cls_aps.append(compute_ap(recalls, precisions))

        per_class_aps[cls] = cls_aps

    idx_50 = 0
    idx_75 = 5
    all_ap50 = [aps[idx_50] for aps in per_class_aps.values()]
    all_ap75 = [aps[idx_75] for aps in per_class_aps.values()]
    all_ap_mean = [float(np.mean(aps)) for aps in per_class_aps.values()]

    metrics = {
        "mAP": float(np.mean(all_ap_mean)) if all_ap_mean else 0.0,
        "mAP50": float(np.mean(all_ap50)) if all_ap50 else 0.0,
        "mAP75": float(np.mean(all_ap75)) if all_ap75 else 0.0,
    }
    if len(best_ious_global) > 0:
        best_ious_concat = np.concatenate(best_ious_global, axis=0)
        metrics["dbg_best_iou_mean"] = float(np.mean(best_ious_concat))
        metrics["dbg_best_iou_median"] = float(np.median(best_ious_concat))
    else:
        metrics["dbg_best_iou_mean"] = 0.0
        metrics["dbg_best_iou_median"] = 0.0
    pred_count_arr = np.asarray(pred_count_per_img, dtype=np.float32)
    pred_score_max_arr = np.asarray(pred_score_max_per_img, dtype=np.float32)
    metrics["dbg_recall_iou50"] = float(matched_gt_iou50 / total_gt_boxes_all) if total_gt_boxes_all > 0 else 0.0
    metrics["dbg_pred_count_mean"] = float(np.mean(pred_count_arr)) if pred_count_arr.size > 0 else 0.0
    metrics["dbg_pred_has_any_frac"] = float(np.mean(pred_count_arr > 0)) if pred_count_arr.size > 0 else 0.0
    metrics["dbg_pred_score_max_p95"] = float(np.percentile(pred_score_max_arr, 95)) if pred_score_max_arr.size > 0 else 0.0

    for cls, aps in per_class_aps.items():
        if class_names is not None and cls < len(class_names):
            key = f"AP50/{class_names[cls]}"
        else:
            key = f"AP50/cls_{cls}"
        metrics[key] = aps[idx_50]

    return metrics


class TestValidationMetrics(unittest.TestCase):
    def test_streaming_accumulator_matches_reference_map(self):
        class_names = ("car", "pedestrian", "cyclist")
        predictions = [
            torch.tensor(
                [
                    [8.0, 8.0, 12.0, 12.0, 0.95, 0.90, 0.0],
                    [30.0, 30.0, 36.0, 36.0, 0.70, 0.80, 1.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor([[40.0, 40.0, 44.0, 44.0, 0.90, 0.85, 0.0]], dtype=torch.float32),
            torch.tensor(
                [
                    [50.0, 50.0, 56.0, 56.0, 0.92, 0.88, 1.0],
                    [49.0, 49.0, 57.0, 57.0, 0.40, 0.90, 1.0],
                    [60.0, 60.0, 66.0, 66.0, 0.96, 0.95, 2.0],
                ],
                dtype=torch.float32,
            ),
            None,
            [torch.tensor([[99.0, 99.0, 105.0, 105.0, 0.80, 0.75, 2.0]], dtype=torch.float32)],
        ]
        targets = [
            torch.tensor([_xyxy_to_target_row(0, 8.0, 8.0, 12.0, 12.0)], dtype=torch.float32),
            torch.zeros((0, 5), dtype=torch.float32),
            torch.tensor(
                [
                    _xyxy_to_target_row(1, 50.0, 50.0, 56.0, 56.0),
                    _xyxy_to_target_row(2, 60.0, 60.0, 66.0, 66.0),
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    _xyxy_to_target_row(0, 20.0, 20.0, 24.0, 24.0),
                    _xyxy_to_target_row(1, 70.0, 70.0, 76.0, 76.0),
                ],
                dtype=torch.float32,
            ),
            torch.tensor([_xyxy_to_target_row(2, 100.0, 100.0, 106.0, 106.0)], dtype=torch.float32),
        ]

        expected = _reference_calculate_map(predictions, targets, num_classes=3, class_names=class_names)
        direct_metrics = calculate_map(predictions, targets, num_classes=3, class_names=class_names)

        accumulator = StreamingMAPAccumulator(num_classes=3)
        accumulator.update(predictions[:2], targets[:2])
        accumulator.update(predictions[2:4], targets[2:4])
        accumulator.update(predictions[4:], targets[4:])
        streamed_metrics = accumulator.compute(class_names=class_names)

        self.assertEqual(set(expected.keys()), set(direct_metrics.keys()))
        self.assertEqual(set(expected.keys()), set(streamed_metrics.keys()))
        for key, value in expected.items():
            self.assertAlmostEqual(direct_metrics[key], value, places=6, msg=key)
            self.assertAlmostEqual(streamed_metrics[key], value, places=6, msg=key)


if __name__ == "__main__":
    unittest.main()
