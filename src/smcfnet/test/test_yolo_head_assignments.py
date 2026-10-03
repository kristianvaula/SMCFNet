import torch

from smcfnet.model.yolo_head import YOLOXHead


def test_get_assignments_does_not_delete_nonexistent_temporaries():
    head = YOLOXHead(num_classes=8, width=0.25, in_channels=[64, 128, 256])

    gt_boxes = torch.tensor([[4.0, 4.0, 8.0, 8.0]], dtype=torch.float32)
    gt_classes = torch.tensor([1.0], dtype=torch.float32)
    pred_boxes = torch.tensor(
        [
            [4.0, 4.0, 8.0, 8.0],
            [12.0, 4.0, 8.0, 8.0],
            [20.0, 4.0, 8.0, 8.0],
            [28.0, 4.0, 8.0, 8.0],
        ],
        dtype=torch.float32,
    )
    x_shifts = torch.arange(4, dtype=torch.float32).view(1, 4)
    y_shifts = torch.zeros((1, 4), dtype=torch.float32)
    expanded_strides = torch.full((1, 4), 8.0, dtype=torch.float32)
    cls_preds = torch.zeros((1, 4, 8), dtype=torch.float32)
    obj_preds = torch.zeros((1, 4, 1), dtype=torch.float32)

    gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = head.get_assignments(
        batch_idx=0,
        num_gt=1,
        gt_bboxes_per_image=gt_boxes,
        gt_classes=gt_classes,
        bboxes_preds_per_image=pred_boxes,
        expanded_strides=expanded_strides,
        x_shifts=x_shifts,
        y_shifts=y_shifts,
        cls_preds=cls_preds,
        obj_preds=obj_preds,
    )

    assert num_fg >= 1
    assert fg_mask.any()
    assert gt_matched_classes.numel() == pred_ious.numel() == matched_gt_inds.numel()
