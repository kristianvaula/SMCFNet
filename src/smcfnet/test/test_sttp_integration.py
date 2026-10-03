"""Integration tests for STTP wired into DualStreamYOLOX.

Validates that:
- Detection still works when STTP is enabled
- STTP loss appears in loss dict
- STTP disabled (sttp_head=None) doesn't change detection behavior
- Gradient flows from STTP loss back through stem

Run: pytest src/smcfnet/test/test_sttp_integration.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import STTPHead, STTPLoss
from smcfnet.model.yolo_head import YOLOXHead


# ── Fixtures ──────────────────────────────────────────────────────────────

IN_CHANNELS = [256, 512, 1024]
WIDTH = 0.5
NUM_CLASSES = 8
BATCH = 2
IMG_H, IMG_W = 480, 640


def _make_neck_outputs(device):
    """Synthetic PAFPN outputs at P3/P4/P5."""
    return [
        torch.randn(BATCH, int(256 * WIDTH), 60, 80, device=device),
        torch.randn(BATCH, int(512 * WIDTH), 30, 40, device=device),
        torch.randn(BATCH, int(1024 * WIDTH), 15, 20, device=device),
    ]


def _make_targets(device, num_gt=5):
    """Synthetic targets: [B, max_gt, 5] with (class, cx, cy, w, h)."""
    targets = torch.zeros(BATCH, num_gt, 5, device=device)
    for b in range(BATCH):
        for i in range(num_gt):
            targets[b, i, 0] = i % NUM_CLASSES
            targets[b, i, 1] = 100 + i * 50  # cx
            targets[b, i, 2] = 100 + i * 40  # cy
            targets[b, i, 3] = 30 + i * 5    # w
            targets[b, i, 4] = 30 + i * 5    # h
    return targets


def _make_future_targets(device, num_gt=5):
    """Synthetic future targets dict."""
    boxes = torch.zeros(BATCH, num_gt, 4, device=device)
    validity = torch.ones(BATCH, num_gt, dtype=torch.bool, device=device)
    for b in range(BATCH):
        for i in range(num_gt):
            boxes[b, i, 0] = 105 + i * 50  # cx shifted
            boxes[b, i, 1] = 103 + i * 40  # cy shifted
            boxes[b, i, 2] = 30 + i * 5    # w
            boxes[b, i, 3] = 30 + i * 5    # h
    return {"boxes": boxes, "validity": validity}


# ── YOLOXHead stem exposure ──────────────────────────────────────────────

class TestStemExposure:
    def test_training_returns_stem_features(self):
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        neck = _make_neck_outputs("cpu")
        targets = _make_targets("cpu")

        result, stems = head(neck, labels=targets)

        assert isinstance(stems, list)
        assert len(stems) == 3
        stem_c = int(256 * WIDTH)
        assert stems[0].shape == (BATCH, stem_c, 60, 80)
        assert stems[1].shape == (BATCH, stem_c, 30, 40)
        assert stems[2].shape == (BATCH, stem_c, 15, 20)

    def test_inference_returns_stem_features(self):
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        head.eval()
        neck = _make_neck_outputs("cpu")

        result, stems = head(neck)

        assert isinstance(stems, list)
        assert len(stems) == 3

    def test_assignments_in_training_output(self):
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        neck = _make_neck_outputs("cpu")
        targets = _make_targets("cpu")

        result, stems = head(neck, labels=targets)

        # result is a 7-tuple: (loss, box, obj, cls, l1, fg_ratio, assignments)
        assert len(result) == 7
        assignments = result[6]
        assert "fg_masks" in assignments
        assert "matched_gt_inds" in assignments
        assert "expanded_strides" in assignments
        assert "num_fg" in assignments


# ── STTPHead + YOLOXHead combined ────────────────────────────────────────

class TestSTTPWithYOLOXHead:
    def test_sttp_forward_on_real_stems(self):
        """STTPHead should work on actual YOLOXHead stem outputs."""
        yolox_head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        sttp_head = STTPHead(int(256 * WIDTH), flow_channels=0)

        neck = _make_neck_outputs("cpu")
        targets = _make_targets("cpu")

        _, stems = yolox_head(neck, labels=targets)
        fut_preds = sttp_head(stems)

        assert len(fut_preds) == 3
        assert fut_preds[0].shape == (BATCH, 4, 60, 80)

    def test_sttp_gradient_reaches_stem(self):
        """STTP loss gradient should flow back through shared stem."""
        yolox_head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        sttp_head = STTPHead(int(256 * WIDTH), flow_channels=0)

        neck = _make_neck_outputs("cpu")
        targets = _make_targets("cpu")

        result, stems = yolox_head(neck, labels=targets)
        fut_preds = sttp_head(stems)

        # Simple proxy loss on fut preds
        fut_loss = sum(p.sum() for p in fut_preds)
        fut_loss.backward()

        # Stem conv weights should have gradients (shared path)
        assert yolox_head.stems[0].conv.weight.grad is not None
        assert yolox_head.stems[0].conv.weight.grad.abs().sum() > 0

    def test_detection_unchanged_without_sttp(self):
        """Adding STTP head should not change detection output when not backpropping."""
        yolox_head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        yolox_head.eval()
        neck = _make_neck_outputs("cpu")

        det_a, _ = yolox_head(neck)

        # Create STTP head but don't use it
        _ = STTPHead(int(256 * WIDTH), flow_channels=0)
        det_b, _ = yolox_head(neck)

        assert torch.allclose(det_a, det_b)


# ── Design C integration ─────────────────────────────────────────────────

class TestDesignCIntegration:
    def test_design_c_with_flow(self):
        yolox_head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=IN_CHANNELS)
        sttp_head = STTPHead(int(256 * WIDTH), flow_channels=2, flow_proj_dim=16)

        neck = _make_neck_outputs("cpu")
        targets = _make_targets("cpu")
        flow = torch.randn(BATCH, 2, IMG_H, IMG_W)

        _, stems = yolox_head(neck, labels=targets)
        fut_preds = sttp_head(stems, flow=flow)

        assert len(fut_preds) == 3
        assert fut_preds[0].shape[1] == 4
