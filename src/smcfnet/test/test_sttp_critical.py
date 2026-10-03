"""Critical STTP tests: decoding correctness, _gather_future_targets,
trajectory metrics, end-to-end training forward, and inference tuple.

These cover the gaps identified in the STTP test audit — the most
dangerous untested code paths.

Run: python -m pytest src/smcfnet/test/test_sttp_critical.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import STTPHead, STTPLoss
from smcfnet.model.yolo_head import YOLOXHead
from smcfnet.utils.trajectory_metrics import (
    compute_ade,
    compute_fde,
    compute_ade_fde,
    compute_ttc_proxy,
)

# ── Constants ────────────────────────────────────────────────────────────

BATCH = 2
NUM_CLASSES = 8
WIDTH = 0.5
STEM_C = int(256 * WIDTH)
HW = [(60, 80), (30, 40), (15, 20)]
STRIDES = [8, 16, 32]
N_ANCHORS = [h * w for h, w in HW]
N_TOTAL = sum(N_ANCHORS)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_neck_outputs(device="cpu"):
    return [
        torch.randn(BATCH, STEM_C, 60, 80, device=device),
        torch.randn(BATCH, int(512 * WIDTH), 30, 40, device=device),
        torch.randn(BATCH, int(1024 * WIDTH), 15, 20, device=device),
    ]


def _make_targets(device="cpu", num_gt=5):
    targets = torch.zeros(BATCH, num_gt, 5, device=device)
    for b in range(BATCH):
        for i in range(num_gt):
            targets[b, i, 0] = i % NUM_CLASSES
            targets[b, i, 1] = 100 + i * 50  # cx
            targets[b, i, 2] = 100 + i * 40  # cy
            targets[b, i, 3] = 30 + i * 5    # w
            targets[b, i, 4] = 30 + i * 5    # h
    return targets


def _make_future_targets(device="cpu", num_gt=5):
    boxes = torch.zeros(BATCH, num_gt, 4, device=device)
    validity = torch.ones(BATCH, num_gt, dtype=torch.bool, device=device)
    current_boxes = torch.zeros(BATCH, num_gt, 4, device=device)
    for b in range(BATCH):
        for i in range(num_gt):
            current_boxes[b, i, 0] = 100 + i * 50
            current_boxes[b, i, 1] = 100 + i * 40
            current_boxes[b, i, 2] = 30 + i * 5
            current_boxes[b, i, 3] = 30 + i * 5
            boxes[b, i, 0] = 105 + i * 50
            boxes[b, i, 1] = 103 + i * 40
            boxes[b, i, 2] = 30 + i * 5
            boxes[b, i, 3] = 30 + i * 5
    # Multi-step contract: boxes [B,num_gt,k,4], validity [B,num_gt,k]. The default
    # STTPHead here has future_steps=1, so k=1.
    return {
        "boxes": boxes.unsqueeze(2),
        "validity": validity.unsqueeze(2),
        "current_boxes": current_boxes,
    }


def _build_grids_and_strides():
    grids_list = []
    strides_list = []
    for (h, w), stride in zip(HW, STRIDES):
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((xv, yv), dim=-1).reshape(1, -1, 2).float()
        grids_list.append(grid)
        strides_list.append(torch.full((1, h * w), stride, dtype=torch.float32))
    return torch.cat(grids_list, dim=1), torch.cat(strides_list, dim=1)


# ══════════════════════════════════════════════════════════════════════════
# 1. STTP DECODING vs YOLOX DECODING — must produce identical transforms
# ══════════════════════════════════════════════════════════════════════════

class TestDecodingConsistency:
    """Verify STTPLoss decoding matches YOLOXHead.decode_outputs exactly."""

    def test_sttp_decode_matches_yolox_decode(self):
        """The grid+stride transform in STTPLoss lines 68-70 must produce the
        same absolute coords as YOLOXHead.decode_outputs for identical raw values."""
        grids, strides = _build_grids_and_strides()

        # Create raw predictions: [B, N_TOTAL, 4] in grid-relative format
        torch.manual_seed(42)
        raw = torch.randn(BATCH, N_TOTAL, 4)

        # --- YOLOX decode path (from yolo_head.py:232-253) ---
        yolox_grids = []
        yolox_strides = []
        for (h, w), stride in zip(HW, STRIDES):
            yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            yolox_grids.append(grid)
            yolox_strides.append(torch.full((*grid.shape[:2], 1), stride))
        yolox_grids_cat = torch.cat(yolox_grids, dim=1).float()
        yolox_strides_cat = torch.cat(yolox_strides, dim=1).float()

        yolox_decoded = torch.cat([
            (raw[..., 0:2] + yolox_grids_cat) * yolox_strides_cat,
            torch.exp(raw[..., 2:4]) * yolox_strides_cat,
        ], dim=-1)

        # --- STTP decode path (from sttp_loss.py:68-70) ---
        sttp_decoded = raw.clone()
        sttp_decoded[..., :2] = (raw[..., :2] + grids) * strides.unsqueeze(-1)
        sttp_decoded[..., 2:4] = torch.exp(raw[..., 2:4].clamp(max=10.0)) * strides.unsqueeze(-1)

        # Centers should match exactly (no clamp involved)
        assert torch.allclose(sttp_decoded[..., :2], yolox_decoded[..., :2], atol=1e-5), \
            "STTP and YOLOX center decoding diverged"

        # Sizes: STTP has clamp(max=10), YOLOX doesn't. For reasonable values they match.
        reasonable = raw[..., 2:4].abs() < 10.0
        assert torch.allclose(
            sttp_decoded[..., 2:4][reasonable],
            yolox_decoded[..., 2:4][reasonable],
            atol=1e-5,
        ), "STTP and YOLOX size decoding diverged for reasonable values"

    def test_clamp_prevents_explosion(self):
        """exp(10) ~ 22026 — without clamp, large raw values produce huge sizes."""
        grids, strides = _build_grids_and_strides()

        raw = torch.zeros(1, N_TOTAL, 4)
        raw[..., 2:4] = 20.0  # extreme value

        decoded = raw.clone()
        decoded[..., :2] = (raw[..., :2] + grids) * strides.unsqueeze(-1)
        decoded[..., 2:4] = torch.exp(raw[..., 2:4].clamp(max=10.0)) * strides.unsqueeze(-1)

        # Should be clamped to exp(10) * stride, not exp(20) * stride
        max_size = decoded[..., 2:4].max().item()
        assert max_size < 1e6, f"Decoded size {max_size} too large — clamp failed"
        # exp(10) * 32 ≈ 704832, which is the max possible
        assert max_size == pytest.approx(torch.exp(torch.tensor(10.0)).item() * 32, rel=1e-3)

    def test_decode_known_values(self):
        """Verify decode on a hand-computed example."""
        # Single anchor at grid position (5, 3) with stride 8
        grids = torch.tensor([[[5.0, 3.0]]])     # [1, 1, 2]
        strides = torch.tensor([[8.0]])            # [1, 1]

        # Raw prediction: dx=0.5, dy=-0.2, dw=1.0, dh=0.5
        raw = torch.tensor([[[0.5, -0.2, 1.0, 0.5]]])  # [1, 1, 4]

        decoded = raw.clone()
        decoded[..., :2] = (raw[..., :2] + grids) * strides.unsqueeze(-1)
        decoded[..., 2:4] = torch.exp(raw[..., 2:4].clamp(max=10.0)) * strides.unsqueeze(-1)

        # cx = (0.5 + 5) * 8 = 44.0
        assert decoded[0, 0, 0].item() == pytest.approx(44.0)
        # cy = (-0.2 + 3) * 8 = 22.4
        assert decoded[0, 0, 1].item() == pytest.approx(22.4)
        # w = exp(1.0) * 8 = 21.746
        assert decoded[0, 0, 2].item() == pytest.approx(torch.exp(torch.tensor(1.0)).item() * 8, rel=1e-4)
        # h = exp(0.5) * 8 = 13.189
        assert decoded[0, 0, 3].item() == pytest.approx(torch.exp(torch.tensor(0.5)).item() * 8, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════════
# 2. _gather_future_targets — SimOTA indexing correctness
# ══════════════════════════════════════════════════════════════════════════

class TestGatherFutureTargets:
    """Test the _gather_future_targets method on DualStreamYOLOX."""

    def _build_model(self):
        from smcfnet.model.yolox import DualStreamYOLOX
        from smcfnet.model.backbone import CSPDarknetDualStream
        from smcfnet.model.yolo_pafpn import YOLOPAFPN

        backbone = CSPDarknetDualStream(dep_mul=0.33, wid_mul=WIDTH)
        neck = YOLOPAFPN(depth=0.33, width=WIDTH, in_channels=[256, 512, 1024])
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=[256, 512, 1024])
        sttp_head = STTPHead(STEM_C, flow_channels=0)
        sttp_loss = STTPLoss()
        model = DualStreamYOLOX(backbone, neck, head, sttp_head=sttp_head, sttp_loss=sttp_loss)
        model._loss_sttp_weight.fill_(0.5)
        return model

    def test_gather_basic_indexing(self):
        """Verify _gather_future_targets returns correct boxes for given gt_inds."""
        model = self._build_model()

        future_targets = {
            "boxes": torch.tensor([
                [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],  # batch 0
                [[11, 21, 31, 41], [51, 61, 71, 81], [91, 101, 111, 121]],  # batch 1
            ], dtype=torch.float32),
            "validity": torch.tensor([
                [True, True, False],
                [True, False, True],
            ]),
            "current_boxes": torch.tensor([
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ], dtype=torch.float32),
        }
        # Multi-step contract: boxes [B,maxgt,k,4], validity [B,maxgt,k] (here k=1).
        future_targets["boxes"] = future_targets["boxes"].unsqueeze(2)
        future_targets["validity"] = future_targets["validity"].unsqueeze(2)

        # Simulate SimOTA: batch 0 matched GTs [2, 0], batch 1 matched GT [1]
        assignments = {
            "matched_gt_inds": [
                torch.tensor([2, 0]),  # batch 0: 2 fg anchors matched to GT 2 and GT 0
                torch.tensor([1]),      # batch 1: 1 fg anchor matched to GT 1
            ],
        }

        fut_targets, validity, current_boxes = model._gather_future_targets(future_targets, assignments)

        # Should be [3, k=1, 4] — 2 from batch 0 + 1 from batch 1
        assert fut_targets.shape == (3, 1, 4)
        assert validity.shape == (3, 1)

        # Batch 0, GT 2 → [90, 100, 110, 120], validity=False
        assert torch.allclose(fut_targets[0, 0], torch.tensor([90.0, 100, 110, 120]))
        assert validity[0, 0].item() == False

        # Batch 0, GT 0 → [10, 20, 30, 40], validity=True
        assert torch.allclose(fut_targets[1, 0], torch.tensor([10.0, 20, 30, 40]))
        assert validity[1, 0].item() == True

        # Batch 1, GT 1 → [51, 61, 71, 81], validity=False
        assert torch.allclose(fut_targets[2, 0], torch.tensor([51.0, 61, 71, 81]))
        assert validity[2, 0].item() == False
        assert torch.allclose(current_boxes[0], torch.tensor([9.0, 10, 11, 12]))
        assert torch.allclose(current_boxes[1], torch.tensor([1.0, 2, 3, 4]))
        assert torch.allclose(current_boxes[2], torch.tensor([17.0, 18, 19, 20]))

    def test_gather_empty_batch(self):
        """Batch with no foreground anchors should be skipped gracefully."""
        model = self._build_model()

        future_targets = {
            "boxes": torch.randn(BATCH, 3, 1, 4),
            "validity": torch.ones(BATCH, 3, 1, dtype=torch.bool),
            "current_boxes": torch.randn(BATCH, 3, 4),
        }

        assignments = {
            "matched_gt_inds": [
                torch.tensor([], dtype=torch.long),  # batch 0: no fg
                torch.tensor([0, 1]),                  # batch 1: 2 fg
            ],
        }

        fut_targets, validity, current_boxes = model._gather_future_targets(future_targets, assignments)

        assert fut_targets.shape == (2, 1, 4)
        assert validity.shape == (2, 1)
        assert current_boxes.shape == (2, 4)

    def test_gather_all_empty(self):
        """No foreground in any batch should return empty tensors."""
        model = self._build_model()

        future_targets = {
            "boxes": torch.randn(BATCH, 3, 1, 4),
            "validity": torch.ones(BATCH, 3, 1, dtype=torch.bool),
            "current_boxes": torch.randn(BATCH, 3, 4),
        }

        assignments = {
            "matched_gt_inds": [
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            ],
        }

        fut_targets, validity, current_boxes = model._gather_future_targets(future_targets, assignments)

        assert fut_targets.shape == (0, 1, 4)
        assert validity.shape == (0, 1)
        assert current_boxes.shape == (0, 4)


# ══════════════════════════════════════════════════════════════════════════
# 3. trajectory_metrics.py — every function
# ══════════════════════════════════════════════════════════════════════════

class TestTrajectoryMetrics:

    def test_ade_zero_error(self):
        """Identical preds and targets → ADE = 0."""
        preds = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])   # [1, 2, 2]
        targets = preds.clone()
        valid = torch.ones(1, 2)
        assert compute_ade(preds, targets, valid).item() == pytest.approx(0.0)

    def test_ade_known_value(self):
        """Hand-computed ADE."""
        preds = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])   # [1, 2, 2]
        targets = torch.zeros(1, 2, 2)
        valid = torch.ones(1, 2)
        # L2 distances: sqrt(9) = 3.0, sqrt(16) = 4.0 → ADE = 3.5
        assert compute_ade(preds, targets, valid).item() == pytest.approx(3.5)

    def test_ade_with_validity_mask(self):
        """Only valid steps should contribute."""
        preds = torch.tensor([[[3.0, 0.0], [100.0, 100.0]]])  # [1, 2, 2]
        targets = torch.zeros(1, 2, 2)
        valid = torch.tensor([[1.0, 0.0]])  # second step invalid
        # Only first step: L2 = 3.0
        assert compute_ade(preds, targets, valid).item() == pytest.approx(3.0)

    def test_ade_all_invalid(self):
        """All-invalid mask → ADE = 0 (clamped denominator)."""
        preds = torch.randn(3, 5, 2)
        targets = torch.randn(3, 5, 2)
        valid = torch.zeros(3, 5)
        assert compute_ade(preds, targets, valid).item() == pytest.approx(0.0)

    def test_fde_known_value(self):
        """FDE = error at last valid step per object."""
        preds = torch.tensor([
            [[1.0, 0.0], [0.0, 3.0], [4.0, 0.0]],  # object 0: last valid=2, L2=4
            [[0.0, 0.0], [5.0, 0.0], [0.0, 0.0]],   # object 1: last valid=1, L2=5
        ])  # [2, 3, 2]
        targets = torch.zeros(2, 3, 2)
        valid = torch.tensor([
            [1.0, 1.0, 1.0],   # all valid → last=2
            [1.0, 1.0, 0.0],   # last valid=1
        ])
        # FDE = mean(4.0, 5.0) = 4.5
        assert compute_fde(preds, targets, valid).item() == pytest.approx(4.5)

    def test_fde_no_valid(self):
        """No valid steps → FDE = 0."""
        preds = torch.randn(2, 3, 2)
        targets = torch.randn(2, 3, 2)
        valid = torch.zeros(2, 3)
        assert compute_fde(preds, targets, valid).item() == pytest.approx(0.0)

    def test_ade_fde_combined(self):
        """compute_ade_fde should match individual functions."""
        torch.manual_seed(99)
        preds = torch.randn(5, 4, 2)
        targets = torch.randn(5, 4, 2)
        valid = (torch.rand(5, 4) > 0.3).float()

        ade_sep = compute_ade(preds, targets, valid)
        fde_sep = compute_fde(preds, targets, valid)
        ade_comb, fde_comb = compute_ade_fde(preds, targets, valid)

        assert ade_comb.item() == pytest.approx(ade_sep.item(), abs=1e-5)
        assert fde_comb.item() == pytest.approx(fde_sep.item(), abs=1e-5)

    def test_ttc_approaching_object(self):
        """Object moving toward ego should have finite TTC."""
        # Object at (300, 240), ego at (320, 240)
        # Moving 5px/step toward ego over 3 steps
        current = torch.tensor([[300.0, 240.0]])
        preds = torch.tensor([[[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]]])  # cumulative-like
        valid = torch.ones(1, 3)
        ego = torch.tensor([320.0, 240.0])

        ttc = compute_ttc_proxy(preds, current, valid, dt=0.05, ego_position=ego)
        assert torch.isfinite(ttc[0])
        assert ttc[0].item() > 0

    def test_ttc_receding_object(self):
        """Object moving away from ego should have TTC = inf."""
        current = torch.tensor([[300.0, 240.0]])
        # Moving left (away from ego at 320, 240)
        preds = torch.tensor([[[-5.0, 0.0], [-10.0, 0.0], [-15.0, 0.0]]])
        valid = torch.ones(1, 3)
        ego = torch.tensor([320.0, 240.0])

        ttc = compute_ttc_proxy(preds, current, valid, dt=0.05, ego_position=ego)
        assert torch.isinf(ttc[0])


# ══════════════════════════════════════════════════════════════════════════
# 4. End-to-end training forward pass with STTP
# ══════════════════════════════════════════════════════════════════════════

class TestEndToEndForward:
    """Full DualStreamYOLOX forward with STTP enabled."""

    def _build_model(self):
        from smcfnet.model.yolox import DualStreamYOLOX
        from smcfnet.model.backbone import CSPDarknetDualStream
        from smcfnet.model.yolo_pafpn import YOLOPAFPN

        backbone = CSPDarknetDualStream(dep_mul=0.33, wid_mul=WIDTH)
        neck = YOLOPAFPN(depth=0.33, width=WIDTH, in_channels=[256, 512, 1024])
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=[256, 512, 1024])
        sttp_head = STTPHead(STEM_C, flow_channels=0)
        sttp_loss = STTPLoss()
        model = DualStreamYOLOX(backbone, neck, head, sttp_head=sttp_head, sttp_loss=sttp_loss)
        model._loss_sttp_weight.fill_(0.5)
        return model

    def test_training_forward_returns_loss_fut(self):
        """Training forward should include loss_fut in loss dict."""
        model = self._build_model()
        model.train()

        rgb = torch.randn(BATCH, 3, 480, 640)
        events = torch.randn(BATCH, 5, 480, 640)
        targets = _make_targets()
        future_targets = _make_future_targets()

        loss_dict = model(rgb, events, targets=targets, future_targets=future_targets)

        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "loss_fut" in loss_dict
        assert loss_dict["loss_fut"].item() >= 0
        assert torch.isfinite(loss_dict["total"])

    def test_training_backward_sttp(self):
        """loss_fut gradient should reach STTP head parameters."""
        model = self._build_model()
        model.train()

        rgb = torch.randn(BATCH, 3, 480, 640)
        events = torch.randn(BATCH, 5, 480, 640)
        targets = _make_targets()
        future_targets = _make_future_targets()

        loss_dict = model(rgb, events, targets=targets, future_targets=future_targets)
        loss_dict["total"].backward()

        sttp_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.sttp_head.parameters()
        )
        assert sttp_has_grad, "STTP head has no gradients after backward"

    def test_inference_returns_tuple(self):
        """Inference should return (detections, fut_preds) when STTP enabled."""
        model = self._build_model()
        model.eval()

        rgb = torch.randn(BATCH, 3, 480, 640)
        events = torch.randn(BATCH, 5, 480, 640)

        with torch.no_grad():
            output = model(rgb, events)

        assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
        detections, fut_preds = output

        # Detections: [B, n_anchors, 5+num_classes]
        assert detections.ndim == 3
        assert detections.shape[0] == BATCH
        assert detections.shape[2] == 5 + NUM_CLASSES

        # fut_preds: list of 3 tensors [B, 4, H, W]
        assert isinstance(fut_preds, list)
        assert len(fut_preds) == 3
        for pred, (h, w) in zip(fut_preds, HW):
            assert pred.shape == (BATCH, 4, h, w)

    def test_no_sttp_returns_tensor(self):
        """Without STTP, inference should return just a tensor (backward compat)."""
        from smcfnet.model.yolox import DualStreamYOLOX
        from smcfnet.model.backbone import CSPDarknetDualStream
        from smcfnet.model.yolo_pafpn import YOLOPAFPN

        backbone = CSPDarknetDualStream(dep_mul=0.33, wid_mul=WIDTH)
        neck = YOLOPAFPN(depth=0.33, width=WIDTH, in_channels=[256, 512, 1024])
        head = YOLOXHead(NUM_CLASSES, width=WIDTH, in_channels=[256, 512, 1024])
        model = DualStreamYOLOX(backbone, neck, head)
        model.eval()

        rgb = torch.randn(BATCH, 3, 480, 640)
        events = torch.randn(BATCH, 5, 480, 640)

        with torch.no_grad():
            output = model(rgb, events)

        assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"


# ══════════════════════════════════════════════════════════════════════════
# 5. Validation decode_sttp_predictions
# ══════════════════════════════════════════════════════════════════════════

class TestValidationDecode:
    """Test decode_sttp_predictions from validation.py."""

    def test_decode_matches_loss_decode(self):
        """validation.py decode must produce same result as sttp_loss decode."""
        from smcfnet.utils.validation import decode_sttp_predictions

        torch.manual_seed(123)
        fut_preds = [
            torch.randn(BATCH, 4, 60, 80),
            torch.randn(BATCH, 4, 30, 40),
            torch.randn(BATCH, 4, 15, 20),
        ]

        # Validation decode -> [B, A, k=1, 4]; collapse the step axis to compare.
        val_decoded = decode_sttp_predictions(fut_preds, HW, STRIDES).squeeze(2)

        # Loss decode (reproduce sttp_loss._decode_absolute, single step)
        grids, strides = _build_grids_and_strides()
        flat_preds = []
        for pred in fut_preds:
            B, C, H, W = pred.shape
            flat_preds.append(pred.permute(0, 2, 3, 1).reshape(B, -1, C))
        flat_preds = torch.cat(flat_preds, dim=1)

        loss_decoded = flat_preds.clone()
        loss_decoded[..., :2] = (flat_preds[..., :2] + grids) * strides.unsqueeze(-1)
        loss_decoded[..., 2:4] = torch.exp(flat_preds[..., 2:4].clamp(max=10.0)) * strides.unsqueeze(-1)

        assert torch.allclose(val_decoded, loss_decoded, atol=1e-5), \
            "Validation decode diverged from loss decode"

    def test_decode_output_shape(self):
        from smcfnet.utils.validation import decode_sttp_predictions

        fut_preds = [
            torch.randn(BATCH, 4, 60, 80),
            torch.randn(BATCH, 4, 30, 40),
            torch.randn(BATCH, 4, 15, 20),
        ]

        decoded = decode_sttp_predictions(fut_preds, HW, STRIDES)
        assert decoded.shape == (BATCH, N_TOTAL, 1, 4)  # [B, A, k=1, 4]

    def test_decode_delta_mode_to_absolute_pixels(self):
        from smcfnet.utils.validation import decode_sttp_predictions

        fut_preds = [
            torch.zeros(BATCH, 4, 60, 80),
            torch.zeros(BATCH, 4, 30, 40),
            torch.zeros(BATCH, 4, 15, 20),
        ]
        fut_preds[0][:, 0, 0, 0] = 0.1   # +64 px in x at 640 width
        fut_preds[0][:, 1, 0, 0] = -0.1  # -48 px in y at 480 height
        fut_preds[0][:, 2, 0, 0] = 0.05  # +32 px in w
        fut_preds[0][:, 3, 0, 0] = -0.05 # -24 px in h

        current_boxes = torch.zeros(BATCH, N_TOTAL, 4)
        current_boxes[:, 0, :] = torch.tensor([100.0, 120.0, 30.0, 40.0])

        decoded = decode_sttp_predictions(
            fut_preds,
            HW,
            STRIDES,
            target_mode="delta",
            current_boxes=current_boxes,
            image_width=640,
            image_height=480,
        )

        # decoded is [B, A, k=1, 4]
        assert decoded[0, 0, 0, 0].item() == pytest.approx(164.0)
        assert decoded[0, 0, 0, 1].item() == pytest.approx(72.0)
        assert decoded[0, 0, 0, 2].item() == pytest.approx(62.0)
        assert decoded[0, 0, 0, 3].item() == pytest.approx(16.0)
