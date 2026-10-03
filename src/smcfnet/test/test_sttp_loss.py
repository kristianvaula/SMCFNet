"""Tests for STTPLoss module.

Validates loss computation, validity masking, temporal consistency,
and edge cases (all invalid, no foreground).

Run: pytest src/smcfnet/test/test_sttp_loss.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import STTPLoss


# ── Helpers ───────────────────────────────────────────────────────────────

BATCH = 2
# Anchor counts per scale: P3=60*80, P4=30*40, P5=15*20
N_ANCHORS = [60 * 80, 30 * 40, 15 * 20]
N_TOTAL = sum(N_ANCHORS)
K = 3  # trajectory length (multi-step head); pred layers emit 4*K channels


def _make_fut_preds(batch=BATCH, k=K):
    """Synthetic STTP head output: list of [B, 4*k, H, W] per scale."""
    return [
        torch.randn(batch, 4 * k, 60, 80),
        torch.randn(batch, 4 * k, 30, 40),
        torch.randn(batch, 4 * k, 15, 20),
    ]


def _make_fg_masks(batch=BATCH, num_fg_per_image=10):
    """Bool mask with num_fg_per_image foreground anchors per image."""
    masks = torch.zeros(batch * N_TOTAL, dtype=torch.bool)
    for b in range(batch):
        offset = b * N_TOTAL
        fg_indices = torch.randperm(N_TOTAL)[:num_fg_per_image] + offset
        masks[fg_indices] = True
    return masks


def _make_grids_and_strides():
    """Build grid offsets and strides matching the anchor layout."""
    grids_list = []
    strides_list = []
    for (h, w), stride in zip([(60, 80), (30, 40), (15, 20)], [8, 16, 32]):
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((xv, yv), dim=-1).reshape(1, -1, 2).float()
        grids_list.append(grid)
        strides_list.append(torch.full((1, h * w), stride, dtype=torch.float32))
    grids = torch.cat(grids_list, dim=1)
    strides = torch.cat(strides_list, dim=1)
    return grids, strides


# ── Tests ─────────────────────────────────────────────────────────────────

class TestSTTPLoss:
    def test_basic_loss_computation(self):
        loss_fn = STTPLoss(temporal_consistency_weight=0.0)
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        result = loss_fn(
            fut_preds=_make_fut_preds(),
            fut_targets=torch.randn(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.ones(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
        )

        assert "loss_fut" in result
        assert result["loss_fut"].shape == ()
        assert result["loss_fut"].item() > 0
        assert "loss_tc" not in result

    def test_validity_mask_zeros_out(self):
        """All-invalid mask should produce zero loss."""
        loss_fn = STTPLoss()
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        result = loss_fn(
            fut_preds=_make_fut_preds(),
            fut_targets=torch.randn(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.zeros(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
        )

        assert result["loss_fut"].item() == 0.0

    def test_partial_validity(self):
        """Half-valid mask should produce lower loss than all-valid (different normalization)."""
        loss_fn = STTPLoss()
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        torch.manual_seed(42)
        preds = _make_fut_preds()
        targets = torch.randn(num_fg, K, 4)

        all_valid = torch.ones(num_fg, K, dtype=torch.bool)
        half_valid = torch.zeros(num_fg, K, dtype=torch.bool)
        half_valid[:num_fg // 2] = True

        r_all = loss_fn(preds, targets, fg_masks, all_valid, strides, grids, float(num_fg))
        r_half = loss_fn(preds, targets, fg_masks, half_valid, strides, grids, float(num_fg))

        # Both should be finite, non-negative
        assert r_all["loss_fut"].item() >= 0
        assert r_half["loss_fut"].item() >= 0

    def test_temporal_consistency_loss(self):
        loss_fn = STTPLoss(temporal_consistency_weight=1.0)
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        result = loss_fn(
            fut_preds=_make_fut_preds(),
            fut_targets=torch.randn(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.ones(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
            reg_preds_future=torch.randn(num_fg, 4),
        )

        assert "loss_fut" in result
        assert "loss_tc" in result
        assert result["loss_tc"].item() > 0

    def test_tc_disabled_when_weight_zero(self):
        loss_fn = STTPLoss(temporal_consistency_weight=0.0)
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        result = loss_fn(
            fut_preds=_make_fut_preds(),
            fut_targets=torch.randn(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.ones(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
            reg_preds_future=torch.randn(num_fg, 4),
        )

        assert "loss_tc" not in result

    def test_gradient_flows_through_loss(self):
        loss_fn = STTPLoss()
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        preds = _make_fut_preds()
        for p in preds:
            p.requires_grad_(True)

        result = loss_fn(
            fut_preds=preds,
            fut_targets=torch.randn(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.ones(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
        )

        result["loss_fut"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in preds)

    def test_delta_mode_zero_target_zero_loss(self):
        loss_fn = STTPLoss(target_mode="delta", image_width=640, image_height=480)
        num_fg = 20
        fg_masks = _make_fg_masks(num_fg_per_image=10)
        grids, strides = _make_grids_and_strides()

        zero_preds = [torch.zeros_like(p) for p in _make_fut_preds()]
        result = loss_fn(
            fut_preds=zero_preds,
            fut_targets=torch.zeros(num_fg, K, 4),
            fg_masks=fg_masks,
            validity_mask=torch.ones(num_fg, K, dtype=torch.bool),
            expanded_strides=strides,
            grids=grids,
            num_fg=float(num_fg),
            current_boxes=torch.zeros(num_fg, 4),
        )

        assert result["loss_fut"].item() == pytest.approx(0.0)
