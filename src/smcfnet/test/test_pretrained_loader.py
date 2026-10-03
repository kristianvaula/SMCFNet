"""Tests for pretrained weight loading, including DualStream checkpoint format."""

import pytest
import torch
import torch.nn as nn

from smcfnet.model.pretrained import (
    PretrainedLoadStats,
    _is_dualstream_checkpoint,
    _load_dualstream_direct,
    load_yolox_pretrained_dualstream,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal models and fake checkpoints
# ---------------------------------------------------------------------------

def _make_dualstream_model():
    """Minimal nn.Module mimicking DualStreamYOLOX key structure."""
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.stem_rgb = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32))
    model.backbone.stem_event = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32))
    model.backbone.dark2_rgb = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1))
    model.backbone.dark2_event = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1))
    model.backbone.dark3_rgb = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1))
    model.backbone.dark3_event = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1))
    model.neck = nn.Module()
    model.neck.lateral_conv0 = nn.Sequential(nn.Conv2d(128, 64, 1))
    model.head = nn.Module()
    model.head.cls_preds = nn.Sequential(nn.Conv2d(64, 8, 1))
    model.sttp_head = nn.Module()
    model.sttp_head.pred = nn.Linear(64, 4)
    return model


def _make_dualstream_checkpoint(model, *, include_mcm=False, include_sttp=False):
    """Create a checkpoint dict with DualStream-format keys."""
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("sttp_head.") and not include_sttp:
            continue
        sd[k] = v.clone()
    if include_mcm:
        sd["backbone.mcm.estimator.conv.weight"] = torch.randn(32, 5, 3, 3)
        sd["backbone.mcm.estimator.conv.bias"] = torch.randn(32)
    return {"model": sd}


def _make_yolox_checkpoint():
    """Create a checkpoint dict with YOLOX-format keys."""
    sd = {
        "backbone.backbone.stem.0.conv.weight": torch.randn(32, 12, 3, 3),
        "backbone.backbone.stem.0.bn.weight": torch.randn(32),
        "backbone.backbone.stem.0.bn.bias": torch.randn(32),
        "backbone.backbone.stem.0.bn.running_mean": torch.zeros(32),
        "backbone.backbone.stem.0.bn.running_var": torch.ones(32),
        "backbone.backbone.stem.0.bn.num_batches_tracked": torch.tensor(0),
        "backbone.backbone.dark2.0.weight": torch.randn(64, 32, 3, 3),
        "backbone.backbone.dark3.0.weight": torch.randn(128, 64, 3, 3),
        "backbone.lateral_conv0.0.weight": torch.randn(64, 128, 1, 1),
        "head.cls_preds.0.weight": torch.randn(8, 64, 1, 1),
    }
    return sd


# ---------------------------------------------------------------------------
# Tests: _is_dualstream_checkpoint
# ---------------------------------------------------------------------------

class TestIsDualstreamCheckpoint:
    def test_dualstream_keys_detected(self):
        sd = {"backbone.stem_rgb.0.conv.weight": torch.randn(1)}
        assert _is_dualstream_checkpoint(sd) is True

    def test_event_marker_detected(self):
        sd = {"backbone.stem_event.0.conv.weight": torch.randn(1)}
        assert _is_dualstream_checkpoint(sd) is True

    def test_dark2_rgb_marker_detected(self):
        sd = {"backbone.dark2_rgb.0.weight": torch.randn(1)}
        assert _is_dualstream_checkpoint(sd) is True

    def test_yolox_keys_not_detected(self):
        sd = {
            "backbone.backbone.stem.0.conv.weight": torch.randn(1),
            "backbone.lateral_conv0.weight": torch.randn(1),
            "head.cls_preds.0.weight": torch.randn(1),
        }
        assert _is_dualstream_checkpoint(sd) is False

    def test_empty_dict(self):
        assert _is_dualstream_checkpoint({}) is False


# ---------------------------------------------------------------------------
# Tests: DualStream direct loading
# ---------------------------------------------------------------------------

class TestLoadDualstreamDirect:
    def test_loads_all_matching_keys(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        stats = load_yolox_pretrained_dualstream(model, ckpt)

        # Should load everything except sttp_head (not in checkpoint)
        model_keys_no_sttp = [k for k in model.state_dict() if not k.startswith("sttp_head.")]
        assert stats.loaded_keys == len(model_keys_no_sttp)
        assert stats.skipped_shape_mismatch == 0
        assert stats.missing_keys > 0  # sttp_head keys are missing
        assert stats.unexpected_keys == 0

    def test_mcm_keys_skipped_when_not_in_model(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model, include_mcm=True)
        stats = load_yolox_pretrained_dualstream(model, ckpt)

        # MCM keys should be skipped (not in model), not cause errors
        assert stats.skipped_unmapped >= 2  # at least the 2 MCM keys
        assert stats.unexpected_keys == 0

    def test_filter_load_rgb_false(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        stats = load_yolox_pretrained_dualstream(
            model, ckpt, load_rgb=False,
        )

        # RGB keys should be skipped
        loaded_keys_set = set()
        sd = ckpt["model"]
        model_dict = model.state_dict()
        for k in sd:
            if "_rgb." not in k and k in model_dict:
                loaded_keys_set.add(k)
        assert stats.loaded_keys == len(loaded_keys_set)

    def test_filter_load_event_false(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        stats = load_yolox_pretrained_dualstream(
            model, ckpt, load_event=False,
        )

        # No event keys should be loaded
        loaded_sd = {k: v for k, v in ckpt["model"].items()
                     if "_event." not in k and k in model.state_dict()}
        assert stats.loaded_keys == len(loaded_sd)

    def test_filter_load_neck_false(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        stats = load_yolox_pretrained_dualstream(
            model, ckpt, load_neck=False,
        )

        loaded_sd = {k: v for k, v in ckpt["model"].items()
                     if not k.startswith("neck.") and k in model.state_dict()}
        assert stats.loaded_keys == len(loaded_sd)

    def test_filter_load_head_false(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        stats = load_yolox_pretrained_dualstream(
            model, ckpt, load_head=False,
        )

        loaded_sd = {k: v for k, v in ckpt["model"].items()
                     if not k.startswith("head.") and k in model.state_dict()}
        assert stats.loaded_keys == len(loaded_sd)

    def test_weights_actually_loaded(self):
        """Verify loaded weights change the model parameters."""
        model = _make_dualstream_model()
        before = model.backbone.stem_rgb[0].weight.clone()

        # Create checkpoint with different random weights
        ckpt_model = _make_dualstream_model()
        nn.init.constant_(ckpt_model.backbone.stem_rgb[0].weight, 42.0)
        ckpt = _make_dualstream_checkpoint(ckpt_model)

        load_yolox_pretrained_dualstream(model, ckpt)
        assert torch.allclose(model.backbone.stem_rgb[0].weight,
                              torch.full_like(before, 42.0))

    def test_shape_mismatch_skipped(self):
        model = _make_dualstream_model()
        ckpt = _make_dualstream_checkpoint(model)
        # Corrupt a non-conv key with wrong shape
        ckpt["model"]["neck.lateral_conv0.0.weight"] = torch.randn(999)
        stats = load_yolox_pretrained_dualstream(model, ckpt)

        assert stats.skipped_shape_mismatch >= 1

    def test_module_prefix_stripped(self):
        """DDP-wrapped checkpoints with module. prefix should still work."""
        model = _make_dualstream_model()
        raw_sd = {k: v.clone() for k, v in model.state_dict().items()
                  if not k.startswith("sttp_head.")}
        ddp_sd = {f"module.{k}": v for k, v in raw_sd.items()}
        ckpt = {"model": ddp_sd}

        stats = load_yolox_pretrained_dualstream(model, ckpt)
        assert stats.loaded_keys == len(raw_sd)


# ---------------------------------------------------------------------------
# Tests: YOLOX path still works (regression guard)
# ---------------------------------------------------------------------------

class TestYoloxPathRegression:
    def test_yolox_checkpoint_uses_remap_path(self):
        model = _make_dualstream_model()
        ckpt = _make_yolox_checkpoint()
        stats = load_yolox_pretrained_dualstream(model, ckpt)

        # Should load via remapping, not direct path
        assert stats.loaded_keys > 0
        assert stats.unexpected_keys == 0

    def test_yolox_stem_replicated_to_both_streams(self):
        model = _make_dualstream_model()
        # Build checkpoint keys that match model's Sequential key naming
        # Model stem_rgb is Sequential(Conv2d, BN) → keys are stem_rgb.0.weight, stem_rgb.1.weight, etc.
        ckpt = {
            "backbone.backbone.stem.0.weight": torch.randn(32, 12, 3, 3),
            "backbone.backbone.stem.1.weight": torch.randn(32),
            "backbone.backbone.stem.1.bias": torch.randn(32),
            "backbone.backbone.stem.1.running_mean": torch.zeros(32),
            "backbone.backbone.stem.1.running_var": torch.ones(32),
            "backbone.backbone.stem.1.num_batches_tracked": torch.tensor(0),
        }

        # Zero out model weights first
        for p in model.parameters():
            nn.init.zeros_(p)

        load_yolox_pretrained_dualstream(model, ckpt)

        # Both stems should have non-zero weights (replicated from YOLOX stem)
        # RGB stem: 3 in_channels adapted from 12 via channel truncation
        assert model.backbone.stem_rgb[0].weight.abs().sum() > 0
        # Event stem: 5 in_channels adapted from 12 via channel truncation
        assert model.backbone.stem_event[0].weight.abs().sum() > 0
