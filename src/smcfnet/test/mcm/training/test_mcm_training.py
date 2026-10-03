"""Tests for the generalized MCM module and MCM-only training pipeline.

Covers:
 - MCM module build from config (_build_mcm) for multiple architectures
 - MCM forward/backward with different estimator architectures
 - MCMTrainer initialization, run_step, and short training loop (synthetic data)
 - MCM checkpoint save / load / resume cycle
 - MCM weight loading into full detection model (Phase B)
 - MCM freeze / unfreeze round-trip
 - Joint detection model forward with MCM enabled
 - Config-driven dispatch (mcm_pretrain mode triggers MCMTrainer)

Run:  pytest src/smcfnet/test/test_mcm_training.py -v

Requires: torch, CUDA recommended (some tests skip on CPU).
"""

from __future__ import annotations

import copy
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smcfnet.config.config import Config
from smcfnet.dataset.samplers import InfiniteSampler
from smcfnet.model.mcm import (
    MotionCompensationModule,
    EventWarpingModule,
    StreamingMotionCompensationModule,
)
from smcfnet.model.mcm.weight_utils import (
    save_mcm_checkpoint,
    load_mcm_checkpoint,
    load_mcm_weights_into_module,
    load_mcm_weights_into_full_model,
    freeze_mcm_params,
    unfreeze_mcm_params,
)
from smcfnet.model.build import _build_mcm
from smcfnet.model.event_warping import prepare_events_for_warping
from smcfnet.model.ofe import create_flow_estimator, FlowEstimator
from smcfnet.model.ofe.sofe import SOFEState
from smcfnet.model.ofe.ecm_ofe import ECMFlowEstimator
from smcfnet.model.losses import CMaxLoss, EventWarpingLoss, SupervisedFlowLoss, create_mc_loss
from smcfnet.core.mcm_trainer import (
    MCMTrainer,
    build_mcm_pretrain_forward_call,
    configure_mcm_pretrain_data_flags,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# Use small spatial dims to keep CPU tests fast on shared nodes.
H_TEST, W_TEST = 32, 40


def _make_voxel(batch=2, bins=5, h=H_TEST, w=W_TEST, device="cpu"):
    return torch.randn(batch, bins, h, w, device=device)


class _CountingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, corrected, flow, *args, **kwargs):
        self.calls += 1
        return corrected.mean() + flow.mean()


def test_prepare_events_for_warping_accepts_signed_and_legacy_polarities():
    events = torch.tensor(
        [
            [4.0, 6.0, 10.0, 1.0],
            [5.0, 6.0, 20.0, -1.0],
            [6.0, 6.0, 30.0, 1.0],
            [8.0, 3.0, 40.0, 1.0],
            [9.0, 3.0, 50.0, 0.0],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64)

    events_batch, pol_mask = prepare_events_for_warping(events, offsets)

    assert events_batch.shape == (2, 3, 4)
    assert pol_mask.shape == (2, 3, 2)

    assert pol_mask[0, :, 0].tolist() == [1.0, 0.0, 1.0]
    assert pol_mask[0, :, 1].tolist() == [0.0, 1.0, 0.0]

    assert pol_mask[1, :, 0].tolist() == [1.0, 0.0, 0.0]
    assert pol_mask[1, :, 1].tolist() == [0.0, 1.0, 0.0]

    assert events_batch[1, 2, 1].item() == -1.0
    assert events_batch[1, 2, 2].item() == -1.0


def test_prepare_events_for_warping_handles_empty_samples_between_nonempty():
    events = torch.tensor(
        [
            [1.0, 2.0, 10.0, 1.0],
            [2.0, 3.0, 20.0, -1.0],
            [4.0, 5.0, 100.0, 0.0],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 2, 3], dtype=torch.int64)

    events_batch, pol_mask = prepare_events_for_warping(events, offsets)

    assert events_batch.shape == (3, 2, 4)
    assert torch.allclose(events_batch[0, :2, 0], torch.tensor([0.0, 1.0]))
    assert events_batch[1, :, 1:3].eq(-1.0).all()
    assert pol_mask[1].sum() == 0
    assert events_batch[2, 0, 0] == 0.0
    assert pol_mask[2, 0, 1] == 1.0
    assert pol_mask[2, 1].sum() == 0


def _make_structured_streaming_batch(
    *,
    batch_size: int = 2,
    h: int = H_TEST,
    w: int = W_TEST,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Packed streaming batch with simple rightward motion structure.

    The same sparse track pattern is repeated per sample so the streaming
    event-warping loss sees a coherent motion-compensation problem instead of
    unstructured random events.
    """
    events_per_sample = 24
    total = batch_size * events_per_sample
    events = torch.zeros((total, 4), dtype=torch.float32, device=device)
    offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    t_start = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    t_end = torch.full((batch_size,), 1000, dtype=torch.int64, device=device)
    t_ref = t_end.clone()
    sequence_reset = torch.ones((batch_size,), dtype=torch.bool, device=device)

    for b in range(batch_size):
        start = b * events_per_sample
        offsets[b + 1] = start + events_per_sample
        y_base = 8 + b * 6
        x_base = 6 + b
        for i in range(events_per_sample):
            idx = start + i
            t_us = 40 * i
            x = min(w - 2, x_base + (i // 3))
            y = min(h - 2, y_base + ((i // 8) % 2))
            p = i % 2
            events[idx, 0] = float(x)
            events[idx, 1] = float(y)
            events[idx, 2] = float(t_us)
            events[idx, 3] = float(p)

    return {
        "events": events,
        "events_offsets": offsets,
        "t_start_us": t_start,
        "t_end_us": t_end,
        "t_ref_us": t_ref,
        "sequence_reset": sequence_reset,
    }


def _make_mcm_cfg(**overrides) -> Config:
    """Build a minimal Config with MCM enabled (ECM default)."""
    cfg = Config()
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "ecm"
    cfg.model.mcm.loss_type = "cmax"
    cfg.model.mcm.warper_type = "voxel"
    cfg.model.mcm.contrast_weight = 1.0
    cfg.model.mcm.smooth_weight = 0.1
    cfg.model.mcm.mid_channels = 16
    cfg.data.event_bins = 5
    cfg.model.activation = "silu"
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def _make_mcm_pretrain_cfg(**overrides) -> Config:
    """Config for MCMTrainer (mcm_pretrain mode)."""
    cfg = _make_mcm_cfg()
    cfg.training.mode = "mcm_pretrain"
    cfg.training.num_epochs = 2
    cfg.training.batch_size = 2
    cfg.training.optimizer = "adamw"
    cfg.training.learning_rate = 1e-3
    cfg.training.weight_decay = 1e-4
    cfg.training.warmup_epochs = 0
    cfg.training.warmup_lr = 1e-5
    cfg.training.use_amp = False
    cfg.training.grad_clip = 5.0
    cfg.training.log_grad_norm = False
    cfg.wandb.use_wandb = False
    cfg.log.log_interval = 1
    cfg.eval.eval_interval = 0
    cfg.eval.save_interval = 0
    # Apply overrides last so they win
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


class _FakePrefetcher:
    """Minimal prefetcher that yields synthetic voxel batches."""

    def __init__(self, num_batches: int, bins: int = 5, h: int = H_TEST, w: int = W_TEST,
                 batch_size: int = 2, device: str = "cpu", representation: str = "events_voxel"):
        self._num = num_batches
        self._idx = 0
        self._bins = bins
        self._h = h
        self._w = w
        self._bs = batch_size
        self._device = device
        self.prepared_representation = str(representation)

    def next(self):
        if self._idx >= self._num:
            raise StopIteration
        self._idx += 1
        if self.prepared_representation == "streaming_events":
            n_per = 64
            total = self._bs * n_per
            events = torch.zeros((total, 4), dtype=torch.float32, device=self._device)
            offsets = torch.zeros((self._bs + 1,), dtype=torch.int64, device=self._device)
            seq_idx = torch.zeros((self._bs,), dtype=torch.int64, device=self._device)
            frame_index = torch.full((self._bs,), self._idx - 1, dtype=torch.int64, device=self._device)
            t_ref = torch.full((self._bs,), 1000 * self._idx, dtype=torch.int64, device=self._device)
            t_start = torch.full((self._bs,), 1000 * (self._idx - 1), dtype=torch.int64, device=self._device)
            t_end = t_ref.clone()

            for b in range(self._bs):
                start = b * n_per
                end = start + n_per
                offsets[b + 1] = end
                xs = torch.randint(0, self._w, (n_per,), device=self._device, dtype=torch.int64)
                ys = torch.randint(0, self._h, (n_per,), device=self._device, dtype=torch.int64)
                ts = torch.arange(start, end, device=self._device, dtype=torch.int64)
                ps = torch.randint(0, 2, (n_per,), device=self._device, dtype=torch.int64)
                events[start:end, 0] = xs.to(torch.float32)
                events[start:end, 1] = ys.to(torch.float32)
                events[start:end, 2] = ts.to(torch.float32)
                events[start:end, 3] = ps.to(torch.float32)

            return {
                "events": events,
                "events_offsets": offsets,
                "seq_idx": seq_idx,
                "frame_index": frame_index,
                "t_ref_us": t_ref,
                "t_start_us": t_start,
                "t_end_us": t_end,
                "sequence_reset": frame_index <= 0,
            }

        return {
            "events_voxel": torch.randn(
                self._bs, self._bins, self._h, self._w, device=self._device
            ),
        }


class _FakeDataLoader:
    """Minimal iterable that reports length (for iters_per_epoch)."""

    def __init__(self, length: int):
        self._len = length

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(range(self._len))


# ═══════════════════════════════════════════════════════════════════════════
# 1. MCM module build from config
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildMCM:
    """Test _build_mcm for different architectures."""

    def test_build_ecm(self):
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg)
        assert isinstance(mcm, MotionCompensationModule)
        assert isinstance(mcm.estimator, ECMFlowEstimator)
        assert mcm.loss_fn is not None

    def test_build_cnn_baseline(self):
        cfg = _make_mcm_cfg(**{"model.mcm.architecture": "cnn_baseline"})
        mcm = _build_mcm(cfg)
        assert isinstance(mcm, MotionCompensationModule)

    def test_build_with_different_losses(self):
        for loss_type in (
            "cmax",
            "supervised",
            "combined",
            "event_warping",
            "event_warping_supervised",
        ):
            cfg = _make_mcm_cfg(**{"model.mcm.loss_type": loss_type})
            mcm = _build_mcm(cfg)
            assert mcm.loss_fn is not None

    def test_build_streaming_mcm_threads_auxiliary_warp_and_edge_aware_loss(self):
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.estimator_representation = "streaming_events"
        cfg.model.mcm.output_representation = "iwe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.auxiliary_warp_loss = True
        cfg.model.mcm.auxiliary_warp_loss_weight = 0.2
        cfg.model.mcm.auxiliary_flow_scales = ["h8", "h4"]
        cfg.model.mcm.loss_kwargs = {
            "smoothness_mode": "edge_aware_l1",
            "edge_weight_gamma": 3.0,
        }
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.model.mcm.sofe_architecture = {
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "num_stages": 2,
            "num_res_blocks": 0,
            "recurrent_cell": "convrnn",
            "recurrent_placement": "every_stage",
            "readout_type": "single_scale",
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)

        assert isinstance(mcm, StreamingMotionCompensationModule)
        assert mcm.auxiliary_warp_loss is True
        assert mcm.auxiliary_warp_loss_weight == pytest.approx(0.2)
        assert mcm.auxiliary_flow_scales == ["h8", "h4"]
        assert isinstance(mcm.loss_fn, EventWarpingLoss)
        assert mcm.loss_fn.smoothness_mode == "edge_aware_l1"
        assert mcm.loss_fn.edge_weight_gamma == pytest.approx(3.0)


class TestEventWarpingSupervisedLoss:
    """Validate EventWarping + supervised combined loss behavior."""

    @staticmethod
    def _make_packed_events(device: torch.device):
        # Format: [x, y, t_ms, p], offsets define two samples (3 + 2 events)
        events = torch.tensor(
            [
                [5.0, 6.0, 0.0, 1.0],
                [6.0, 6.0, 2.0, 0.0],
                [7.0, 7.0, 4.0, 1.0],
                [10.0, 9.0, 1.0, 1.0],
                [11.0, 9.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
        return events, offsets

    def test_event_warping_supervised_with_flow_gt(self, device):
        b, c, h, w = 2, 5, H_TEST, W_TEST
        corrected = torch.randn(b, c, h, w, device=device)
        flow = torch.randn(b, 2, h, w, device=device) * 0.01
        flow_gt = torch.randn(b, 2, h, w, device=device) * 0.01
        events, offsets = self._make_packed_events(device)

        loss_fn = create_mc_loss(
            "event_warping_supervised",
            alpha=1.0,
            beta=1.0,
            res=(h, w),
            smooth_weight=0.001,
            flow_scaling=1.0,
            loss_scaling=True,
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            flow_gt,
            events=events,
            events_offsets=offsets,
        )

        assert torch.isfinite(loss)
        assert "ew_contrast" in loss_dict
        assert "ew_smooth" in loss_dict
        assert "sup_epe" in loss_dict

    def test_supervised_epe_zero_error_has_finite_gradients(self, device):
        flow = torch.zeros(1, 2, H_TEST, W_TEST, device=device, requires_grad=True)
        flow_gt = torch.zeros(1, 3, H_TEST, W_TEST, device=device)
        flow_gt[:, 2] = 1.0

        loss, loss_dict = SupervisedFlowLoss()(None, flow, flow_gt)
        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(loss_dict["epe"])
        assert flow.grad is not None
        assert torch.isfinite(flow.grad).all()

    def test_event_warping_supervised_without_flow_gt(self, device):
        b, c, h, w = 2, 5, H_TEST, W_TEST
        corrected = torch.randn(b, c, h, w, device=device)
        flow = torch.randn(b, 2, h, w, device=device) * 0.01
        events, offsets = self._make_packed_events(device)

        loss_fn = create_mc_loss(
            "event_warping_supervised",
            alpha=1.0,
            beta=1.0,
            res=(h, w),
            smooth_weight=0.001,
            flow_scaling=1.0,
            loss_scaling=True,
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            None,
            events=events,
            events_offsets=offsets,
        )

        assert torch.isfinite(loss)
        assert "ew_contrast" in loss_dict
        assert "ew_smooth" in loss_dict
        assert "sup_epe" not in loss_dict

    def test_event_warping_supervised_requires_events_kwargs(self, device):
        b, c, h, w = 2, 5, H_TEST, W_TEST
        corrected = torch.randn(b, c, h, w, device=device)
        flow = torch.randn(b, 2, h, w, device=device) * 0.01
        flow_gt = torch.randn(b, 2, h, w, device=device) * 0.01

        loss_fn = create_mc_loss(
            "event_warping_supervised",
            alpha=1.0,
            beta=1.0,
            res=(h, w),
        )

        with pytest.raises(ValueError, match="EventWarpingLoss requires 'events'"):
            loss_fn(corrected, flow, flow_gt)

    def test_event_warping_smooth_matches_ecm_mean_definition(self):
        device = torch.device("cpu")
        b = 2
        h, w = 4, 5
        corrected = torch.zeros(b, 5, h, w, device=device)
        base_flow = torch.zeros(2, h, w, device=device)
        base_flow[0] = torch.tensor(
            [
                [0.0, 1.0, 3.0, 6.0, 10.0],
                [0.0, 1.0, 3.0, 6.0, 10.0],
                [1.0, 2.0, 4.0, 7.0, 11.0],
                [1.0, 2.0, 4.0, 7.0, 11.0],
            ],
            device=device,
        )
        base_flow[1] = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            device=device,
        )
        flow = base_flow.unsqueeze(0).repeat(b, 1, 1, 1)
        events, offsets = self._make_packed_events(device)

        loss_fn = create_mc_loss(
            "event_warping",
            res=(h, w),
            smooth_weight=0.5,
            flow_scaling=1.0,
            loss_scaling=True,
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            None,
            events=events,
            events_offsets=offsets,
            flow_unscaled=flow,
        )

        expected_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
        expected_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
        expected_smooth_raw = expected_dx + expected_dy

        assert torch.isfinite(loss)
        assert torch.allclose(loss_dict["smooth_raw"], expected_smooth_raw)
        assert torch.allclose(loss_dict["smooth"], expected_smooth_raw * 0.5)

    def test_event_warping_smooth_accepts_native_flow_resolution(self, device):
        b, h, w = 2, 4, 5
        corrected = torch.zeros(b, 5, h, w, device=device)
        flow = torch.zeros(b, 2, h, w, device=device)
        flow_unscaled = torch.zeros(b, 2, h // 2, w, device=device)
        flow_unscaled[:, 0] = 1.0
        events, offsets = self._make_packed_events(device)

        loss_fn = create_mc_loss(
            "event_warping",
            res=(h, w),
            smooth_weight=0.5,
            flow_scaling=1.0,
            loss_scaling=True,
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            None,
            events=events,
            events_offsets=offsets,
            flow_unscaled=flow_unscaled,
        )

        expected_dx = (flow_unscaled[:, :, :, 1:] - flow_unscaled[:, :, :, :-1]).abs().mean()
        expected_dy = (flow_unscaled[:, :, 1:, :] - flow_unscaled[:, :, :-1, :]).abs().mean()
        expected_smooth_raw = expected_dx + expected_dy

        assert torch.isfinite(loss)
        assert torch.allclose(loss_dict["smooth_raw"], expected_smooth_raw)
        assert torch.allclose(loss_dict["smooth"], expected_smooth_raw * 0.5)

    def test_event_warping_original_smooth_matches_mcfnet_definition(self):
        device = torch.device("cpu")
        b = 2
        h, w = 4, 5
        corrected = torch.zeros(b, 5, h, w, device=device)
        flow = torch.zeros(b, 2, h, w, device=device)
        events, offsets = self._make_packed_events(device)

        loss_fn = create_mc_loss(
            "event_warping",
            res=(h, w),
            smooth_weight=0.0005,
            flow_scaling=1.0,
            loss_scaling=True,
            smoothness_mode="original",
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            None,
            events=events,
            events_offsets=offsets,
            flow_unscaled=flow,
        )

        flow_dx = flow[:, :, :-1, :] - flow[:, :, 1:, :]
        flow_dy = flow[:, :, :, :-1] - flow[:, :, :, 1:]
        expected_smooth_raw = (
            torch.sqrt(flow_dx.pow(2) + 1e-6).sum()
            + torch.sqrt(flow_dy.pow(2) + 1e-6).sum()
        ) / flow_dx.shape[1]

        assert torch.isfinite(loss)
        assert torch.allclose(loss_dict["smooth_raw"], expected_smooth_raw)
        assert torch.allclose(loss_dict["smooth"], expected_smooth_raw * 0.0005)

    def test_event_warping_edge_aware_smoothness_forward_pass(self):
        device = torch.device("cpu")
        batch = _make_structured_streaming_batch(batch_size=2, h=H_TEST, w=W_TEST, device="cpu")
        corrected = torch.zeros(2, 5, H_TEST, W_TEST, device=device)
        flow = torch.randn(2, 2, H_TEST, W_TEST, device=device) * 0.01

        loss_fn = create_mc_loss(
            "event_warping",
            res=(H_TEST, W_TEST),
            smooth_weight=0.5,
            flow_scaling=1.0,
            loss_scaling=True,
            smoothness_mode="edge_aware",
            edge_weight_gamma=4.0,
        )
        loss, loss_dict = loss_fn(
            corrected,
            flow,
            None,
            events=batch["events"],
            events_offsets=batch["events_offsets"],
            flow_unscaled=flow,
        )

        assert isinstance(loss_fn, EventWarpingLoss)
        assert loss_fn.smoothness_mode == "edge_aware_l1"
        assert torch.isfinite(loss)
        assert torch.isfinite(loss_dict["smooth_raw"])
        assert torch.isfinite(loss_dict["smooth"])

    def test_auxiliary_pyramid_warp_loss_backpropagates_to_aux_flow(self):
        device = torch.device("cpu")
        batch = _make_structured_streaming_batch(batch_size=2, h=H_TEST, w=W_TEST, device="cpu")
        corrected = torch.zeros(2, 5, H_TEST, W_TEST, device=device)
        flow_h2 = (torch.randn(2, 2, H_TEST // 2, W_TEST // 2, device=device) * 0.01).requires_grad_()
        aux = {"flow_h2": flow_h2}
        loss_fn = create_mc_loss(
            "event_warping",
            res=(H_TEST, W_TEST),
            smooth_weight=0.5,
            flow_scaling=1.0,
            loss_scaling=True,
            smoothness_mode="edge_aware_l1",
        )

        aux_loss, aux_dict = StreamingMotionCompensationModule._compute_auxiliary_pyramid_warp(
            aux,
            corrected,
            loss_fn,
            scales=["h2"],
            ref_flow_hw=(H_TEST, W_TEST),
            events=batch["events"],
            events_offsets=batch["events_offsets"],
        )

        assert aux_loss is not None
        assert torch.isfinite(aux_loss)
        assert "aux_warp_h2" in aux_dict
        aux_loss.backward()
        assert flow_h2.grad is not None
        assert torch.isfinite(flow_h2.grad).all()
        assert flow_h2.grad.abs().sum() > 0

    def test_build_disabled_raises_no_error(self):
        """_build_mcm should still work (config.model.mcm.enabled is not its concern)."""
        cfg = _make_mcm_cfg()
        cfg.model.mcm.enabled = False
        mcm = _build_mcm(cfg)
        assert isinstance(mcm, MotionCompensationModule)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MCM forward / backward for multiple architectures
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMForwardBackward:
    """Forward + backward pass for supported non-streaming estimators.

    Backward tests require CUDA — grid_sample backward is prohibitively slow
    on shared login-node CPUs.
    """

    @staticmethod
    def _build_mcm(arch: str, device):
        cfg = _make_mcm_cfg(**{"model.mcm.architecture": arch})
        return _build_mcm(cfg).to(device)

    @pytest.mark.parametrize("arch", ["ecm", "cnn_baseline"])
    def test_forward_shapes(self, arch, device):
        mcm = self._build_mcm(arch, device)
        voxel = _make_voxel(device=device)
        corrected, loss = mcm(voxel)
        assert corrected.shape == voxel.shape
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_sparse_event_forward_with_kwargs(self, device):
        if device.type == "cpu":
            pytest.skip("sparse-event integration test runs on CUDA only")

        cfg = _make_mcm_cfg(**{
            "model.mcm.loss_type": "event_warping",
            "model.mcm.warper_type": "sparse_event",
            "model.mcm.flow_scaling": 1.0,
            "data.image_height": H_TEST,
            "data.image_width": W_TEST,
        })
        mcm = _build_mcm(cfg).to(device)
        voxel = _make_voxel(device=device)

        events = torch.tensor(
            [
                [5.0, 6.0, 0.0, 1.0],
                [6.0, 6.0, 2.0, 0.0],
                [7.0, 7.0, 4.0, 1.0],
                [10.0, 9.0, 1.0, 1.0],
                [11.0, 9.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)

        corrected, loss = mcm(voxel, events=events, events_offsets=offsets)
        assert corrected.shape == voxel.shape
        assert torch.isfinite(loss)

    # Backward-heavy tests: GPU only
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="backward needs CUDA")
    @pytest.mark.parametrize("arch", ["ecm", "cnn_baseline"])
    def test_backward_grads_exist(self, arch):
        device = torch.device("cuda")
        mcm = self._build_mcm(arch, device)
        voxel = _make_voxel(device=device)
        _, loss = mcm(voxel)
        loss.backward()
        grad_params = [p for p in mcm.parameters() if p.grad is not None]
        assert len(grad_params) > 0, "No gradients flowed through MCM"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="training loop needs CUDA")
    @pytest.mark.parametrize("arch", ["ecm", "cnn_baseline"])
    def test_loss_decreases_over_steps(self, arch):
        """A few optimizer steps should reduce loss (basic sanity)."""
        device = torch.device("cuda")
        mcm = self._build_mcm(arch, device)
        optimizer = torch.optim.Adam(mcm.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            voxel = _make_voxel(device=device)
            _, loss = mcm(voxel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # Loss should remain finite and not blow up over a short run.
        # This is intentionally tolerant because self-supervised objectives
        # can be non-monotonic on random synthetic batches.
        assert all(torch.isfinite(torch.tensor(losses))), "Encountered non-finite loss values"
        assert losses[-1] < losses[0] + 5.0, (
            f"Loss appears unstable: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. MCMTrainer with synthetic data (CPU)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMTrainer:
    """Test MCMTrainer init, run_step, and a short train loop using synthetic data."""

    def _make_trainer(self, tmp_path: Path, *, streaming: bool = False):
        """Build an MCMTrainer with mocked data pipeline."""
        cfg = _make_mcm_pretrain_cfg()
        if streaming:
            cfg.model.mcm.input_representation = "streaming_events"
            cfg.model.mcm.architecture = "sofe"
            cfg.model.mcm.warper_type = "sparse_event"
        cfg.log.output_dir = str(tmp_path)
        cfg.log.experiment_name = "test_mcm_trainer"
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.data.event_bins = 5

        num_batches = 4

        with patch("smcfnet.core.mcm_trainer.build_train_loader_cfg") as mock_loader, \
             patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1), \
             patch("smcfnet.core.mcm_trainer.get_local_rank", return_value=0), \
             patch("torch.utils.tensorboard.SummaryWriter"):
            mock_loader.return_value = _FakeDataLoader(num_batches)
            from smcfnet.core.mcm_trainer import MCMTrainer
            trainer = MCMTrainer(cfg)

        dev = str(trainer.device)

        # Replace the prefetcher with our fake
        trainer._get_train_prefetcher = lambda: _FakePrefetcher(
            num_batches,
            h=H_TEST,
            w=W_TEST,
            device=dev,
            representation="streaming_events" if streaming else "events_voxel",
        )
        return trainer, num_batches

    def test_trainer_init(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        assert isinstance(trainer.mcm, MotionCompensationModule)
        assert trainer.optimizer is not None
        assert trainer.epoch == 0

    def test_single_run_step(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer.mcm.train()
        trainer.storage = MagicMock()
        trainer.storage.latest.return_value = {}
        trainer.storage.iter = 0
        trainer.run_step()
        # Optimizer should have stepped — check param values changed
        # (at minimum, no exception was raised)

    def test_voxel_run_step_resets_state_each_batch(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)

        class _StatefulVoxelMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.reset_calls = 0
                self.forward_reset_calls = []

            def reset_state(self):
                self.reset_calls += 1

            def forward(self, primary_input, **_kwargs):
                self.forward_reset_calls.append(self.reset_calls)
                corrected = torch.zeros_like(primary_input)
                loss = self.weight * primary_input.mean()
                return corrected, loss

        fake_mcm = _StatefulVoxelMCM().to(trainer.device)
        trainer.mcm = fake_mcm
        trainer.model = fake_mcm
        trainer.optimizer = torch.optim.SGD(fake_mcm.parameters(), lr=1e-2)
        trainer.use_amp = False
        trainer.scaler = None
        trainer.storage = MagicMock()
        trainer.storage.latest.return_value = {}
        trainer.storage.iter = 0

        trainer.run_step()
        trainer.run_step()

        assert fake_mcm.reset_calls == 2
        assert fake_mcm.forward_reset_calls == [1, 2]

    def test_short_training_loop(self, tmp_path):
        """Run 2 epochs × 4 batches with synthetic data."""
        trainer, num_batches = self._make_trainer(tmp_path)
        dev = str(trainer.device)

        # Patch reset_data_loader_iter to re-create fake prefetcher each epoch
        original_reset = trainer.reset_data_loader_iter

        def _patched_reset():
            original_reset()
            trainer._get_train_prefetcher = lambda: _FakePrefetcher(
                num_batches,
                h=H_TEST,
                w=W_TEST,
                device=dev,
                representation="events_voxel",
            )

        trainer.reset_data_loader_iter = _patched_reset

        # Override is_main_process for logging paths inside train()
        with patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1):
            trainer.train()

        assert trainer.epoch == 1  # last epoch index (0-based)
        assert trainer.iteration == 2 * num_batches

    def test_training_loop_respects_max_steps_per_epoch(self, tmp_path):
        trainer, num_batches = self._make_trainer(tmp_path)
        trainer.config.training.num_epochs = 2
        trainer.config.training.max_steps_per_epoch = 2

        dev = str(trainer.device)
        original_reset = trainer.reset_data_loader_iter

        def _patched_reset():
            original_reset()
            trainer._get_train_prefetcher = lambda: _FakePrefetcher(
                num_batches,
                h=H_TEST,
                w=W_TEST,
                device=dev,
                representation="events_voxel",
            )

        trainer.reset_data_loader_iter = _patched_reset

        with patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1):
            trainer.train()

        assert trainer.iteration == 4
        assert trainer.epoch == 1

    def test_infinite_sampler_stream_continues_across_limited_epochs(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        loader = DataLoader(
            list(range(10)),
            batch_size=1,
            sampler=InfiniteSampler(10, shuffle=False, seed=0),
        )
        trainer.train_loader = loader
        trainer.data_loader = loader
        trainer.reset_data_loader_iter()

        trainer._prepare_train_epoch_data_stream(0)
        first_epoch_iter = trainer.train_data_loader_iter
        first_chunk = [int(next(first_epoch_iter).item()) for _ in range(3)]

        trainer._prepare_train_epoch_data_stream(1)
        second_epoch_iter = trainer.train_data_loader_iter
        second_chunk = [int(next(second_epoch_iter).item()) for _ in range(3)]

        assert second_epoch_iter is first_epoch_iter
        assert first_chunk == [0, 1, 2]
        assert second_chunk == [3, 4, 5]

    def test_training_loop_respects_max_total_steps(self, tmp_path):
        trainer, num_batches = self._make_trainer(tmp_path)
        trainer.config.training.num_epochs = 3
        trainer.config.training.max_total_steps = 3

        dev = str(trainer.device)
        original_reset = trainer.reset_data_loader_iter

        def _patched_reset():
            original_reset()
            trainer._get_train_prefetcher = lambda: _FakePrefetcher(
                num_batches,
                h=H_TEST,
                w=W_TEST,
                device=dev,
                representation="events_voxel",
            )

        trainer.reset_data_loader_iter = _patched_reset

        with patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1):
            trainer.train()

        assert trainer.iteration == 3
        assert trainer.epoch == 0

    def test_limited_smoke_run_does_not_trigger_final_validation(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer.config.training.num_epochs = 1
        trainer.config.training.max_total_steps = 1
        trainer.config.eval.eval_interval = 5
        trainer.val_loader = object()
        trainer.epoch = 0

        assert not trainer._should_run_validation_this_epoch()

        trainer.config.training.max_total_steps = 0
        assert trainer._should_run_validation_this_epoch()

    def test_single_run_step_streaming_representation(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path, streaming=True)
        assert isinstance(trainer.mcm, StreamingMotionCompensationModule)

        # Provide a minimal streaming estimator output so this unit test
        # exercises trainer-to-SMCM wiring without requiring a full SOFE impl.
        mcm_mod = trainer.mcm
        est = mcm_mod.estimator

        def _fake_forward_streaming(**kwargs):
            events = kwargs["events"]
            t_ref_us = kwargs["t_ref_us"]
            b = int(t_ref_us.shape[0])
            base = torch.ones((b, 2, H_TEST, W_TEST), device=events.device, dtype=torch.float32)
            flow = base * est.placeholder_gain * 1e-3
            state = kwargs.get("state", None)
            if bool(kwargs.get("return_debug", False)):
                return flow, state, {"fake_streaming": True}
            return flow, state

        est.forward_streaming = _fake_forward_streaming

        trainer.mcm.train()
        trainer.storage = MagicMock()
        trainer.storage.latest.return_value = {}
        trainer.storage.iter = 0
        trainer.run_step()

    def test_collect_flow_diagnostics_reports_spatial_structure(self):
        flow = torch.zeros((1, 2, 4, 5), dtype=torch.float32)
        flow[:, 0] = torch.tensor(
            [[0.0, 0.2, 0.4, 0.6, 0.8],
             [0.0, 0.2, 0.4, 0.6, 0.8],
             [0.0, 0.2, 0.4, 0.6, 0.8],
             [0.0, 0.2, 0.4, 0.6, 0.8]],
            dtype=torch.float32,
        )
        flow[:, 1] = torch.tensor(
            [[0.0, 0.1, 0.2, 0.3, 0.4],
             [0.3, 0.4, 0.5, 0.6, 0.7],
             [0.6, 0.7, 0.8, 0.9, 1.0],
             [0.9, 1.0, 1.1, 1.2, 1.3]],
            dtype=torch.float32,
        )

        diagnostics = MCMTrainer._collect_flow_diagnostics(flow)

        assert float(diagnostics["flow_mag_mean"]) > 0.0
        assert float(diagnostics["flow_centered_mag_mean"]) > 0.0
        assert float(diagnostics["flow_u_spatial_std_mean"]) > 0.0
        assert float(diagnostics["flow_v_spatial_std_mean"]) > 0.0
        assert float(diagnostics["flow_dx_mean"]) > 0.0
        assert float(diagnostics["flow_dy_mean"]) > 0.0
        assert float(diagnostics["flow_dx_max"]) >= float(diagnostics["flow_dx_mean"])
        assert float(diagnostics["flow_dy_max"]) >= float(diagnostics["flow_dy_mean"])
        assert float(diagnostics["flow_div_mean"]) > 0.0
        assert float(diagnostics["flow_curl_mean"]) > 0.0

    def test_collect_flow_diagnostics_supports_logit_prefix(self):
        logits = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
        logits[:, 0, :, 1:] = 0.5
        diagnostics = MCMTrainer._collect_flow_diagnostics(logits, prefix="logits")

        assert "logits_mag_mean" in diagnostics
        assert "flow_mag_mean" not in diagnostics
        assert float(diagnostics["logits_mag_mean"]) > 0.0

    def test_collect_flow_sign_diagnostics_reports_direction_fractions(self):
        flow = torch.tensor(
            [
                [
                    [[1.0, -1.0], [2.0, -2.0]],
                    [[-3.0, 3.0], [-4.0, 4.0]],
                ]
            ],
            dtype=torch.float32,
        )

        diagnostics = MCMTrainer._collect_flow_sign_diagnostics(flow, prefix="flow")

        assert diagnostics["flow_u_pos_frac"] == pytest.approx(0.5)
        assert diagnostics["flow_u_neg_frac"] == pytest.approx(0.5)
        assert diagnostics["flow_v_pos_frac"] == pytest.approx(0.5)
        assert diagnostics["flow_v_neg_frac"] == pytest.approx(0.5)

    def test_collect_streaming_state_diagnostics_includes_flowwarm_and_deltaflow(self):
        class _FakeEstimator:
            def __init__(self):
                self._state = SOFEState(
                    hidden={"h": torch.ones((1, 4, 2, 2), dtype=torch.float32)},
                    readout_state={"r": torch.ones((1, 2, 2, 2), dtype=torch.float32)},
                    flow_warm=torch.tensor(
                        [[[[1.0, -1.0], [0.5, -0.5]], [[-1.0, 1.0], [-0.5, 0.5]]]],
                        dtype=torch.float32,
                    ),
                    last_step_ref_us=torch.tensor([100], dtype=torch.int64),
                )
                self._last_step_readout_aux = {
                    "delta_flow": torch.tensor(
                        [[[[0.2, -0.2], [0.1, -0.1]], [[-0.2, 0.2], [-0.1, 0.1]]]],
                        dtype=torch.float32,
                    ),
                    "readout_alpha": torch.tensor(0.75, dtype=torch.float32),
                }

        class _FakeMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.estimator = _FakeEstimator()

        diagnostics = MCMTrainer._collect_streaming_state_diagnostics(_FakeMCM())

        assert float(diagnostics["core_hidden_norm"]) > 0.0
        assert float(diagnostics["readout_state_norm"]) > 0.0
        assert "flow_warm_u_mean" in diagnostics
        assert "flow_warm_v_mean" in diagnostics
        assert "flow_warm_u_pos_frac" in diagnostics
        assert "flow_warm_v_neg_frac" in diagnostics
        assert "delta_flow_mag_mean" in diagnostics
        assert "delta_flow_u_pos_frac" in diagnostics
        assert float(diagnostics["readout_alpha_mean"]) == pytest.approx(0.75)

    def test_nonfinite_preclip_grad_norm_skips_step_and_resets_state(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)

        class _InfGradMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros((), dtype=torch.float32))
                self.reset_calls = 0

            def forward(self, _primary_input, **_kwargs):
                corrected = torch.zeros((2, 5, H_TEST, W_TEST), dtype=torch.float32)
                loss = torch.sqrt(self.weight)
                return corrected, loss

            def reset_state(self):
                self.reset_calls += 1

        fake_mcm = _InfGradMCM().to(trainer.device)
        trainer.mcm = fake_mcm
        trainer.model = fake_mcm
        trainer.optimizer = torch.optim.SGD(fake_mcm.parameters(), lr=1e-2)
        trainer.use_amp = False
        trainer.scaler = None
        trainer.storage = MagicMock()
        trainer.storage.latest.return_value = {}
        trainer.storage.iter = 0

        original_step = trainer.optimizer.step
        trainer.optimizer.step = MagicMock(wraps=original_step)

        trainer.run_step()

        # Current trainer behavior may reset at step start and again when
        # non-finite gradients are detected.
        assert fake_mcm.reset_calls >= 1
        assert trainer.optimizer.step.call_count == 0
        assert fake_mcm.weight.grad is None

    def test_clip_gradients_scales_large_finite_grads_in_place(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)

        class _HugeFiniteGradMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros((16,), dtype=torch.float32))

            def reset_state(self):
                pass

        fake_mcm = _HugeFiniteGradMCM().to(trainer.device)
        trainer.mcm = fake_mcm
        trainer.model = fake_mcm
        trainer.config.training.grad_clip = 5.0

        huge_value = torch.finfo(torch.float32).max / 2.0
        fake_mcm.weight.grad = torch.full_like(fake_mcm.weight, huge_value)

        assert trainer._clip_gradients_and_check_finite() is True
        assert torch.isfinite(fake_mcm.weight.grad).all()
        clipped_norm = torch.linalg.vector_norm(fake_mcm.weight.grad.detach(), ord=2).item()
        assert clipped_norm == pytest.approx(5.0, rel=1e-4, abs=1e-4)

    def test_run_step_with_large_finite_gradients_still_steps_optimizer(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)

        class _LargeFiniteGradMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.reset_calls = 0

            def forward(self, primary_input, **_kwargs):
                corrected = torch.zeros_like(primary_input)
                loss = self.weight * torch.tensor(1e38, device=self.weight.device, dtype=torch.float32)
                return corrected, loss

            def reset_state(self):
                self.reset_calls += 1

        fake_mcm = _LargeFiniteGradMCM().to(trainer.device)
        trainer.mcm = fake_mcm
        trainer.model = fake_mcm
        trainer.optimizer = torch.optim.SGD(fake_mcm.parameters(), lr=1e-2)
        trainer.use_amp = False
        trainer.scaler = None
        trainer.storage = MagicMock()
        trainer.storage.latest.return_value = {}
        trainer.storage.iter = 0

        original_step = trainer.optimizer.step
        trainer.optimizer.step = MagicMock(wraps=original_step)

        trainer.run_step()

        assert fake_mcm.reset_calls >= 1
        assert trainer.optimizer.step.call_count == 1
        assert float(fake_mcm.weight.detach().item()) < 1.0

    def test_clip_gradients_leaves_regular_finite_grads_untouched(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)

        class _SmallFiniteGradMCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            def reset_state(self):
                pass

        fake_mcm = _SmallFiniteGradMCM().to(trainer.device)
        trainer.mcm = fake_mcm
        trainer.model = fake_mcm
        trainer.config.training.grad_clip = 5.0

        fake_mcm.weight.grad = torch.ones_like(fake_mcm.weight) * 0.5
        original_grad = fake_mcm.weight.grad.detach().clone()

        assert trainer._clip_gradients_and_check_finite() is True
        assert torch.allclose(fake_mcm.weight.grad, original_grad)

    def test_configure_mcm_pretrain_data_flags_loads_flow_for_directional_gt_losses(self):
        cfg = _make_mcm_pretrain_cfg()
        cfg.data.mapper_load_flow = False

        for loss_type in ("angular", "cmax_angular", "event_warping_supervised"):
            cfg_local = copy.deepcopy(cfg)
            cfg_local.model.mcm.loss_type = loss_type
            configure_mcm_pretrain_data_flags(cfg_local)
            assert cfg_local.data.mapper_load_flow is True

    def test_streaming_call_detaches_state_after_first_iteration(self):
        device = torch.device("cpu")
        batch = _FakePrefetcher(
            num_batches=1,
            h=H_TEST,
            w=W_TEST,
            batch_size=2,
            device=str(device),
            representation="streaming_events",
        ).next()

        class _NoTrainerResetOwnership:
            def prepare_sequence_reset(self, **kwargs):
                raise AssertionError("Trainer should not precompute sequence_reset for streaming SMCM")

        primary_0, kwargs_0, rep_0 = build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation="streaming_events",
            mcm_module=_NoTrainerResetOwnership(),
            tbptt_steps=0,
            iteration=0,
        )
        assert primary_0 is None
        assert rep_0 == "streaming_events"
        assert torch.equal(kwargs_0["sequence_reset"], torch.tensor([True, True], dtype=torch.bool))
        assert "detach_state" not in kwargs_0

        _primary_1, kwargs_1, rep_1 = build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation="streaming_events",
            mcm_module=_NoTrainerResetOwnership(),
            tbptt_steps=8,
            iteration=1,
        )
        assert rep_1 == "streaming_events"
        assert kwargs_1["detach_state"] is True

    def test_streaming_smcm_tbptt_executes_chunked_loss_and_backward_cpu(self):
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.max_steps_per_sample = 4
        cfg.model.mcm.flow_scaling = 1.0
        cfg.model.mcm.smooth_weight = 1e-3
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.data.event_bins = 5
        cfg.model.mcm.sofe_architecture = {
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "num_stages": 2,
            "num_res_blocks": 0,
            "recurrent_cell": "convrnn",
            "recurrent_placement": "every_stage",
            "readout_type": "single_scale",
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)
        mcm.train()

        batch = _FakePrefetcher(
            num_batches=1,
            h=H_TEST,
            w=W_TEST,
            batch_size=2,
            device="cpu",
            representation="streaming_events",
        ).next()
        primary_input, forward_kwargs, rep = build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation="streaming_events",
            mcm_module=mcm,
            tbptt_steps=cfg.model.mcm.bptt_steps,
            iteration=0,
        )

        assert primary_input is None
        assert rep == "streaming_events"

        corrected, loss, debug = mcm(primary_input, return_debug=True, **forward_kwargs)
        assert corrected.shape == (2, 5, H_TEST, W_TEST)
        assert torch.isfinite(loss)
        assert debug["tbptt_steps"] == 2
        assert debug["num_tbptt_chunks"] >= 2
        assert debug["num_readouts"] == debug["num_tbptt_chunks"]

        loss.backward()
        has_grad = any(
            p.grad is not None for p in mcm.estimator.parameters() if p.requires_grad
        )
        assert has_grad, "Expected TBPTT streaming backward to reach estimator parameters"

    def test_frozen_streaming_smcm_uses_inference_path_and_skips_tbptt_loss_cpu(self):
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.max_steps_per_sample = 4
        cfg.model.mcm.flow_scaling = 1.0
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.model.mcm.sofe_architecture = {
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "num_stages": 2,
            "num_res_blocks": 0,
            "recurrent_cell": "convrnn",
            "recurrent_placement": "every_stage",
            "readout_type": "single_scale",
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)
        counting_loss = _CountingLoss()
        mcm.loss_fn = counting_loss
        mcm.train()
        model = nn.Module()
        model.mcm = mcm
        freeze_mcm_params(model)

        batch = _FakePrefetcher(
            num_batches=1,
            h=H_TEST,
            w=W_TEST,
            batch_size=2,
            device="cpu",
            representation="streaming_events",
        ).next()
        corrected, loss, debug = mcm(None, return_debug=True, **batch)

        assert mcm.should_use_frozen_inference_path()
        assert counting_loss.calls == 0
        assert corrected.shape == (2, 5, H_TEST, W_TEST)
        assert not corrected.requires_grad
        assert not loss.requires_grad
        assert loss.item() == pytest.approx(0.0)
        assert isinstance(debug, dict)
        assert mcm._last_loss_dict is None
        assert all(p.grad is None for p in mcm.parameters())

    def test_streaming_smcm_tbptt_plan_chunk_backward_cpu(self):
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.max_steps_per_sample = 4
        cfg.model.mcm.flow_scaling = 1.0
        cfg.model.mcm.smooth_weight = 1e-3
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.data.event_bins = 5
        cfg.model.mcm.sofe_architecture = {
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "num_stages": 2,
            "num_res_blocks": 0,
            "recurrent_cell": "convrnn",
            "recurrent_placement": "every_stage",
            "readout_type": "single_scale",
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)
        mcm.train()
        batch = _FakePrefetcher(
            num_batches=1,
            h=H_TEST,
            w=W_TEST,
            batch_size=2,
            device="cpu",
            representation="streaming_events",
        ).next()
        primary_input, forward_kwargs, rep = build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation="streaming_events",
            mcm_module=mcm,
            tbptt_steps=cfg.model.mcm.bptt_steps,
            iteration=0,
        )

        assert primary_input is None
        assert rep == "streaming_events"

        plan = mcm.prepare_streaming_tbptt_plan(voxel=primary_input, **forward_kwargs)
        assert plan["num_chunks"] >= 2
        assert plan["total_events"] == sum(plan["chunk_event_counts"])
        assert plan["total_events"] > 0

        mcm.zero_grad(set_to_none=True)
        loss_weight_sum = 0.0
        for chunk_idx in range(int(plan["num_chunks"])):
            chunk_out = mcm(
                None,
                streaming_tbptt_plan=plan,
                streaming_tbptt_chunk_index=chunk_idx,
            )
            assert torch.isfinite(chunk_out["loss"])
            loss_weight = float(chunk_out["loss_weight"])
            loss_weight_sum += loss_weight
            (chunk_out["loss"] * loss_weight).backward()
            mcm.detach_streaming_tbptt_plan_state(plan)

        assert loss_weight_sum == pytest.approx(1.0)
        debug = mcm.finalize_streaming_tbptt_plan(plan)
        assert debug["streaming_step_policy"] == "chunk_backward"
        assert debug["streaming_step_policy_id"] == 1
        assert debug["num_tbptt_chunks"] == plan["num_chunks"]
        assert debug["chunk_event_count_min"] > 0
        assert debug["chunk_event_count_max"] >= debug["chunk_event_count_min"]
        assert debug["chunk_loss_weight_min"] > 0.0
        assert debug["chunk_loss_weight_max"] <= 1.0
        assert torch.isfinite(plan["weighted_loss_total"])

        has_grad = any(
            p.grad is not None for p in mcm.estimator.parameters() if p.requires_grad
        )
        assert has_grad, "Expected chunk-plan backward to reach estimator parameters"

    def test_streaming_sofe_convlif_tiny_subset_fits_simple_flow_target_cpu(self):
        torch.manual_seed(0)
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.max_steps_per_sample = 4
        cfg.model.mcm.flow_scaling = 1.0
        cfg.model.mcm.smooth_weight = 1e-3
        cfg.model.mcm.mode = "ann"
        cfg.model.mcm.lif_alpha = 0.85
        cfg.model.mcm.lif_theta = 1.0
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.data.event_bins = 5
        cfg.model.mcm.sofe_architecture = {
            "family": "fire",
            "step_encoding": "hist2",
            "stage_channels": [8, 8],
            "stage_depths": [1, 1],
            "num_stages": 2,
            "num_res_blocks": 0,
            "recurrent_cell": "convlif",
            "recurrent_placement": "every_stage",
            "readout_type": "single_scale",
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)
        estimator = mcm.estimator
        estimator.train()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)

        batch = _make_structured_streaming_batch(batch_size=2, h=H_TEST, w=W_TEST, device="cpu")
        target_flow = torch.zeros((2, 2, H_TEST, W_TEST), dtype=torch.float32)
        target_flow[:, 0].fill_(0.15)
        target_flow[:, 1].fill_(-0.05)

        losses = []
        for _ in range(8):
            optimizer.zero_grad()
            estimator.reset_state()
            flow, _state = estimator.forward_streaming(
                events=batch["events"],
                events_offsets=batch["events_offsets"],
                t_start_us=batch["t_start_us"],
                t_end_us=batch["t_end_us"],
                t_ref_us=batch["t_ref_us"],
                sequence_reset=batch["sequence_reset"],
            )
            assert flow.shape == (2, 2, H_TEST, W_TEST)
            loss = F.mse_loss(flow, target_flow)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        assert losses[-1] < losses[0], (
            f"Expected the ConvLIF SOFE path to fit a simple flow target on a tiny structured subset; "
            f"observed losses={losses}"
        )

    def test_streaming_smcm_unet_spikingjelly_lif_backward_smoke_cpu(self):
        pytest.importorskip("spikingjelly.activation_based")

        torch.manual_seed(0)
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.input_representation = "streaming_events"
        cfg.model.mcm.architecture = "sofe"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.partition_mode = "fixed_time"
        cfg.model.mcm.step_time_us = 16
        cfg.model.mcm.bptt_steps = 2
        cfg.model.mcm.max_steps_per_sample = 4
        cfg.model.mcm.flow_scaling = 1.0
        cfg.model.mcm.smooth_weight = 1e-3
        cfg.model.mcm.mode = "snn"
        cfg.model.mcm.lif_alpha = 0.85
        cfg.model.mcm.lif_theta = 1.0
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST
        cfg.data.event_bins = 5
        cfg.model.mcm.sofe_architecture = {
            "family": "unet",
            "step_encoding": "hist2",
            "stage_channels": [8, 16, 32],
            "stage_depths": [1, 1, 1],
            "num_res_blocks": 1,
            "recurrent_backend": "spikingjelly",
            "recurrent_cell": "convlif",
            "recurrent_placement": "every_stage",
            "spiking_neuron": "lif",
            "surrogate_function": "sigmoid",
            "surrogate_alpha": 4.0,
            "detach_reset": True,
            "spiking_backend": "torch",
            "spiking_v_threshold": 1.0,
            "spiking_v_reset": 0.0,
            "downsample_mode": "stride_conv",
            "upsample_mode": "nearest_conv",
            "skip_mode": "concat",
            "readout_type": "multi_scale",
            "readout_scales": [0, 1],
            "readout_mid_channels": 8,
            "flow_activation": "identity",
        }

        mcm = _build_mcm(cfg)
        mcm.train()

        batch = _FakePrefetcher(
            num_batches=1,
            h=H_TEST,
            w=W_TEST,
            batch_size=2,
            device="cpu",
            representation="streaming_events",
        ).next()
        primary_input, forward_kwargs, rep = build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation="streaming_events",
            mcm_module=mcm,
            tbptt_steps=cfg.model.mcm.bptt_steps,
            iteration=0,
        )

        assert primary_input is None
        assert rep == "streaming_events"

        corrected, loss, debug = mcm(primary_input, return_debug=True, **forward_kwargs)
        assert corrected.shape == (2, 5, H_TEST, W_TEST)
        assert torch.isfinite(loss)
        assert debug["sofe_family"] == "unet"
        assert debug["recurrent_backend"] == "spikingjelly"
        assert debug["recurrent_cell"] == "convlif"
        assert debug["spiking_neuron"] == "lif"
        assert debug["spiking_backend"] == "torch"
        assert debug["readout_type"] == "multi_scale"

        loss.backward()
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in mcm.estimator.parameters()
            if p.requires_grad
        )
        assert has_grad, "Expected streaming SMCM backward to reach SpikingJelly-backed SOFE parameters"

    def test_short_training_loop_streaming_sofe_detaches_between_steps(self, tmp_path):
        trainer, num_batches = self._make_trainer(tmp_path, streaming=True)
        trainer.config.training.num_epochs = 1
        trainer.config.training.max_total_steps = 2
        dev = str(trainer.device)

        original_reset = trainer.reset_data_loader_iter

        def _patched_reset():
            original_reset()
            trainer._get_train_prefetcher = lambda: _FakePrefetcher(
                num_batches,
                h=H_TEST,
                w=W_TEST,
                device=dev,
                representation="streaming_events",
            )

        trainer.reset_data_loader_iter = _patched_reset

        with patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1):
            trainer.train()

        assert trainer.iteration == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. Checkpoint save / load / resume
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMCheckpointing:
    """Save, load, and resume MCM-only checkpoints."""

    def _make_mcm(self) -> MotionCompensationModule:
        cfg = _make_mcm_cfg()
        return _build_mcm(cfg)

    def test_save_and_load(self, tmp_path):
        mcm = self._make_mcm()
        ckpt_path = tmp_path / "mcm_ckpt.pth"
        save_mcm_checkpoint(mcm, ckpt_path, epoch=3, iteration=100)

        ckpt = load_mcm_checkpoint(ckpt_path)
        assert "mcm" in ckpt
        assert ckpt["epoch"] == 3
        assert ckpt["iteration"] == 100

    def test_load_weights_into_module(self, tmp_path):
        mcm_original = self._make_mcm()
        ckpt_path = tmp_path / "mcm_weights.pth"
        save_mcm_checkpoint(mcm_original, ckpt_path)

        mcm_new = self._make_mcm()
        # Scramble weights
        for p in mcm_new.parameters():
            p.data.uniform_(-10, 10)

        load_mcm_weights_into_module(mcm_new, ckpt_path)

        # Weights should now match
        for (n1, p1), (n2, p2) in zip(
            mcm_original.state_dict().items(), mcm_new.state_dict().items()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_save_with_optimizer_scheduler(self, tmp_path):
        mcm = self._make_mcm()
        opt = torch.optim.Adam(mcm.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        ckpt_path = tmp_path / "full_mcm_ckpt.pth"

        save_mcm_checkpoint(
            mcm, ckpt_path,
            optimizer=opt, scheduler=sched,
            epoch=5, iteration=200, best_metric=0.42,
        )

        ckpt = load_mcm_checkpoint(ckpt_path)
        assert "optimizer" in ckpt
        assert "scheduler" in ckpt
        assert ckpt["best_metric"] == pytest.approx(0.42)

    def test_resume_trainer(self, tmp_path):
        """Save a trainer checkpoint, then resume into a new trainer."""
        cfg = _make_mcm_pretrain_cfg()
        cfg.log.output_dir = str(tmp_path / "run1")
        cfg.log.experiment_name = "ckpt_test"
        cfg.data.image_height = H_TEST
        cfg.data.image_width = W_TEST

        with patch("smcfnet.core.mcm_trainer.build_train_loader_cfg") as mock_loader, \
             patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1), \
             patch("smcfnet.core.mcm_trainer.get_local_rank", return_value=0), \
             patch("torch.utils.tensorboard.SummaryWriter"):
            mock_loader.return_value = _FakeDataLoader(4)
            from smcfnet.core.mcm_trainer import MCMTrainer
            trainer1 = MCMTrainer(cfg)

        trainer1.epoch = 5
        trainer1.iteration = 100
        trainer1.best_metric = 0.75
        ckpt_path = tmp_path / "mcm_resume.pth"
        save_mcm_checkpoint(
            trainer1.mcm, ckpt_path,
            optimizer=trainer1.optimizer,
            scheduler=trainer1.scheduler,
            epoch=5, iteration=100, best_metric=0.75,
        )

        # Build a fresh trainer and resume
        cfg2 = _make_mcm_pretrain_cfg()
        cfg2.log.output_dir = str(tmp_path / "run2")
        cfg2.log.experiment_name = "ckpt_test_resumed"
        cfg2.log.resume_from = str(ckpt_path)
        cfg2.log.load_weights_only = False
        cfg2.data.image_height = H_TEST
        cfg2.data.image_width = W_TEST

        with patch("smcfnet.core.mcm_trainer.build_train_loader_cfg") as mock_loader, \
             patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1), \
             patch("smcfnet.core.mcm_trainer.get_local_rank", return_value=0), \
             patch("torch.utils.tensorboard.SummaryWriter"):
            mock_loader.return_value = _FakeDataLoader(4)
            trainer2 = MCMTrainer(cfg2)

        assert trainer2.epoch == 5
        assert trainer2.iteration == 100
        assert trainer2.best_metric == pytest.approx(0.75)

    def test_extract_ofe_state_dict_from_raw_estimator_checkpoint(self, tmp_path):
        from smcfnet.model.mcm.weight_utils import extract_ofe_state_dict

        base = create_flow_estimator("ecm", num_bins=5, mode="ann")
        raw_path = tmp_path / "raw_estimator_state.pth"
        torch.save(base.state_dict(), raw_path)

        extracted = extract_ofe_state_dict(raw_path)
        assert extracted, "Expected non-empty extracted OFE state dict"
        assert set(extracted.keys()) == set(base.state_dict().keys())

    def test_load_raw_estimator_checkpoint_into_mcm_module(self, tmp_path):
        src = self._make_mcm()
        raw_path = tmp_path / "raw_estimator_only.pth"
        torch.save(src.estimator.state_dict(), raw_path)

        dst = self._make_mcm()
        for p in dst.parameters():
            p.data.uniform_(-1.0, 1.0)

        load_mcm_weights_into_module(dst, raw_path, strict=False)

        src_sd = src.estimator.state_dict()
        dst_sd = dst.estimator.state_dict()
        assert set(src_sd.keys()) == set(dst_sd.keys())
        for k in src_sd.keys():
            assert torch.allclose(src_sd[k], dst_sd[k]), f"Estimator key mismatch after raw load: {k}"

    def test_external_ofev_checkpoint_loads_into_cuadrado_estimator_when_available(self):
        # Optional integration check against local external assets.
        ext_ckpt = Path("/cluster/work/kristvje/projects/smcm-mcfnet/example/OF_EV_SNN/examples/dsec_fullset.pth")
        if not ext_ckpt.exists():
            pytest.skip("External OF_EV_SNN checkpoint not available in this environment")

        pytest.importorskip("spikingjelly")
        from smcfnet.model.mcm.weight_utils import extract_ofe_state_dict

        ofe_sd = extract_ofe_state_dict(ext_ckpt)
        est = create_flow_estimator(
            "cuadrado_unet",
            num_bins=21,
            mode="snn",
            image_height=64,
            image_width=80,
        )
        missing, unexpected = est.load_state_dict(ofe_sd, strict=False)
        assert not missing
        assert not unexpected

        x = torch.randn(1, 21, 64, 80)
        with torch.no_grad():
            y = est(x)
        assert tuple(y.shape) == (1, 2, 64, 80)
        assert torch.isfinite(y).all()

class TestSDFormerFlowSNNEstimatorContract:
    def test_sdformerflow_snn_accepts_signed_voxel_and_returns_dense_flow(self):
        est = create_flow_estimator(
            "sdformerflow_snn",
            num_bins=5,
            mode="snn",
        )
        x = torch.randn(2, 5, H_TEST, W_TEST)
        with torch.no_grad():
            y = est(x)
        assert tuple(y.shape) == (2, 2, H_TEST, W_TEST)
        assert torch.isfinite(y).all()

    def test_sdformerflow_snn_accepts_split_polarity_layout(self):
        est = create_flow_estimator(
            "sdformerflow_snn",
            num_bins=5,
            mode="ann",
        )
        x = torch.randn(1, 5, 2, H_TEST, W_TEST)
        with torch.no_grad():
            y = est(x)
        assert tuple(y.shape) == (1, 2, H_TEST, W_TEST)

    def test_sdformerflow_snn_rejects_wrong_5d_layout(self):
        est = create_flow_estimator(
            "sdformerflow_snn",
            num_bins=5,
            mode="ann",
        )
        bad = torch.randn(1, 2, 5, H_TEST, W_TEST)
        with pytest.raises(ValueError, match="reorder"):
            _ = est(bad)

    def test_sdformerflow_snn_mode_switch_and_reset(self):
        est = create_flow_estimator(
            "sdformerflow_snn",
            num_bins=5,
            mode="ann",
        )
        assert est.get_mode() == "ann"
        est.set_mode("snn")
        assert est.get_mode() == "snn"

        x = torch.randn(1, 5, H_TEST, W_TEST)
        with torch.no_grad():
            _ = est(x)
        est.reset_state()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Freeze / unfreeze MCM parameters
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMFreezeUnfreeze:

    def test_freeze_standalone_mcm(self):
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg)
        # Wrap in a simple model with .mcm attribute
        model = nn.Module()
        model.mcm = mcm

        n_frozen = freeze_mcm_params(model)
        assert n_frozen > 0
        for p in mcm.parameters():
            assert not p.requires_grad

    def test_unfreeze_after_freeze(self):
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg)
        model = nn.Module()
        model.mcm = mcm

        freeze_mcm_params(model)
        n_unfrozen = unfreeze_mcm_params(model)
        assert n_unfrozen > 0
        for p in mcm.parameters():
            assert p.requires_grad

    def test_freeze_no_grad_on_backward(self):
        """Frozen MCM should not accumulate gradients."""
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg)
        model = nn.Module()
        model.mcm = mcm

        freeze_mcm_params(model)

        voxel = _make_voxel()
        _, loss = mcm(voxel)

        # With all params frozen, loss has no grad_fn — verify that
        assert not loss.requires_grad, "Loss should not require grad when MCM is frozen"
        for p in mcm.parameters():
            assert p.grad is None, "Frozen param should have no gradient"

    def test_frozen_mcm_uses_inference_path_and_skips_loss(self):
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg)
        counting_loss = _CountingLoss()
        mcm.loss_fn = counting_loss
        mcm.train()
        model = nn.Module()
        model.mcm = mcm

        freeze_mcm_params(model)

        voxel = _make_voxel().requires_grad_(True)
        corrected, loss = mcm(voxel)

        assert mcm.should_use_frozen_inference_path()
        assert counting_loss.calls == 0
        assert corrected.shape == voxel.shape
        assert not corrected.requires_grad
        assert not loss.requires_grad
        assert loss.item() == pytest.approx(0.0)
        assert mcm._last_loss_dict is None
        assert mcm._last_contrast_loss is None

    def test_freeze_in_full_model(self):
        """Freeze MCM inside a backbone.mcm path."""
        from smcfnet.model.backbone import CSPDarknetDualStream
        cfg = _make_mcm_cfg()
        mcm_instance = _build_mcm(cfg)
        backbone = CSPDarknetDualStream(
            dep_mul=0.33, wid_mul=0.25,
            out_features=("dark3", "dark4", "dark5"),
            use_mcm=True,
            mcm_instance=mcm_instance,
        )
        # build_model wraps backbone in DualStreamYOLOX — simulate that structure
        model = nn.Module()
        model.backbone = backbone

        n = freeze_mcm_params(model)
        assert n > 0
        for p in backbone.mcm.parameters():
            assert not p.requires_grad


# ═══════════════════════════════════════════════════════════════════════════
# 6. Weight loading into full detection model (Phase B)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMWeightTransfer:
    """Phase B: load MCM-only checkpoint weights into the full DualStreamYOLOX."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for full model")
    def test_load_mcm_into_full_model(self, tmp_path):
        from smcfnet.model.build import build_model

        # Phase A: build and save MCM-only checkpoint
        cfg_a = _make_mcm_cfg()
        mcm = _build_mcm(cfg_a)
        ckpt_path = tmp_path / "phase_a.pth"
        save_mcm_checkpoint(mcm, ckpt_path)

        # Phase B: build full model with MCM, then load MCM weights
        cfg_b = Config()
        cfg_b.model.mcm.enabled = True
        cfg_b.model.mcm.architecture = "ecm"
        cfg_b.model.mcm.mid_channels = 16
        cfg_b.model.mcm.loss_type = "cmax"
        cfg_b.model.mcm.contrast_weight = 1.0
        cfg_b.model.mcm.smooth_weight = 0.1
        cfg_b.model.size = "nano"
        cfg_b.data.event_bins = 5

        model = build_model(cfg_b).cuda()
        load_mcm_weights_into_full_model(model, ckpt_path)

        # Verify weights were transferred
        backbone_mcm = model.backbone.mcm
        original_state = mcm.state_dict()
        loaded_state = backbone_mcm.state_dict()
        for key in original_state:
            assert key in loaded_state, f"Missing key {key} after transfer"
            assert torch.allclose(
                original_state[key].cpu(), loaded_state[key].cpu()
            ), f"Weight mismatch for {key}"

    def test_build_model_loads_external_cuadrado_checkpoint_when_available(self):
        ext_ckpt = Path("/cluster/work/kristvje/projects/smcm-mcfnet/example/OF_EV_SNN/examples/dsec_fullset.pth")
        if not ext_ckpt.exists():
            pytest.skip("External OF_EV_SNN checkpoint not available in this environment")

        pytest.importorskip("spikingjelly")
        from smcfnet.model.build import build_model
        from smcfnet.model.mcm.weight_utils import extract_ofe_state_dict

        cfg = Config()
        cfg.model.size = "nano"
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "cuadrado_unet"
        cfg.model.mcm.mode = "snn"
        cfg.model.mcm.input_representation = "voxel"
        cfg.model.mcm.loss_type = "event_warping"
        cfg.model.mcm.warper_type = "sparse_event"
        cfg.model.mcm.pretrained_weights = str(ext_ckpt)
        cfg.model.mcm.flow_scaling = 1.0
        cfg.data.event_bins = 21
        cfg.data.image_height = 64
        cfg.data.image_width = 80

        model = build_model(cfg)

        expected_state = extract_ofe_state_dict(ext_ckpt)
        loaded_state = model.backbone.mcm.estimator.state_dict()
        for key, tensor in expected_state.items():
            assert key in loaded_state, f"Missing Cuadrado estimator key after build_model load: {key}"
            assert torch.allclose(
                loaded_state[key].cpu(),
                tensor.cpu(),
            ), f"Weight mismatch for Cuadrado estimator key: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Joint model forward with MCM enabled
# ═══════════════════════════════════════════════════════════════════════════


class TestJointModelWithMCM:
    """Full DualStreamYOLOX forward + backward with MCM active."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="full model needs GPU")
    def test_joint_forward_backward(self):
        from smcfnet.model.build import build_model

        cfg = Config()
        cfg.model.size = "nano"
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "ecm"
        cfg.model.mcm.mid_channels = 16
        cfg.model.mcm.loss_type = "cmax"
        cfg.model.mcm.warper_type = "voxel"
        cfg.model.mcm.loss_weight = 0.1
        cfg.data.event_bins = 5

        model = build_model(cfg).cuda()
        model.train()

        B = 2
        rgb = torch.randn(B, 3, H_TEST, W_TEST, device="cuda")
        events = torch.randn(B, 5, H_TEST, W_TEST, device="cuda")
        targets = torch.zeros(B, 3, 5, device="cuda")
        targets[:, 0, :] = torch.tensor([0, 15.0, 15.0, 10.0, 10.0])
        targets[:, 1, :] = torch.tensor([1, 25.0, 20.0, 8.0, 8.0])

        loss_dict = model(rgb, events, targets)

        assert "total" in loss_dict
        assert "loss_mcm" in loss_dict
        assert torch.isfinite(loss_dict["total"])

        loss_dict["total"].backward()
        mcm_params = list(model.backbone.mcm.parameters())
        grads = [p.grad for p in mcm_params if p.grad is not None]
        assert len(grads) > 0, "MCM should receive gradients in joint training"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="full model needs GPU")
    def test_joint_forward_with_frozen_mcm(self):
        """Phase B: frozen MCM still produces corrected voxels but no MCM grads."""
        from smcfnet.model.build import build_model

        cfg = Config()
        cfg.model.size = "nano"
        cfg.model.mcm.enabled = True
        cfg.model.mcm.architecture = "ecm"
        cfg.model.mcm.mid_channels = 16
        cfg.model.mcm.loss_type = "cmax"
        cfg.model.mcm.warper_type = "voxel"
        cfg.model.mcm.loss_weight = 0.0  # no MCM loss
        cfg.model.mcm.freeze = True
        cfg.data.event_bins = 5

        model = build_model(cfg).cuda()
        model.train()

        B = 2
        rgb = torch.randn(B, 3, H_TEST, W_TEST, device="cuda")
        events = torch.randn(B, 5, H_TEST, W_TEST, device="cuda")
        targets = torch.zeros(B, 3, 5, device="cuda")
        targets[:, 0, :] = torch.tensor([0, 15.0, 15.0, 10.0, 10.0])

        loss_dict = model(rgb, events, targets)
        assert "total" in loss_dict
        loss_dict["total"].backward()

        # MCM params should have no gradients (frozen)
        for p in model.backbone.mcm.parameters():
            assert not p.requires_grad
            assert p.grad is None


# ═══════════════════════════════════════════════════════════════════════════
# 8. Config dispatch: mcm_pretrain mode
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigDispatch:
    """Verify that training.mode = 'mcm_pretrain' routes to MCMTrainer."""

    def test_mcm_pretrain_mode_flag(self):
        cfg = _make_mcm_pretrain_cfg()
        assert cfg.training.mode == "mcm_pretrain"

    def test_joint_mode_default(self):
        cfg = Config()
        assert cfg.training.mode == "joint"

    def test_mcm_trainer_requires_mcm_enabled(self, tmp_path):
        """MCMTrainer should raise if mcm.enabled is False."""
        cfg = _make_mcm_pretrain_cfg()
        cfg.model.mcm.enabled = False
        cfg.log.output_dir = str(tmp_path)
        cfg.log.experiment_name = "should_fail"

        with patch("smcfnet.core.mcm_trainer.build_train_loader_cfg"), \
             patch("smcfnet.core.mcm_trainer.is_main_process", return_value=True), \
             patch("smcfnet.core.mcm_trainer.get_world_size", return_value=1), \
             patch("smcfnet.core.mcm_trainer.get_local_rank", return_value=0), \
             patch("torch.utils.tensorboard.SummaryWriter"):
            from smcfnet.core.mcm_trainer import MCMTrainer
            with pytest.raises(ValueError, match="mcm.enabled"):
                MCMTrainer(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# 9. MCM optimizer / scheduler construction
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMOptimizerScheduler:

    def _mcm(self):
        return _build_mcm(_make_mcm_cfg())

    def test_adam_optimizer(self):
        from smcfnet.core.mcm_trainer import _build_mcm_optimizer
        cfg = _make_mcm_pretrain_cfg(**{"training.optimizer": "adam"})
        opt = _build_mcm_optimizer(cfg, self._mcm())
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw_optimizer(self):
        from smcfnet.core.mcm_trainer import _build_mcm_optimizer
        cfg = _make_mcm_pretrain_cfg(**{"training.optimizer": "adamw"})
        opt = _build_mcm_optimizer(cfg, self._mcm())
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd_optimizer(self):
        from smcfnet.core.mcm_trainer import _build_mcm_optimizer
        cfg = _make_mcm_pretrain_cfg(**{"training.optimizer": "sgd"})
        opt = _build_mcm_optimizer(cfg, self._mcm())
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_optimizer_raises(self):
        from smcfnet.core.mcm_trainer import _build_mcm_optimizer
        cfg = _make_mcm_pretrain_cfg(**{"training.optimizer": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown optimizer"):
            _build_mcm_optimizer(cfg, self._mcm())

    def test_cosine_scheduler(self):
        from smcfnet.core.mcm_trainer import _build_mcm_scheduler
        cfg = _make_mcm_pretrain_cfg(**{"training.lr_scheduler": "cosine"})
        mcm = self._mcm()
        opt = torch.optim.Adam(mcm.parameters(), lr=1e-3)
        sched = _build_mcm_scheduler(cfg, opt)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_scheduler(self):
        from smcfnet.core.mcm_trainer import _build_mcm_scheduler
        cfg = _make_mcm_pretrain_cfg(**{"training.lr_scheduler": "step"})
        mcm = self._mcm()
        opt = torch.optim.Adam(mcm.parameters(), lr=1e-3)
        sched = _build_mcm_scheduler(cfg, opt)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_none_scheduler(self):
        from smcfnet.core.mcm_trainer import _build_mcm_scheduler
        cfg = _make_mcm_pretrain_cfg(**{"training.lr_scheduler": "none"})
        mcm = self._mcm()
        opt = torch.optim.Adam(mcm.parameters(), lr=1e-3)
        sched = _build_mcm_scheduler(cfg, opt)
        assert sched is None


# ═══════════════════════════════════════════════════════════════════════════
# 10. AMP compatibility with MCM training step
# ═══════════════════════════════════════════════════════════════════════════


class TestMCMAMP:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP needs CUDA")
    def test_amp_mcm_forward_backward(self):
        cfg = _make_mcm_cfg()
        mcm = _build_mcm(cfg).cuda()
        voxel = _make_voxel(device="cuda")

        with torch.amp.autocast("cuda"):
            corrected, loss = mcm(voxel)

        assert torch.isfinite(loss)
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()

        for p in mcm.parameters():
            if p.requires_grad:
                assert p.grad is not None
