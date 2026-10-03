"""Tests for configurable MCM/SMCM detector output renderers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from smcfnet.config.config import Config
from smcfnet.model.build import _build_mcm, build_model
from smcfnet.model.event_outputs import (
    LiuRGBIWEEventOutput,
    VoxelGridEventOutput,
    create_event_output_renderer,
)
from smcfnet.model.sparse_event_warper import SparseEventWarpingModule


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _packed_events(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 1.0, 1.0, -1.0],
            [2.0, 2.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 3, 4], dtype=torch.int64, device=device)
    return events, offsets


def _mcm_cfg(output_type: str = "iwe_liu_rgb") -> Config:
    cfg = Config()
    cfg.data.event_bins = 5
    cfg.data.image_height = 32
    cfg.data.image_width = 32
    cfg.model.size = "nano"
    cfg.model.mcm.enabled = True
    cfg.model.mcm.architecture = "ecm"
    cfg.model.mcm.mid_channels = 8
    cfg.model.mcm.loss_type = "event_warping"
    cfg.model.mcm.flow_scaling = 1.0
    cfg.model.mcm.output.type = output_type
    cfg.model.mcm.output.iwe_normalize = "none"
    cfg.model.mcm.output.iwe_scale = 1.0
    cfg.model.mcm.output.iwe_align_to_rgb = "none"
    return cfg


def test_voxel_renderer_matches_legacy_sparse_event_warper(device):
    events, offsets = _packed_events(device)
    flow = torch.zeros(2, 2, 4, 4, device=device)
    voxel = torch.zeros(2, 5, 4, 4, device=device)

    renderer = VoxelGridEventOutput(num_bins=5, res=(4, 4)).to(device)
    legacy = SparseEventWarpingModule(num_bins=5, res=(4, 4)).to(device)

    rendered = renderer(flow=flow, events=events, events_offsets=offsets, voxel=voxel)
    legacy_out = legacy(voxel, flow, events=events, events_offsets=offsets)

    assert rendered.shape == (2, 5, 4, 4)
    assert torch.equal(rendered, legacy_out)


def test_iwe_renderer_keeps_polarities_separate_without_cancellation(device):
    events = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, -1.0]],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 2], dtype=torch.int64, device=device)
    flow = torch.zeros(1, 2, 4, 4, device=device)

    out = LiuRGBIWEEventOutput(res=(4, 4), normalize="none").to(device)(
        flow=flow,
        events=events,
        events_offsets=offsets,
    )

    assert out.shape == (1, 3, 4, 4)
    assert torch.count_nonzero(out[:, 0]) == 0
    assert out[0, 1, 1, 1] > 0
    assert out[0, 2, 1, 1] > 0


def test_iwe_renderer_bilinear_splat_distributes_to_four_neighbors(device):
    events = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device=device)
    flow = torch.zeros(1, 2, 4, 4, device=device)
    flow[0, 0, 1, 1] = 0.5   # x displacement
    flow[0, 1, 1, 1] = 0.25  # y displacement

    out = LiuRGBIWEEventOutput(res=(4, 4), normalize="none").to(device)(
        flow=flow,
        events=events,
        events_offsets=offsets,
    )
    pos = out[0, 1]

    assert torch.isclose(pos.sum(), torch.tensor(1.0, device=device))
    assert torch.isclose(pos[1, 1], torch.tensor(0.375, device=device))
    assert torch.isclose(pos[1, 2], torch.tensor(0.375, device=device))
    assert torch.isclose(pos[2, 1], torch.tensor(0.125, device=device))
    assert torch.isclose(pos[2, 2], torch.tensor(0.125, device=device))


def test_renderers_handle_zero_events_and_out_of_bounds_warps(device):
    empty_events = torch.zeros(0, 4, dtype=torch.float32, device=device)
    empty_offsets = torch.tensor([0, 0], dtype=torch.int64, device=device)
    flow = torch.zeros(1, 2, 4, 4, device=device)

    voxel_out = VoxelGridEventOutput(num_bins=5, res=(4, 4)).to(device)(
        flow=flow,
        events=empty_events,
        events_offsets=empty_offsets,
    )
    iwe_out = LiuRGBIWEEventOutput(res=(4, 4), normalize="p99").to(device)(
        flow=flow,
        events=empty_events,
        events_offsets=empty_offsets,
    )
    assert voxel_out.shape == (1, 5, 4, 4)
    assert iwe_out.shape == (1, 3, 4, 4)
    assert torch.count_nonzero(voxel_out) == 0
    assert torch.count_nonzero(iwe_out) == 0

    events = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int64, device=device)
    flow[:, 0].fill_(100.0)
    out_of_bounds = LiuRGBIWEEventOutput(res=(4, 4), normalize="none").to(device)(
        flow=flow,
        events=events,
        events_offsets=offsets,
    )
    assert torch.count_nonzero(out_of_bounds) == 0


def test_iwe_renderer_p99_normalization_and_mapping_validation(device):
    events, offsets = _packed_events(device)
    flow = torch.zeros(2, 2, 4, 4, device=device)

    out = LiuRGBIWEEventOutput(res=(4, 4), normalize="p99", scale=255.0).to(device)(
        flow=flow,
        events=events,
        events_offsets=offsets,
    )
    assert torch.isfinite(out).all()
    assert out[:, 1:].max() <= 255.0

    mapping_renderer = LiuRGBIWEEventOutput(
        res=(4, 4),
        normalize="none",
        align_to_rgb="mapping",
    ).to(device)
    with pytest.raises(ValueError, match="requires an 'iwe_mapping'"):
        mapping_renderer(flow=flow, events=events, events_offsets=offsets)


def test_renderer_factory_rejects_invalid_config_values():
    with pytest.raises(ValueError, match="Unsupported MCM output renderer type"):
        create_event_output_renderer(output_type="unknown")
    with pytest.raises(ValueError, match="Unsupported IWE normalization"):
        LiuRGBIWEEventOutput(normalize="median")
    with pytest.raises(ValueError, match="num_bins must be > 0"):
        VoxelGridEventOutput(num_bins=0)


def test_build_mcm_exposes_renderer_metadata_and_channel_count():
    cfg = _mcm_cfg("iwe_liu_rgb")
    mcm = _build_mcm(cfg)

    assert mcm.output_type == "iwe_liu_rgb"
    assert mcm.output_channels == 3
    assert mcm.required_batch_keys == ("events_voxel", "events", "events_offsets")

    cfg = _mcm_cfg("voxel_grid")
    cfg.model.mcm.output.voxel_bins = 7
    mcm = _build_mcm(cfg)
    assert mcm.output_type == "voxel_grid"
    assert mcm.output_channels == 7


def test_build_model_uses_mcm_output_channels_for_backbone_and_edum():
    cfg = _mcm_cfg("iwe_liu_rgb")
    cfg.model.mcm.loss_type = "cmax"
    cfg.model.edum.enabled = True

    model = build_model(cfg)

    assert model.backbone.event_in_channels == 3
    assert model.backbone.stem_event.conv.conv.in_channels == 12
    assert model.backbone.edum.num_bins == 3


def test_mcm_training_with_iwe_renderer_keeps_event_warping_loss_path(device):
    cfg = _mcm_cfg("iwe_liu_rgb")
    mcm = _build_mcm(cfg).to(device)
    mcm.train()

    events, offsets = _packed_events(device)
    voxel = torch.randn(2, 5, 32, 32, device=device)

    corrected, loss = mcm(voxel, events=events, events_offsets=offsets)
    total = loss + 1e-3 * corrected.sum()
    total.backward()

    assert corrected.shape == (2, 3, 32, 32)
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in mcm.estimator.parameters())
    assert mcm._last_corrected.shape == corrected.shape


def test_mcm_inference_with_iwe_renderer_returns_zero_loss(device):
    cfg = _mcm_cfg("iwe_liu_rgb")
    mcm = _build_mcm(cfg).to(device)
    mcm.eval()

    events, offsets = _packed_events(device)
    voxel = torch.randn(2, 5, 32, 32, device=device)

    with torch.inference_mode():
        corrected, loss = mcm(voxel, events=events, events_offsets=offsets)

    assert corrected.shape == (2, 3, 32, 32)
    assert torch.isfinite(corrected).all()
    assert loss.item() == 0.0


class _TinyHead(nn.Module):
    """Stable detector-head test double for DualStreamYOLOX train/eval plumbing."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, targets=None):
        score = sum(feat.float().mean() for feat in features) * self.weight
        if self.training:
            total = score.square() + 1.0
            outputs = (
                total,
                total * 0.2,
                total * 0.3,
                total * 0.4,
                total.new_zeros(()),
                total.new_tensor(1.0),
                [],
            )
            return outputs, None
        detections = torch.stack([feat.float().mean() for feat in features]).reshape(1, -1)
        return detections, None


def test_detector_training_and_inference_paths_with_iwe_renderer(device):
    if device.type != "cuda":
        pytest.skip("Full detector train/eval renderer smoke is CUDA-only")

    cfg = _mcm_cfg("iwe_liu_rgb")
    cfg.model.mcm.loss_type = "event_warping"
    cfg.model.mcm.loss_weight = 0.1
    model = build_model(cfg).to(device)
    model.head = _TinyHead().to(device)

    events, offsets = _packed_events(device)
    rgb = torch.randn(2, 3, 32, 32, device=device)
    voxel = torch.randn(2, 5, 32, 32, device=device)
    targets = torch.zeros(2, 1, 5, device=device)

    model.train()
    loss_dict = model(rgb, voxel, targets, events=events, events_offsets=offsets)
    assert torch.isfinite(loss_dict["total"])
    assert loss_dict["loss_mcm"].ndim == 0
    loss_dict["total"].backward()
    assert any(p.grad is not None for p in model.backbone.mcm.estimator.parameters())

    model.eval()
    with torch.inference_mode():
        pred = model(rgb, voxel, events=events, events_offsets=offsets)
    assert pred.shape == (1, 3)
    assert model._mcm_viz["output"].shape == (2, 3, 32, 32)
