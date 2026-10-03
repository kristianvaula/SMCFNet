"""Tests for STTPHead module.

Validates forward pass shapes, Design B vs C modes, gradient flow,
and near-zero initialization.

Run: pytest src/smcfnet/test/test_sttp_head.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import STTPHead


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# YOLOX-s: width=0.5 → stem channels = int(256 * 0.5) = 128
IN_CHANNELS = 128
# P3, P4, P5 for 640x480 input
SCALE_SIZES = [(60, 80), (30, 40), (15, 20)]
BATCH = 2


def _make_stem_features(device, batch=BATCH):
    return [
        torch.randn(batch, IN_CHANNELS, h, w, device=device)
        for h, w in SCALE_SIZES
    ]


def _make_flow(device, batch=BATCH, h=480, w=640):
    return torch.randn(batch, 2, h, w, device=device)


# ── Design B ─────────────────────────────────────────────────────────────

class TestDesignB:
    def test_forward_shapes(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=0).to(device)
        outputs = head(_make_stem_features(device))

        assert len(outputs) == 3
        for k, out in enumerate(outputs):
            h, w = SCALE_SIZES[k]
            assert out.shape == (BATCH, 4, h, w)

    def test_no_flow_encoders(self):
        head = STTPHead(IN_CHANNELS, flow_channels=0)
        for enc in head.flow_encoders:
            assert enc is None

    def test_flow_ignored_when_passed(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=0).to(device)
        head.eval()
        stems = _make_stem_features(device)

        out_a = head(stems, flow=None)
        out_b = head(stems, flow=_make_flow(device))
        for a, b in zip(out_a, out_b):
            assert torch.allclose(a, b)

    def test_gradient_flow_to_stem(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=0).to(device)
        stems = _make_stem_features(device)
        for s in stems:
            s.requires_grad_(True)

        loss = sum(o.sum() for o in head(stems))
        loss.backward()

        for s in stems:
            assert s.grad is not None
            assert s.grad.abs().sum() > 0

    def test_near_zero_init(self):
        head = STTPHead(IN_CHANNELS, flow_channels=0)
        for pred in head.fut_preds:
            assert pred.weight.abs().max() < 0.1
            assert pred.bias.abs().max() < 1e-6


# ── Design C ─────────────────────────────────────────────────────────────

class TestDesignC:
    def test_forward_shapes(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=2, flow_proj_dim=16).to(device)
        outputs = head(_make_stem_features(device), flow=_make_flow(device))

        assert len(outputs) == 3
        for k, out in enumerate(outputs):
            h, w = SCALE_SIZES[k]
            assert out.shape == (BATCH, 4, h, w)

    def test_flow_encoders_exist(self):
        head = STTPHead(IN_CHANNELS, flow_channels=2, flow_proj_dim=16)
        for enc in head.flow_encoders:
            assert enc is not None

    def test_gradient_flow_to_flow_input(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=2).to(device)
        flow = _make_flow(device)
        flow.requires_grad_(True)

        loss = sum(o.sum() for o in head(_make_stem_features(device), flow=flow))
        loss.backward()

        assert flow.grad is not None
        assert flow.grad.abs().sum() > 0

    def test_flow_none_skips_injection(self, device):
        head = STTPHead(IN_CHANNELS, flow_channels=2).to(device)
        outputs = head(_make_stem_features(device), flow=None)
        assert len(outputs) == 3


# ── Param count ──────────────────────────────────────────────────────────

class TestParams:
    def test_design_c_has_more_params(self):
        b = STTPHead(IN_CHANNELS, flow_channels=0)
        c = STTPHead(IN_CHANNELS, flow_channels=2, flow_proj_dim=16)
        assert sum(p.numel() for p in c.parameters()) > sum(p.numel() for p in b.parameters())

    def test_per_scale_independence(self):
        head = STTPHead(IN_CHANNELS, flow_channels=0)
        assert head.fut_preds[0].weight.data_ptr() != head.fut_preds[1].weight.data_ptr()
