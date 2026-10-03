"""Tests for FlowProjectionHead (zero-parameter trivial baseline).

Validates output shape, decode-math correctness against the STTPLoss
absolute-mode decoder, and the no-flow fallback.

Run: pytest src/smcfnet/test/test_sttp_head_trivial.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import FlowProjectionHead


# YOLOX-s strides for 640×480
STRIDES = (8, 16, 32)
SCALE_SIZES = [(60, 80), (30, 40), (15, 20)]
IN_CHANNELS = 128
BATCH = 2
H_FULL, W_FULL = 480, 640


def _make_stem_features(device, batch=BATCH):
    return [
        torch.zeros(batch, IN_CHANNELS, h, w, device=device)
        for h, w in SCALE_SIZES
    ]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def test_zero_parameters():
    head = FlowProjectionHead(future_steps=5)
    assert sum(p.numel() for p in head.parameters()) == 0


def test_constructor_validates_strides():
    with pytest.raises(ValueError):
        FlowProjectionHead(future_steps=5, num_scales=3, strides=(8, 16))


def test_forward_shapes(device):
    K = 5
    head = FlowProjectionHead(future_steps=K).to(device)
    stem = _make_stem_features(device)
    flow = torch.zeros(BATCH, 2, H_FULL, W_FULL, device=device)
    outs = head(stem, flow=flow)
    assert len(outs) == 3
    for k, o in enumerate(outs):
        h, w = SCALE_SIZES[k]
        # Multi-step head packs 4*K channels (one box per step t+1..t+K).
        assert tuple(o.shape) == (BATCH, 4 * K, h, w)


def test_forward_validates_num_features(device):
    head = FlowProjectionHead(future_steps=5).to(device)
    stem = _make_stem_features(device)[:2]  # only 2 instead of 3
    flow = torch.zeros(BATCH, 2, H_FULL, W_FULL, device=device)
    with pytest.raises(ValueError):
        head(stem, flow=flow)


def test_decode_math_constant_flow(device):
    """Per step s: raw_cxcy = 0.5 + s * flow / stride for constant flow."""
    K = 5
    head = FlowProjectionHead(future_steps=K).to(device)
    stem = _make_stem_features(device)
    flow = torch.zeros(BATCH, 2, H_FULL, W_FULL, device=device)
    flow[:, 0] = 3.0   # 3 px / window in x
    flow[:, 1] = -2.0  # -2 px / window in y

    outs = head(stem, flow=flow)
    for k, stride in enumerate(STRIDES):
        h, w = SCALE_SIZES[k]
        o = outs[k].view(BATCH, K, 4, h, w)  # split packed steps
        for s in range(1, K + 1):
            expected_raw_cx = 0.5 + s * 3.0 / stride
            expected_raw_cy = 0.5 + s * -2.0 / stride
            for gy, gx in [(0, 0), (1, 5), (-1, -1)]:
                assert torch.allclose(
                    o[0, s - 1, 0, gy, gx], torch.tensor(expected_raw_cx, device=device),
                    atol=1e-4,
                ), f"scale {k}, step {s}, anchor ({gy},{gx}): raw_cx mismatch"
                assert torch.allclose(
                    o[0, s - 1, 1, gy, gx], torch.tensor(expected_raw_cy, device=device),
                    atol=1e-4,
                ), f"scale {k}, step {s}, anchor ({gy},{gx}): raw_cy mismatch"
        # raw_w, raw_h must be zero for every step (decoded w,h = stride; ADE ignores them)
        assert (o[:, :, 2] == 0).all()
        assert (o[:, :, 3] == 0).all()


def test_decode_consistent_with_sttp_loss_decoder(device):
    """Verify that FlowProjectionHead's raw output decodes to anchor + k*flow.

    Mirrors the inner loop of STTPLoss._decode_absolute, so a regression in
    either side surfaces here.
    """
    head = FlowProjectionHead(future_steps=5).to(device)
    stem = _make_stem_features(device)
    flow = torch.zeros(BATCH, 2, H_FULL, W_FULL, device=device)
    flow[:, 0] = 4.0
    flow[:, 1] = 1.5

    K = 5
    outs = head(stem, flow=flow)
    for k, stride in enumerate(STRIDES):
        h, w = SCALE_SIZES[k]
        o = outs[k][0].view(K, 4, h, w)  # [K, 4, h, w] for batch 0
        gy_grid, gx_grid = torch.meshgrid(
            torch.arange(h, device=device, dtype=o.dtype),
            torch.arange(w, device=device, dtype=o.dtype),
            indexing="ij",
        )
        for s in range(1, K + 1):
            decoded_cx = (o[s - 1, 0] + gx_grid) * stride
            decoded_cy = (o[s - 1, 1] + gy_grid) * stride
            # Expected: anchor center (g + 0.5) * stride + s * flow
            expected_cx = (gx_grid + 0.5) * stride + s * 4.0
            expected_cy = (gy_grid + 0.5) * stride + s * 1.5
            assert torch.allclose(decoded_cx, expected_cx, atol=1e-3), f"scale {k} step {s}: cx"
            assert torch.allclose(decoded_cy, expected_cy, atol=1e-3), f"scale {k} step {s}: cy"


def test_no_flow_returns_anchor_centers(device):
    """When flow is None, every step's predicted future == current anchor center."""
    K = 5
    head = FlowProjectionHead(future_steps=K).to(device)
    stem = _make_stem_features(device)
    outs = head(stem, flow=None)
    for k in range(3):
        h, w = SCALE_SIZES[k]
        o = outs[k].view(BATCH, K, 4, h, w)
        # raw_cxcy = 0.5 → decoded = (g + 0.5) * stride = anchor center, every step
        assert (o[:, :, 0] == 0.5).all()
        assert (o[:, :, 1] == 0.5).all()
        assert (o[:, :, 2] == 0).all()
        assert (o[:, :, 3] == 0).all()


def test_future_steps_scales_displacement(device):
    """Within one head, step 2k displaces twice as far as step k (linear ramp)."""
    K = 10
    flow = torch.zeros(BATCH, 2, H_FULL, W_FULL, device=device)
    flow[:, 0] = 2.0

    head = FlowProjectionHead(future_steps=K).to(device)
    stem = _make_stem_features(device)
    o = head(stem, flow=flow)[0].view(BATCH, K, 4, *SCALE_SIZES[0])  # scale-0

    step5 = o[0, 4, 0, 0, 0]   # step s=5 raw_cx at anchor (0,0)
    step10 = o[0, 9, 0, 0, 0]  # step s=10 raw_cx
    # raw_cx_s = 0.5 + s * flow / stride → displacement scales linearly with s
    assert torch.allclose(step10 - 0.5, 2 * (step5 - 0.5), atol=1e-5)
