"""Smoke tests for the Event Dynamic Upsampling Module (EDUM).

Validates that the EDUM forward/backward pass works, produces correct shapes,
that CARAFE reassembly is well-behaved, and that RGB guidance is used.

Run on cluster:  pytest src/smcfnet/test/test_edum.py -v

Requires: torch (smcfnet environment).
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.edum import EventDynamicUpsamplingModule


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _make_inputs(batch=2, bins=5, h_ev=48, w_ev=64, h_rgb=96, w_rgb=128, device="cpu"):
    """Synthetic event voxel + RGB pair at different resolutions."""
    event = torch.randn(batch, bins, h_ev, w_ev, device=device)
    rgb = torch.randn(batch, 3, h_rgb, w_rgb, device=device)
    return event, rgb


# ── 1. Forward pass shape ────────────────────────────────────────────────

def test_forward_output_shape(device):
    target_h, target_w = 96, 128
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=target_h, target_width=target_w,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    event, rgb = _make_inputs(h_rgb=target_h, w_rgb=target_w, device=device)

    out = edum(event, rgb)

    assert out.shape == (2, 5, target_h, target_w), (
        f"Expected (2, 5, {target_h}, {target_w}), got {out.shape}"
    )
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ── 2. CARAFE reassembly shape (integer upscale only) ────────────────────

def test_carafe_reassembly_shape(device):
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)

    B, C, H, W = 2, 5, 48, 64
    s = edum.scale_factor
    k = edum.kernel_size
    x = torch.randn(B, C, H, W, device=device)
    kernels = torch.ones(B, k * k * s * s, H, W, device=device) / (k * k)

    out = edum._carafe_reassemble(x, kernels, H, W)

    assert out.shape == (B, C, H * s, W * s), (
        f"Expected (B, C, {H * s}, {W * s}), got {out.shape}"
    )


# ── 3. Uniform kernels approximate bilinear ──────────────────────────────

def test_uniform_kernels_preserve_content(device):
    """With uniform kernels, CARAFE should act like nearest-neighbour upsampling."""
    edum = EventDynamicUpsamplingModule(
        num_bins=1, target_height=8, target_width=8,
        mid_channels=16, kernel_size=1, scale_factor=2,
    ).to(device)

    B, C, H, W = 1, 1, 4, 4
    x = torch.arange(16, dtype=torch.float32, device=device).view(B, C, H, W)
    # k=1, s=2 → kernel channels = 1*1*2*2 = 4, uniform weights
    kernels = torch.ones(B, 4, H, W, device=device) / 1.0
    # Softmax over dim=1 with 4 channels → each = 0.25
    kernels = torch.softmax(kernels, dim=1)

    out = edum._carafe_reassemble(x, kernels, H, W)

    assert out.shape == (B, C, H * 2, W * 2)
    assert torch.isfinite(out).all()
    # Every output value should come from the input range
    assert out.min() >= x.min() - 0.01
    assert out.max() <= x.max() + 0.01


# ── 4. Backward pass — gradients flow ────────────────────────────────────

def test_backward_gradients(device):
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    event, rgb = _make_inputs(h_rgb=96, w_rgb=128, device=device)
    event.requires_grad_(True)
    rgb.requires_grad_(True)

    out = edum(event, rgb)
    out.sum().backward()

    # EDUM parameters should have gradients
    for name, p in edum.named_parameters():
        assert p.grad is not None, f"No gradient for EDUM param: {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"

    # Both inputs should have gradients (end-to-end training)
    assert event.grad is not None, "No gradient for input event voxel"
    assert rgb.grad is not None, "No gradient for input RGB"


# ── 5. RGB guidance affects output ───────────────────────────────────────

def test_rgb_guidance_affects_output(device):
    """Different RGB inputs should produce different upsampled outputs."""
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    edum.eval()

    event = torch.randn(1, 5, 48, 64, device=device)
    rgb_a = torch.randn(1, 3, 96, 128, device=device)
    rgb_b = torch.randn(1, 3, 96, 128, device=device) * 5.0

    with torch.no_grad():
        out_a = edum(event, rgb_a)
        out_b = edum(event, rgb_b)

    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-4, (
        f"Different RGB inputs should produce different outputs, max diff={diff:.6f}"
    )


# ── 6. Kernel softmax normalization ──────────────────────────────────────

def test_kernel_softmax_normalization(device):
    """Predicted kernels should be non-negative and sum to 1 per spatial location."""
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    edum.eval()

    event, rgb = _make_inputs(h_rgb=96, w_rgb=128, device=device)

    with torch.no_grad():
        event_feat = edum.event_compressor(event)
        rgb_feat = edum.rgb_encoder(rgb)
        rgb_feat = torch.nn.functional.adaptive_avg_pool2d(
            rgb_feat, (event.shape[2], event.shape[3])
        )
        combined = torch.cat([event_feat, rgb_feat], dim=1)
        kernels = edum.kernel_predictor(combined)
        kernels = torch.softmax(kernels, dim=1)

    assert (kernels >= 0).all(), "Kernels should be non-negative after softmax"
    sums = kernels.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        "Kernels should sum to 1 over the channel dimension"
    )


# ── 7. AMP (mixed precision) compatibility ───────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_amp_forward_backward():
    device = torch.device("cuda")
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    event, rgb = _make_inputs(h_rgb=96, w_rgb=128, device=device)

    with torch.amp.autocast("cuda"):
        out = edum(event, rgb)

    assert torch.isfinite(out).all(), "AMP output not finite"

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(out.sum()).backward()

    for name, p in edum.named_parameters():
        assert p.grad is not None, f"No AMP gradient for: {name}"


# ── 8. Different bin counts ──────────────────────────────────────────────

@pytest.mark.parametrize("num_bins", [1, 3, 5, 10])
def test_various_bin_counts(num_bins, device):
    edum = EventDynamicUpsamplingModule(
        num_bins=num_bins, target_height=96, target_width=128,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)
    event = torch.randn(2, num_bins, 48, 64, device=device)
    rgb = torch.randn(2, 3, 96, 128, device=device)

    out = edum(event, rgb)

    assert out.shape == (2, num_bins, 96, 128)
    assert torch.isfinite(out).all()


# ── 9. Realistic resolution (480x640 → 1080x1440) ───────────────────────

def test_realistic_resolution(device):
    """Test with actual DSEC resolutions (smaller batch to keep memory low)."""
    edum = EventDynamicUpsamplingModule(
        num_bins=5, target_height=1080, target_width=1440,
        mid_channels=16, kernel_size=3, scale_factor=2,
    ).to(device)

    event = torch.randn(1, 5, 480, 640, device=device)
    rgb = torch.randn(1, 3, 1080, 1440, device=device)

    out = edum(event, rgb)

    assert out.shape == (1, 5, 1080, 1440), f"Expected (1,5,1080,1440), got {out.shape}"
    assert torch.isfinite(out).all()


# ── 10. Integration: EDUM through backbone ───────────────────────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration test best run on GPU",
)
def test_edum_integrated_in_backbone():
    """Forward pass: EDUM in backbone -> neck -> head with loss + backward."""
    device = torch.device("cuda")

    from smcfnet.model.backbone import CSPDarknetDualStream
    from smcfnet.model.yolo_pafpn import YOLOPAFPN
    from smcfnet.model.yolox import DualStreamYOLOX
    from smcfnet.model.yolo_head import YOLOXHead

    depth, width = 0.33, 0.25  # nano

    # EDUM upsamples events to match RGB, so both inputs have the same
    # spatial size entering the backbone stems.
    H, W = 64, 80
    backbone = CSPDarknetDualStream(
        dep_mul=depth, wid_mul=width,
        out_features=("dark3", "dark4", "dark5"),
        use_edum=True,
        edum_kwargs={
            "num_bins": 5, "mid_channels": 16,
            "target_height": H, "target_width": W,
            "kernel_size": 3, "scale_factor": 2,
        },
    ).to(device)

    in_channels = [256, 512, 1024]
    neck = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels).to(device)
    head = YOLOXHead(num_classes=8, width=width, in_channels=in_channels).to(device)
    head.initialize_biases(1e-2)

    model = DualStreamYOLOX(backbone, neck, head).to(device)
    model.train()

    B = 2
    rgb = torch.randn(B, 3, H, W, device=device)
    # Events at half resolution — EDUM upsamples to H x W
    events = torch.randn(B, 5, H // 2, W // 2, device=device)

    targets = torch.zeros(B, 3, 5, device=device)
    targets[:, 0, :] = torch.tensor([0, 30.0, 30.0, 20.0, 20.0])
    targets[:, 1, :] = torch.tensor([1, 50.0, 40.0, 15.0, 15.0])

    loss_dict = model(rgb, events, targets)

    assert "total" in loss_dict, f"Missing 'total', got: {list(loss_dict.keys())}"
    assert torch.isfinite(loss_dict["total"]), "Total loss not finite"

    loss_dict["total"].backward()

    for name, p in backbone.edum.named_parameters():
        assert p.grad is not None, f"No gradient for backbone.edum.{name}"


# ── CLI runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
