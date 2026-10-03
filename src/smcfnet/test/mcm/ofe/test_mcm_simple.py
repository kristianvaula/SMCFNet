"""Smoke tests for MCM with the lightweight ECM estimator.

Validates that ECMFlowEstimator + MotionCompensationModule forward/backward
pass works, produces correct shapes, and that the self-supervised contrast
maximization loss is well-behaved.

Run on cluster:  pytest src/smcfnet/test/mcm/ofe/test_mcm_simple.py -v

Requires: torch (smcfnet environment).
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.ecm_ofe import ECMFlowEstimator
from smcfnet.model.mcm import MotionCompensationModule, EventWarpingModule
from smcfnet.model.losses import CMaxLoss


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _make_voxel(batch=2, bins=5, h=64, w=80, device="cpu"):
    """Synthetic voxel grid with non-trivial content."""
    return torch.randn(batch, bins, h, w, device=device)


def _make_mcm(num_bins=5, mid_channels=16, contrast_weight=1.0, smooth_weight=0.01):
    """Build MCM with the lightweight ECM estimator."""
    estimator = ECMFlowEstimator(num_bins=num_bins, mid_channels=mid_channels)
    loss_fn = CMaxLoss(contrast_weight=contrast_weight, smooth_weight=smooth_weight)
    return MotionCompensationModule(estimator=estimator, loss_fn=loss_fn)


# ── 1. Forward pass shape ────────────────────────────────────────────────

def test_forward_output_shapes(device):
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert corrected.shape == voxel.shape, (
        f"Expected corrected shape {voxel.shape}, got {corrected.shape}"
    )
    assert loss.ndim == 0, "MCM loss should be a scalar"
    assert torch.isfinite(loss), f"MCM loss is not finite: {loss.item()}"


# ── 2. Flow head output shape ────────────────────────────────────────────

def test_flow_head_shape(device):
    estimator = ECMFlowEstimator(num_bins=5, mid_channels=16).to(device)
    voxel = _make_voxel(device=device)

    flow = estimator(voxel)

    assert flow.shape == (2, 2, 64, 80), f"Expected (B,2,H,W), got {flow.shape}"
    assert torch.isfinite(flow).all(), "Flow contains non-finite values"


# ── 3. Zero-init flow ≈ identity warping ─────────────────────────────────

def test_zero_init_near_identity(device):
    """With zero-initialized flow head, corrected output should approx input."""
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    # Flow head is zero-initialized by default
    flow = mcm.estimator(voxel)
    assert flow.abs().max() < 0.01, (
        f"Zero-init flow should be near-zero, got max={flow.abs().max().item():.4f}"
    )

    corrected, _ = mcm(voxel)
    diff = (corrected - voxel).abs().max().item()
    assert diff < 0.05, (
        f"With near-zero flow, corrected should be close to input. Max diff={diff:.4f}"
    )


# ── 4. Backward pass — gradients flow ────────────────────────────────────

def test_backward_gradients(device):
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device).requires_grad_(True)

    corrected, loss = mcm(voxel)
    total = loss + corrected.sum() * 0.001  # use both outputs
    total.backward()

    # Estimator parameters should have gradients
    for name, p in mcm.estimator.named_parameters():
        assert p.grad is not None, f"No gradient for estimator param: {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"

    # Input voxel should also have gradients (end-to-end training)
    assert voxel.grad is not None, "No gradient for input voxel"


# ── 5. Loss components are sensible ──────────────────────────────────────

def test_loss_components(device):
    mcm = _make_mcm(contrast_weight=1.0, smooth_weight=0.01).to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert torch.isfinite(loss), "Loss not finite"
    assert hasattr(mcm, "_last_contrast_loss"), "Missing cached contrast loss"
    assert hasattr(mcm, "_last_smooth_loss"), "Missing cached smooth loss"


# ── 6. AMP (mixed precision) compatibility ───────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_amp_forward_backward():
    device = torch.device("cuda")
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    with torch.amp.autocast("cuda"):
        corrected, loss = mcm(voxel)

    assert torch.isfinite(loss), f"AMP loss not finite: {loss}"

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()

    for name, p in mcm.estimator.named_parameters():
        assert p.grad is not None, f"No AMP gradient for: {name}"


# ── 7. Different bin counts ──────────────────────────────────────────────

@pytest.mark.parametrize("num_bins", [1, 3, 5, 10])
def test_various_bin_counts(num_bins, device):
    mcm = _make_mcm(num_bins=num_bins).to(device)
    voxel = _make_voxel(bins=num_bins, device=device)

    corrected, loss = mcm(voxel)
    assert corrected.shape == voxel.shape
    assert torch.isfinite(loss)


# ── 8. Integration: MCM (ECM) through DualStreamYOLOX ───────────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration test best run on GPU",
)
def test_mcm_integrated_in_backbone():
    """Full forward pass: MCM (ecm) -> backbone -> neck -> head with loss + backward."""
    device = torch.device("cuda")

    from smcfnet.model.backbone import CSPDarknetDualStream
    from smcfnet.model.yolo_pafpn import YOLOPAFPN
    from smcfnet.model.yolox import DualStreamYOLOX
    from smcfnet.model.yolo_head import YOLOXHead

    depth, width = 0.33, 0.25  # nano

    mcm_instance = _make_mcm()

    backbone = CSPDarknetDualStream(
        dep_mul=depth, wid_mul=width,
        out_features=("dark3", "dark4", "dark5"),
        use_mcm=True,
        mcm_instance=mcm_instance,
    ).to(device)

    in_channels = [256, 512, 1024]
    neck = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels).to(device)
    head = YOLOXHead(num_classes=8, width=width, in_channels=in_channels).to(device)
    head.initialize_biases(1e-2)

    model = DualStreamYOLOX(backbone, neck, head).to(device)
    model._loss_mcm_weight.fill_(0.1)
    model.train()

    B, H, W = 2, 64, 80
    rgb = torch.randn(B, 3, H, W, device=device)
    events = torch.randn(B, 5, H, W, device=device)

    targets = torch.zeros(B, 3, 5, device=device)
    targets[:, 0, :] = torch.tensor([0, 30.0, 30.0, 20.0, 20.0])
    targets[:, 1, :] = torch.tensor([1, 50.0, 40.0, 15.0, 15.0])

    loss_dict = model(rgb, events, targets)

    assert "total" in loss_dict, f"Missing 'total', got: {list(loss_dict.keys())}"
    assert "loss_mcm" in loss_dict, "Missing 'loss_mcm'"

    assert torch.isfinite(loss_dict["total"]), f"Total loss not finite"
    assert loss_dict["loss_mcm"].item() != 0.0, "MCM loss is zero — not being computed?"

    loss_dict["total"].backward()

    for name, p in backbone.mcm.estimator.named_parameters():
        assert p.grad is not None, f"No gradient for backbone.mcm.estimator.{name}"


# ── 9. Flow & IWE visualization (smoke test) ─────────────────────────────

def test_flow_viz_functions(tmp_path):
    """Verify flow_viz functions produce valid PNGs without crashing."""
    from smcfnet.utils.flow_viz import visualize_flow, visualize_iwe, visualize_mcm_debug

    C, H, W = 5, 32, 40
    flow = torch.randn(2, H, W) * 2.0
    voxel_before = torch.randn(C, H, W)
    voxel_after = torch.randn(C, H, W)

    p1 = visualize_flow(flow, tmp_path / "flow.png")
    assert p1.exists() and p1.stat().st_size > 0

    p2 = visualize_iwe(voxel_before, voxel_after, tmp_path / "iwe.png")
    assert p2.exists() and p2.stat().st_size > 0

    p3 = visualize_mcm_debug(voxel_before, voxel_after, flow, tmp_path / "debug.png")
    assert p3.exists() and p3.stat().st_size > 0


# ── 10. MCM caches _last_flow for visualization ──────────────────────────

def test_mcm_last_flow_cached(device):
    """MCM.forward() should cache detached flow for downstream viz."""
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert hasattr(mcm, "_last_flow"), "MCM should cache _last_flow after forward"
    assert mcm._last_flow.shape == (2, 2, 64, 80)
    assert not mcm._last_flow.requires_grad, "_last_flow should be detached"


# ── 11. ECM mode methods (no-ops) ──────────────────────────────────

def test_ecm_mode_methods():
    """ECM has no temporal state and remains ANN-only."""
    estimator = ECMFlowEstimator(num_bins=5, mid_channels=16)
    assert estimator.get_mode() == "ann"
    estimator.set_mode("snn")  # should be no-op
    assert estimator.get_mode() == "ann"
    estimator.reset_state()  # should be no-op


# ── CLI runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
