"""Smoke tests for MCM with STFlowNet (spiking) estimator.

Tests cover the EWM wrapper, the full MCM pipeline (estimator + EWM + loss),
and integration through the backbone -> neck -> head pipeline.

Run on cluster:  pytest src/smcfnet/test/mcm/ofe/test_mcm_spiking.py -v

Requires: torch (smcfnet environment).
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.mcm import EventWarpingModule, MotionCompensationModule
from smcfnet.model.ofe.st_flownet import STFlowNet
from smcfnet.model.losses import CMaxLoss


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _make_voxel(batch=2, bins=5, h=64, w=80, device="cpu"):
    """Synthetic voxel grid with non-trivial content."""
    return torch.randn(batch, bins, h, w, device=device)


def _make_mcm(num_bins=5, encoder_channels=None, num_res_blocks=1,
              contrast_weight=1.0, smooth_weight=0.1, mode="ann"):
    """Helper to build MCM with the modular API."""
    if encoder_channels is None:
        encoder_channels = [16, 32, 64, 128]
    estimator = STFlowNet(
        num_bins=num_bins,
        encoder_channels=encoder_channels,
        num_res_blocks=num_res_blocks,
        mode=mode,
    )
    loss_fn = CMaxLoss(
        contrast_weight=contrast_weight,
        smooth_weight=smooth_weight,
    )
    return MotionCompensationModule(
        estimator=estimator,
        loss_fn=loss_fn,
    )


# -- 1. EWM output shape -----------------------------------------------------

def test_ewm_output_shape(device):
    ewm = EventWarpingModule().to(device)
    voxel = _make_voxel(device=device)
    flow = torch.randn(2, 2, 64, 80, device=device)

    out = ewm(voxel, flow)

    assert out.shape == (2, 5, 64, 80), f"Expected (B,5,H,W), got {out.shape}"
    assert torch.isfinite(out).all(), "EWM output contains non-finite values"


# -- 2. EWM zero-flow identity -----------------------------------------------

def test_ewm_zero_flow_identity(device):
    """Zero flow should produce near-identity output."""
    ewm = EventWarpingModule().to(device)
    voxel = _make_voxel(device=device)
    flow = torch.zeros(2, 2, 64, 80, device=device)

    out = ewm(voxel, flow)

    diff = (out - voxel).abs().max().item()
    assert diff < 1e-4, f"Zero flow should give identity warp. Max diff={diff:.6f}"


# -- 3. MCM forward shapes --------------------------------------------------

def test_mcm_forward_shapes(device):
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert corrected.shape == voxel.shape, (
        f"Expected corrected shape {voxel.shape}, got {corrected.shape}"
    )
    assert loss.ndim == 0, "MCM loss should be a scalar"
    assert torch.isfinite(loss), f"MCM loss is not finite: {loss.item()}"


# -- 4. MCM zero-init near-identity -----------------------------------------

def test_mcm_zero_init_near_identity(device):
    """With zero-initialized generator, corrected output should approx input."""
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    corrected, _ = mcm(voxel)
    diff = (corrected - voxel).abs().max().item()
    # Tolerance is looser because ConvGRU + spiking blocks
    # introduce some deviation even with zero-init generator
    assert diff < 1.0, (
        f"With zero-init generator, corrected should be close to input. Max diff={diff:.4f}"
    )


# -- 5. MCM backward gradients ----------------------------------------------

def test_mcm_backward_gradients(device):
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device).requires_grad_(True)

    corrected, loss = mcm(voxel)
    total = loss + corrected.sum() * 0.001
    total.backward()

    # Estimator parameters (inside MCM) should have gradients
    params_with_grad = 0
    for name, p in mcm.estimator.named_parameters():
        if p.grad is not None:
            params_with_grad += 1
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"

    assert params_with_grad > 0, "No estimator parameters received gradients"

    # Input voxel should also have gradients (end-to-end training)
    assert voxel.grad is not None, "No gradient for input voxel"


# -- 6. MCM loss components -------------------------------------------------

def test_mcm_loss_components(device):
    mcm = _make_mcm(contrast_weight=1.0, smooth_weight=0.1).to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert torch.isfinite(loss), "MCM loss not finite"
    assert hasattr(mcm, "_last_contrast_loss"), "Missing cached contrast loss"
    assert hasattr(mcm, "_last_smooth_loss"), "Missing cached smooth loss"
    assert torch.isfinite(mcm._last_contrast_loss), "Contrast loss not finite"
    assert torch.isfinite(mcm._last_smooth_loss), "Smooth loss not finite"
    assert mcm._last_smooth_loss >= 0, "Smooth loss should be non-negative"


# -- 7. MCM last flow cached ------------------------------------------------

def test_mcm_last_flow_cached(device):
    """MCM.forward() should cache detached flow for downstream viz."""
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    corrected, loss = mcm(voxel)

    assert hasattr(mcm, "_last_flow"), "MCM should cache _last_flow after forward"
    assert mcm._last_flow.shape == (2, 2, 64, 80)
    assert not mcm._last_flow.requires_grad, "_last_flow should be detached"


# -- 8. MCM mode switching --------------------------------------------------

def test_mcm_mode_switching(device):
    mcm = _make_mcm(mode="ann").to(device)
    voxel = _make_voxel(device=device)

    # ANN mode
    assert mcm.get_mode() == "ann"
    corrected_ann, loss_ann = mcm(voxel)
    assert torch.isfinite(loss_ann)

    # Switch to SNN mode — should propagate to estimator
    mcm.set_mode("snn")
    assert mcm.get_mode() == "snn"
    assert mcm.estimator.get_mode() == "snn"
    corrected_snn, loss_snn = mcm(voxel)
    assert torch.isfinite(loss_snn)

    # Switch back to ANN
    mcm.set_mode("ann")
    assert mcm.get_mode() == "ann"


# -- 9. AMP (mixed precision) compatibility -----------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_mcm_amp_forward_backward():
    device = torch.device("cuda")
    mcm = _make_mcm().to(device)
    voxel = _make_voxel(device=device)

    with torch.amp.autocast("cuda"):
        corrected, loss = mcm(voxel)

    assert torch.isfinite(loss), f"AMP loss not finite: {loss}"

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(loss).backward()

    has_grad = False
    for name, p in mcm.named_parameters():
        if p.grad is not None:
            has_grad = True
    assert has_grad, "No parameters received AMP gradients"


# -- 10. Various bin counts ---------------------------------------------------

@pytest.mark.parametrize("num_bins", [1, 3, 5, 10])
def test_mcm_various_bin_counts(num_bins, device):
    mcm = _make_mcm(num_bins=num_bins).to(device)
    voxel = _make_voxel(bins=num_bins, device=device)

    corrected, loss = mcm(voxel)
    assert corrected.shape == voxel.shape
    assert torch.isfinite(loss)


# -- 11. Integration: MCM through DualStreamYOLOX ----------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration test best run on GPU",
)
def test_mcm_integrated_in_backbone():
    """Full forward pass: MCM -> backbone -> neck -> head with loss + backward."""
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

    assert torch.isfinite(loss_dict["total"]), "Total loss not finite"
    assert loss_dict["loss_mcm"].item() != 0.0, "MCM loss is zero -- not being computed?"

    loss_dict["total"].backward()

    has_estimator_grad = False
    for name, p in backbone.mcm.estimator.named_parameters():
        if p.grad is not None:
            has_estimator_grad = True
    assert has_estimator_grad, "No gradients for backbone.mcm.estimator parameters"


# -- CLI runner ---------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
