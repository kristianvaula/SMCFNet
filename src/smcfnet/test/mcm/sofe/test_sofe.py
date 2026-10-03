"""Smoke tests for the STFlowNet flow estimator.

STFlowNet is a pure flow predictor: forward(voxel) -> flow (B, 2, H, W).
Loss computation and voxel warping are handled by MCM (see test_mcm_spiking.py).

Run on cluster:  pytest src/smcfnet/test/mcm/sofe/test_sofe.py -v

Requires: torch (smcfnet environment).
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe.st_flownet import STFlowNet
from smcfnet.model.spiking_blocks import (
    LIFNeuron,
    SpikingConvBlock,
    SpikingResBlock,
    surrogate_heaviside,
    clip_floor,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


def _make_voxel(batch=2, bins=5, h=64, w=80, device="cpu"):
    """Synthetic voxel grid with non-trivial content."""
    return torch.randn(batch, bins, h, w, device=device)


# -- 1. Forward pass shape ---------------------------------------------------

def test_forward_output_shapes(device):
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device)

    flow = model(voxel)

    assert flow.shape == (2, 2, 64, 80), (
        f"Expected flow shape (2, 2, 64, 80), got {flow.shape}"
    )
    assert torch.isfinite(flow).all(), "Flow contains non-finite values"


# -- 2. Flow prediction shape ------------------------------------------------

def test_flow_prediction_shape(device):
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device)

    model.reset_state()
    flow = model._predict_flow(voxel)

    assert flow.shape == (2, 2, 64, 80), f"Expected (B,2,H,W), got {flow.shape}"
    assert torch.isfinite(flow).all(), "Flow contains non-finite values"


# -- 3. Zero-init flow ~ near-zero ------------------------------------------

def test_zero_init_near_identity(device):
    """With zero-initialized generator, flow should be near zero."""
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device)

    flow = model(voxel)
    # Tolerance is loose because ConvGRU + spiking blocks introduce some deviation
    assert flow.abs().max().item() < 1.0, (
        f"With zero-init generator, flow should be near zero. Max={flow.abs().max().item():.4f}"
    )


# -- 4. Backward pass -- gradients flow --------------------------------------

def test_backward_gradients(device):
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device).requires_grad_(True)

    flow = model(voxel)
    flow.sum().backward()

    # Parameters should have gradients
    params_with_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            params_with_grad += 1
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for: {name}"

    assert params_with_grad > 0, "No parameters received gradients"

    # Input voxel should also have gradients (end-to-end training)
    assert voxel.grad is not None, "No gradient for input voxel"


# -- 5. ANN/SNN mode switching -----------------------------------------------

def test_mode_switching(device):
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
        mode="ann",
    ).to(device)
    voxel = _make_voxel(device=device)

    # ANN mode
    assert model.get_mode() == "ann"
    flow_ann = model(voxel)
    assert torch.isfinite(flow_ann).all()

    # Switch to SNN mode
    model.set_mode("snn")
    assert model.get_mode() == "snn"
    flow_snn = model(voxel)
    assert torch.isfinite(flow_snn).all()

    # Switch back to ANN
    model.set_mode("ann")
    assert model.get_mode() == "ann"


# -- 6. Surrogate gradient flows through Heaviside ---------------------------

def test_surrogate_gradient():
    x = torch.randn(4, 8, requires_grad=True)
    s = surrogate_heaviside(x)
    s.sum().backward()
    assert x.grad is not None, "No gradient through surrogate Heaviside"
    assert torch.isfinite(x.grad).all(), "Non-finite surrogate gradients"
    # Gradient should be non-zero for at least some elements
    assert x.grad.abs().sum() > 0, "All surrogate gradients are zero"


# -- 7. ClipFloor activation -------------------------------------------------

def test_clip_floor_activation():
    x = torch.randn(4, 8, requires_grad=True)
    y = clip_floor(x, 4, 1.0)  # L=4, lambda_l=1.0
    y.sum().backward()
    assert x.grad is not None, "No gradient through ClipFloor"
    # Straight-through: gradient should be exactly 1.0
    assert torch.allclose(x.grad, torch.ones_like(x.grad)), "ClipFloor STE should pass gradient through"


# -- 8. AMP (mixed precision) compatibility -----------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
def test_amp_forward_backward():
    device = torch.device("cuda")
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device)

    with torch.amp.autocast("cuda"):
        flow = model(voxel)

    assert torch.isfinite(flow).all(), f"AMP flow not finite"

    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(flow.sum()).backward()

    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            has_grad = True
    assert has_grad, "No parameters received AMP gradients"


# -- 9. Various bin counts ---------------------------------------------------

@pytest.mark.parametrize("num_bins", [1, 3, 5, 10])
def test_various_bin_counts(num_bins, device):
    model = STFlowNet(
        num_bins=num_bins, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(bins=num_bins, device=device)

    flow = model(voxel)
    assert flow.shape == (2, 2, 64, 80)
    assert torch.isfinite(flow).all()


# -- 10. Caches _last_flow for visualization ----------------------------

def test_last_flow_cached(device):
    """forward() should cache detached flow for downstream viz."""
    model = STFlowNet(
        num_bins=5, encoder_channels=[16, 32, 64, 128], num_res_blocks=1,
    ).to(device)
    voxel = _make_voxel(device=device)

    flow = model(voxel)

    assert hasattr(model, "_last_flow"), "Should cache _last_flow after forward"
    assert model._last_flow.shape == (2, 2, 64, 80)
    assert not model._last_flow.requires_grad, "_last_flow should be detached"


# -- 11. LIF neuron state reset -----------------------------------------------

def test_lif_state_reset(device):
    """LIF membrane potential resets correctly."""
    lif = LIFNeuron(alpha=0.9, theta=1.0).to(device)
    x = torch.randn(2, 8, 4, 4, device=device)

    # Run forward to build up state
    _ = lif(x)
    assert lif.v is not None, "LIF should have membrane state after forward"

    lif.reset_membrane()
    assert lif.v is None, "LIF membrane should be None after reset"


# -- CLI runner ---------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
