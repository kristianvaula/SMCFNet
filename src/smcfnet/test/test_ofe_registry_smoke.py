"""Smoke forward/backward coverage for the new flow estimators (issue #111).

The PR added new OFE registry keys (PEOD Cuadrado U-Net variants + PEOD ECM) and
new streaming spiking families (ssnn_net, firenet_lif, hagenaars_firenet_lif).
None had a test that constructs and runs them and checks output shape/finiteness
plus gradient connectivity. This is the family that previously produced silent
zero-flow / no-learning, so the key guard is: a backward must reach at least one
parameter.

Gradient connectivity is probed with a ``flow.mean()`` loss on purpose: a
``flow.pow(2).mean()`` loss has zero gradient when the flow head is zero-init and
the output is exactly zero, which would falsely look like a dead network. The
``mean()`` loss backpropagates a constant signal regardless of output value, so a
zero gradient genuinely means a broken graph (detach) or fully dead spiking.

All cases are small-resolution and CPU-runnable; a CUDA-gated case exercises the
full 480x640 resolution.
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.ofe import create_flow_estimator
from smcfnet.model.ofe.sofe import StreamingSpikingFlowEstimator

# Registry keys added by the PR. cuadrado_* consume a (B, 2, T>=21, H, W) signed
# event tensor; ecm consumes a (B, 5, H, W) voxel.
CUADRADO_KEYS = [
    "cuadrado_unet_peod",
    "cuadrado_unet_bottom_sep",
    "cuadrado_unet_factorized",
]
ECM_KEY = "ecm_peod"
VOXEL_KEYS = CUADRADO_KEYS + [ECM_KEY]

SOFE_FAMILIES = ["ssnn_net", "firenet_lif", "hagenaars_firenet_lif"]


def _voxel_input(key: str, *, height: int, width: int, device="cpu") -> torch.Tensor:
    if key == ECM_KEY:
        return torch.randn(1, 5, height, width, device=device)
    return torch.randn(1, 2, 21, height, width, device=device)


def _packed_event_batch(device="cpu"):
    # Two samples packed together (x, y, t_us, polarity).
    events = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 200.0, 0.0],
            [3.0, 3.0, 900.0, 1.0],
            [3.0, 4.0, 1200.0, 1.0],
            [4.0, 4.0, 1900.0, 0.0],
            [1.0, 2.0, 100.0, 1.0],
            [2.0, 3.0, 1100.0, 0.0],
            [3.0, 2.0, 2100.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.tensor([0, 5, 8], dtype=torch.int64, device=device)
    t_start = torch.tensor([0, 0], dtype=torch.int64, device=device)
    t_end = torch.tensor([2000, 3000], dtype=torch.int64, device=device)
    t_ref = torch.tensor([2000, 3000], dtype=torch.int64, device=device)
    return events, offsets, t_start, t_end, t_ref


def _assert_finite_flow_with_grad(flow: torch.Tensor, params) -> None:
    assert flow.dim() == 4 and flow.shape[1] == 2, f"expected (B,2,H,W) flow, got {tuple(flow.shape)}"
    assert torch.isfinite(flow).all(), "flow contains NaN/Inf"
    flow.float().mean().backward()
    nonzero_grad = sum(
        1 for p in params if p.grad is not None and torch.count_nonzero(p.grad) > 0
    )
    assert nonzero_grad > 0, "backward reached no parameter (broken graph / dead network)"


@pytest.mark.parametrize("key", VOXEL_KEYS)
def test_voxel_ofe_forward_backward_cpu(key):
    est = create_flow_estimator(key)
    est.train()
    flow = est(_voxel_input(key, height=64, width=64))
    assert tuple(flow.shape) == (1, 2, 64, 64)
    _assert_finite_flow_with_grad(flow, list(est.parameters()))


@pytest.mark.parametrize("family", SOFE_FAMILIES)
def test_streaming_sofe_forward_backward_cpu(family):
    est = StreamingSpikingFlowEstimator(
        step_time_us=1000,
        partition_mode="fixed_time",
        image_height=8,
        image_width=8,
        architecture_spec={"family": family},
    )
    est.train()
    events, offsets, t_start, t_end, t_ref = _packed_event_batch()
    flow = est.forward_streaming(
        events=events,
        events_offsets=offsets,
        t_start_us=t_start,
        t_end_us=t_end,
        t_ref_us=t_ref,
        return_debug=False,
    )[0]
    assert tuple(flow.shape) == (2, 2, 8, 8)
    _assert_finite_flow_with_grad(flow, list(est.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full-res case")
@pytest.mark.parametrize("key", VOXEL_KEYS)
def test_voxel_ofe_forward_backward_full_res_cuda(key):
    est = create_flow_estimator(key).cuda()
    est.train()
    flow = est(_voxel_input(key, height=480, width=640, device="cuda"))
    assert tuple(flow.shape) == (1, 2, 480, 640)
    _assert_finite_flow_with_grad(flow, list(est.parameters()))
